from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Generator

import numpy as np
import onnxruntime
import setfit.exporters.onnx as onnx_exporters
import torch
from loguru import logger
from sentence_transformers import SentenceTransformer
from setfit import SetFitModel
from sklearn.linear_model import LogisticRegression
from torch.nn import Module


def load_setfit_model(model_path: Path, use_cuda: bool = False) -> SetFitModel:
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    model = SetFitModel.from_pretrained(model_path, device=device)
    logger.info(f"Model loaded on {device}.")
    return model


def patched_export_onnx_setfit_model(
    setfit_model: SetFitModel, inputs: dict[str, torch.Tensor], output_path: Path, opset: int = 12
) -> None:
    input_names = list(inputs.keys())
    output_names = ["logits"]

    dynamic_axes_input = {name: {0: "batch_size", 1: "sequence"} for name in input_names}
    dynamic_axes_output = {name: {0: "batch_size"} for name in output_names}

    target = setfit_model.model_body.device
    args = tuple(value.to(target) for value in inputs.values())

    logger.info(f"Exporting ONNX model to {output_path} (opset={opset})...")
    setfit_model.eval()
    with torch.no_grad():
        torch.onnx.export(
            setfit_model,
            args=args,
            f=output_path,
            opset_version=opset,
            input_names=["input_ids", "attention_mask", "token_type_ids"],
            output_names=output_names,
            dynamic_axes={**dynamic_axes_input, **dynamic_axes_output},
            dynamo=False,
        )
    logger.success(f"ONNX model exported successfully to {output_path}.")


@contextmanager
def patched_setfit_exporter(
    new_func: Callable[[SetFitModel, dict[str, torch.Tensor], Path, int], None],
) -> Generator[None, None, None]:
    original_func = onnx_exporters.export_onnx_setfit_model
    onnx_exporters.export_onnx_setfit_model = new_func
    logger.info("Applied temporary patch for SetFit ONNX exporter.")
    try:
        yield
    finally:
        onnx_exporters.export_onnx_setfit_model = original_func
        logger.info("Restored original SetFit ONNX exporter.")


def export_onnx(
    model_body: SentenceTransformer, model_head: Module | LogisticRegression, output_path: Path, opset: int = 12
) -> None:
    with patched_setfit_exporter(patched_export_onnx_setfit_model):
        onnx_exporters.export_onnx(
            model_body=model_body,
            model_head=model_head,
            output_path=output_path,
            opset=opset,
        )


def validate_onnx_export(
    model: SetFitModel,
    session: onnxruntime.InferenceSession,
    tokenizer_inputs: dict[str, np.ndarray],
    input_text: list[str],
    atol: float = 1e-5,
) -> dict[str, np.ndarray]:
    logger.info("Validating ONNX export against Torch model outputs...")

    try:
        onnx_outputs = session.run(None, dict(tokenizer_inputs))
        onnx_preds, onnx_prob_parts = onnx_outputs
        onnx_probs = np.stack([p[:, 1] for p in onnx_prob_parts], axis=1)

        torch_probs = model.predict_proba(input_text).numpy()[:, :, 1]
        torch_preds = model(input_text).numpy()

        diff_probs = np.abs(torch_probs - onnx_probs)
        mean_diff = float(diff_probs.mean())
        max_diff = float(diff_probs.max())

        preds_equal = bool(np.array_equal(onnx_preds, torch_preds))
        probs_close = bool(np.allclose(torch_probs, onnx_probs, atol=atol))

        logger.info(f"Prediction equality: {preds_equal}")
        logger.info(f"Probabilities close:  {probs_close}")
        logger.info(f"Mean probability diff: {mean_diff:.8f}")
        logger.info(f"Max probability diff:  {max_diff:.8f}")

        if not preds_equal or not probs_close:
            logger.warning("ONNX and Torch predictions differ beyond tolerance.")

        return {
            "torch_preds": torch_preds,
            "torch_probs": torch_probs,
            "onnx_preds": onnx_preds,
            "onnx_probs": onnx_probs,
        }

    except Exception as e:
        logger.exception("Error during ONNX validation.")
        raise RuntimeError(f"ONNX validation failed due to unexpected error: {e}")
