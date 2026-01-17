import warnings

import mlflow
import onnxruntime
from dotenv import load_dotenv
from loguru import logger

from src.mlflow_helper import MLFlowHelper as ml_helper
from src.setfit_onnx.onnx_utils import export_onnx, load_setfit_model, validate_onnx_export
from src.setfit_onnx.tokenizer_utils import tokenize_input
from src.settings import MLflowSettings
from src.utils import get_config

warnings.filterwarnings("ignore")


def main() -> None:
    load_dotenv()
    logger.info("Starting ONNX export pipeline...")

    onnx_config = get_config("onnx_config.yaml")
    mlflow.set_tracking_uri(MLflowSettings().tracking_uri)

    run_id: str = onnx_config.run_id
    model_dirname: str = onnx_config.model_dirname
    use_cuda: bool = onnx_config.use_cuda
    onnx_model_name: str = onnx_config.model_name
    opset: int = onnx_config.opset
    input_text: list[str] = list(onnx_config.input_text_example)
    validate_export: bool = onnx_config.validate_export

    model_path, run_id_model_dir = ml_helper.download_mlflow_model(run_id, model_dirname)
    model = load_setfit_model(model_path, use_cuda)

    onnx_output_path = run_id_model_dir / onnx_model_name
    export_onnx(model.model_body, model.model_head, output_path=onnx_output_path, opset=opset)

    if validate_export:
        tokenizer_inputs = tokenize_input(model_path, input_text)
        session = onnxruntime.InferenceSession(str(onnx_output_path))
        validate_onnx_export(model, session, tokenizer_inputs, input_text)


if __name__ == "__main__":
    main()
