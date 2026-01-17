import json
import re
from email import policy
from email.parser import BytesParser
from pathlib import Path

import langid
import mlflow
import numpy as np
import stanza
import torch
from bs4 import BeautifulSoup
from constants import FEATURES_SUBPATH, MODEL_SUBPATH, MODELS_LOCAL_DIR
from loguru import logger
from omegaconf import OmegaConf
from setfit import SetFitModel
from tqdm import tqdm

from settings import MLflowSettings


class MessagePreparer:
    def __init__(self, messages_to_process: Path | list[str]) -> None:
        self.messages_to_process = messages_to_process
        self._validate_messages()

    def prepare_message_data(self) -> dict[str, str]:
        source = self.messages_to_process
        messages = {}

        if isinstance(source, list):
            for i, text in tqdm(enumerate(source, start=1), total=len(source), desc="Prepare messages"):
                messages[f"message_{i}"] = text.strip()
            return messages

        if isinstance(source, Path):
            if source.is_file() and source.suffix.lower() == ".eml":
                logger.info(f"Parsing file: {source.name}")
                messages[str(source)] = self._parse_eml(source)
            elif source.is_dir():
                files = [f for f in source.rglob("*.eml") if f.is_file() and f.stat().st_size > 0]
                for file_path in tqdm(files, desc=f"Parsing .eml from {source.name}"):
                    messages[str(file_path)] = self._parse_eml(file_path)
            else:
                raise ValueError(f"Path {source} is neither a valid .eml file nor directory.")

        logger.info(f"Prepared {len(messages)} messages.")
        return messages

    def _parse_eml(self, file_path: Path) -> str:
        try:
            raw_data = file_path.read_bytes()
            msg = BytesParser(policy=policy.default).parsebytes(raw_data)  # type: ignore[arg-type]
            text_parts, html_parts = [], []

            for part in msg.walk():
                content_type = part.get_content_type()
                if content_type not in ("text/plain", "text/html"):
                    continue

                payload = part.get_payload(decode=True)
                if not payload:
                    continue

                charset = part.get_content_charset() or "utf-8"
                try:
                    decoded = payload.decode(charset, errors="replace")  # type: ignore[union-attr]
                except LookupError:
                    decoded = payload.decode("utf-8", errors="replace")  # type: ignore[union-attr]

                if content_type == "text/html" or any(
                    tag in decoded.lower() for tag in ("<html", "<body", "<div", "<p>", "href=", "<br>")
                ):
                    html_parts.append(self._text_from_html(decoded, file_name=file_path.name))
                else:
                    text_parts.append(decoded.strip())

            combined = "\n".join(filter(None, text_parts + html_parts)).strip()
            return combined
        except Exception as e:
            logger.error(f"Failed to parse {file_path.name}: {e}")
            return ""

    def _text_from_html(self, text: str, file_name: str = "") -> str:
        try:
            return " ".join(BeautifulSoup(text, "html.parser").stripped_strings)
        except Exception as e:
            logger.warning(f"HTML parsing failed in file {file_name}: {e}")
            return text

    def _validate_messages(self) -> None:
        if not isinstance(self.messages_to_process, (Path, list)):
            raise ValueError(
                f"Variable `messages_to_process` must be `Path` or `list[str]`. Got: {type(self.messages_to_process)}"
            )
        elif isinstance(self.messages_to_process, Path) and not self.messages_to_process.exists():
            raise ValueError(f"Messages dir/path {self.messages_to_process} doesn't exists")
        elif isinstance(self.messages_to_process, list) and len(self.messages_to_process) == 0:
            raise ValueError("The list of messages to process is empty")
        logger.success("Variable `messages_to_process` validated successfully")


class SentenceSplitter:
    def __init__(self, languages: list[str] = ["en", "ru"], use_gpu: bool = False, use_txt_split: bool = True) -> None:
        langid.set_languages(languages)
        self.nlp_pipelines: dict[str, stanza.Pipeline] = {}
        self._initialize_pipelines(languages, use_gpu)
        self.use_txt_split = use_txt_split

    def batch_split(self, messages: dict[str, str]) -> dict[str, list[str]]:
        results = {}
        for msg_id, text in tqdm(messages.items(), desc="Splitting sentences"):
            results[msg_id] = self._split(text)
        return results

    def _split(self, text: str) -> list[str]:
        if not text:
            return []
        try:
            sentences = []
            if self.use_txt_split:
                for line in text.splitlines():
                    sentences.extend(self._run_nlp_pipeline(line))
            else:
                sentences.extend(self._run_nlp_pipeline(text))
            return self._remove_dups(sentences)
        except Exception as e:
            logger.error(f"Sentence split error: {e}")
            return [text]

    def _run_nlp_pipeline(self, text: str) -> list[str]:
        sentences = []
        lang, _ = langid.classify(text)
        doc = self.nlp_pipelines[lang](text.strip())
        for sent in doc.sentences:
            sentences.append(self._clean_text(sent.text))
        return sentences

    def _clean_text(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"https?://\S+|www\.\S+", "[link]", text, flags=re.IGNORECASE)
        text = re.sub(r"[\ud800-\udfff]", "\ufffd", text)
        text = re.sub(r"\b[0-9a-fA-F]{5,}\b", "", text)
        text = re.sub(
            r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F700-\U0001F77F\U0001F900-\U0001F9FF]",
            "",
            text,
        )
        return text.strip()

    def _remove_dups(self, sentences: list[str]) -> list:
        return list(dict.fromkeys(sentences))

    def _initialize_pipelines(self, languages: list[str], use_gpu: bool) -> None:
        for lang in languages:
            try:
                self.nlp_pipelines[lang] = stanza.Pipeline(lang, processors="tokenize", use_gpu=use_gpu, verbose=False)
                logger.info(f"Initialized stanza pipeline for: {lang}")
            except Exception as e:
                raise ValueError(f"Failed to init stanza for {lang}: {e}")


class ModelHandler:
    def __init__(self, run_id: str, mlflow_settings: MLflowSettings, use_gpu: bool):
        self.use_gpu = use_gpu
        self.run_id = run_id

        mlflow.set_tracking_uri(mlflow_settings.tracking_uri)
        self.client = mlflow.tracking.MlflowClient()
        self._validate_run()

        self.local_run_dir = MODELS_LOCAL_DIR / run_id
        self.local_run_dir.mkdir(parents=True, exist_ok=True)

        self.local_model_dir = self.local_run_dir / MODEL_SUBPATH
        self.local_features_path = self.local_run_dir / Path(FEATURES_SUBPATH).name

    def load_model_info(self) -> tuple[SetFitModel, list[str], Path]:
        feature_names = self._load_feature_names()
        logger.info(f"Got {len(feature_names)} features")
        model = self._load_model()
        return model, feature_names, self.local_run_dir

    def _load_feature_names(self) -> list[str]:
        self._dwnld_artifact(FEATURES_SUBPATH, self.local_features_path)

        cfg = OmegaConf.load(self.local_features_path)

        features = cfg.get("features")  # type: ignore[arg-type]
        if not features:
            raise ValueError("Missing or empty 'features' section in features.yaml")
        excludes = {e.lower() for e in cfg.get("excludes") or []}  # type: ignore[arg-type]

        return [f["name"] for f in features if f.get("name", "").lower() not in excludes]

    def _load_model(self) -> SetFitModel:
        self._dwnld_artifact(MODEL_SUBPATH, self.local_model_dir)
        device = torch.device("cuda" if self.use_gpu and torch.cuda.is_available() else "cpu")
        model = SetFitModel.from_pretrained(self.local_model_dir, device=device)
        logger.success(f"SetFit model loaded on {device}.")
        return model

    def _dwnld_artifact(self, mlflow_path: str, local_path: Path) -> None:
        if local_path.exists():
            logger.info(f"Artifact already exists locally: {local_path}")
        else:
            logger.info(f"Downloading artifact for run {self.run_id} ...")
            self._dwnld_mlflow_artifact(mlflow_path)
        if not local_path.exists():
            raise FileNotFoundError(f"Artifact '{local_path}' not found")

    def _dwnld_mlflow_artifact(self, artifact_path: str) -> str:
        downloaded_path = self.client.download_artifacts(
            run_id=self.run_id, path=artifact_path, dst_path=str(self.local_run_dir)
        )
        logger.info(f"Mlflow artifact {artifact_path} downloaded to {downloaded_path}")
        return downloaded_path

    def _validate_run(self) -> None:
        try:
            run = self.client.get_run(self.run_id)
        except Exception:
            raise ValueError(f"Run '{self.run_id}' does not exist in MLflow")

        if run.info.lifecycle_stage == "deleted":
            raise ValueError(f"Run '{self.run_id}' is deleted and cannot be used")


class InferencePipeline:
    def __init__(self, splitter: SentenceSplitter, run_id: str, mlflow_settings: MLflowSettings, use_gpu: bool) -> None:
        self.model, self.feature_names, self.results_dir = ModelHandler(
            run_id, mlflow_settings, use_gpu
        ).load_model_info()
        self.splitter = splitter

    def run(self, messages: dict[str, str]) -> list[dict]:
        message_sentences = self.splitter.batch_split(messages)

        results = []
        for msg_id, sents in message_sentences.items():
            msg_info = {
                "message_id": msg_id,
                "sentences": sents,
            }
            if not sents:
                msg_info["predicted_features"] = []
            else:
                msg_info["predicted_features"] = self._predict_message(sents)
            results.append(msg_info)
        logger.success("Inference completed.")
        self._save_json(results, self.results_dir / "inference_output.json")
        return results

    def _predict_message(self, sentences: list[str]) -> list[str]:
        preds = (np.sum(self.model(sentences).numpy(), axis=0) > 0).astype(int)
        return [name for i, name in enumerate(self.feature_names) if preds[i] == 1]

    def _save_json(self, data: list[dict], output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.success(f"Results saved to {output_path}")
