import pandas as pd
import stanza
from loguru import logger
from tqdm import tqdm

from src.paths import sentences_dir
from src.prepare_data.data_processing import clean_text
from src.utils import get_config


class SentenceSplitter:
    def __init__(self) -> None:
        self.nlp_pipelines: dict[str, stanza.Pipeline] = {}
        self._initialize_pipelines()

    def process_and_save_chunks(self, df: pd.DataFrame, chunk_size: int) -> None:
        sentences_dir.mkdir(parents=True, exist_ok=True)

        for i in tqdm(range(0, len(df), chunk_size), desc="Processing chunks"):
            chunk = df.iloc[i : i + chunk_size].copy()
            chunk["sentence"] = chunk.progress_apply(self._split_text_content, axis=1)
            chunk.to_pickle(sentences_dir / f"chunk_{i}.pkl")

    def _split_text_content(self, row: pd.Series) -> str | list[str]:
        content = row["content"]
        html_content = row["html_content"]
        lang = row["lang"]

        if len(content) > 0:
            normalized_content = clean_text(content, esc_seq_only=True)
            normalized_html = clean_text(html_content, esc_seq_only=True)

            if normalized_html != normalized_content:
                combined = content + " " + html_content
                return self._split_sentences(combined, lang)

            return self._split_sentences(content, lang)

        elif len(html_content) > 0:
            return self._split_sentences(html_content, lang)

        return ""

    def _split_sentences(self, text: str, lang: str) -> list[str]:
        if not text:
            return []

        if lang not in self.nlp_pipelines:
            logger.warning(f"No pipeline for language: {lang}. Returning raw text.")
            return [text]

        try:
            nlp = self.nlp_pipelines[lang]
            doc = nlp(text)
            return [sentence.text.strip() for sentence in doc.sentences if sentence.text.strip()]
        except Exception as e:
            logger.error(f"Split sentences error: {e}")
            return [text]

    def _initialize_pipelines(self) -> None:
        config = get_config("lang_config.yaml")
        languages = config.get("languages", [])

        for lang in languages:
            try:
                stanza.download(lang)
                self.nlp_pipelines[lang] = stanza.Pipeline(lang, processors="tokenize", use_gpu=True)
                logger.info(f"Initialized stanza pipeline for language: {lang}")
            except Exception as e:
                logger.error(f"Failed to initialize stanza for {lang}: {e}")
