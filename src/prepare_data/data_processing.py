import re
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from numpy.typing import NDArray
from tqdm import tqdm

from src.paths import eml_df_path, new_data_labels_path


def get_new_data_labels(labels_df_path: Path) -> pd.DataFrame:
    labels_df = pd.read_csv(labels_df_path, sep=";")
    labels_df = labels_df.rename(
        columns={
            "Column1": "filename",
            "Column2": "hash",
            "Column3": "headers",
            "Column4": "rank",
            "Column5": "verdict",
        }
    )
    return labels_df[~labels_df.filename.duplicated()]


def map_new_data_labels(new_data_df: pd.DataFrame, labels_df: pd.DataFrame) -> pd.DataFrame:
    new_data_df = new_data_df.assign(filename=new_data_df.file_path.apply(lambda x: Path(x).stem))
    new_data_df = pd.merge(new_data_df, labels_df[["filename", "verdict"]], how="left")

    without_label = sum(new_data_df.verdict.isna())
    logger.info(f"Files without label: {without_label} ({without_label / len(new_data_df) * 100:.2f}%)")

    new_data_df["verdict"] = new_data_df.verdict.fillna("clean")
    label_mapping = {
        "clean": 0,
        "suspicious": -1,
    }
    new_data_df["label"] = new_data_df["verdict"].map(label_mapping).fillna(1)
    return new_data_df


def get_df_mails_statistics(df: pd.DataFrame) -> pd.DataFrame:
    df_mails = df.copy()
    df_mails["content_length"] = df_mails["content"].apply(len)
    df_mails["html_content_length"] = df_mails["html_content"].apply(lambda x: len(x) if isinstance(x, str) else None)

    total_records = len(df_mails)
    clean_count = len(df_mails[df_mails["label"] == 0])
    phishing_count = len(df_mails[df_mails["label"] == 1])
    susp_count = len(df_mails[df_mails["label"] == -1])
    error_count = len(df_mails[df_mails.get("error", "").notna()])

    avg_content_len = df_mails["content_length"].mean()
    median_content_len = df_mails["content_length"].median()
    avg_html_len = df_mails["html_content_length"].mean()
    median_html_len = df_mails["html_content_length"].median()

    output = f"""
    {"=" * 50}
    EMAIL DATASET STATISTICS
    {"=" * 50}
    Total records: {total_records:_}
    - Clean (0): {clean_count:_}
    - Phishing (1): {phishing_count:_}
    - Suspicious (-1): {susp_count:_}

    Files with errors: {error_count:_}

    Text content length:
      - Average: {avg_content_len:_.0f} characters
      - Median: {median_content_len:_.0f} characters

    HTML content length:
      - Average: {avg_html_len:_.0f} characters
      - Median: {median_html_len:_.0f} characters
    {"=" * 50}
    """

    logger.info(output)
    logger.info("Label distribution:\n{}", df_mails.label.value_counts(normalize=True).to_string())

    return df_mails


def get_df_mails(
    old_emls_parsed: dict, new_emls_parsed: dict, new_ru_emls_parsed: dict, save_df: bool = True
) -> pd.DataFrame:
    df_old_emls = pd.DataFrame(old_emls_parsed)
    df_new_emls = pd.DataFrame(new_emls_parsed)
    df_new_ru_emls = pd.DataFrame(new_ru_emls_parsed)

    labels_df = get_new_data_labels(new_data_labels_path)
    df_new_emls = map_new_data_labels(df_new_emls, labels_df)

    eml_df = pd.concat([df_old_emls, df_new_emls, df_new_ru_emls])
    eml_df = get_df_mails_statistics(eml_df)

    if save_df:
        eml_df.to_pickle(eml_df_path)
        logger.info(f"df_mails saved to {eml_df_path}")

    return eml_df


def filter_df_mails(df_mails: pd.DataFrame, delete_susp: bool = True, cutoff_q: float = 0.95) -> pd.DataFrame:
    df_filtered = df_mails.copy()

    empty_mask = (df_filtered.content_length == 0) & (df_filtered.html_content_length == 0)
    df_filtered = df_filtered[~empty_mask]
    logger.info(f"Filtered {empty_mask.sum()} rows as empty mails")

    if delete_susp:
        susp_mask = df_filtered.label == -1
        df_filtered = df_filtered[~susp_mask]
        logger.info(f"Filtered {susp_mask.sum()} rows as suspicious mails")

    dup_mask = df_filtered.duplicated(subset=["content", "html_content"])
    df_filtered = df_filtered[~dup_mask]
    logger.info(f"Filtered {dup_mask.sum()} rows as duplicated content mails")

    html_content_length_cutoff = df_filtered.html_content_length.quantile(cutoff_q)
    content_length_cutoff = df_filtered.content_length.quantile(cutoff_q)

    logger.info(f"Content length cutoff ({cutoff_q}): {content_length_cutoff}")
    logger.info(f"HTML content length cutoff ({cutoff_q}): {html_content_length_cutoff}")

    df_filtered = df_filtered[df_filtered.content_length < content_length_cutoff]
    df_filtered = df_filtered[df_filtered.html_content_length < html_content_length_cutoff]

    logger.info("Label distribution:\n{}", df_filtered.label.value_counts().to_string())
    return df_filtered


def clean_df_content(df: pd.DataFrame, esc_seq_only: bool = False) -> pd.DataFrame:
    df_cleaned = df.copy()
    df_cleaned["content"] = df_cleaned["content"].progress_apply(lambda x: clean_texts(x, esc_seq_only))
    df_cleaned["html_content"] = df_cleaned["html_content"].progress_apply(lambda x: clean_texts(x, esc_seq_only))
    df_cleaned = get_df_mails_statistics(df_cleaned)
    return df_cleaned


def remove_duplicates_preserve_order(arr: NDArray[np.str_]) -> NDArray[np.str_]:
    return np.array(list(dict.fromkeys(arr)))


def clean_text(text: str, esc_seq_only: bool) -> str:
    if esc_seq_only:
        text = re.sub(r"\s+", " ", text)
    else:
        text = re.sub(r"https?://\S+|www\.\S+", "[link]", text, flags=re.IGNORECASE)
        text = re.sub(r"[\ud800-\udfff]", "\ufffd", text)
        text = re.sub(r"\b[0-9a-fA-F]{5,}\b", "", text)
        text = re.sub(
            r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F700-\U0001F77F\U0001F900-\U0001F9FF]",
            "",
            text,
        )
    return text.strip()


def clean_texts(input_data: str | list[str], esc_seq_only: bool = True) -> str | list:
    if isinstance(input_data, str):
        return clean_text(input_data, esc_seq_only)

    elif isinstance(input_data, (list, tuple, np.ndarray)):
        return [clean_texts(item, esc_seq_only) for item in input_data]

    else:
        return clean_text(str(input_data), esc_seq_only)


def get_sentence_df(sentences_dir: Path) -> pd.DataFrame:
    chunk_files = list(sentences_dir.rglob("*.pkl"))
    return pd.concat([pd.read_pickle(f) for f in tqdm(chunk_files)], ignore_index=True)


def clean_sentence_col(df_column: pd.Series) -> pd.Series:
    return df_column.apply(clean_texts).apply(remove_duplicates_preserve_order)


def prepare_sentence_df(df: pd.DataFrame, cutoff_q: float = 0.95) -> pd.DataFrame:
    include_columns = ["file_path", "label", "content_length", "sentence", "lang"]
    df_sentences = df.copy()
    df_sentences = df_sentences[include_columns]

    df_sentences["sentence"] = clean_sentence_col(df_sentences["sentence"])

    df_sentences = df_sentences.assign(sentences_count=df_sentences.sentence.apply(len))

    msg_count_cutoff = df_sentences.sentences_count.quantile(cutoff_q)
    files_count_cutoff = (df_sentences.sentences_count >= msg_count_cutoff).sum()
    logger.info(f"Sentences cutoff ({cutoff_q}): {msg_count_cutoff:.2f} ({files_count_cutoff} files)")
    df_sentences = df_sentences[df_sentences.sentences_count < msg_count_cutoff]

    return df_sentences


def explode_sentence_df(df_sentences: pd.DataFrame, min_unique_chars_count: int = 5) -> pd.DataFrame:
    df_sentences_exp = (
        df_sentences.reset_index().drop(columns=["content_length", "sentences_count"]).explode("sentence")
    )
    df_sentences_exp = df_sentences_exp.assign(unique_chars_count=df_sentences_exp.sentence.apply(set).apply(len))
    return df_sentences_exp[df_sentences_exp.unique_chars_count >= min_unique_chars_count]


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text
