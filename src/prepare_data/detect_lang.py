import pandas as pd
from langdetect import DetectorFactory, LangDetectException, detect

from src.logger import logger
from src.paths import lang_df_path
from src.utils import get_config

DetectorFactory.seed = 42


def detect_lang(text: str) -> str | None:
    if not text or not isinstance(text, str) or text.strip() == "":
        return None
    try:
        return detect(text)
    except LangDetectException as e:
        logger.warning(f"Language detection failed for text: '{text[:100]}[TRUNCATED]'. Error: {e}")
        return None


def row_detect_lang(row: pd.Series) -> str | None:
    lang = detect_lang(row["content"])

    if not lang:
        lang = detect_lang(row["html_content"])

    return lang


def get_content_lang(df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    df_lang = df.copy()
    df_lang["lang"] = df_lang.progress_apply(row_detect_lang, axis=1)
    langs_c = len(df_lang["lang"].unique())
    logger.info(f"Found {langs_c} languages in content")
    logger.info(
        f"Language distribution (top_{top_n}):\n" + df_lang.lang.value_counts(normalize=True).head(top_n).to_string()
    )
    df_lang.to_pickle(lang_df_path)
    logger.info(f"df_lang saved to {lang_df_path}")
    return df_lang


def get_valid_languages(df_mails: pd.DataFrame) -> pd.DataFrame:
    valid_langs = get_config("lang_config.yaml")["languages"]
    logger.info(f"Valid languages: {valid_langs}")

    df_mails_cleaned = df_mails.copy()
    if len(valid_langs) > 0:
        df_mails_cleaned["valid_language"] = df_mails_cleaned["lang"].isin(valid_langs)
    else:
        df_mails_cleaned["valid_language"] = True
    logger.info(f"Language detection error files: {(df_mails_cleaned.lang.isna()).sum()}")
    logger.info(f"Not valid languages files: {(df_mails_cleaned.valid_language == False).sum()}")  # noqa: E712

    len_before = len(df_mails_cleaned)
    df_mails_cleaned = df_mails_cleaned[~(df_mails_cleaned.valid_language == False)]  # noqa: E712
    len_after = len(df_mails_cleaned)

    logger.info(f"Filtered {len_before - len_after} rows as not valid language messages")
    logger.info("Language distribution:\n" + df_mails_cleaned.lang.value_counts(normalize=True).to_string())
    return df_mails_cleaned
