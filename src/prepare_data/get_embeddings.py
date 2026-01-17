import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

from src.paths import emb_df_path


def get_encoder_model(
    sentence_encoder_name: str = "cointegrated/LaBSE-en-ru",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> SentenceTransformer:
    return SentenceTransformer(sentence_encoder_name, device=device)


def encode_on_device(
    sentences: list[str] | NDArray[np.str_], model: SentenceTransformer, batch_size: int = 32
) -> NDArray[np.float32]:
    """
    Encodes sentences into embeddings using a SentenceTransformer model, running on GPU if available.

    :param sentences: List of sentences to encode.
    :param model: SentenceTransformer model.
    :param batch_size: Batch size.
    :return: Array of sentence embeddings.
    """
    embeddings = (
        model.encode(sentences, convert_to_tensor=True, show_progress_bar=True, batch_size=batch_size).cpu().numpy()
    )
    return embeddings


def save_emb_df(emb_df: pd.DataFrame, embeddings: NDArray[np.float32]) -> pd.DataFrame:
    result_emb_df = emb_df.copy()
    result_emb_df = result_emb_df.assign(emb=embeddings.tolist())
    result_emb_df.to_pickle(emb_df_path)
    return result_emb_df
