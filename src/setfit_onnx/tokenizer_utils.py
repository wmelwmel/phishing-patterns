from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


def tokenize_input(model_path: Path, input_text: list[str]) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    inputs = tokenizer(
        input_text,
        padding=True,
        truncation=True,
        return_attention_mask=True,
        return_token_type_ids=True,
        return_tensors="np",
    )
    return inputs
