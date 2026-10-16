"""GSM8K loading and answer checking.

Every method is scored by the same two functions below, so differences in
accuracy can only come from the method, never from parsing.
"""
import re
from typing import Dict, List, Optional

_BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
_NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")


def load_gsm8k(split: str = "test", n: Optional[int] = None, offset: int = 0) -> List[Dict]:
    """Return [{'id', 'question', 'gold'}]. Use the same n/offset for every method."""
    from datasets import load_dataset  # imported lazily so tests run without it

    ds = load_dataset("openai/gsm8k", "main", split=split)
    end = len(ds) if n is None else min(len(ds), offset + n)
    items = []
    for i in range(offset, end):
        row = ds[i]
        items.append({
            "id": f"gsm8k-{split}-{i}",
            "question": row["question"],
            "gold": normalize_number(row["answer"].split("####")[-1]),
        })
    return items


def normalize_number(text: str) -> Optional[str]:
    """'$1,234.50.' -> '1234.5'. Returns None if no number is present."""
    if text is None:
        return None
    match = _NUMBER.findall(text.replace("$", ""))
    if not match:
        return None
    raw = match[-1].replace(",", "").rstrip(".")
    try:
        value = float(raw)
    except ValueError:
        return None
    return str(int(value)) if value == int(value) else str(value)


def extract_prediction(output: str) -> Optional[str]:
    """Prefer the last \\boxed{...}; fall back to the last number in the text."""
    if not output:
        return None
    boxed = _BOXED.findall(output)
    if boxed:
        value = normalize_number(boxed[-1])
        if value is not None:
            return value
    return normalize_number(output)


def is_correct(pred: Optional[str], gold: Optional[str]) -> bool:
    if pred is None or gold is None:
        return False
    try:
        return abs(float(pred) - float(gold)) < 1e-6
    except ValueError:
        return False
