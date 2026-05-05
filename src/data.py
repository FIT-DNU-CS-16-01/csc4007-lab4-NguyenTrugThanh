from __future__ import annotations

import re
from collections import Counter
from typing import Iterable

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
LABEL_ID_TO_NAME = {0: "negative", 1: "positive"}
LABEL_NAME_TO_ID = {"negative": 0, "positive": 1, "neg": 0, "pos": 1, "0": 0, "1": 1}

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?|[^\w\s]", flags=re.UNICODE)


def simple_tokenize(text: str) -> list[str]:
    """A small tokenizer for teaching purposes."""
    text = str(text).lower().strip()
    return _TOKEN_RE.findall(text)


def normalize_label(value) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and int(value) in (0, 1):
        return int(value)
    key = str(value).strip().lower()
    if key in LABEL_NAME_TO_ID:
        return LABEL_NAME_TO_ID[key]
    raise ValueError(f"Unsupported label value: {value!r}. Expected 0/1 or negative/positive.")


def _standardize_dataframe(df: pd.DataFrame, text_col: str, label_col: str) -> pd.DataFrame:
    if text_col not in df.columns:
        raise ValueError(f"Missing text column {text_col!r}. Available columns: {list(df.columns)}")
    if label_col not in df.columns:
        raise ValueError(f"Missing label column {label_col!r}. Available columns: {list(df.columns)}")
    out = df[[text_col, label_col]].copy()
    out.columns = ["text", "label"]
    out["text"] = out["text"].astype(str)
    out["label_id"] = out["label"].apply(normalize_label)
    out["label"] = out["label_id"].map(LABEL_ID_TO_NAME)
    out = out.dropna(subset=["text", "label_id"]).reset_index(drop=True)
    return out


def _safe_stratify(series: pd.Series):
    counts = series.value_counts()
    if len(counts) < 2 or counts.min() < 2:
        return None
    return series


def _split_local_csv(df: pd.DataFrame, seed: int) -> dict[str, pd.DataFrame]:
    if len(df) < 10:
        raise ValueError("local_csv should contain at least 10 rows for a train/val/test split.")
    train_val, test = train_test_split(
        df,
        test_size=0.2,
        random_state=seed,
        stratify=_safe_stratify(df["label_id"]),
    )
    train, val = train_test_split(
        train_val,
        test_size=0.2,
        random_state=seed,
        stratify=_safe_stratify(train_val["label_id"]),
    )
    return {
        "train": train.reset_index(drop=True),
        "val": val.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }


def prepare_splits(
    name: str,
    data_path: str | None = None,
    text_col: str = "text",
    label_col: str = "label",
    max_rows: int | None = None,
    seed: int = 42,
    val_size: float = 0.15,
) -> dict[str, pd.DataFrame]:
    """Load data and return train/val/test DataFrames with text, label, label_id."""
    if name == "local_csv":
        if data_path is None:
            raise ValueError("--data_path is required when --dataset local_csv")
        df = pd.read_csv(data_path)
        df = _standardize_dataframe(df, text_col=text_col, label_col=label_col)
        if max_rows is not None:
            df = df.sample(n=min(max_rows, len(df)), random_state=seed).reset_index(drop=True)
        return _split_local_csv(df, seed=seed)

    if name == "imdb":
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError("Please install datasets: pip install datasets") from exc
        ds = load_dataset("imdb")
        train_df = pd.DataFrame(ds["train"])[["text", "label"]].copy()
        test_df = pd.DataFrame(ds["test"])[["text", "label"]].copy()
        train_df = _standardize_dataframe(train_df, text_col="text", label_col="label")
        test_df = _standardize_dataframe(test_df, text_col="text", label_col="label")
        if max_rows is not None:
            train_df = train_df.sample(n=min(max_rows, len(train_df)), random_state=seed).reset_index(drop=True)
            test_n = max(20, min(max_rows, len(test_df)))
            test_df = test_df.sample(n=test_n, random_state=seed).reset_index(drop=True)
        train_df, val_df = train_test_split(
            train_df,
            test_size=val_size,
            random_state=seed,
            stratify=_safe_stratify(train_df["label_id"]),
        )
        return {
            "train": train_df.reset_index(drop=True),
            "val": val_df.reset_index(drop=True),
            "test": test_df.reset_index(drop=True),
        }

    raise ValueError(f"Unsupported dataset: {name}")


def build_vocab(texts: Iterable[str], max_vocab_size: int = 20000, min_freq: int = 1) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(simple_tokenize(text))
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for token, freq in counter.most_common(max_vocab_size - len(vocab)):
        if freq >= min_freq and token not in vocab:
            vocab[token] = len(vocab)
    return vocab


def encode_text(text: str, vocab: dict[str, int], max_len: int) -> tuple[list[int], int, int]:
    tokens = simple_tokenize(text)
    raw_len = len(tokens)
    ids = [vocab.get(tok, vocab[UNK_TOKEN]) for tok in tokens[:max_len]]
    if not ids:
        ids = [vocab[UNK_TOKEN]]
    length = max(1, len(ids))
    if len(ids) < max_len:
        ids = ids + [vocab[PAD_TOKEN]] * (max_len - len(ids))
    return ids[:max_len], length, raw_len


def encode_dataframe(df: pd.DataFrame, vocab: dict[str, int], max_len: int) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        ids, length, raw_len = encode_text(row["text"], vocab=vocab, max_len=max_len)
        rows.append({
            "text": row["text"],
            "label": row["label"],
            "label_id": int(row["label_id"]),
            "input_ids": ids,
            "length": int(length),
            "raw_length": int(raw_len),
            "is_truncated": bool(raw_len > max_len),
        })
    return pd.DataFrame(rows)


class TextSequenceDataset(Dataset):
    def __init__(self, encoded_df: pd.DataFrame):
        self.df = encoded_df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        return {
            "input_ids": torch.tensor(row["input_ids"], dtype=torch.long),
            "length": torch.tensor(int(row["length"]), dtype=torch.long),
            "label": torch.tensor(int(row["label_id"]), dtype=torch.long),
            "index": torch.tensor(idx, dtype=torch.long),
        }


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    batch_size: int,
    seed: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        TextSequenceDataset(train_df), batch_size=batch_size, shuffle=True, generator=generator
    )
    val_loader = DataLoader(TextSequenceDataset(val_df), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TextSequenceDataset(test_df), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader
