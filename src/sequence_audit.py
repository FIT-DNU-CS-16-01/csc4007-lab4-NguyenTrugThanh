from __future__ import annotations

from pathlib import Path

import numpy as np

from src.data import UNK_TOKEN, simple_tokenize
from src.utils import ensure_dir


def build_sequence_audit(train_texts: list[str], vocab: dict[str, int], max_len: int, encoded_train) -> dict:
    raw_lengths = [len(simple_tokenize(t)) for t in train_texts]
    truncated = [int(x > max_len) for x in raw_lengths]
    unk_id = vocab.get(UNK_TOKEN, 1)
    total_tokens = 0
    total_unk = 0
    for ids in encoded_train["input_ids"]:
        for tok_id in ids:
            if tok_id != 0:
                total_tokens += 1
                if tok_id == unk_id:
                    total_unk += 1
    def pct(q):
        return float(np.percentile(raw_lengths, q)) if raw_lengths else 0.0
    return {
        "num_train_examples": int(len(train_texts)),
        "vocab_size_actual": int(len(vocab)),
        "max_len": int(max_len),
        "length_min": int(min(raw_lengths)) if raw_lengths else 0,
        "length_p25": pct(25),
        "length_median": pct(50),
        "length_p75": pct(75),
        "length_p90": pct(90),
        "length_max": int(max(raw_lengths)) if raw_lengths else 0,
        "truncated_count": int(sum(truncated)),
        "truncated_rate": float(sum(truncated) / max(1, len(truncated))),
        "unk_token_count": int(total_unk),
        "non_pad_token_count": int(total_tokens),
        "unk_rate_after_vocab": float(total_unk / max(1, total_tokens)),
    }


def render_sequence_audit_md(path: str | Path, audit: dict) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    lines = [
        "# Sequence Audit",
        "",
        "Báo cáo này giúp sinh viên kiểm tra độ dài chuỗi, tỷ lệ bị cắt ngắn và tỷ lệ `<unk>`.",
        "",
        f"- Number of train examples: `{audit['num_train_examples']}`",
        f"- Actual vocab size: `{audit['vocab_size_actual']}`",
        f"- max_len: `{audit['max_len']}`",
        f"- Length min / median / max: `{audit['length_min']}` / `{audit['length_median']:.1f}` / `{audit['length_max']}`",
        f"- Length p75 / p90: `{audit['length_p75']:.1f}` / `{audit['length_p90']:.1f}`",
        f"- Truncated count: `{audit['truncated_count']}`",
        f"- Truncated rate: `{audit['truncated_rate']:.4f}`",
        f"- UNK rate after vocab: `{audit['unk_rate_after_vocab']:.4f}`",
        "",
        "## Gợi ý đọc kết quả",
        "",
        "- Nếu `truncated_rate` quá cao, hãy thử tăng `max_len`.",
        "- Nếu `unk_rate_after_vocab` quá cao, hãy thử tăng `vocab_size` hoặc cải thiện tokenizer.",
        "- Nếu mô hình overfit, hãy thử tăng dropout, giảm hidden_dim, hoặc dùng early stopping.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
