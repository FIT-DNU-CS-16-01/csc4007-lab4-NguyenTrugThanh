# Sequence Audit

Báo cáo này giúp sinh viên kiểm tra độ dài chuỗi, tỷ lệ bị cắt ngắn và tỷ lệ `<unk>`.

- Number of train examples: `21250`
- Actual vocab size: `20000`
- max_len: `256`
- Length min / median / max: `11` / `213.0` / `2819`
- Length p75 / p90: `348.0` / `561.0`
- Truncated count: `8324`
- Truncated rate: `0.3917`
- UNK rate after vocab: `0.0203`

## Gợi ý đọc kết quả

- Nếu `truncated_rate` quá cao, hãy thử tăng `max_len`.
- Nếu `unk_rate_after_vocab` quá cao, hãy thử tăng `vocab_size` hoặc cải thiện tokenizer.
- Nếu mô hình overfit, hãy thử tăng dropout, giảm hidden_dim, hoặc dùng early stopping.