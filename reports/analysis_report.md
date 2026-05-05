# CSC4007 — Lab 4 Analysis Report

> Sinh viên điền báo cáo này sau khi chạy baseline và các biến thể nâng cấp.

## 1. Thông tin chung

- Họ tên:
- MSSV:
- Lớp:
- Link GitHub repo:
- Link W&B project/run nếu có:

## 2. Baseline bắt buộc

Mô hình baseline trong Lab 4:

```text
Tokenized text → Embedding → 1-layer LSTM → Dropout → Linear classifier
```

Điền cấu hình đã chạy:

| Tham số | Giá trị |
|---|---:|
| seed |  |
| vocab_size |  |
| max_len |  |
| embed_dim |  |
| hidden_dim |  |
| num_layers | 1 |
| bidirectional | False |
| dropout |  |
| lr |  |
| batch_size |  |
| epochs_trained |  |

Kết quả baseline:

| Split | Loss | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| Validation |  |  |  |
| Test |  |  |  |

Nhận xét ngắn về baseline:

- ...

## 3. Bảng ablation

Sinh viên cần thử ít nhất 2 biến thể nâng cấp so với baseline.

| Run | model_type | bidirectional | num_layers | max_len | hidden_dim | dropout | Test Accuracy | Test Macro-F1 | Nhận xét |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| baseline_lstm | lstm | False | 1 |  |  |  |  |  |  |
| variant_1 |  |  |  |  |  |  |  |  |  |
| variant_2 |  |  |  |  |  |  |  |  |  |

## 4. So sánh công bằng

Trả lời ngắn:

1. Các run có dùng cùng dataset không?
2. Các run có dùng cùng train/validation/test split không?
3. Các run có dùng cùng seed không?
4. Metric chính để chọn mô hình là gì?
5. Có dùng test set để chọn mô hình không? Vì sao không nên?

## 5. Phân tích learning curves

Dựa vào `outputs/figures/loss_curve.png` và `outputs/figures/metric_curve.png`:

- Mô hình có dấu hiệu overfitting không?
- Validation macro-F1 cải thiện đến epoch nào?
- Có nên tăng epoch, giảm learning rate, hoặc tăng dropout không?

Nhận xét:

- ...

## 6. Confusion matrix

Dựa vào `outputs/figures/confusion_matrix.png`:

- Mô hình nhầm lớp nào nhiều hơn?
- False positive và false negative có cân bằng không?
- Điều này ảnh hưởng gì nếu triển khai vào bài toán thực tế?

Nhận xét:

- ...

## 7. Error analysis

Chọn ít nhất 10 mẫu sai từ `outputs/error_analysis/error_analysis.csv`.

| STT | Trích đoạn review | Nhãn đúng | Mô hình dự đoán | Confidence | Nguyên nhân giả định |
|---:|---|---|---|---:|---|
| 1 |  |  |  |  |  |
| 2 |  |  |  |  |  |
| 3 |  |  |  |  |  |
| 4 |  |  |  |  |  |
| 5 |  |  |  |  |  |
| 6 |  |  |  |  |  |
| 7 |  |  |  |  |  |
| 8 |  |  |  |  |  |
| 9 |  |  |  |  |  |
| 10 |  |  |  |  |  |

Gợi ý nhóm lỗi:

- phủ định;
- câu dài;
- chuyển ý bằng “but/however”;
- sarcasm/mỉa mai;
- từ hiếm hoặc tên riêng;
- review có cả ý tích cực và tiêu cực.

## 8. Kết luận

Mô hình tốt nhất của em là:

- Run name:
- Cấu hình:
- Test accuracy:
- Test macro-F1:

Giải thích vì sao mô hình này tốt hơn baseline:

- ...

## 9. Tự đánh giá

- [ ] Em đã chạy baseline LSTM.
- [ ] Em đã thử ít nhất 2 biến thể nâng cấp.
- [ ] Em đã lưu checkpoint tốt nhất.
- [ ] Em đã phân tích learning curves.
- [ ] Em đã phân tích confusion matrix.
- [ ] Em đã phân tích ít nhất 10 mẫu sai.
- [ ] Em đã commit code và report lên GitHub.
