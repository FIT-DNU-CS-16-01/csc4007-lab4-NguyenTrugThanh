# CSC4007 — Lab 4 Analysis Report

> Sinh viên điền báo cáo này sau khi chạy baseline và các biến thể nâng cấp.

## 1. Thông tin chung

- Họ tên: Nguyễn Trung Thành
- MSSV: 1671040025
- Lớp: KHMT-1601
- Link GitHub repo: https://github.com/NguyenTrugThanh
- Link W&B project/run nếu có: https://wandb.ai/thanhfifo24-vietcombank/csc4007-lab4-lstm-gru?nw=nwuserthanhfifo24

## 2. Baseline bắt buộc

Mô hình baseline trong Lab 4:

```text
Tokenized text → Embedding → 1-layer LSTM → Dropout → Linear classifier
```

Điền cấu hình đã chạy:

| Tham số | Giá trị |
|---|---:|
| seed | 42 |
| vocab_size | 20000 |
| max_len | 256 |
| embed_dim | 128 |
| hidden_dim | 128 |
| num_layers | 1 |
| bidirectional | False |
| dropout | 0.3 |
| lr | 0.001 |
| batch_size | 64 |
| epochs_trained | 6 |

Kết quả baseline:

| Split | Loss | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| Validation | 0.4397 | 0.8181 | 0.8181 |
| Test | 0.4495 | 0.8156 | 0.8155 |

Nhận xét ngắn về baseline:

- Mô hình LSTM 1 lớp đạt độ chính xác 81.56% trên test set, cho thấy hiệu suất khá tốt.
- Validation Macro-F1 đạt cao nhất ở epoch 5 (0.8181), sau đó giảm nhẹ ở epoch 6, cho thấy dấu hiệu overfitting nhẹ.
- Mô hình hội tụ nhanh chóng, với validation loss giảm mạnh trong 5 epoch đầu tiên.
- Dropout 0.3 và weight decay 0.0001 giúp kiểm soát overfitting hiệu quả.

## 3. Bảng ablation

Sinh viên cần thử ít nhất 2 biến thể nâng cấp so với baseline.

| Run | model_type | bidirectional | num_layers | max_len | hidden_dim | dropout | Test Accuracy | Test Macro-F1 | Nhận xét |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| baseline_lstm | lstm | False | 1 | 256 | 128 | 0.3 | 0.8156 | 0.8155 | Baseline |
| variant_1 |  |  |  |  |  |  |  |  | *Điền kết quả chạy thứ 2* |
| variant_2 |  |  |  |  |  |  |  |  | *Điền kết quả chạy thứ 3* |

## 4. So sánh công bằng

Trả lời ngắn:

1. Các run có dùng cùng dataset không? **Có, tất cả đều dùng IMDB dataset với cùng split: train=21250, val=3750, test=25000**
2. Các run có dùng cùng train/validation/test split không? **Có, tất cả dùng cùng splits từ seed=42**
3. Các run có dùng cùng seed không? **Có, tất cả dùng seed=42 để đảm bảo tính tái tạo được**
4. Metric chính để chọn mô hình là gì? **Validation Macro-F1 - chọn checkpoint có Macro-F1 cao nhất trên validation set**
5. Có dùng test set để chọn mô hình không? Vì sao không nên? **Không, vì test set là vô hình (unseen data). Nếu dùng test set để chọn mô hình sẽ dẫn đến data leakage và ước lượng hiệu suất bị lạc quan hóa (overfitting trên test set)**

## 5. Phân tích learning curves

Dựa vào `outputs/figures/loss_curve.png` và `outputs/figures/metric_curve.png`:

- Mô hình có dấu hiệu overfitting không? **Có dấu hiệu nhẹ**: Val loss tăng từ epoch 5 (0.4397) lên epoch 6 (0.4447), val macro-F1 giảm từ 0.8181 xuống 0.8103
- Validation macro-F1 cải thiện đến epoch nào? **Epoch 5** (đạt 0.8181, cao nhất), sau đó giảm ở epoch 6
- Có nên tăng epoch, giảm learning rate, hoặc tăng dropout không? **Dừng tại epoch 5 là hợp lý (early stopping đã được áp dụng với patience=2)**

Nhận xét:

- Training loss giảm liên tục từ 0.65 (epoch 1) xuống 0.29 (epoch 6), cho thấy mô hình học tốt
- Validation loss ổn định quanh 0.44 từ epoch 4 trở đi, không giảm thêm nhiều
- Sự khác biệt nhỏ giữa train loss và val loss cho thấy dropout (0.3) hoạt động tốt
- Early stopping với patience=2 giúp tránh overfitting, checkpoint tốt nhất được lưu ở epoch 5

## 6. Confusion matrix

Dựa vào `outputs/figures/confusion_matrix.png`:

- Mô hình nhầm lớp nào nhiều hơn? **False Positive** (dự đoán POSITIVE nhưng là NEGATIVE) chiếm ~9% (2,242/25,000), tức là mô hình có xu hướng đánh giá quá tích cực
- False positive và false negative có cân bằng không? **Không hoàn toàn cân bằng**: FP ≈ 2,242 trong khi FN ≈ 2,147, FP cao hơn FN nhẹ
- Điều này ảnh hưởng gì nếu triển khai vào bài toán thực tế? **Nếu ứng dụng vào bài toán kinh doanh**: FP cao có thể dẫn đến lãng phí tài nguyên (ví dụ: gợi ý sản phẩm cho khách không hài lòng), cần cân nhắc threshold dự đoán

Nhận xét:

- Mô hình cân bằng khá tốt giữa 2 lớp (Accuracy = Sensitivity ≈ Specificity ≈ 81.5%)
- Tỷ lệ lỗi thấp cho thấy LSTM học được các đặc trưng ngôn ngữ tốt
- Có thể cải thiện bằng: tăng mô hình phức tạp, điều chỉnh class weight, hoặc thay đổi decision threshold

Chọn ít nhất 10 mẫu sai từ `outputs/error_analysis/error_analysis.csv`.

| STT | Trích đoạn review | Nhãn đúng | Mô hình dự đoán | Confidence | Nguyên nhân giả định |
|---:|---|---|---|---:|---|
| 1 | "Whew! What can one say about this bizarre, stupefying mock-u-mentary... a plea for tolerance" | Negative | Positive | 0.989 | False Positive: Review chỉ trích kịch liệt nhưng dùng từ "plea for tolerance" dương tính, mô hình nhầm |
| 2 | "Yes, it might be not historically accurate... it's strongest point in showing soldier's life" | Negative | Positive | 0.988 | False Positive: Đoạn khoá "it's strongest point" có tone dương tính, làm mô hình nhầm |
| 3 | "Curse of Michael Myers is very frustrating... It's too bad, as this is the last movie for Donald Pleasance" | Negative | Positive | 0.984 | False Positive: Từ "frustrating" được context xung quanh làm giảm tác động, mô hình không bắt được |
| 4 | "China Syndrome claim... Before you try to find weak point in the film, you should watch it first!!!!" | Positive | Negative | 0.983 | False Negative: Kết luận tích cực "should watch it first" nhưng trước đó có nhiều lời chỉ trích |
| 5 | "Viewers of independent films... this one paled in comparison" | Negative | Positive | 0.983 | False Positive: Có từ dương tính "fan", "good", nhưng kết luận tiêu cực |
| 6 | "The Comic Strip featured actors... The soundtrack features great music... 5/10" | Negative | Positive | 0.982 | False Positive: Có "great music" nhưng điểm thấp (5/10) cho thấy review tiêu cực |
| 7 | "It would appear some reviewers had expectations set too high... was pleasantly surprised" | Positive | Negative | 0.982 | False Negative: "pleasantly surprised" là tích cực nhưng mô hình bắt từ "disappointing" trước đó |
| 8 | "Lonely, disconnected... tropical storm makes them true lovers" | Negative | Positive | 0.980 | False Positive: Có từ lãng mạn "lovers" nhưng review chỉ trích diễn xuất và cốt truyện |
| 9 | "This film doesn't have a clear picture... By the last fifteen minutes, plot twists are clichés" | Negative | Positive | 0.980 | False Positive: Mô hình bị confuse bởi từ tích cực nhưng context tiêu cực |
| 10 | "Last night I got to see early preview... Christina Ricci performance was fantastic... rate 4/10" | Negative | Positive | 0.978 | False Positive: "fantastic" làm mô hình dự đoán dương tính, nhưng kết quả 4/10 là tiêu cực |

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
