# Rubric — CSC4007 Lab 4: LSTM/GRU & Long-range Dependencies

Tổng điểm: 100

## 1. Chạy đúng baseline LSTM — 20 điểm

| Tiêu chí | Điểm |
|---|---:|
| Cài đặt và chạy được pipeline end-to-end | 5 |
| Dùng đúng baseline Embedding + 1-layer LSTM | 5 |
| Lưu đầy đủ checkpoint, predictions, metrics | 5 |
| Cố định seed và mô tả split dữ liệu | 5 |

## 2. Thử nghiệm nâng cấp và ablation — 25 điểm

| Tiêu chí | Điểm |
|---|---:|
| Có ít nhất 2 biến thể nâng cấp | 8 |
| So sánh công bằng cùng split/seed/metric | 7 |
| Bảng ablation rõ ràng, có nhận xét | 7 |
| Biết giải thích vì sao một biến thể tốt/kém hơn | 3 |

## 3. Đánh giá mô hình — 20 điểm

| Tiêu chí | Điểm |
|---|---:|
| Báo cáo accuracy và macro-F1 | 5 |
| Có confusion matrix | 5 |
| Có learning curves | 5 |
| Chọn mô hình bằng validation, không chọn bằng test | 5 |

## 4. Error analysis — 15 điểm

| Tiêu chí | Điểm |
|---|---:|
| Phân tích ít nhất 10 mẫu sai | 5 |
| Nhận diện nhóm lỗi hợp lý | 5 |
| Liên hệ lỗi với giới hạn của LSTM/GRU/tokenizer/max_len | 5 |

## 5. Báo cáo và trình bày repo — 15 điểm

| Tiêu chí | Điểm |
|---|---:|
| Điền đầy đủ `reports/analysis_report.md` | 5 |
| Repo sạch, có commit rõ ràng | 4 |
| Có link W&B hoặc log cấu hình thí nghiệm | 3 |
| Kết luận có căn cứ từ kết quả | 3 |

## 6. Bonus — tối đa 5 điểm

| Tiêu chí | Điểm |
|---|---:|
| Thử scheduler, pretrained embeddings, hoặc tokenizer tốt hơn | +2 |
| Có phân tích thêm về câu dài/phủ định/chuyển ý | +2 |
| Có script tổng hợp nhiều run thành bảng ablation tự động | +1 |
