# Đặc tả tổng quan dự án: Hệ thống huấn luyện phát hiện Audio Thật/Giả sử dụng WavLM

## 1. Mục tiêu dự án

Xây dựng một hệ thống Machine Learning hoàn chỉnh để phát hiện **audio thật (Real)** và **audio giả (Fake/Deepfake)**, tập trung vào việc xây dựng một pipeline ổn định, có khả năng mở rộng và dễ bảo trì.

Giai đoạn đầu hướng đến việc kiểm chứng toàn bộ quy trình từ xử lý dữ liệu đến huấn luyện mô hình, **không đặt mục tiêu tối ưu độ chính xác cho production**.

Toàn bộ hệ thống phải:

* Chuẩn hóa dữ liệu audio.
* Trích xuất đặc trưng bằng **WavLM Base**.
* Huấn luyện classifier trên embedding.
* Đánh giá mô hình.
* Lưu checkpoint.
* Quản lý nhiều lần huấn luyện.
* Theo dõi quá trình huấn luyện theo thời gian thực.
* Phát hiện overfitting sớm.
* Chạy hoàn toàn trong Docker.

---

# 2. Phạm vi giai đoạn 1 (MVP)

Sử dụng tập dữ liệu nhỏ để xác nhận pipeline hoạt động đúng.

Dataset đề xuất:

* **SEA-Spoof Dataset**

  * [https://huggingface.co/datasets/Jack-ppkdczgx/SEA-Spoof](https://huggingface.co/datasets/Jack-ppkdczgx/SEA-Spoof)

Quy mô ban đầu:

* 200 audio Real
* 200 audio Fake

Chia dữ liệu:

* Train: 70%
* Validation: 15%
* Test: 15%

Yêu cầu:

* Cân bằng Real/Fake.
* Nhiều speaker khác nhau.
* Không để cùng một speaker xuất hiện ở cả Train và Test (speaker-disjoint).
* Ưu tiên dữ liệu tiếng Việt khi có thể.

---

# 3. Kiến trúc tổng thể

```text
                  SEA-Spoof Dataset
                          │
                          ▼
                Data Preprocessing
      (Validate / Resample / VAD / Manifest)
                          │
                          ▼
                WavLM Base (Frozen)
                 Feature Extraction
                          │
                          ▼
                 Embedding Cache (.pt)
                          │
                          ▼
                 MLP Classifier Training
                          │
          ┌───────────────┴───────────────┐
          ▼                               ▼
     Model Evaluation             Checkpoint Saving
          │                               │
          └───────────────┬───────────────┘
                          ▼
             MLflow + TensorBoard
                          │
                          ▼
          Streamlit Realtime Dashboard
```

---

# 4. Quy trình xử lý dữ liệu

Pipeline preprocessing phải tự động thực hiện:

1. Kiểm tra file audio.
2. Loại bỏ file lỗi.
3. Chuyển Stereo → Mono.
4. Resample về 16 kHz.
5. Chuẩn hóa độ dài (3–5 giây).
6. Cắt hoặc Padding.
7. Voice Activity Detection (VAD).
8. Loại bỏ audio quá ngắn hoặc không có tiếng nói.
9. Sinh Manifest.
10. Chia Train / Validation / Test.
11. Sinh thống kê dữ liệu.

Chuẩn dữ liệu:

* WAV
* 16 kHz
* Mono
* 16-bit PCM
* 3–5 giây

---

# 5. WavLM Feature Extraction

Dự án chỉ sử dụng **WavLM Base** làm mô hình trích xuất đặc trưng.

Trong giai đoạn đầu:

* WavLM hoạt động ở chế độ **Frozen**.
* Không fine-tune.
* Chỉ dùng để sinh embedding.

Pipeline:

```text
Audio
    │
    ▼
Preprocessing
    │
    ▼
WavLM Base
(Frozen)
    │
    ▼
Frame Embedding
    │
Mean Pooling
    │
    ▼
Embedding
```

---

# 6. Embedding Cache

Embedding được sinh một lần và lưu xuống ổ cứng.

Ví dụ:

```text
embeddings/
├── train/
├── validation/
└── test/
```

Mỗi file:

```text
sample_001.pt
```

Lợi ích:

* Không phải chạy lại WavLM mỗi epoch.
* Giảm đáng kể thời gian huấn luyện.
* Cho phép thử nhiều classifier khác nhau.

---

# 7. Classifier

Classifier ban đầu sử dụng kiến trúc đơn giản:

```text
Embedding
      │
Mean Pooling
      │
Linear
      │
ReLU
      │
Dropout
      │
Linear
      │
Sigmoid
```

Huấn luyện:

* AdamW
* Binary Cross Entropy
* Early Stopping
* Learning Rate Scheduler

---

# 8. Đánh giá mô hình

Sau mỗi epoch cần đánh giá trên Validation.

Theo dõi:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* Equal Error Rate (EER)
* Confusion Matrix

Sau khi kết thúc huấn luyện:

Sinh báo cáo và lưu:

* Metrics
* ROC Curve
* Precision-Recall Curve
* Confusion Matrix
* Classification Report

---

# 9. Checkpoint

Sau mỗi epoch:

Lưu:

* Model.
* Optimizer.
* Scheduler.
* Epoch.
* Metrics.
* Config.

Hỗ trợ:

* Resume Training.
* Best Checkpoint.
* Last Checkpoint.

---

# 10. Dashboard giám sát thời gian thực

Hệ thống phải cung cấp giao diện web để theo dõi toàn bộ pipeline.

## Pipeline

Hiển thị:

* Tiến độ từng bước.
* ETA.
* Trạng thái.
* Thời gian chạy.

---

## Dataset

Hiển thị:

* Real/Fake Distribution.
* Speaker Distribution.
* Duration Distribution.
* Source Distribution.
* Audio lỗi.
* Audio bị loại.

---

## WavLM

Theo dõi:

* Tiến độ sinh embedding.
* Audio đang xử lý.
* Tốc độ xử lý.
* ETA.
* Thiết bị đang sử dụng (CPU / CUDA / MPS).
* RAM / VRAM.
* Dung lượng embedding cache.

---

## Training

Theo dõi:

* Epoch.
* Batch.
* Learning Rate.
* Train Loss.
* Validation Loss.
* Accuracy.
* Precision.
* Recall.
* F1.
* ROC-AUC.
* EER.
* ETA.

---

## Learning Curves

Hiển thị:

* Train Loss.
* Validation Loss.
* Train Accuracy.
* Validation Accuracy.
* Precision.
* Recall.
* F1.
* ROC-AUC.
* EER.

---

## Overfitting Detection

Dashboard phải tự động cảnh báo khi phát hiện dấu hiệu overfitting.

Ví dụ:

* Validation Loss tăng liên tục.
* Train Loss tiếp tục giảm.
* Train Accuracy tăng nhưng Validation Accuracy không cải thiện.
* Generalization Gap vượt ngưỡng cấu hình.

Hiển thị cảnh báo trực quan để người dùng cân nhắc dừng huấn luyện hoặc điều chỉnh tham số.

---

## Early Stopping

Hiển thị:

* Best Validation Loss.
* Current Validation Loss.
* Patience còn lại.
* Epoch tốt nhất.

---

## Prediction Samples

Hiển thị ngẫu nhiên một số mẫu validation:

* Audio.
* Prediction.
* Ground Truth.
* Probability.

Giúp đánh giá trực quan chất lượng mô hình.

---

## Resource Monitoring

Theo dõi:

* CPU.
* RAM.
* GPU/MPS.
* Disk.
* I/O.

---

## Experiment Tracking

Quản lý toàn bộ lịch sử huấn luyện:

* Hyperparameters.
* Learning Rate.
* Batch Size.
* Optimizer.
* Metrics.
* Checkpoints.
* Artifacts.
* Dataset Version.

Cho phép so sánh nhiều lần train.

---

# 11. Docker

Toàn bộ hệ thống chạy trong Docker.

Khởi động bằng một lệnh:

```bash
docker compose up
```

Hệ thống tự động:

1. Chuẩn hóa dữ liệu.
2. Sinh Manifest.
3. Trích xuất Embedding bằng WavLM.
4. Huấn luyện Classifier.
5. Validation.
6. Evaluation.
7. Lưu Checkpoint.
8. Ghi nhận Experiment.
9. Cập nhật Dashboard theo thời gian thực.

---

# 12. Cấu trúc thư mục

```text
project/
│
├── configs/
├── data/
│   ├── raw/
│   ├── processed/
│   ├── manifests/
│   └── embeddings/
│
├── checkpoints/
├── reports/
├── experiments/
│
├── src/
│   ├── preprocessing/
│   ├── dataset/
│   ├── wavlm/
│   ├── models/
│   ├── trainer/
│   ├── evaluator/
│   ├── monitoring/
│   ├── dashboard/
│   ├── api/
│   └── utils/
│
├── docker/
├── docker-compose.yml
└── README.md
```

---

# 13. Công nghệ sử dụng

| Thành phần          | Công nghệ                                        |
| ------------------- | ------------------------------------------------ |
| Deep Learning       | PyTorch                                          |
| Feature Extractor   | **WavLM Base**                                   |
| Audio Processing    | torchaudio, librosa                              |
| Dataset             | SEA-Spoof (Hugging Face)                         |
| Experiment Tracking | MLflow                                           |
| Visualization       | TensorBoard                                      |
| Dashboard           | Streamlit                                        |
| Logging             | Python Logging + Rich                            |
| System Monitoring   | psutil, pynvml (NVIDIA), MPS API (Apple Silicon) |
| Container           | Docker + Docker Compose                          |

## Định hướng phát triển

Sau khi hoàn thành giai đoạn MVP, hệ thống sẽ tiếp tục được mở rộng theo các hướng:

* **Fine-tune WavLM**: thay vì giữ WavLM ở trạng thái frozen, mở khóa một phần hoặc toàn bộ encoder để tăng khả năng học đặc trưng trên dữ liệu deepfake.
* **Nâng cấp classifier**: thay MLP bằng các kiến trúc chuyên biệt cho anti-spoofing như **AASIST**, **RawNet3** hoặc các mô hình dựa trên attention.
* **Mở rộng dữ liệu**: sử dụng toàn bộ SEA-Spoof và kết hợp thêm các bộ dữ liệu tiếng Việt hoặc đa ngôn ngữ để tăng khả năng tổng quát hóa.
* **Triển khai suy luận (Inference)**: xây dựng API dự đoán thời gian thực và tối ưu mô hình để phục vụ môi trường production.

Cách tiếp cận này giúp dự án có một nền tảng vững chắc: đơn giản ở giai đoạn đầu nhưng đủ linh hoạt để phát triển thành một hệ thống phát hiện audio deepfake quy mô lớn.


/Users/mac02/Documents/ai-detector/WavLM-Base.pt