# Huấn luyện WavLM detector trên dataset VIVOS + Fish Speech S2

Bộ phát hiện giọng nói giả (real/fake) dùng **WavLM-base đóng băng** (frozen) làm bộ
trích đặc trưng + một **MLP** nhẹ làm bộ phân loại. Nhãn: `0 = real`, `1 = fake`.

Pipeline 5 bước, chạy từ một file config duy nhất `configs/mvp.yaml`:

```
download(ingest local) → preprocess → extract(WavLM) → train → evaluate
```

## Dữ liệu (đã có sẵn trong repo tại `dataset/`)
- 1000 real (VIVOS) + 1000 fake (Fish Speech S2 clone chính giọng speaker đó), 16kHz mono.
- `dataset/metadata/metadata.csv`: `audio_path,label,speaker,text,generator,split`.
- Đã chia **train/test speaker-disjoint sẵn** (train = speaker `VIVOSSPK*`, test = `VIVOSDEV*`,
  không trùng speaker). Pipeline **tự tách thêm val** (15%) từ speaker của train, vẫn
  speaker-disjoint → không rò rỉ danh tính speaker giữa các tập.

## Checkpoint WavLM
Không cần tải tay. Lần chạy đầu, `transformers` tự tải `microsoft/wavlm-base` (~360MB)
và cache lại. Nếu có sẵn thư mục `./models/wavlm-base` (config + `pytorch_model.bin`)
thì nó được dùng offline tự động (weights không đẩy qua git vì vượt giới hạn 100MB/file
của GitHub).

## Cách 1 — Docker (CPU-only)
```bash
docker compose up --build        # chạy trọn pipeline; cần internet lần đầu để tải WavLM
```
Kết quả nằm ở `reports/` (mount ra host). WavLM cache lưu trong volume `hf_cache`.

## Cách 2 — Native (nhanh hơn; có MPS/CUDA thì tự dùng)
```bash
python -m venv .venv && source .venv/bin/activate
pip install torch torchaudio        # bản mặc định đã có MPS trên Apple Silicon
pip install -r requirements-train.txt
python run_pipeline.py --config configs/mvp.yaml
```
Chạy lại một phần:
```bash
python run_pipeline.py --only extract train evaluate   # tái dùng data đã preprocess
python run_pipeline.py --skip download preprocess
```

## Xem tiến độ + kết quả trực tiếp trên dashboard
Bảng điều khiển realtime (tiếng Việt) đọc các artifact pipeline ghi ra: tiến độ từng
bước + ETA, đường cong loss/val_loss theo epoch, cảnh báo overfitting, và kết quả test
(EER + biểu đồ ROC/PR/confusion). Cập nhật mỗi vài giây.

- **Docker**: `docker compose up --build` chạy sẵn cả `train` lẫn `dashboard`.
  Mở **http://localhost:8501** để theo dõi trong lúc train.
- **Native**: mở terminal thứ hai (cùng venv) và chạy song song lúc train:
  ```bash
  streamlit run src/dashboard/app.py     # hoặc: python scripts/dashboard.py
  ```

## Kết quả
- `reports/metrics.json` — Accuracy / Precision / Recall / F1 / ROC-AUC / **EER** (metric chính) + ngưỡng EER.
- `reports/roc.png`, `pr.png`, `confusion_matrix.png`, `classification_report.txt`, `predictions.csv`.
- `checkpoints/best.pt` — mô hình để suy luận (kèm optimizer/scheduler, resume được).
- Lịch sử huấn luyện: `reports/history.csv`.

## Các nút chỉnh quan trọng (`configs/mvp.yaml`)
- `extract.output_layer: 6` — layer giữa của WavLM tách cue giả tốt nhất (layer cuối kém hơn);
  `extract.pooling: mean` (đổi `mean_std` để bắt động lực thời gian → nhớ đổi `model.input_dim: 1536`).
- `preprocess.target_seconds: 3.0` — độ dài chuẩn hoá (crop/pad); `min_seconds: 1.0` loại clip quá ngắn.
- `train.{epochs,batch_size,lr,weight_decay}`, `train.early_stopping` — dừng sớm theo `val_loss`.
- `dataset.val_ratio: 0.15` — tỉ lệ val tách từ train (theo speaker).

## Lưu ý
- Embedding được cache theo chỉ số ở `data/embeddings/{split}/sample_XXXX.pt` (idempotent).
  Nếu đổi số lượng mẫu hoặc cách chia split, **xoá `data/embeddings/`** trước khi extract lại.
- Đã smoke-test native 2 bước `download`+`preprocess` (train 591+591 / val 105+105 / test 304+304,
  leakage=0). Các bước `extract/train/evaluate` dùng backbone HF WavLM — chạy lần đầu trên máy đích.
