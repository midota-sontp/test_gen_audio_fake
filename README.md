# ai-detector — corpus REAL tiếng Việt (Demo v1)

Pipeline gom **audio người thật** từ Common Voice 20, VietMed và VIVOS, lọc theo
chuẩn chất lượng, chuẩn hoá về **16 kHz mono WAV PCM 16-bit**, rồi đồng bộ lên
**Kaggle Dataset**. Chạy trong Docker, checkpoint dày, kill lúc nào cũng resume đúng chỗ.

```
Raw Audio → decode? → 2–10s? → speech ≥50%? → clipping <1%? → SHA256 trùng?
          → normalize 16k/mono/PCM16 → ghi audio → ghi metadata → checkpoint ≤10
```

## Chạy nhanh

```bash
cp docker/.env.example docker/.env          # KAGGLE_USERNAME / KAGGLE_KEY / HF_TOKEN
C="docker compose -f docker/docker-compose.yml"

$C up -d --build                     # 1. REAL: lấy MỌI audio đạt chuẩn từ 3 nguồn
                                     #    dashboard: http://<máy-docker>:8000
$C run --rm monitor --watch=10       #    hoặc theo dõi trong terminal
$C run --rm build-fake               # 2. FAKE: lấy đúng bằng số REAL đã nhận
$C run --rm export --verify          # 3. bung ra cây dataset cuối
```

Thứ tự bắt buộc: FAKE cần biết số REAL nên phải chạy sau, và chỉ chạy khi
`monitor` báo cả ba nguồn REAL đã `unit x/x`.

Chạy thử không đụng Kaggle:

```bash
docker run --rm -v $PWD/data:/data ai-detector/corpus-builder:latest \
  --out /data/out --cache /data/cache --sources common_voice --limit 50 --no-kaggle
```

Chạy trực tiếp (không Docker): `pip install -r requirements.txt` rồi
`python scripts/build_corpus.py --out ./data/out --cache ./data/cache --no-kaggle`.

## Theo dõi tiến độ

Hai cách, cùng đọc một nguồn số liệu:

```bash
$C up -d dashboard                   # UI web  -> http://<máy-docker>:8000
$C run --rm monitor --watch=10       # bản terminal, cho lúc chỉ có ssh
```

Dashboard tự làm mới mỗi 5 giây, có nút tạm dừng và nút đổi nền sáng/tối. Nội dung:

| Khối | Trả lời câu hỏi gì |
|---|---|
| Hàng chỉ số | tổng audio, REAL, FAKE, **có cân bằng chưa**, tốc độ, số bị loại, đĩa còn trống |
| Tiến độ quét theo nguồn | mỗi nguồn đã duyệt tới đâu; VieNeu tách riêng pha `index` và `write` |
| Phân bố duration REAL ↔ FAKE | hai phân bố có bám nhau không — chỗ dễ hỏng nhất của dataset |
| Chọn mẫu FAKE | mục tiêu / đã lấy / có sẵn từng bucket, phần bù chéo bucket, seed |
| Lý do bị loại | loại vì gì, rê chuột ra chi tiết theo nguồn |
| Shard & đồng bộ Kaggle | shard nào đã đẩy, đẩy lúc nào, vào dataset nào |

Mỗi biểu đồ có nút **Bảng số liệu** để đọc bằng số thay vì bằng thanh.
Server dashboard mở sqlite ở chế độ **read-only**, không xin khoá ghi nên không
làm job chậm lại hay hỏng tiến độ.

## Nguồn dữ liệu

| Nguồn | Lấy từ | Số audio | Speaker | Recording |
|---|---|---|---|---|
| `common_voice` | HF `hataphu/common-voice-corpus-20` (parquet, 244 MB) | 8,963 | **không có** → mỗi clip 1 speaker, `speaker_known=false` | clip id |
| `vietmed` | HF `leduckhai/VietMed` (parquet, 185 MB) | 9,207 | `speaker_name` (13 speaker ở train) | `audio_name` |
| `vivos` | HF `AILAB-VNUHCM/vivos` (tar.gz, 1.47 GB) | 12,420 | thư mục `VIVOSSPK*` | tên file |

VIVOS lấy bản HuggingFace thay vì Kaggle: **cùng nội dung** nhưng không cần Kaggle
credential để tải. File mp3 của Common Voice thực tế là 32/48 kHz (dataset card ghi
16 kHz là sai) nên bước resample là bắt buộc.

| Nguồn FAKE | Lấy từ | Số audio | Speaker |
|---|---|---|---|
| `vieneu_tts` | HF `pnnbao-ump/VieNeu-TTS-140h` (49 file arrow, **~24 GB**) | 74,858 → lấy đúng bằng số REAL | 193 |

Thêm generator FAKE khác = thêm một file trong `src/corpus/sources/` rồi đăng ký ở
`registry.py`, đặt `label = 1` và `generator = "<tên>"`.

## Ngưỡng lọc

| Điều kiện | Mặc định | Cờ |
|---|---|---|
| Duration | 2.0 – 10.0 s | `--min-duration` / `--max-duration` |
| Speech ratio | ≥ 0.50 (WebRTC VAD, 30 ms) | `--min-speech-ratio`, `--vad-aggressiveness` |
| Clipping ratio | < 0.01, đo trên audio **gốc** | `--max-clipping-ratio` |
| Duplicate | SHA256 audio gốc **và** SHA256 sau chuẩn hoá | — |
| Corrupt | decode lỗi / rỗng / sample-rate lạ → loại | — |
| RMS normalize | **tắt** (spec Demo v1 không yêu cầu) | `--rms-dbfs -23` để bật |

`--vad-aggressiveness` đổi kết quả khá nhiều — đo trên 300 clip Common Voice:

| aggressiveness | speech_ratio p50 | loại ở ngưỡng 0.5 | loại ở ngưỡng 0.6 |
|---|---|---|---|
| 0 | 0.758 | 1.0% | 9.3% |
| 1 | 0.754 | 2.0% | 10.0% |
| **2 (mặc định)** | 0.642 | 9.0% | 33.3% |
| 3 | 0.572 | 28.7% | 61.7% |

VietMed gần như không bị loại vì speech ratio (thoại điện thoại, nói liên tục).

> **Cảnh báo về sample rate nguồn.** VietMed là thoại điện thoại **8 kHz gốc**, còn
> Common Voice là 32/48 kHz. Upsample 8→16 kHz để lại vách phổ cứng ở 4 kHz. Vì
> toàn bộ VietMed là REAL còn FAKE (VieNeu-TTS) là full-band, model rất dễ học tắt
> theo băng thông thay vì học đặc trưng AI-generated. Cột `original_sample_rate`
> trong metadata giữ lại thông tin này — nên kiểm tra khi đánh giá, và cân nhắc
> lowpass đồng đều cả REAL lẫn FAKE khi train.

## Output trong lúc chạy

```
/data/out/
├── dist/                              # thứ được đẩy lên Kaggle
│   ├── audio/real_shard_0000.tar      # WAV 16k mono PCM16, gói trong tar không nén
│   └── metadata/real_shard_0000.jsonl
├── state/corpus.sqlite                # NGUỒN SỰ THẬT: con trỏ, hash, item, shard
├── state/progress.json                # bản chiếu cho người/monitor đọc
└── stage/                             # hardlink tới dist/, không tốn thêm đĩa
```

Mỗi dòng metadata theo schema Demo v1: `id, audio, label, speaker_id, source,
generator, recording_id, duration, sample_rate, channels, split, text, gender,
speech_ratio, rms_db, clipping_ratio, original_sample_rate, original_split,
sha256_raw, sha256_norm, shard, bytes`.

`split` để `null` — chia train/val/test speaker-disjoint là bước sau, làm trên
metadata chứ không di chuyển file.

## Theo dõi tiến độ

Hai cách, cùng đọc một nguồn dữ liệu:

```bash
$C up -d dashboard                   # web: http://<máy-docker>:8000, tự làm mới 5 giây
$C run --rm monitor --watch=10       # terminal
```

Dashboard là một trang tĩnh `dashboard/index.html` + một API JSON `/api/progress`.
Server **chỉ đọc**: sqlite mở ở chế độ read-only nên không đụng vào tiến độ của job,
và mọi lỗi phía dashboard đều bị nuốt để không bao giờ làm sập job đang chạy.
Đổi cổng bằng `DASHBOARD_PORT` trong `.env`.

## Phần FAKE — cân bằng với REAL

VieNeu-TTS-140h là dataset **gated**: accept terms tại
<https://huggingface.co/datasets/pnnbao-ump/VieNeu-TTS-140h> rồi đặt `HF_TOKEN`.
Kiểm tra tên cột trước khi chạy thật (repo không công khai schema):

```bash
docker compose -f docker/docker-compose.yml run --rm build-fake --probe
```

Chạy sau khi REAL xong — số FAKE **lấy đúng bằng số REAL đã nhận**:

```bash
docker compose -f docker/docker-compose.yml run --rm build-fake
```

Ba pha, pha nào cũng resume được (`--phase index|select|write`):

| Pha | Làm gì | Vì sao tách |
|---|---|---|
| `index` | quét cả 74,858 audio, chạy **đúng bộ lọc của REAL**, ghi sổ ứng viên vào bảng `candidates` — chưa ghi audio | chưa biết cần bao nhiêu: N chỉ xác định sau khi REAL quét xong |
| `select` | N = số REAL; chọn N ứng viên bám **phân bố duration thực tế của REAL**, chia đều cho speaker theo vòng tròn | tránh model học duration hoặc học một giọng nào đó thay vì học đặc trưng AI-generated |
| `write` | quét lại, chỉ ghi N audio đã chọn | item không được chọn bị bỏ qua **trước khi decode** nên pha này nhanh hơn pha index nhiều |

Chọn mẫu là **xác định** (`--seed`), chạy lại cho ra đúng tập cũ. Nếu một bucket
duration thiếu ứng viên, phần thiếu được bù từ bucket còn dư và ghi rõ trong
`spill` của báo cáo. Nếu tổng ứng viên < N thì báo `short_by` và FAKE sẽ ít hơn REAL
— khi đó phải giảm REAL hoặc thêm generator, script không tự ý nới ngưỡng lọc.

**Dung lượng:** mặc định giữ cache 24 GB để pha `write` đọc lại từ đĩa.
`--no-keep-cache` xoá từng file arrow sau khi xử lý (đĩa ~500 MB thay vì 24 GB,
đổi lại phải tải lại toàn bộ ở pha `write`).

## Dataset cuối (spec §12)

Tar shard chỉ là **dạng vận chuyển** — Kaggle upload 30k file WAV rời rất chậm và
hay lỗi. Bung ra cây dataset chuẩn bằng một lệnh riêng, chạy lại bao nhiêu lần cũng
được mà không đụng tới tiến độ:

```bash
docker compose -f docker/docker-compose.yml run --rm export --verify
# hoặc: python scripts/export_dataset.py --out /data/out --dest /data/dataset --verify
```

```
/data/dataset/vietnamese-audio-deepfake-demo/
├── audio/
│   ├── real/{common_voice,vietmed,vivos}/*.wav
│   └── fake/<generator>/*.wav
├── metadata/
│   ├── metadata.csv
│   ├── metadata.parquet
│   └── metadata.jsonl
└── README.md                    # thống kê + attribution, sinh tự động
```

`--verify` đối chiếu `sha256_norm` của từng file sau khi bung. Bước này cần thêm
~5 GB đĩa vì dữ liệu tồn tại đồng thời ở cả `dist/` lẫn `dataset/`.
`--metadata-only` dựng lại ba file metadata mà không đụng WAV.

## Checkpoint & resume

Cứ **10 audio nhận được** (`--checkpoint-every`) — hoặc 250 audio đã xử lý, hoặc
60 giây — pipeline chốt một lần:

1. `fsync` file tar + jsonl, ghi khối EOF tar → **shard đang mở vẫn là tar hợp lệ**;
2. **một transaction sqlite duy nhất** ghi con trỏ, hash, item, reject, offset shard;
3. ghi `progress.json` kiểu atomic (tmp → fsync → rename).

Chạy lại: đọc con trỏ `(nguồn, unit, dòng)` từ sqlite, **truncate** tar và jsonl về
đúng offset đã chốt rồi ghi tiếp. Không quét lại input, không có member tar cụt,
không trùng file. Đã test bằng `kill -9` giữa chừng: sau khi chạy lại,
số member tar = số dòng metadata = số item trong DB, không trùng id.

`SIGTERM`/`Ctrl-C` → chốt checkpoint cuối rồi thoát sạch (compose để
`stop_grace_period: 120s`). `--max-seconds` đặt ngân sách thời gian cho một lần chạy.

## Đồng bộ Kaggle — giới hạn cần biết

Kaggle **không có upload delta**: mỗi `datasets version` đẩy lại toàn bộ thư mục và
mất vài phút tạo version. Đẩy thật sự mỗi 10 audio là bất khả thi. Pipeline chia ba nhịp:

| Cái gì | Nhịp | Kích thước |
|---|---|---|
| Checkpoint xuống đĩa | **mỗi 10 audio** | — |
| Dataset `…-index` (`progress.json` + `manifest.json`) | `--push-index-every` (mặc định 120 s) | vài KB |
| Dataset `…-NNNN` (một shard) | `--push-shard-every` (mặc định 900 s) + khi shard đóng | ≤ `--shard-max-mb` (256 MB) |

**Mỗi shard là một dataset Kaggle riêng** (`<slug>-0000`, `<slug>-0001`, …). Nhờ vậy
mỗi lần đẩy chỉ tốn đúng một shard chứ không phải toàn bộ corpus đã tích luỹ — nếu
gom hết vào một dataset thì lần đẩy cuối sẽ phải upload lại cả ~5 GB.

Đẩy tay khi job đã dừng hoặc push lỗi:

```bash
python scripts/push_kaggle.py --out /data/out --owner <user> --slug vi-real-audio-demo-v1
python scripts/push_kaggle.py --out /data/out ... --retry-failed   # chỉ shard chưa đẩy
```

## Dung lượng

| Phần | Dung lượng |
|---|---|
| Cache nguồn REAL (parquet + tarball VIVOS) | ~1.9 GB |
| Cache nguồn FAKE (49 file arrow VieNeu) | **~24 GB** (hoặc ~0.5 GB với `--no-keep-cache`) |
| `dist/` — shard REAL + FAKE | ~9–10 GB |
| `dataset/` — bung ra file rời (bước export) | ~9–10 GB nữa |

Máy build nên có **~50 GB trống** cho volume `corpus-data` nếu giữ cache,
hoặc ~25 GB nếu dùng `--no-keep-cache` và xoá `dist/` sau khi export.

## License

Ba nguồn có điều kiện sử dụng khác nhau. Tải về được **không** đồng nghĩa với được
re-upload công khai — mặc định `docker-compose` tạo dataset **private** (`-u`);
`--kaggle-public` chỉ dùng sau khi đã kiểm tra quyền redistribution của từng nguồn
và giữ attribution.
