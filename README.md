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
cp docker/.env.example docker/.env          # điền KAGGLE_USERNAME / KAGGLE_KEY
docker compose -f docker/docker-compose.yml up -d --build

# theo dõi
docker compose -f docker/docker-compose.yml run --rm monitor --out=/data/out --watch=10
docker compose -f docker/docker-compose.yml logs -f build-corpus
```

Chạy thử không đụng Kaggle:

```bash
docker run --rm -v $PWD/data:/data ai-detector/corpus-builder:latest \
  --out /data/out --cache /data/cache --sources common_voice --limit 50 --no-kaggle
```

Chạy trực tiếp (không Docker): `pip install -r requirements.txt` rồi
`python scripts/build_corpus.py --out ./data/out --cache ./data/cache --no-kaggle`.

## Nguồn dữ liệu

| Nguồn | Lấy từ | Số audio | Speaker | Recording |
|---|---|---|---|---|
| `common_voice` | HF `hataphu/common-voice-corpus-20` (parquet, 244 MB) | 8,963 | **không có** → mỗi clip 1 speaker, `speaker_known=false` | clip id |
| `vietmed` | HF `leduckhai/VietMed` (parquet, 185 MB) | 9,207 | `speaker_name` (13 speaker ở train) | `audio_name` |
| `vivos` | HF `AILAB-VNUHCM/vivos` (tar.gz, 1.47 GB) | 12,420 | thư mục `VIVOSSPK*` | tên file |

VIVOS lấy bản HuggingFace thay vì Kaggle: **cùng nội dung** nhưng không cần Kaggle
credential để tải. File mp3 của Common Voice thực tế là 32/48 kHz (dataset card ghi
16 kHz là sai) nên bước resample là bắt buộc.

Thêm nguồn FAKE (vd. VieNeu-TTS-140h) = thêm một file trong `src/corpus/sources/`
rồi đăng ký ở `registry.py`, đặt `label = 1` và `generator = "<tên generator>"`.
VieNeu-TTS-140h là dataset *gated*: cần accept terms rồi truyền `HF_TOKEN`.

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

## Output

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

Ước tính output đầy đủ ~**4.5–5 GB** WAV (~30k audio). Cache nguồn thêm ~1.9 GB.
Máy build cần ít nhất ~8 GB trống cho volume `corpus-data`.

## License

Ba nguồn có điều kiện sử dụng khác nhau. Tải về được **không** đồng nghĩa với được
re-upload công khai — mặc định `docker-compose` tạo dataset **private** (`-u`);
`--kaggle-public` chỉ dùng sau khi đã kiểm tra quyền redistribution của từng nguồn
và giữ attribution.
