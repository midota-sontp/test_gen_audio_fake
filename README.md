# ai-detector — corpus REAL tiếng Việt (Demo v1)

Pipeline gom **audio người thật** từ Common Voice 20, VietMed và VIVOS, lọc theo
chuẩn chất lượng, chuẩn hoá về **16 kHz mono WAV PCM 16-bit**, rồi đồng bộ lên
**Kaggle Dataset**. Chạy trong Docker, checkpoint dày, kill lúc nào cũng resume đúng chỗ.

```
Raw Audio → decode? → 2–10s? → speech ≥50%? → clipping <1%? → SHA256 trùng?
          → normalize 16k/mono/PCM16 → ghi audio → ghi metadata → checkpoint ≤10
```

## Chạy nhanh

Yêu cầu máy build: Docker + **~50 GB trống** (24 GB trong đó là cache VieNeu).

```bash
git clone <repo> && cd ai-detector
cp docker/.env.example docker/.env      # rồi điền KAGGLE_API_TOKEN và HF_TOKEN
C="docker compose -f docker/docker-compose.yml"

$C build                          # 0. dựng image
$C up -d corpus dashboard         # 1. REAL — chạy nền, kèm UI :8000
$C run --rm cli fake --probe      # 2. kiểm tên cột VieNeu trước
$C run --rm cli fake              #    FAKE — lấy đúng bằng số REAL
$C run --rm cli export --verify   # 3. bung ra cây dataset cuối
$C run --rm cli split             # 4. chia train/validation/test
$C run --rm cli push              # 5. đẩy lên Kaggle — MỘT dataset
```

Chỉ có **3 service**: `corpus` (dựng REAL, chạy nền), `dashboard` (UI), và `cli`
(mọi việc còn lại, chạy một lần rồi thoát). `docker compose run --rm cli` không
tham số sẽ liệt kê các lệnh con.

Thứ tự bắt buộc: FAKE cần biết số REAL nên phải chạy **sau**, và chỉ chạy khi
dashboard báo cả ba nguồn REAL đã xong (`unit x/x`, thanh đầy 100%).

Chạy thử 50 audio, không đụng Kaggle, không đụng volume thật:

```bash
docker run --rm -v "$PWD/data":/data ai-detector/corpus-builder:latest \
  --out /data/out --cache /data/cache --sources common_voice --limit 50 --no-kaggle
```

### Dừng, chạy lại, lấy kết quả

```bash
$C stop corpus     # SIGTERM -> chốt checkpoint cuối (có 120s ân hạn)
$C start corpus    # chạy tiếp từ đúng con trỏ, không quét lại
$C logs -f corpus
```

Dữ liệu nằm trong volume `ai-detector_corpus-data`. Copy ra máy host:

```bash
docker run --rm -v ai-detector_corpus-data:/data -v "$PWD/out":/host \
  alpine cp -r /data/dataset /host/
```

Muốn ghi thẳng ra thư mục host thì đổi `corpus-data:/data` trong
`docker-compose.yml` thành `./data:/data`.

Chạy không cần Docker: `pip install -r requirements.txt` rồi
`python scripts/build_corpus.py --out ./data/out --cache ./data/cache --no-kaggle`.

## Theo dõi tiến độ

Hai cách, cùng đọc một nguồn số liệu:

```bash
$C up -d dashboard                   # UI web  -> http://<máy-docker>:8000
$C run --rm cli monitor --watch=10       # bản terminal, cho lúc chỉ có ssh
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
| `common_voice` | HF `hataphu/common-voice-corpus-20` (parquet, 244 MB) | 8,963 dòng nhưng chỉ **5,388 clip khác nhau** | **không có** → mỗi clip 1 speaker, `speaker_known=false` | clip id |
| `vietmed` | HF `leduckhai/VietMed` (parquet, 185 MB) | 9,207 | `speaker_name` (13 speaker ở train) | `audio_name` |
| `vivos` | HF `AILAB-VNUHCM/vivos` (tar.gz, 1.47 GB) | 12,420 | thư mục `VIVOSSPK*` | tên file |

> **Mirror Common Voice 20 đếm trùng.** Split `validation` (5,388) chứa **trọn cả**
> `train` (2,219) lẫn `test` (1,356) — kiểm bằng `audio.path`: `train ∩ validation`
> = 2,219, `validation ∩ test` = 1,356. Cộng ba split ra 8,963 nhưng chỉ có 5,388
> clip khác nhau. Con số "8,963 Common Voice" trong spec Demo v1 vì thế đếm thừa
> 3,575, và mục tiêu 30,590 REAL không thể đạt bằng ba nguồn này. Pipeline bỏ qua
> bản lặp theo `item_id` và đếm riêng ở cột **Bỏ qua**, nên luôn có
> `quét = nhận + loại + bỏ qua`.

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

`split` để `null` cho tới khi chạy bước chia (xem dưới).

## Phần FAKE — cân bằng với REAL

VieNeu-TTS-140h là dataset **gated**: accept terms tại
<https://huggingface.co/datasets/pnnbao-ump/VieNeu-TTS-140h> rồi đặt `HF_TOKEN`.
Kiểm tra tên cột trước khi chạy thật (repo không công khai schema):

```bash
docker compose -f docker/docker-compose.yml run --rm cli fake --probe
```

Chạy sau khi REAL xong — số FAKE **lấy đúng bằng số REAL đã nhận**:

```bash
docker compose -f docker/docker-compose.yml run --rm cli fake
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

## Kiểm tra lại audio đã ghi

Audio chỉ được ghi vào shard **sau khi qua đủ 6 cổng lọc**, và chỉ shard đã
checkpoint (đã `fsync` + commit sqlite) mới được đẩy lên Kaggle — nên mọi thứ trên
Kaggle đều đã qua kiểm. Nhưng đó là kiểm ở *đầu vào*; muốn kiểm ở *đầu ra* thì:

```bash
$C run --rm cli audit                      # mở lại từng WAV trong tar, đo lại từ đầu
$C run --rm cli audit --pushed-only        # chỉ các shard đã đẩy lên Kaggle
$C run --rm cli audit --sample 2000        # kiểm nhanh 2000 file đầu
```

Script **không tin metadata**: nó giải mã lại từng file trong tar rồi đối chiếu
sample rate / channels / encoding / duration / speech ratio / `sha256_norm`,
kiểm trùng `id` + hai loại hash trên toàn corpus, và kiểm tar ↔ metadata khớp
hai chiều. Sai lệch nào cũng làm mã thoát ≠ 0, dùng được trong CI hoặc chặn trước
khi push.

## Dataset cuối (spec §12)

Tar shard chỉ là **dạng vận chuyển** — Kaggle upload 30k file WAV rời rất chậm và
hay lỗi. Bung ra cây dataset chuẩn bằng một lệnh riêng, chạy lại bao nhiêu lần cũng
được mà không đụng tới tiến độ:

```bash
docker compose -f docker/docker-compose.yml run --rm cli export --verify
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

## Chia train / validation / test

```bash
$C run --rm cli split --dry-run     # xem kết quả trước, chưa ghi
$C run --rm cli split               # ghi cột `split` vào 3 file metadata
```

Mặc định 70/15/15 (`--train/--val/--test`), ghi vào cột `split` của
`metadata.csv` / `.parquet` / `.jsonl` — **không di chuyển file audio**, nên đổi
cách chia lúc nào cũng được.

**Đơn vị chia không phải speaker, cũng không phải recording, mà là cụm liên thông
của đồ thị speaker ↔ recording.** Hai ràng buộc của spec §10 không suy ra nhau:
VietMed là hội thoại, mỗi `audio_name` chứa nhiều speaker (`VietMed_011` gồm
`_a`, `_b`, `_c`). Chia theo speaker sẽ xé một recording ra hai split; chia theo
recording sẽ để một speaker xuất hiện ở hai split. Gom cụm liên thông thì cả hai
ràng buộc cùng thoả.

Script tự kiểm lại sau khi chia: nếu còn bất kỳ `speaker_id` hay `recording_id`
nào nằm ở hai split thì báo lỗi và thoát ≠ 0.

> **Cụm to làm tỷ lệ lệch.** VietMed gom cả nghìn utterance vào vài cụm, nên
> 70/15/15 không thể bám sát — thậm chí một split có thể nhận 0 mẫu VietMed.
> Đây là giới hạn của dữ liệu chứ không phải lỗi chia (một recording hội thoại
> không tách đôi được). Script in cảnh báo rõ chứ không im lặng.

## Xác thực Kaggle

Kaggle CLI 2.x nhận nhiều kiểu credential; chọn **một**:

| Cách | Biến | Lấy ở đâu |
|---|---|---|
| Token mới (khuyến nghị) | `KAGGLE_API_TOKEN=KGAT_...` | <https://www.kaggle.com/settings/api> → Generate New Token |
| API key cũ | `KAGGLE_USERNAME` + `KAGGLE_KEY` | `key` trong `kaggle.json`, 32 ký tự hex |
| File | mount `~/.kaggle/kaggle.json` hoặc `~/.kaggle/access_token` | — |

`KAGGLE_USERNAME` **luôn** phải có dù dùng cách nào: nó là phần `<owner>` của
dataset id. Đặt token `KGAT_...` vào `KAGGLE_KEY` sẽ không chạy — đó là hai định
dạng khác nhau, và script báo đúng lỗi này.

Mọi lệnh động tới Kaggle đều chạy `kaggle quota` để **xác thực thật** trước khi
bắt đầu, thay vì chỉ kiểm biến môi trường có tồn tại hay không.

## Đẩy lên Kaggle

Dataset cuối là **một** dataset, đúng cây trong spec §12:

```bash
$C run --rm cli export --verify     # bung shard thành file WAV rời + metadata
$C run --rm cli push        # đẩy cả cây lên Kaggle
```

Tên dataset lấy từ `KAGGLE_DATASET_SLUG` (mặc định `vietnamese-audio-deepfake-demo`),
owner từ `KAGGLE_USERNAME`. Tạo ra ở chế độ **private**; `--public` mới công khai —
kiểm quyền redistribution của từng nguồn trước khi dùng cờ đó.

Kaggle CLI bỏ qua thư mục con trừ khi có `--dir-mode`. Script dùng `-r zip`: mỗi
thư mục con được gói thành một file rồi Kaggle bung lại phía server, nhờ vậy
`audio/real/...` giữ nguyên cấu trúc thay vì phải upload 50 nghìn file WAV rời.
`--dir-mode tar` nhanh hơn (WAV nén gần như vô ích) nếu muốn.

Chạy lại lệnh đó lần nữa sẽ tạo **version mới** của cùng dataset, không tạo cái thứ hai.

## Đẩy theo shard (không khuyến nghị)

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
