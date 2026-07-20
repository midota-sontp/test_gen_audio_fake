# VIVOS → Fish Speech S2 fake-speech dataset generator

Sinh dataset **phát hiện deepfake giọng nói tiếng Việt**: mỗi audio thật trong VIVOS được
ghép một audio **giả** do **Fish Speech S2** clone lại — **cùng transcript, cùng speaker, giọng
được nhân bản** nhưng là audio do model sinh ra. Kết quả là các cặp Real/Fake.

Vì mỗi speaker xuất hiện ở cả hai lớp (speaker-matched) và real/fake nói **cùng nội dung**
(content-matched), model detector không thể "đi tắt" theo danh tính giọng hay nội dung.

## Cấu trúc đầu ra

```
dataset/
├── real/
│   └── VIVOSSPK01/ VIVOSSPK01_R001.wav ...
├── fake/
│   └── fishspeech/
│       └── VIVOSSPK01/ VIVOSSPK01_R001_fake.wav ...
├── reference/
│   └── VIVOSSPK01.wav          # giọng mẫu 10-20s để clone
└── metadata/
    └── metadata.csv
```

`metadata.csv`:

| audio_path | label | speaker | text | generator | split |
|---|---|---|---|---|---|
| real/VIVOSSPK01/VIVOSSPK01_R001.wav | 0 | VIVOSSPK01 | Xin chào | real | train |
| fake/fishspeech/VIVOSSPK01/VIVOSSPK01_R001_fake.wav | 1 | VIVOSSPK01 | Xin chào | FishSpeechS2 | train |

Mọi audio được chuẩn hoá **16 kHz / mono / PCM-16**. File hỏng bị bỏ qua và ghi log; fake chỉ
được ghi vào metadata sau khi xác nhận file tồn tại & đọc được.

## Cài đặt

```bash
pip install -r requirements.txt
bash setup_fish.sh          # clone fish-speech + tải weights fishaudio/s2-pro (public)
```

`setup_fish.sh` cài `fish_speech` và tải checkpoint vào `third_party/fish-speech/checkpoints/s2-pro`.

## Dữ liệu VIVOS

Tải từ Kaggle: <https://www.kaggle.com/datasets/kynthesis/vivos-vietnamese-speech-corpus-for-asr>
rồi giải nén sao cho có `vivos/train/{prompts.txt,waves/}` và `vivos/test/...`.

## Chạy

```bash
python cli.py --dataset vivos --output dataset --config config.yaml
```

- `--dataset` đè `dataset_root`, `--output` đè `output_root`, `--splits train` để chỉ chạy 1 split.
- **Resume**: chạy lại sẽ **bỏ qua** audio đã sinh (kiểm tra file hợp lệ + đã có trong metadata).
  Dùng `--overwrite` để sinh lại từ đầu.
- Thanh tiến độ (tqdm) cho pha real và pha fake; log ở `dataset/logs/generate.log`.

> **Hiệu năng**: S2 (4B) trên MPS ~25–70s/câu → sinh full VIVOS (~11.6k câu) mất nhiều giờ; nên
> chạy overnight. Sinh fake được **tuần tự hoá trên accelerator** (chạy nhiều tiến trình 4B song
> song sẽ OOM); `num_workers` chỉ tăng tốc chuẩn hoá audio thật + dựng reference (I/O).

## Chạy HẾT trong Docker (một lệnh)

Image dựng trên **image CPU chính thức của Fish Audio** (`fishaudio/fish-speech:server-cpu-v2.0.0-beta`,
đã có sẵn `fish_speech` + torch-cpu trong venv `/app/.venv`), chỉ thêm 3 gói nhẹ — nên build nhanh,
không dính dependency hell. Container **tự bootstrap** ([docker/bootstrap.sh](docker/bootstrap.sh)):
tự tải weights S2 + tự tải VIVOS từ Kaggle (nếu chưa có) rồi sinh dataset. Không cần setup native gì cả.

```bash
docker compose run --rm generate
```

Lần đầu sẽ: tải weights `fishaudio/s2-pro` → tải VIVOS → sinh. Mọi thứ **persist trên host** (weights,
VIVOS, output) nên chạy lại là **resume**. Kết quả ở `./dataset`.

Các volume trong [docker-compose.yml](docker-compose.yml):

| host | container | vai trò |
|---|---|---|
| `./vivos` | `/data/vivos` | VIVOS — **tự tải vào đây** nếu trống (persist host) |
| `./dataset` | `/data/output` | output ghi ngược ra host |
| `./third_party/fish-speech/checkpoints` | `/work/.../checkpoints` | weights S2 (tải 1 lần, giữ lại) |
| `./config.yaml` | `/work/config.yaml` (ro) | sửa config không cần build lại |
| `~/.kaggle` | `/home/fish/.kaggle` (ro) | token Kaggle để tải VIVOS |

Tải VIVOS cần token Kaggle: mount sẵn `~/.kaggle`, hoặc đặt `KAGGLE_USERNAME`/`KAGGLE_KEY`. Nếu bạn
đã có VIVOS sẵn, cứ đặt vào `./vivos/train/...` — container sẽ bỏ qua bước tải.

> ⚠️ **Model S2 là 4B — cần nhiều RAM/VRAM.** Đây là điểm dễ hỏng nhất.
>
> **Lỗi thường gặp: `t2s step failed (exit -9)`** = SIGKILL = **hết RAM (OOM)**. Load 4B trên CPU fp32
> cần **~16GB**. Cách xử lý:
> - **CPU**: tăng RAM cho Docker (Desktop → Settings → Resources → Memory ≥ 16GB). `half: true` (fp16, ~8GB)
>   có thể giúp nhưng trên CPU đôi khi lỗi `not implemented for 'Half'` → nếu vậy quay lại tăng RAM.
> - **GPU (Linux + NVIDIA, khuyến nghị)**: sửa dòng `FROM` trong [Dockerfile](Dockerfile) sang
>   `fishaudio/fish-speech:server-cuda-v2.0.0-beta`, đặt `half: true`, rồi:
>   ```bash
>   FISH_DEVICE=cuda docker compose run --rm generate   # + bỏ comment khối deploy: GPU trong compose
>   ```
>
> **Chậm**: wrapper gọi CLI theo từng câu nên model bị **nạp lại mỗi clip** (~40-60s/clip chỉ để load).
> Trên CPU nên để `limit` nhỏ; sinh số lượng lớn nên dùng GPU.
>
> Bật fp16 trong `config.yaml`:
> ```yaml
> fishspeech:
>   half: true
> ```

## Cấu hình (`config.yaml`)

| khoá | mặc định | ý nghĩa |
|---|---|---|
| `dataset_root` | `vivos` | thư mục chứa `train/`, `test/` |
| `output_root` | `dataset` | nơi ghi `real/ fake/ reference/ metadata/` |
| `splits` | `[train, test]` | các split VIVOS xử lý |
| `limit` | `null` | **số audio cần gen**: `null`=tất cả, hoặc số nguyên (vd `200` → 200 real + 200 fake) |
| `max_per_speaker` | `null` | giới hạn số câu mỗi speaker (`null`=không giới hạn) |
| `reference_seconds` | `15` | độ dài reference/speaker (giới hạn 10–20s) |
| `sample_rate` | `16000` | tần số lấy mẫu đầu ra |
| `generator` | `FishSpeechS2` | backend sinh giả (factory trong `src/fishspeech.py`) |
| `num_workers` | `4` | song song hoá pha real + reference |
| `overwrite` | `false` | `false` = resume (bỏ qua file đã có) |
| `fishspeech.*` | | device/temperature/top_p/top_k/seed/step_timeout của S2 |

## Kiến trúc mã (module hoá)

| module | vai trò |
|---|---|
| `src/parser.py` | quét corpus, đọc `prompts.txt` → `Utterance(audio_id, speaker, text, wav, split)`, gom theo speaker |
| `src/preprocess.py` | load/chuẩn hoá 16k·mono·PCM-16, validate, bỏ file hỏng |
| `src/reference_builder.py` | dựng giọng mẫu 10–20s/speaker (ghép clip dài nhất) |
| `src/fishspeech.py` | backend Fish Speech S2 (CLI 3 bước: encode → text2semantic → decode), cache prompt tokens |
| `src/metadata.py` | ghi `metadata.csv` tăng dần, resume-safe, thread-safe |
| `src/generator.py` | điều phối: pha real+reference (song song) → pha fake (tuần tự) |
| `cli.py` | argparse + đọc YAML + chạy |

Muốn thêm backend khác: cài đặt lớp `Generator` (2 hàm `prepare_speaker`, `generate`) và đăng ký
trong `get_generator` (`src/fishspeech.py`) — phần còn lại không đổi.
