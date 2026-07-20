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
