# ai-detector — phát hiện giọng nói giả tiếng Việt

Phát hiện audio do TTS / voice-cloning sinh ra, tập trung vào tiếng Việt.

```
                    Vietnamese real speech
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
        REAL dataset                    Voice cloning / TTS
                                    ┌─────────┼─────────┐
                                    ▼         ▼         ▼
                                OmniVoice  Kokoro     Piper      (+ engine mới…)
                                    └─────────┼─────────┘
                                              ▼
                                        FAKE dataset
                                              │
                              └───────────────┬───────────────┘
                                              ▼
                                    Audio augmentation
                                     (noise · reverb · MP3/AAC)
                                              ▼
                                       WavLM  (đóng băng)
                                     (hoặc XLS-R / HuBERT / Whisper)
                                              ▼
                                         Classifier
                                              │
                                    ┌─────────┴─────────┐
                                    ▼                   ▼
                                  REAL                FAKE
```

Hướng dẫn chạy từng bước: **[run.md](run.md)**.
Chạy trên GPU miễn phí của Kaggle: hai notebook tự chứa toàn bộ mã nguồn, import lên
Kaggle là chạy — **[notebooks/aidetector_dataset.ipynb](notebooks/aidetector_dataset.ipynb)**
(tạo dataset) rồi **[notebooks/aidetector_train.ipynb](notebooks/aidetector_train.ipynb)**
(huấn luyện).

## Ý tưởng thiết kế

**Một chuẩn dữ liệu duy nhất.** Mọi nguồn — VIVOS, Common Voice, một thư mục wav
bất kỳ, dataset trên HuggingFace, hay audio vừa do TTS sinh ra — đều đi qua đúng
một hàm chuẩn hoá rồi ghi vào `corpus/<bộ>/metadata.csv`. Các tầng sau không cần biết
dữ liệu đến từ đâu.

| Thuộc tính | Chuẩn |
|---|---|
| Sample rate | 16 000 Hz |
| Channels | mono |
| Format / bit depth | WAV, 16-bit PCM |
| Duration | 3–10 giây (file dài hơn được cắt thành nhiều đoạn) |
| Peak / RMS | RMS ≈ −23 dBFS, trần peak −1 dBFS |
| Silence | cắt bớt im lặng đầu/cuối |
| Clipping / NaN / Inf | không được có (`validate` kiểm tra lại) |
| Background noise | cho phép — nhưng luôn giữ **cả bản clean lẫn bản noisy** |
| Compression | MP3/AAC sinh ở tầng augmentation, không nằm trong corpus gốc |

Cột của `metadata.csv` theo **chuẩn corpus Việt**, nên một bảng dựng sẵn ngoài dự án
nạp thẳng được — không cần bước map tên cột:

| Nhóm | Cột |
|---|---|
| Định danh & nội dung | `id` `audio` `label` (0=real, 1=fake) `speaker_id` `speaker_known` `source` `generator` `recording_id` `duration` `sample_rate` `channels` `split` `text` `gender` |
| Số đo — tuỳ chọn | `speech_ratio` `rms_db` `clipping_ratio` `original_sample_rate` `original_channels` `original_split` `original_duration` `norm_gain` `norm_peak` `sha256_raw` `sha256_norm` `shard` `bytes` `extra` |
| Riêng pipeline này | `ref_id` `language` `augment` `parent_id` `checked` `schema_version` |

Nhóm **số đo** giữ nguyên khi nạp từ bảng ngoài và để trống khi `ingest`/`generate`
tự sinh bản ghi — rỗng đọc ra `None`, phân biệt được với một giá trị 0 đo thật. Bản
sao có audio khác (augment) bị xoá sạch nhóm này thay vì thừa hưởng, vì một
`sha256_norm` chép lại từ bản gốc là số đo sai một cách im lặng.

Nhóm **riêng pipeline** không có trong bảng ngoài nhưng chịu lực: `augment` +
`parent_id` giữ bất biến "bản augment nằm cùng split với bản gốc", `checked` cho
`validate` soi tăng dần thay vì đọc lại cả corpus mỗi phiên.

Schema cũ (`utt_id` `path` `speaker` `ref_utt_id` `parent_utt_id`, `label` real/fake,
split `val`) vẫn **đọc được** — `Record.from_row` đổi tên khi nạp, lượt `save()` kế
tiếp ghi ra chuẩn mới. Mọi `corpus.zip` đã đẩy lên Kaggle vì thế không thành rác.

**Mọi tầng đều cắm rời.** Nguồn dữ liệu, engine sinh fake, phép augment, backbone
và classifier head đều nằm trong registry riêng. Thêm cái mới = thêm một file +
một dòng config, không phải sửa pipeline.

| Tầng | Đang có | Thêm mới |
|---|---|---|
| Nguồn dữ liệu | `vivos`, `common_voice`, `folder`, `labeled_folder`, `hf` | `aidetector/ingest/` |
| Engine sinh fake | `piper`, `kokoro`, `omnivoice` | `aidetector/generate/` |
| Augment | codec, background noise, reverb, band-limit, gain, speed | `aidetector/augment/ops.py` |
| Backbone | `wavlm`, `wav2vec2`/XLS-R, `hubert`, `whisper` | `aidetector/features/backbones.py` |
| Head | `linear`, `mlp`, `deep_mlp` | `aidetector/models.py` |

`python -m aidetector info` in ra toàn bộ danh sách kèm trạng thái đã cài hay chưa.

**Chống rò rỉ, chống "ăn gian".** Đây là phần dễ làm sai nhất của bài toán này:

- Fake được sinh từ **chính transcript của real** và mang **cùng speaker** ⇒ mỗi
  fake có một real đối chứng cùng nội dung, cùng giọng. Mô hình không thể phân loại
  dựa vào chủ đề câu nói hay danh tính người nói.
- Chia tập **speaker-disjoint**; bản augment luôn nằm cùng split với bản gốc.
- Real và fake đi qua **cùng một chuỗi chuẩn hoá** (16 kHz, cùng mức RMS) ⇒ không
  còn manh mối kiểu "fake thì 24 kHz" hay "fake thì to hơn".
- `splits.holdout_generators` giữ hẳn một engine riêng cho test ⇒ đo được khả năng
  tổng quát hoá sang engine **chưa từng thấy** — con số quan trọng nhất khi triển khai.

## Cấu trúc

```
aidetector/
  corpus/       CHUẨN dữ liệu: spec audio, schema manifest, đọc/ghi
  ingest/       nguồn thô  → chuẩn corpus (có tự nhận diện loại dataset)
  generate/     REAL       → FAKE bằng TTS / voice cloning
  augment/      thêm bản nhiễu · vang · nén, giữ nguyên bản clean
  features/     backbone đóng băng + cache embedding theo id
  models.py     classifier head
  splits.py     chia train/validation/test speaker-disjoint + holdout engine
  train.py      huấn luyện (early-stopping theo val EER)
  evaluate.py   EER/AUC + breakdown theo từng generator
  detect.py     suy luận trên file bất kỳ
  env.py        nhận diện local / Kaggle / Colab + cảnh báo giới hạn phiên
  packaging.py  gói/bung corpus thành 1 zip để chuyển giữa các phiên
  cli.py        `python -m aidetector <lệnh>`
configs/default.yaml   toàn bộ tham số
configs/kaggle.yaml    preset cho Kaggle (kế thừa default, đổi path + bật GPU)
notebooks/             notebook chạy trọn pipeline trên Kaggle
tests/                 pytest — chạy trọn pipeline bằng engine/backbone giả
```

## Nơi chạy được

| Môi trường | Engine sinh fake dùng được | Ghi chú |
|---|---|---|
| macOS / Apple Silicon | `piper`, `kokoro` | WavLM chạy MPS; OmniVoice quá chậm |
| Kaggle GPU (T4/P100) | cả ba, kể cả `omnivoice` | `configs/kaggle.yaml` + notebook sẵn |
| Linux + CUDA | cả ba | |
| Docker CPU | `piper`, `kokoro` | `docker compose` |

## Số đo

**EER** (Equal Error Rate) là số đo chính. `reports/metrics.json` còn có AUC,
min-DCF, confusion matrix và breakdown theo `generator` / `source` /
clean-vs-augmented; `reports/curves.png` có ROC, PR, DET và phân bố điểm.
