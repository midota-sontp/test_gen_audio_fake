# Hướng dẫn chạy

Toàn bộ pipeline gọi qua một lệnh duy nhất: `python -m aidetector <stage>`.
Mọi tham số nằm trong [configs/default.yaml](configs/default.yaml), ghi đè nhanh
bằng `--set khoá=giá_trị`.

## 0. Cài đặt

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # lõi: chuẩn hoá → WavLM → classifier
pip install -r requirements-generate.txt # engine sinh fake (cài riêng, xem mục 2)
brew install ffmpeg                      # cần cho augment MP3/AAC
```

Kiểm tra máy đã có gì:

```bash
python -m aidetector info
```

In ra thiết bị (`cpu`/`mps`/`cuda`), có ffmpeg hay chưa, và **engine nào đã cài
được / thiếu gì thì cài bằng lệnh nào**. Engine chưa cài sẽ tự bị bỏ qua chứ không
làm chết pipeline.

---

## 1. REAL dataset — nạp giọng thật về chuẩn corpus

Lệnh `ingest` tự nhận diện loại dataset rồi ép mọi file về đúng chuẩn (16 kHz,
mono, PCM_16, 3–10 giây, chuẩn mức âm lượng, cắt im lặng, không clipping/NaN).

```bash
# Thư mục bất kỳ — tự dò loại (VIVOS / Common Voice / real+fake / thư mục thường)
python -m aidetector ingest /đường/dẫn/vivos

# Chỉ định thẳng adapter khi cần
python -m aidetector ingest /đường/dẫn/data --adapter common_voice --name cv-vi

# Tải trực tiếp từ HuggingFace Hub
python -m aidetector ingest --hf AILAB-VNUHCM/vivos --name vivos

# Giới hạn khi muốn thử nhanh
python -m aidetector ingest /data/vivos --limit 2000 --per-speaker 40
```

Kết quả nằm ở `corpus/`:

```
corpus/
  manifest.csv                    ← nguồn sự thật duy nhất về dữ liệu
  audio/real/<nguồn>/<speaker>/<utt_id>.wav
```

Chạy lại `ingest` là **idempotent**: `utt_id` suy ra từ (nguồn, speaker, khoá) nên
chỉ phần thiếu được bổ sung.

Kiểm tra corpus có đúng chuẩn:

```bash
python -m aidetector validate           # báo cáo mọi vi phạm
python -m aidetector validate --fix     # loại bỏ bản ghi hỏng khỏi manifest
```

> **Có transcript là tốt nhất.** VIVOS/Common Voice có sẵn transcript, và tầng
> `generate` dùng chính transcript đó để TTS đọc lại → mỗi fake có một real đối
> chứng cùng nội dung, cùng speaker. Không có transcript thì pipeline vẫn chạy
> (dùng danh sách câu dự phòng) nhưng chất lượng dữ liệu huấn luyện kém hơn.

---

## 2. FAKE dataset — sinh audio giả

### Engine đang hỗ trợ

| Engine | Loại | Máy cần | Ghi chú |
|---|---|---|---|
| `piper` | TTS giọng cố định | CPU (rất nhanh) | 3 giọng `vi_VN`, tự tải từ `rhasspy/piper-voices` |
| `kokoro` | TTS giọng cố định | CPU | Kokoro-Vietnamese, 13 giọng vi |
| `omnivoice` | **voice cloning** zero-shot | GPU / Apple MPS | clone chính giọng speaker trong corpus REAL |

```bash
pip install piper-tts
pip install git+https://github.com/iamdinhthuan/Kokoro-Vietnamese.git
pip install omnivoice
```

### Sinh

```bash
# Mặc định: tổng fake ≈ tổng real, chia đều cho các engine trong config
python -m aidetector generate

# Chỉ định engine + số lượng mỗi engine
python -m aidetector generate --engines piper kokoro --count 1000

# Voice cloning trên Apple Silicon
python -m aidetector generate --engines omnivoice --device mps --count 500
```

Fake được ghi vào `corpus/audio/fake/<engine>/<voice>/`, mỗi bản ghi mang:
`generator` (vd `piper:vi_VN-vais1000-medium`), `ref_utt_id` (utterance real gốc)
và **cùng `speaker` + cùng `text` với real** — nhờ vậy mô hình không thể phân loại
dựa vào nội dung câu nói hay danh tính người nói.

### Thêm engine mới về sau

Tạo `aidetector/generate/<tên>.py`:

```python
from .base import KIND_TTS, Availability, Generator, register

@register
class MyEngine(Generator):
    id = "my_engine"
    kind = KIND_TTS            # hoặc KIND_CLONE nếu cần audio tham chiếu
    native_sample_rate = 24_000

    @classmethod
    def availability(cls):
        try:
            import my_tts_lib  # noqa: F401
        except ImportError:
            return Availability(False, "chưa cài my-tts-lib", "pip install my-tts-lib")
        return Availability(True)

    def voices(self):
        return ["giong_1", "giong_2"]

    def synthesize(self, text, voice=None, ref_audio=None, ref_text=None):
        audio = ...                       # numpy float32 mono
        return audio, self.native_sample_rate
```

Thêm `my_engine` vào `generate.engines` trong config rồi import trong
`aidetector/generate/__init__.py`. Không phải sửa gì khác — resample, chuẩn hoá,
đặt tên file, ghi manifest đều dùng chung.

---

## 3. Chia tập

```bash
python -m aidetector split

# Giữ hẳn một engine riêng cho test: đo khả năng tổng quát sang engine LẠ
python -m aidetector split --holdout omnivoice
```

Chia **speaker-disjoint**; bản augment luôn theo bản gốc. Lệnh in ra số lượng từng
split và kiểm tra rò rỉ speaker (`Không có rò rỉ speaker giữa các split ✔`).

## 4. Augmentation

```bash
python -m aidetector augment              # 1 bản biến dạng cho mỗi utterance train
python -m aidetector augment --copies 3
```

Bản clean **luôn được giữ lại**, bản augment là bản ghi thêm (`augment` ghi rõ chuỗi
phép đã dùng, vd `mp3-64k+bg12db`). Mặc định chỉ augment `train`; val/test giữ sạch
để số đo phản ánh dữ liệu thật.

Muốn dùng nhiễu nền và vang phòng thật, bỏ chú thích trong config và trỏ vào thư
mục có sẵn file audio:

```yaml
augment:
  noise_dir: noise    # MUSAN, WHAM, DEMAND...
  rir_dir: rir        # impulse response
```

> Chạy `augment` **sau** `split` để bản augment thừa hưởng đúng split của bản gốc.

## 5. Trích đặc trưng (WavLM)

```bash
python -m aidetector features
```

Lần đầu tải `microsoft/wavlm-base-plus` (~380 MB) và cache lại. Embedding lưu theo
`utt_id` tại `features/<backbone>-<layer>-<pooling>/`, nên thêm dữ liệu chỉ trích
phần mới.

Đổi sang backbone khác — cache tách riêng, không đè lên nhau:

```bash
python -m aidetector features --set features.backbone.name=wav2vec2
python -m aidetector features --set features.backbone.checkpoint=microsoft/wavlm-large \
                              --set features.backbone.output_layer=9
python -m aidetector features --set features.backbone.pooling=mean_std
```

## 6. Huấn luyện + đánh giá

```bash
python -m aidetector train
python -m aidetector evaluate
```

`train` dừng sớm theo **val EER**, lưu `checkpoints/best.pt` (kèm ngưỡng quyết định
đã chốt trên val, thống kê chuẩn hoá, và cấu hình backbone để suy luận tái lập được).

`evaluate` ghi ra `reports/`:

| File | Nội dung |
|---|---|
| `metrics.json` | EER, AUC, min-DCF, confusion matrix + **breakdown theo từng generator / nguồn / clean-vs-augmented** |
| `predictions.csv` | điểm từng utterance trong test |
| `curves.png` | ROC · PR · DET · phân bố điểm |
| `confusion_matrix.png` | confusion matrix tại ngưỡng đang dùng |
| `history.csv` | loss/EER theo epoch |

## 7. Suy luận

```bash
python -m aidetector detect audio.wav
python -m aidetector detect *.mp3 --json
```

```
🔴 FAKE · P(fake)=0.958 · 4.20s · audio.wav
```

File dài hơn 10 giây được chấm từng đoạn rồi lấy trung bình; file ngắn hơn 3 giây
vẫn chấm được nhưng kèm cảnh báo độ tin cậy.

---

## Chạy trọn pipeline một lệnh

```bash
python -m aidetector run --path /đường/dẫn/vivos --engines piper kokoro
```

Thứ tự: `ingest → generate → split → augment → features → train → evaluate`
(`split` trước `augment` để bản augment chỉ sinh cho train và bám đúng split của
bản gốc; `features` sau khi corpus đã đủ).

Chạy lại một phần:

```bash
python -m aidetector run features train evaluate
python -m aidetector run --skip ingest generate
```

## Chạy trên Kaggle (GPU miễn phí)

**Một file duy nhất: [notebooks/aidetector_kaggle.ipynb](notebooks/aidetector_kaggle.ipynb)**
— `File → Import Notebook` trên Kaggle rồi chạy. Notebook **nhúng sẵn toàn bộ mã
nguồn** (base64 của `aidetector/` + `configs/`), không cần clone repo hay thêm
dataset chứa code.

**Trong panel bên phải phải bật:** Accelerator = `GPU T4 x2`/`P100`, Internet = `On`.
Rồi Add Input → Datasets một bộ giọng thật tiếng Việt. Không có GPU thì OmniVoice
(voice cloning) gần như không chạy nổi — bỏ nó khỏi `--engines`, Piper và Kokoro vẫn
chạy tốt trên CPU.

Notebook chia hai phần, **chạy phần A trước**:

| | Làm gì | Ghi chú |
|---|---|---|
| **PHẦN A** | ingest → generate → validate → **nghe thử + xem phổ** → `pack` | có công tắc `SMOKE = True` chạy thử ~40 mẫu vài phút |
| **PHẦN B** | split → augment → features → train → evaluate | chỉ chạy khi dataset đã ưng |

Phần A dừng lại ở một dataset đóng gói sẵn, nên bạn kiểm tra được dữ liệu **trước
khi** tốn giờ GPU huấn luyện: thống kê cân bằng real/fake, số lượng theo từng engine,
tỉ lệ fake có real đối chứng, và nghe trực tiếp từng cặp real/fake cùng câu cùng giọng.

Sau khi sửa code trong repo, build lại notebook cho khớp:

```bash
python scripts/build_kaggle_notebook.py
```

Payload tất định (cùng code ⇒ cùng hash) và có test chặn notebook lệch khỏi repo.

Config riêng: **[configs/kaggle.yaml](configs/kaggle.yaml)** — kế thừa `default.yaml`,
chỉ đổi đường dẫn sang `/kaggle/working`, tăng batch và bật thêm `omnivoice`. Thiết bị
để `auto` nên quên bật Accelerator cũng chỉ chậm chứ không chết giữa chừng.

### Giữ dữ liệu giữa các phiên

Phiên Kaggle ~9 giờ và **xoá sạch `/kaggle/working` khi kết thúc**. Commit output với
hàng chục nghìn file wav rời rạc thì rất chậm, nên gói vào một zip:

```python
# Cuối phiên 1 — sau đó: Output → New Dataset
!python -m aidetector pack -c {CFG} --out /kaggle/working/corpus.zip

# Đầu phiên 2 — add dataset vừa tạo rồi bung ra, khỏi ingest/generate lại
!python -m aidetector unpack /kaggle/input/<tên-dataset>/corpus.zip -c {CFG}
!python -m aidetector run split augment features train evaluate -c {CFG}
```

`unpack` chỉ ghi những file chưa có nên chạy lại được, và tự loại khỏi manifest các
bản ghi thiếu audio.

## Docker (CPU-only)

```bash
docker compose build
docker compose run --rm ingest /data/vivos     # mount dataset host vào /data
docker compose run --rm generate
docker compose up pipeline                     # split → features → train → evaluate
docker compose run --rm cli detect /data/thu.wav
```

`corpus/`, `features/`, `checkpoints/`, `reports/`, `models/` được mount ra host nên
kết quả không mất khi container tắt.

## Test

```bash
pytest -q
```

59 test chạy trọn pipeline bằng engine/backbone giả — không tải model, không cần
mạng, xong trong ~12 giây.

## Bảng tham số hay dùng

| Khoá | Ý nghĩa |
|---|---|
| `audio.min_seconds` / `max_seconds` | khoảng độ dài chuẩn (3–10 s) |
| `audio.long_policy` | `split` cắt nhiều đoạn · `crop` lấy giữa · `drop` bỏ |
| `generate.fake_to_real_ratio` | tổng fake so với tổng real |
| `generate.voices.<engine>` | danh sách giọng dùng cho engine đó |
| `augment.ops.<phép>.p` | xác suất áp dụng từng phép |
| `augment.copies` | số bản biến dạng mỗi utterance |
| `splits.holdout_generators` | engine chỉ xuất hiện ở test |
| `features.backbone.name` | `wavlm` · `wav2vec2` · `hubert` · `whisper` |
| `features.backbone.output_layer` | layer lấy đặc trưng (mặc định 6) |
| `features.backbone.pooling` | `mean` · `mean_std` · `max` · `first` |
| `model.head` | `linear` · `mlp` · `deep_mlp` |
| `train.early_stopping.monitor` | `val_eer` (mặc định) hoặc `val_loss` |

## Đọc kết quả

- **EER** là số chính. Nhìn thêm `by_generator` trong `metrics.json`: engine nào
  EER cao là engine mô hình còn yếu → cần thêm dữ liệu từ engine đó.
- Nếu dùng `--holdout <engine>`, EER của engine đó là con số **sát thực tế nhất**
  cho việc triển khai: nó đo khả năng bắt được engine chưa từng thấy.
- `by_condition` so sánh clean vs augmented — chênh lệch lớn nghĩa là mô hình chưa
  bền với nhiễu/nén, nên tăng `augment.copies` hoặc xác suất `codec`.
- Cảnh báo `khoảng cách val-train` trong log train là chỉ báo overfit.
