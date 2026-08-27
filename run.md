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
  <bộ>/metadata.csv               ← nguồn sự thật về dữ liệu CỦA BỘ ĐÓ
  <bộ>/real/<speaker>/0001.wav
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
| `omnivoice` | **voice cloning** zero-shot | GPU / Apple MPS | clone chính giọng speaker trong corpus REAL; mặc định dùng fine-tune tiếng Việt `splendor1811/omnivoice-vietnamese` |

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

#### Khi giọng clone nghe không giống người nói gốc

Biến quyết định là **reference**, không phải engine. Pipeline ghép nhiều utterance của
cùng speaker cho tới ~12 giây (`TARGET_REF_SECONDS` trong `aidetector/generate/`), nên
mỗi speaker cần có đủ bản ghi trong corpus — log sẽ in `Reference clone: trung bình
N giây/mẫu` và cảnh báo nếu quá ngắn.

| Triệu chứng | Việc cần làm |
|---|---|
| log báo reference trung bình < 7 giây | ingest lại với `--per-speaker` cao hơn để mỗi speaker có nhiều câu |
| giọng đúng chất nhưng vẫn "lệch người" | tăng bám prompt: `--set generate.options.omnivoice.guidance_scale=3.0` |
| phát âm chuẩn nhưng danh tính sai hẳn | thử bản đa ngữ: `--set generate.options.omnivoice.checkpoint=k2-fsa/OmniVoice` — fine-tune một-ngôn-ngữ đọc tiếng Việt tốt hơn nhưng clone kém hơn |
| audio rè / mất chi tiết | `--set generate.options.omnivoice.num_step=48` |

Checkpoint khác mặc định được ghi thẳng vào cột `generator` (`omnivoice:k2-fsa-omnivoice`)
và vào `utt_id`, nên hai lượt A/B nằm cạnh nhau trong cùng corpus thay vì lượt sau bị bỏ
qua vì trùng id. Knob (`guidance_scale`, `num_step`) thì không vào id — đổi knob phải
`--overwrite` và lượt mới đè lên lượt cũ.

Corpus 16 kHz cũng là một trần cứng: OmniVoice sinh ở 24 kHz nên phần trên 8 kHz của
reference không tồn tại và model phải tự bịa — đó là phần "chất giọng" nghe khác nhất.

#### Chạy dở rồi tiếp tục ở phiên sau

`generate` idempotent theo `utt_id`: chạy lại cùng `--count` thì nó bỏ qua phần đã có và
chỉ làm phần còn thiếu. Bốn thứ đỡ mất công:

```bash
# Còn thiếu bao nhiêu? Không nạp model, xong trong vài giây, kèm tiến độ theo speaker.
python -m aidetector generate --engines omnivoice --count 4000 --dry-run

# Chốt tiến độ ở ranh giới mỗi speaker rồi gọi lệnh đẩy dữ liệu ra ngoài
python -m aidetector generate --engines omnivoice --count 4000 \
    --after-speaker "python sync_corpus.py"

# Mang corpus giữa hai phiên
python -m aidetector pack --out corpus.zip
python -m aidetector unpack corpus.zip
```

Engine cloning chạy **gom theo speaker** (phần chọn câu vẫn round-robin nên phân bổ
không đổi), nhờ đó có ranh giới rõ ràng để chốt: xong một giọng ⇒ lưu manifest ⇒ gọi
hook. Hook nhận `AIDETECTOR_SPEAKER`, `AIDETECTOR_KEPT`, `AIDETECTOR_CORPUS` qua biến môi
trường, và hook hỏng **không** làm dừng lượt sinh — mất kết nối lúc đẩy thì corpus vẫn
còn trên đĩa, còn bỏ dở nhiều giờ GPU thì không lấy lại được.

Manifest còn được lưu sau mỗi `SAVE_EVERY = 50` mẫu, nên bị ngắt giữa hai speaker vẫn
giữ được phần đã sinh. Nếu chỉ lưu lúc engine chạy xong thì mất vài giờ GPU là chuyện
thường.

Fake được ghi vào `corpus/<bộ>/fake/<engine>/<speaker>/` — thư mục của **chính bộ dữ
liệu đã sinh ra nó**, vì `source` thừa hưởng từ real gốc. Mỗi bản ghi mang:
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

**Hai file, một cho mỗi việc** — `File → Import Notebook` trên Kaggle rồi chạy. Cả hai
**nhúng sẵn toàn bộ mã nguồn** (base64 của `aidetector/` + `configs/`), không cần clone
repo hay thêm dataset chứa code:

| Notebook | Làm gì | Input cần add |
|---|---|---|
| [notebooks/aidetector_dataset.ipynb](notebooks/aidetector_dataset.ipynb) | ingest → generate → validate → **nghe thử + xem phổ** → đẩy lên Kaggle Dataset | một bộ giọng thật (VIVOS…) |
| [notebooks/aidetector_train.ipynb](notebooks/aidetector_train.ipynb) | split → augment → features → train → evaluate | corpus do file trên đẩy lên |

**Trong panel bên phải phải bật:** Accelerator = `GPU T4 x2`/`P100`, Internet = `On`.
Rồi Add Input → Datasets một bộ giọng thật tiếng Việt. Không có GPU thì OmniVoice
(voice cloning) gần như không chạy nổi — bỏ nó khỏi `--engines`, Piper và Kokoro vẫn
chạy tốt trên CPU.

**Chạy file dataset trước.** Nó dừng lại ở một corpus đã đóng gói và đẩy lên Kaggle
Dataset, nên bạn kiểm tra được dữ liệu **trước khi** tốn giờ GPU huấn luyện: thống kê cân
bằng real/fake, số lượng theo từng engine, tỉ lệ fake có real đối chứng, và nghe trực tiếp
từng cặp real/fake cùng câu cùng giọng. Có công tắc `SMOKE = True` chạy thử ~40 mẫu trong
vài phút. Xong rồi mới mở file train, add đúng dataset đó vào Input và Save & Run All.

#### Vì sao hai file chứ không một file với công tắc

Sinh fake bằng voice cloning mất nhiều giờ (~4 giây/mẫu trên T4) nên hai việc không bao
giờ nằm cùng một phiên. Tách file thì **mọi ô trong file đang mở đều là ô phải chạy** —
Save & Run All không còn phải bỏ qua nửa notebook, và không có đường nào chạy lộn phần.

Ba chỗ hai file thật sự khác nhau, chứ không chỉ ẩn ô đi:

* **File train không cài engine sinh audio nào** — đỡ vài phút và tránh hẳn màn
  `transformers` 4.x/5.x giằng nhau, vì WavLM chạy được trên cả hai nhánh.
* **File train không dò dataset REAL** — nó mount corpus đã sinh sẵn chứ không mount
  VIVOS, nên đòi cho được một bộ giọng thật là dừng oan.
* **File train bắt buộc phải có corpus** — không nạp được gì, hoặc nạp ra corpus thiếu hẳn
  một lớp, thì ô A1b dừng ngay thay vì đốt mấy giờ GPU vào tay không.

Đổi lại, `MODE = "both"` (chạy trọn pipeline trong một phiên) không còn: muốn thử
end-to-end thì chạy file dataset với `SMOKE = True` rồi chạy file train.

Hai file **không được sửa tay** — cả hai do `scripts/build_kaggle_notebook.py` sinh ra từ
cùng một danh sách ô, nên ô dùng chung (bung mã nguồn, `run()`, **A1b** nạp corpus) giống
nhau từng byte. Đó là thứ giữ cho file train không huấn luyện theo một chuẩn dữ liệu khác
chuẩn lúc sinh; có test chặn cả ba việc: file lệch script, ô dùng chung lệch nhau, và file
dùng một tên mà chính nó không định nghĩa.

#### Chạy full nguồn, nhiều phiên

Lượt chạy thật mặc định **không áp trần**:

```python
N_REAL, PER_SPEAKER, N_FAKE_TTS, N_FAKE_CLONE = None, None, None, None
```

`None` ⇒ không truyền `--limit`/`--count`, nên `ingest` lấy mọi utterance đạt chuẩn và
`generate` để `fake_to_real_ratio: 1.0` tự tính đúng một fake cho mỗi real. Không phải
đoán nguồn cấp được bao nhiêu — đoán sai là lệch lớp (ingest 6.300 real mà chỉ sinh 4.000
fake là 1,6× nghiêng về real).

Vòng một phiên:

1. **A1b** bung corpus phiên trước rồi in `Tiến độ gen: 4200/6300 (67%) · còn 2100 mẫu
   ≈ 2,2 giờ trên T4`. Đọc từ manifest, không nạp engine, xong trong vài giây.
2. **A3b** `--dry-run` xác nhận lại bằng đúng phép đếm của lượt sinh (theo `utt_id`), kèm
   tiến độ theo speaker.
3. **A3b** sinh phần còn thiếu, đẩy lên dataset ở ranh giới mỗi speaker.
4. **A3c** đếm lại: `còn 0` ⇒ xong, phiên sau mở file train. Còn dương ⇒ hết giờ
   giữa đường, vào lại là tiếp đúng chỗ.

Số đo thật trên VIVOS: 12.420 file → **7.367** utterance đạt chuẩn (59,3%; phần bỏ là
clip ngắn hơn `min_seconds = 3s`), 65 speaker. Với 3,7 giây/mẫu trên T4 thì sinh đủ 1:1 là
**~7,6 giờ** — vừa trong một phiên tạo dataset 9 giờ, nhưng sát; hết giờ giữa đường
thì phiên sau tiếp, không mất gì.

`PER_SPEAKER = None` là chọn "mọi utterance" thay vì "phần công bằng của mỗi giọng". Trần
120 cho 5.395 utterance rải đều; bỏ trần cho 7.367 nhưng phần thêm dồn vào các giọng nói
nhiều. Vì split là speaker-disjoint và test đo tổng quát hoá sang giọng mới, nếu thấy EER
trên test tệ hơn val rõ rệt thì đặt lại `PER_SPEAKER = 120–200`.

Sau khi sửa code trong repo, build lại notebook cho khớp:

```bash
python scripts/build_kaggle_notebook.py
```

Một lệnh sinh lại **cả hai** file. Payload tất định (cùng code ⇒ cùng hash) và có test
chặn notebook lệch khỏi repo. Thử chạy trọn cả hai ngay trên máy, trong một `/kaggle` giả:

```bash
python scripts/run_notebook_locally.py /tmp/sandbox <thư-mục-VIVOS>
```

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

### Cấu trúc corpus

```
corpus/
├── vivos/
│   ├── metadata.csv                ← nguồn sự thật về dữ liệu của bộ này
│   ├── real/<speaker>/0001.wav
│   ├── fake/<engine>/<speaker>/0001.wav
│   └── augment/[<engine>/]<speaker>/0001.wav
└── abc/
    ├── metadata.csv
    └── real/<speaker>/0001.wav
```

**Mỗi bộ dữ liệu một thư mục tự chứa.** Thêm hay bỏ một bộ là thêm hay bỏ một thư mục;
`metadata.csv` của một bộ chỉ kể bản ghi của bộ đó. Fake nằm trong thư mục của chính bộ
đã sinh ra nó (`source` thừa hưởng từ real gốc), rồi mới tách theo engine. Tầng cuối luôn
là **speaker** — kể cả fake — nên đứng ở một giọng là thấy cả hai lớp cạnh nhau.

Trong bộ nhớ thì `Manifest` vẫn là **một bảng hợp nhất**: chia tập speaker-disjoint, cân
bằng lớp và huấn luyện đều phải nhìn toàn bộ dữ liệu cùng lúc.

Cột `path` tính từ **gốc corpus** (`vivos/real/…`), không phải từ thư mục bộ. Nhờ vậy
`path` là khoá tra cứu duy nhất ở mọi chỗ, mà thư mục bộ vẫn dời được sang corpus khác
miễn giữ nguyên tên thư mục.

Tên file là số thứ tự trong thư mục, **cấp một lần rồi nằm trong cột `path`**. Suy lại số
từ thứ tự duyệt sẽ phá idempotency: ingest lần sau có thể gán `0003` cho một utterance
khác. `utt_id` vẫn là khoá chính và vẫn sinh từ `stable_id`, nên resume không đổi gì.

Corpus cũ vẫn **đọc được** vì cột `path` là nguồn sự thật — cả cây `audio/<label>/…` lẫn
cây `<label>/<nguồn>/…` với một `metadata.csv` gộp ở gốc. Lượt `save` đầu tiên tách bảng
gộp ra từng bộ (bản gốc được giữ lại dưới tên `metadata.goc-cu.csv`, không xoá); `migrate`
dời file audio về cây hiện hành:

```bash
python -m aidetector migrate          # idempotent; --dry-run để chỉ đếm
```

Notebook tự gọi `migrate` ngay sau `unpack`.

### Nhập một bộ dữ liệu mới

Cây trên cũng là định dạng trao đổi: convert một bộ dữ liệu về đó **một lần**, rồi mọi
thứ phía sau không cần biết nó vốn có cấu trúc gì.

```bash
python -m aidetector ingest /đường/dẫn/cây-chuẩn          # tự nhận adapter `canonical`
```

Adapter `canonical` chỉ nhập phần **real**. Fake luôn do `generate` của chính pipeline
sinh — fake nhập từ ngoài không có `ref_utt_id` nên không ghép cặp được với real nào,
đúng thứ mà cả thiết kế corpus tránh. Muốn mang nguyên corpus (cả fake, giữ nguyên
`utt_id`) thì dùng `pack`/`unpack`.

### `MIN_SECONDS`

Ngưỡng độ dài tối thiểu, áp cho **cả real và fake ở mọi stage** — ô `run` của notebook tự
dán `--set audio.min_seconds` vào từng lệnh. Chỉ hạ cho `ingest` mà không hạ cho
`generate` là real giữ tới 2s trong khi fake dưới 3s bị bỏ: chính độ dài thành dấu hiệu
phân biệt hai lớp.

Số đo thật trên VIVOS ở `3.0`: giữ 8.246/12.421 clip (66%), bỏ 4.175 vì quá ngắn — trong
đó 2.865 clip vẫn dài ≥2s. Hạ xuống `2.0` lấy lại chừng đó (corpus ~11.100) và thêm khoảng
3 giờ sinh, đổi lại mỗi clip mang ít bằng chứng hơn.

#### Notebook làm việc đó tự động

`DATASET_ID` ở ô setup là **một biến duy nhất** cho cả hai chiều: ô A1b nạp về từ đó, ô
A2b đẩy lên đó. Không có đường nào để đẩy lên một chỗ rồi phiên sau nạp từ chỗ khác.

Chu kỳ đẩy có ba mốc:

| Mốc | Ở đâu | Bịt lỗ nào |
|---|---|---|
| sau `ingest` | ô A2b, chỉ khi ingest thêm bản ghi | out lúc sinh giọng đầu — đúng lúc chưa có mốc nào được chốt |
| **xong mỗi speaker** | `generate --after-speaker`, **chạy nền** | out giữa lượt sinh nhiều giờ |
| cuối phiên | ô A5, `--force` (chặn, đợi lượt nền) | phần lẻ sau mốc cuối |

Speaker là mốc dày nhất corpus có: trước ranh giới đó, phần đã xong chỉ là một nhúm mẫu
lẻ giữa chừng. 4000 mẫu trên ~46 speaker ⇒ mỗi giọng ~6 phút, nên out bất ngờ mất tối đa
cỡ 6 phút GPU.

Nhịp dày đó chỉ khả thi vì **lượt đẩy chạy nền**. Gói ~1 GB rồi upload mất 1–3 phút; đẩy
chặn dòng sinh thì 46 lượt cộng lại là hơn một giờ GPU đứng chờ — trả hơn một giờ để rút
cửa sổ mất mát từ 20 phút xuống 6 phút là lỗ. Chạy nền thì việc đó là của CPU với mạng,
GPU sinh tiếp. Hook dùng `nohup … >> sync.log 2>&1 &`: cả ba thành phần đều bắt buộc, vì
hook gọi bằng `capture_output` nên chỉ `&` thôi là nó vẫn đứng chờ ống stdout của con cháu.

Hai bất biến đi kèm: **khoá PID** (speaker xong sớm hơn thời gian đẩy thì bỏ lượt — mỗi
lần đẩy là ảnh chụp toàn bộ corpus nên mốc sau gói cả phần vừa bỏ), và **ảnh chụp nhất
quán** (`pack` đọc manifest — ghi bằng `tmp` + `os.replace` — rồi zip đúng những file
trong đó).

`SYNC_EVERY_MINUTES = 0` là không chặn nhịp; đặt > 0 nếu mạng chậm. `kaggle datasets
version` bị từ chối khi version trước còn đang xử lý — vô hại vì lượt sau là ảnh chụp đầy
đủ. Xem các lượt đẩy nền bằng `sync_log()`.

Thứ duy nhất tăng theo nhịp mà không tự dọn là **số version**: mỗi lượt đẩy là một version
~1 GB, nhịp theo speaker ⇒ vài chục version mỗi phiên. `KEEP_OLD_VERSIONS = False` thêm
`--delete-old-versions` để dataset chỉ giữ bản mới nhất; mất mát duy nhất là đường lùi, vì
bản mới nhất luôn là superset của mọi bản cũ. Mặc định `True` — xoá version là không lấy
lại được.

Hai notebook dùng chung dataset đó, nhưng chỉ một chiều đẩy:

| Notebook | Nạp về | Đẩy lên |
|---|---|---|
| `aidetector_dataset.ipynb` | A1b nạp corpus phiên trước | ba mốc ở trên |
| `aidetector_train.ipynb` | A1b nạp corpus — **bắt buộc**, không có thì dừng ngay | không đẩy |

File train không đẩy là có chủ ý: phần B chạy `augment`, nó ghi thêm bản nhiễu/nén vào
corpus. Đẩy sau đó là bơm dữ liệu phái sinh vào dataset, buộc mọi phiên sau tải thêm phần
mà một lệnh `augment` sinh lại được trong vài phút. Mô hình và báo cáo đi đường Output
(ô B4 gói `model.zip`, `reports_bundle.zip`).

Chuỗi thường dùng: file dataset vài phiên cho tới khi đủ mẫu (mỗi phiên chỉ sinh phần còn
thiếu — xem `--dry-run` để biết còn bao nhiêu), rồi một phiên file train.

Cần Kaggle token: [kaggle.com/settings](https://www.kaggle.com/settings) → Create New
Token, rồi Add-ons → Secrets thêm `KAGGLE_USERNAME` và `KAGGLE_KEY`. Không có token thì
phần đẩy tự tắt và bạn dùng đường Output → New Dataset như đoạn trên.

### Flow làm việc nhiều phiên

Corpus đầy đủ mất nhiều giờ GPU hơn một phiên Kaggle, nên mọi thứ dưới đây được thiết kế
quanh một câu: **phiên sau phải tiếp được đúng chỗ phiên trước dừng, và không được đè mất
công của nó.**

#### Chuẩn bị một lần cho cả tài khoản

1. [kaggle.com/settings](https://www.kaggle.com/settings) → **API Tokens** → Generate New
   Token → chuỗi `KGAT_…`
2. Trong notebook: **Add-ons → Secrets** → thêm `KAGGLE_API_TOKEN`, **tick attach**
   (secret là của tài khoản nhưng attach theo từng notebook). `HF_TOKEN` là tuỳ chọn —
   không có thì tải checkpoint bị throttle chứ không hỏng.
3. Đặt `DATASET_ID` ở ô setup thành kho của bạn.

#### Mỗi phiên — thứ tự không đổi

| | Ô | Làm gì | Dừng phiên khi |
|---|---|---|---|
| 1 | — | **Add Input**: kho lưu trữ + bộ dữ liệu nguồn | — |
| 2 | setup | `TTS_ENGINES` · `DATASET_ID` · `MIN_SECONDS` (`MODE` đã chốt theo file) | `MODE` bị sửa tay thành giá trị lạ |
| 3 | A1 | quy mô (`None` = toàn bộ nguồn) | không dataset nào đọc được |
| 4 | **A1b** | nạp kho → `migrate` → in trạng thái, cả tổng lẫn **theo từng nguồn** | kho có dữ liệu mà **không mount được**; file train mà không có corpus; corpus chỉ một lớp |
| 5 | A2 | `ingest` — bỏ qua phần đã có **trước khi giải mã** | <10 real · <3 speaker · 0 transcript |
| 6 | **A2c** | `validate --fix` — **chỉ soi phần chưa đóng dấu** | >20% corpus hỏng (lỗi hệ thống, không tự xoá) |
| 7 | A2b | kiểm `Đồng bộ: BẬT` → chốt mốc sau ingest | — |
| 8 | A3b | `--dry-run` → `generate` (đẩy nền mỗi speaker) | cloning hỏng khi nó là nguồn fake duy nhất |
| 9 | A3c | đếm lại: `còn 0` ⇒ xong | — |
| 10 | A4 | `validate` + thống kê + nghe thử + đo giống giọng/phát âm | corpus không đạt chuẩn |
| 11 | A5 | `sync_now()` (chặn, đợi lượt nền) + `sync_log()` | — |
| 12 | B | split → augment → features → train → evaluate | ở file train, phiên riêng |

Bảng trên là hai file gộp lại: ô 1–11 thuộc `aidetector_dataset.ipynb`, ô 12 thuộc
`aidetector_train.ipynb`; ô 2 và 4 có ở cả hai và giống nhau từng byte.

Chuỗi điển hình: file dataset vài phiên cho tới khi **A3c** báo `còn 0 phải sinh`, rồi một
phiên file train.

**Trước khi thả lượt sinh nhiều giờ đầu tiên**, gọi tay `sync_now()` ngay ở A2b khi corpus
mới chỉ có real: thấy `✔ thêm version` là đường ống thông. Đường đẩy có thể hỏng vì token,
vì quyền, vì tên dataset — biết điều đó sau 8 giờ thì quá muộn.

#### Cái gì làm cho "tiếp đúng chỗ" thành sự thật

| Bước | Không lặp lại được nhờ |
|---|---|
| chuẩn hoá | `utt_id = stable_id(source, speaker, key)` — `ingest` bỏ qua trước khi giải mã file |
| đánh giá | cột `checked` giữ **vân tay chuẩn audio**; đổi `MIN_SECONDS` là vân tay đổi ⇒ soi lại toàn bộ |
| sinh fake | đối chiếu `utt_id` trong manifest; `--dry-run` đếm bằng đúng phép đếm của lượt sinh |
| chốt tiến độ | `--after-speaker` đẩy nền ở ranh giới mỗi speaker, khoá PID chống chồng lượt |

#### Cái gì bảo vệ dữ liệu cũ

`kaggle datasets version` tạo ảnh chụp **toàn bộ** thư mục staging, không cộng dồn — nên
một phiên lỡ bắt đầu từ đầu mà đẩy lên là xoá công của mọi phiên trước. Ba tầng chặn:

1. **A1b dừng hẳn** khi kho có `corpus.zip`/`metadata.csv`/`progress.json` mà phiên này
   không mount được. Chặn ở phút đầu, trước cả ingest.
2. **Đếm trước khi đẩy**: tải `progress.json` (vài KB) so `dataset_records`; nhỏ hơn thì
   `TỪ CHỐI ĐẨY`. Đọc không được thì không chặn — trục trặc mạng không được làm đứng lượt
   sinh nhiều giờ.
3. **`KEEP_OLD_VERSIONS = True`**: version cũ vẫn nằm trên Kaggle, lấy lại được từ tab
   Data. Đặt `False` chỉ khi dung lượng thành vấn đề.

Còn giữa các **nguồn** với nhau thì cách ly sẵn: `utt_id` mang tên nguồn, đường dẫn tách
theo nguồn (`real/dataset_a/…`), và `ingest` chỉ thêm chứ không ghi đè.

#### Thêm một bộ dữ liệu mới

Cây chuẩn cũng là định dạng trao đổi, nên bộ dữ liệu lạ chỉ cần convert **một lần**:

```bash
python -m aidetector ingest /đường/dẫn/dataset_B --name dataset_b
python -m aidetector validate --fix        # chỉ soi phần mới
python -m aidetector generate --engines omnivoice
```

Trong notebook, ô A1 tự dò và chọn **một** dataset có điểm cao nhất. Muốn nạp bộ thứ hai
thì đặt tay `RAW = "/kaggle/input/<tên>"` rồi chạy lại A2 — notebook chưa nạp nhiều nguồn
trong một lượt. Tên nguồn mặc định lấy theo tên thư mục, nên đặt tên thư mục cho gọn hoặc
truyền `--name`.

`progress.json` tách theo nguồn, nên phiên sau nhìn một dòng là biết bộ nào đã tới đâu:

```
nguồn dataset_a            real   8246 · fake   4200 · đã duyệt   8246
nguồn dataset_b            real   3100 · fake      0 · đã duyệt   3100
```

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

### Thử notebook ngay trên máy, không cần Kaggle

```bash
python scripts/run_notebook_locally.py /tmp/kaggle-sandbox /đường/dẫn/vivos
```

Thay `/kaggle/working` và `/kaggle/input` bằng thư mục sandbox rồi chạy tuần tự mọi
ô code trong cùng một namespace, đúng như Jupyter. Bắt được những lỗi mà kiểm tra
cú pháp bỏ sót: stage nuốt lỗi, ô đọc phải dữ liệu cũ, engine tuỳ chọn làm dừng cả
phiên. Chạy lại sau mỗi lần `build_kaggle_notebook.py`.
