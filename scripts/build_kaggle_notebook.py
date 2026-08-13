#!/usr/bin/env python3
"""Sinh notebook Kaggle **độc lập** — nhúng cả package vào trong một file .ipynb.

Notebook do script này tạo ra không cần clone repo, không cần Kaggle Dataset chứa
code: toàn bộ `aidetector/` + `configs/` được nén, mã hoá base64 và đặt thẳng
trong ô bung mã nguồn. Người dùng chỉ việc `File → Import Notebook` rồi chạy.

Notebook chia làm hai phần tách bạch:

    PHẦN A — tạo & kiểm tra dataset   (ingest → generate → validate → nghe thử)
    PHẦN B — huấn luyện               (split → augment → features → train → evaluate)

Phần A có công tắc `SMOKE` để chạy thử với vài chục mẫu trước khi làm thật, và
đóng gói dataset ra zip. Chỉ khi dataset đã ưng ý mới chạy tiếp phần B.

    python scripts/build_kaggle_notebook.py

Chạy lại sau mỗi lần sửa code để notebook không bị lệch với repo.
"""

from __future__ import annotations

import base64
import gzip
import hashlib
import io
import json
import tarfile
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "notebooks" / "aidetector_kaggle.ipynb"

#: Những gì được nhúng. Cố ý KHÔNG nhúng tests/ và notebooks/ cho nhẹ.
INCLUDE_GLOBS = ("aidetector/**/*.py", "configs/*.yaml", "requirements.txt",
                 "requirements-generate.txt")
EXCLUDE_PARTS = ("__pycache__", ".pytest_cache")


# --------------------------------------------------------------------- payload
def collect_files() -> list[Path]:
    files: list[Path] = []
    for pattern in INCLUDE_GLOBS:
        for path in sorted(REPO.glob(pattern)):
            if path.is_file() and not any(part in EXCLUDE_PARTS for part in path.parts):
                files.append(path)
    if not files:
        raise SystemExit("Không tìm thấy file nguồn nào để nhúng — chạy từ gốc repo?")
    return files


def build_payload(files: list[Path]) -> tuple[str, str, int]:
    """Nén danh sách file thành tar.gz rồi mã hoá base64.

    Phải **tất định**: cùng mã nguồn ⇒ cùng payload, để notebook chỉ đổi khi code
    thật sự đổi (diff sạch, và test phát hiện được notebook lệch repo). Muốn vậy
    cần xoá mốc thời gian ở CẢ hai lớp — `mode="w:gz"` của tarfile vẫn để gzip ghi
    thời điểm hiện tại vào header, nên ở đây tar và gzip được tách riêng.
    """
    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode="w") as tf:
        for path in files:
            info = tf.gettarinfo(str(path), arcname=str(path.relative_to(REPO)))
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            with path.open("rb") as fh:
                tf.addfile(info, fh)

    gz_buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=gz_buffer, mode="wb", compresslevel=9, mtime=0) as gz:
        gz.write(tar_buffer.getvalue())

    raw = gz_buffer.getvalue()
    return base64.b64encode(raw).decode("ascii"), hashlib.sha256(raw).hexdigest(), len(raw)


def payload_literal(b64: str, width: int = 96) -> str:
    """Chuỗi base64 dài → nhiều literal nối ngầm cho dễ đọc trong notebook."""
    body = "\n".join(f'    "{line}"' for line in textwrap.wrap(b64, width))
    return f"_PAYLOAD = (\n{body}\n)"


# -------------------------------------------------------------------- notebook
def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {},
            "source": text.strip("\n").splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


def build_cells(payload: str, sha: str, size: int, n_files: int) -> list[dict]:
    return [
md(f"""
# ai-detector — phát hiện giọng nói giả tiếng Việt

Notebook **tự chứa toàn bộ mã nguồn** ({n_files} file, {size / 1024:.0f} KB nhúng sẵn) —
không cần clone repo, không cần dataset chứa code. Import lên Kaggle là chạy được.

```
REAL (giọng thật tiếng Việt)
   └── Piper · Kokoro · OmniVoice ──> FAKE
                 └── augmentation ──> WavLM ──> Classifier ──> REAL / FAKE
```

## Notebook chia làm hai phần — chạy phần A trước

| | Làm gì | Khi nào chạy |
|---|---|---|
| **PHẦN A** | tạo dataset: ingest → generate → **kiểm tra + nghe thử** → đóng gói | chạy trước, xem dataset có ổn không |
| **PHẦN B** | huấn luyện: split → augment → WavLM → classifier → đánh giá | chỉ chạy khi dataset đã ưng ý |

Phần A có công tắc **`SMOKE = True`**: chạy thử ~40 mẫu trong vài phút để xem
engine nào hoạt động, audio nghe ra sao. Ưng rồi mới đặt `SMOKE = False` chạy thật.

## Cần bật trong panel bên phải

| Mục | Đặt thành | Vì sao |
|---|---|---|
| **Accelerator** | `GPU T4 x2` hoặc `P100` | OmniVoice (voice cloning) không chạy nổi trên CPU |
| **Internet** | `On` | tải WavLM, giọng Piper/Kokoro, cài thư viện |

Rồi **Add Input → Datasets** một bộ giọng thật tiếng Việt (VIVOS, Common Voice vi…).
Pipeline tự nhận diện định dạng — không cần chỉnh gì thêm.

> Phiên Kaggle ~9 giờ rồi **xoá sạch `/kaggle/working`**. Ô cuối phần A đóng gói
> dataset thành một zip để bạn lưu ra Dataset, phiên sau train mà khỏi tạo lại.
"""),

# ─────────────────────────────────────────────────────────── chuẩn bị
md("## 0. Chuẩn bị"),
code(f"""
# Toàn bộ package aidetector + configs, nén tar.gz rồi base64.
# sha256(payload) = {sha[:16]}…
{payload}

import base64, hashlib, importlib, io, os, shutil, sys, tarfile
from pathlib import Path

_raw = base64.b64decode(_PAYLOAD)
assert hashlib.sha256(_raw).hexdigest() == "{sha}", "payload hỏng khi sao chép notebook"

WORK = Path("/kaggle/working/ai-detector")
WORK.mkdir(parents=True, exist_ok=True)

# Xoá sạch cây mã nguồn cũ trước khi bung: chạy đè lên bản cũ sẽ để sót những file
# đã bị bỏ ở bản mới, và để lại __pycache__ cũ.
for _old in ("aidetector", "configs"):
    shutil.rmtree(WORK / _old, ignore_errors=True)

with tarfile.open(fileobj=io.BytesIO(_raw), mode="r:gz") as _tf:
    try:
        _tf.extractall(WORK, filter="data")     # Python >= 3.12
    except TypeError:
        _tf.extractall(WORK)

os.chdir(WORK)
if str(WORK) not in sys.path:
    sys.path.insert(0, str(WORK))

# Kernel Kaggle sống xuyên suốt nhiều lần chạy. Nếu phiên trước đã import
# aidetector, Python giữ nguyên module cũ trong sys.modules và lờ đi mã vừa bung —
# biểu hiện là những lỗi rất khó hiểu kiểu "cannot import name X" dù X có trong
# file. Phải gỡ chúng ra để lần import sau đọc lại từ đĩa.
_stale = [m for m in sys.modules if m == "aidetector" or m.startswith("aidetector.")]
for _m in _stale:
    del sys.modules[_m]
importlib.invalidate_caches()

CFG = "configs/kaggle.yaml"


# Chạy một stage của pipeline và DỪNG notebook ngay nếu nó lỗi.
# Không dùng `!python -m aidetector ...`: trong Jupyter, lệnh shell lỗi vẫn để
# notebook chạy tiếp các ô sau, nên một stage hỏng sẽ âm thầm kéo theo cả loạt lỗi
# vô nghĩa ở dưới — hoặc tệ hơn, chạy tiếp trên dữ liệu cũ còn sót lại.
def run(*args):
    import subprocess

    cmd = [sys.executable, "-m", "aidetector", *[str(a) for a in args], "-c", CFG]
    print("$ python -m aidetector " + " ".join(str(a) for a in args) + f" -c {{CFG}}\\n")
    if subprocess.run(cmd).returncode != 0:
        raise SystemExit(f"✖ Stage {{args[0]!r}} thất bại — xem log ngay phía trên, "
                         f"đừng chạy tiếp các ô sau.")


print(f"Đã bung {{len(_raw) / 1024:.0f}} KB mã nguồn vào {{WORK}}")
if _stale:
    print(f"Đã gỡ {{len(_stale)}} module aidetector cũ khỏi bộ nhớ kernel")
"""),

md("""
Cài thư viện. Kaggle có sẵn torch + CUDA nên chỉ cài phần thiếu; ba engine sinh
fake cài riêng — cái nào lỗi thì bỏ qua, ô `info` ngay dưới cho biết cái nào dùng được.
"""),
code("""
# Engine cài trước, requirements.txt cài SAU CÙNG: nó ghim transformers <5 (bản
# Kaggle cài sẵn là 5.x, mà kokoro-vietnamese chỉ chạy trên 4.x), nên phải để nó
# nói tiếng nói cuối cùng.
!pip install -q piper-tts                                                 || true
!pip install -q git+https://github.com/iamdinhthuan/Kokoro-Vietnamese.git || true
!pip install -q omnivoice                                                 || true

!pip install -q -r requirements.txt

!apt-get -qq install -y ffmpeg > /dev/null 2>&1 || true   # cần cho augment MP3/AAC

import transformers, torch
print(f"transformers {transformers.__version__} · torch {torch.__version__} "
      f"· CUDA {torch.cuda.is_available()}")
"""),
code("""
run("info")
"""),

# ─────────────────────────────────────────────────────────── PHẦN A
md("""
---
# PHẦN A — Tạo dataset

Mục tiêu của phần này là ra được một corpus **đạt chuẩn và cân bằng**, kiểm tra tận
tai trước khi tốn thời gian huấn luyện.
"""),

md("""
## A1. Chọn dataset thật + đặt quy mô

`SMOKE = True` chạy thử nhanh (~40 real + 40 fake, vài phút). Xem kết quả ở A4–A5,
ưng rồi đặt `SMOKE = False` và chạy lại từ A2 để làm thật.
"""),
code("""
import logging
from pathlib import Path

from aidetector.ingest import detect_adapter
from aidetector.ingest.base import describe_directory

SMOKE = True        # ← True: chạy thử nhanh · False: chạy thật
RAW = None          # ← đặt tay nếu tự dò không đúng, vd "/kaggle/input/vivos"

if SMOKE:
    N_REAL, PER_SPEAKER, N_FAKE_TTS, N_FAKE_CLONE = 60, 8, 30, 15
else:
    N_REAL, PER_SPEAKER, N_FAKE_TTS, N_FAKE_CLONE = 4000, 120, 1200, 800

# Soi TỪNG dataset đang mount rồi chọn cái dùng được, thay vì lấy bừa cái đầu tiên:
# một dataset rỗng hay sai định dạng đứng đầu bảng chữ cái sẽ làm hỏng cả phiên.
logging.getLogger("aidetector.ingest").setLevel(logging.WARNING)
mounted = sorted(p for p in Path("/kaggle/input").glob("*") if p.is_dir())
if not mounted:
    raise SystemExit("Chưa add dataset nào — Add Input → Datasets ở panel bên phải.")

print("Dataset đang mount:")
usable = []
for folder in mounted:
    try:
        adapter, score, effective = detect_adapter(folder)
    except ValueError as exc:
        reason = next((l.strip() for l in str(exc).splitlines()[1:] if l.strip()),
                      "không nhận diện được")
        print(f"  ✖ {folder.name:<26} {reason}")
        continue
    where = "" if effective == folder else f" tại {effective.relative_to(folder)}/"
    print(f"  ✔ {folder.name:<26} {adapter.name} (điểm {score:.2f}){where}")
    usable.append((score, folder))

if RAW is None:
    if not usable:
        raise SystemExit(
            "Không dataset nào chứa audio đọc được. Chi tiết:\\n"
            + "\\n".join(f"[{p.name}]\\n" + describe_directory(p) for p in mounted)
        )
    usable.sort(key=lambda pair: -pair[0])
    RAW = str(usable[0][1])

print(f"\\nNguồn REAL : {RAW}")
print(f"Chế độ     : {'CHẠY THỬ' if SMOKE else 'CHẠY THẬT'}")
print(f"Quy mô     : {N_REAL} real · {N_FAKE_TTS} fake TTS · {N_FAKE_CLONE} fake cloning")
"""),

md("""
## A2. REAL — nạp giọng thật về chuẩn corpus

`ingest` tự nhận diện loại dataset (VIVOS / Common Voice / thư mục wav / real+fake
chia sẵn) rồi ép mọi file về đúng một chuẩn:

| | |
|---|---|
| Sample rate · kênh | 16 000 Hz · mono |
| Định dạng | WAV, 16-bit PCM |
| Độ dài | 3–10 giây (file dài hơn cắt thành nhiều đoạn) |
| Mức âm lượng | RMS −23 dBFS, trần peak −1 dBFS |
| Im lặng · clipping · NaN | cắt bớt · không được có · không được có |

Real và fake dùng **chung** chuỗi chuẩn hoá này, nên mô hình không thể phân biệt hai
lớp bằng định dạng hay độ to.
"""),
code("""
run("ingest", RAW, "--limit", N_REAL, "--per-speaker", PER_SPEAKER)
"""),
code("""
# Chặn sớm: ba điều kiện dưới đây mà không đạt thì mọi bước sau đều vô nghĩa.
from aidetector.config import Config
from aidetector.corpus.manifest import Manifest

manifest = Manifest.load(Config.load(CFG)["paths.corpus"], required=True)
n_real = len(manifest.reals)
n_speakers = len(manifest.speakers("real"))
n_text = sum(1 for r in manifest.reals if r.text)

print(f"real={n_real} · speaker={n_speakers} · có transcript={n_text}")
problems = []
if n_real < 10:
    problems.append(f"Chỉ nạp được {n_real} audio thật — kiểm tra RAW có trỏ đúng dataset không.")
if n_speakers < 3:
    problems.append(
        f"Chỉ có {n_speakers} speaker — không chia được train/val/test speaker-disjoint. "
        "Adapter có thể đang đọc sai cấu trúc thư mục.")
if n_text == 0:
    problems.append(
        "Không có transcript nào — fake sẽ phải dùng câu dự phòng và không ghép cặp "
        "được với real. Hãy dùng bộ dữ liệu có transcript (VIVOS, Common Voice).")
if problems:
    raise SystemExit("DỪNG LẠI:\\n" + "\\n".join(f"  • {p}" for p in problems))
print("✔ dataset thật đủ điều kiện để sinh fake")
"""),

md("""
## A3. FAKE — sinh audio giả

Mỗi audio giả sinh từ **chính transcript và speaker của một utterance thật**, nên
luôn có bản real đối chứng cùng nội dung cùng giọng — mô hình không thể phân loại
theo chủ đề câu nói hay theo danh tính người nói.

`generate` là idempotent: dừng giữa chừng rồi chạy lại chỉ sinh phần còn thiếu.
"""),
code("""
# Hai engine TTS giọng cố định — nhanh, chạy được cả trên CPU.
run("generate", "--engines", "piper", "kokoro", "--count", N_FAKE_TTS)
"""),
code("""
# OmniVoice: voice cloning zero-shot, clone thẳng giọng speaker thật từ một câu
# khác của họ. Đây là engine TUỲ CHỌN — chậm hơn nhiều và cần GPU. Chưa cài được
# thì ô này chỉ báo bỏ qua, Piper/Kokoro ở trên đã đủ để đi tiếp.
run("generate", "--engines", "omnivoice", "--count", N_FAKE_CLONE)
"""),

md("""
## A4. Kiểm tra dataset

Ba việc: soi toàn corpus xem có file nào phạm chuẩn, xem thống kê, và **nghe thử**.
"""),
code("""
run("validate")
"""),
code("""
# Thống kê chi tiết: số lượng, thời lượng, cân bằng hai lớp, phủ speaker
from collections import Counter

from aidetector.config import Config
from aidetector.corpus.manifest import Manifest

cfg = Config.load(CFG)
manifest = Manifest.load(cfg["paths.corpus"], required=True)
stats = manifest.stats()

n_real = stats["by_label"].get("real", 0)
n_fake = stats["by_label"].get("fake", 0)
print(f"Tổng      : {stats['total']} utt · {stats['hours']} giờ")
print(f"REAL/FAKE : {n_real} / {n_fake}"
      + (f"   ⚠ lệch {max(n_real, n_fake) / max(min(n_real, n_fake), 1):.1f}×"
         if min(n_real, n_fake) and max(n_real, n_fake) / min(n_real, n_fake) > 1.3 else "   ✔ cân bằng"))
print(f"Speaker   : {stats['speakers_real']}")

print("\\nTheo engine:")
for name, count in sorted(stats["by_generator"].items()):
    print(f"  {name:<42} {count}")

durations = [r.duration for r in manifest]
print(f"\\nĐộ dài    : {min(durations):.1f}–{max(durations):.1f}s "
      f"(trung bình {sum(durations) / len(durations):.1f}s)")

paired = sum(1 for r in manifest.fakes if r.ref_utt_id in manifest)
print(f"Ghép cặp  : {paired}/{len(manifest.fakes)} fake có real đối chứng cùng nội dung")

no_text = sum(1 for r in manifest.reals if not r.text)
if no_text:
    print(f"⚠ {no_text} utt real không có transcript — không dùng làm khuôn sinh fake được")
"""),
code("""
# NGHE THỬ: mỗi cặp là cùng một câu, cùng một speaker — real trước, fake sau.
from IPython.display import Audio, display

pairs = []
for fake in manifest.fakes:
    real = manifest.get(fake.ref_utt_id)
    if real is not None:
        pairs.append((real, fake))
    if len(pairs) >= 3:
        break

if not pairs:
    print("Chưa có fake nào — chạy lại ô A3.")
for real, fake in pairs:
    print("=" * 90)
    print(f"Câu    : {real.text[:110]}")
    print(f"Speaker: {real.speaker}   ·   engine: {fake.generator}")
    print(f"REAL ({real.duration:.1f}s)")
    display(Audio(str(manifest.abs_path(real))))
    print(f"FAKE ({fake.duration:.1f}s)")
    display(Audio(str(manifest.abs_path(fake))))
"""),
code("""
# Dạng sóng + phổ của một cặp — fake thường mượt và đều hơn ở vùng tần số cao.
import matplotlib.pyplot as plt
import numpy as np

from aidetector.corpus.spec import load_audio

if pairs:
    real, fake = pairs[0]
    fig, axes = plt.subplots(2, 2, figsize=(13, 6))
    for col, (rec, title) in enumerate([(real, "REAL"), (fake, f"FAKE · {fake.generator}")]):
        audio = load_audio(manifest.abs_path(rec), 16_000)
        axes[0, col].plot(np.arange(len(audio)) / 16_000, audio, lw=0.4)
        axes[0, col].set(title=f"{title} — dạng sóng", xlabel="giây", ylim=(-1, 1))
        axes[1, col].specgram(audio, Fs=16_000, NFFT=512, noverlap=256, cmap="magma")
        axes[1, col].set(title=f"{title} — phổ", xlabel="giây", ylabel="Hz")
    fig.tight_layout()
    plt.show()
"""),

md("""
## A5. Đóng gói dataset

`/kaggle/working` bị xoá khi hết phiên, và commit output với hàng chục nghìn file wav
rời rạc thì rất chậm — nên gói tất cả vào **một** zip.

Chạy xong notebook: **Output → New Dataset**. Phiên sau chỉ cần add dataset đó rồi
`unpack`, khỏi phải ingest và generate lại.
"""),
code("""
run("pack", "--out", "/kaggle/working/corpus.zip")
!ls -lh /kaggle/working/corpus.zip
"""),

md("""
> ### Dừng lại ở đây nếu chỉ cần dataset
>
> Xem lại A4: hai lớp có cân bằng không, engine nào sinh được bao nhiêu, nghe thử
> thấy hợp lý chưa. Nếu đang ở `SMOKE = True` thì giờ đặt `SMOKE = False` ở ô A1 và
> chạy lại A2–A5 để làm thật. Ưng rồi mới sang phần B.
"""),

# ─────────────────────────────────────────────────────────── PHẦN B
md("""
---
# PHẦN B — Huấn luyện

Chạy phần này khi dataset đã ưng. Nếu dataset đến từ phiên trước, chạy ô ngay dưới
để bung nó ra rồi bỏ qua toàn bộ phần A.
"""),
code("""
# Chỉ chạy khi dùng lại dataset của phiên trước:
# run("unpack", "/kaggle/input/<tên-dataset>/corpus.zip")
"""),

md("""
## B1. Chia tập → augment

`split` chạy **trước** `augment`: bản augment chỉ sinh cho train và bám đúng split
của bản gốc, còn val/test giữ audio sạch để số đo phản ánh dữ liệu thật. Chia
speaker-disjoint nên không có speaker nào xuất hiện ở hai tập.

Thêm `--holdout omnivoice` nếu muốn giữ hẳn một engine riêng cho test — đó là phép
đo sát thực tế nhất: mô hình có bắt được engine **chưa từng thấy** hay không.
"""),
code("""
run("split")
run("augment", "--copies", 1)
"""),

md("""
## B2. WavLM → Classifier

Embedding cache theo `utt_id` nên chạy lại chỉ trích phần mới. Đổi backbone chỉ cần
`--set features.backbone.name=wav2vec2` — cache tách riêng, không đè lên nhau.
"""),
code("""
run("features")
run("train")
run("evaluate")
"""),

md("## B3. Kết quả"),
code("""
import json
from pathlib import Path
from IPython.display import Image, display

metrics = json.loads(Path("/kaggle/working/reports/metrics.json").read_text())
overall = metrics["overall"]
print(f"EER      : {overall['eer'] * 100:.2f}%      ← số đo chính")
print(f"ROC-AUC  : {overall['roc_auc']:.4f}")
print(f"min-DCF  : {overall['min_dcf']:.4f}")
print(f"Accuracy : {overall['accuracy'] * 100:.2f}%  (ngưỡng {overall['threshold']:.3f})")

print("\\nTheo từng generator:")
for name, entry in metrics["by_generator"].items():
    if "eer_vs_all_real" in entry:
        print(f"  {name:<42} n={entry['n']:>5} · EER {entry['eer_vs_all_real'] * 100:6.2f}%"
              f" · bắt được {entry['detection_rate'] * 100:5.1f}%")
    elif "false_alarm_rate" in entry:
        print(f"  {name:<42} n={entry['n']:>5} · báo nhầm {entry['false_alarm_rate'] * 100:5.1f}%")

print("\\nClean vs augmented:")
for name, entry in metrics["by_condition"].items():
    print(f"  {name:<12} n={entry['n']:>5} · điểm trung bình {entry['mean_score']:.3f}")

display(Image("/kaggle/working/reports/curves.png"))
display(Image("/kaggle/working/reports/confusion_matrix.png"))
"""),

md("## B4. Thử trên file bất kỳ + lưu mô hình"),
code("""
import glob

mau = sorted(glob.glob("/kaggle/working/corpus/audio/fake/piper/*/*.wav"))[:5]
mau += sorted(glob.glob("/kaggle/working/corpus/audio/real/*/*/*.wav"))[:5]
run("detect", *mau)
"""),
code("""
import shutil
shutil.make_archive("/kaggle/working/model",          "zip", "/kaggle/working/checkpoints")
shutil.make_archive("/kaggle/working/reports_bundle", "zip", "/kaggle/working/reports")
!ls -lh /kaggle/working/*.zip
"""),

md("""
---
### Vài nút chỉnh hay dùng

```python
# Đổi backbone (cache đặc trưng tách riêng nên không đụng nhau)
run("run", "features", "train", "evaluate", "--set", "features.backbone.name=wav2vec2")

# Đo khả năng tổng quát sang engine chưa từng thấy
run("split", "--holdout", "omnivoice")
run("run", "features", "train", "evaluate")

# Augment mạnh tay hơn nếu clean và augmented chênh lệch nhiều
run("augment", "--copies", 3, "--set", "augment.ops.codec.p=0.8")
```

Toàn bộ tham số nằm trong `configs/default.yaml` (bản Kaggle kế thừa nó qua
`configs/kaggle.yaml`) — xem bằng `!cat configs/default.yaml`.
"""),
    ]


def main() -> None:
    files = collect_files()
    b64, sha, size = build_payload(files)
    notebook = {
        "cells": build_cells(payload_literal(b64), sha, size, len(files)),
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "accelerator": "GPU",
            "kaggle": {"accelerator": "nvidiaTeslaT4", "dataSources": [],
                       "isInternetEnabled": True, "language": "python",
                       "sourceType": "notebook"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Đã nhúng {len(files)} file ({size / 1024:.0f} KB nén) · sha256 {sha[:16]}…")
    print(f"→ {OUT}  ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
