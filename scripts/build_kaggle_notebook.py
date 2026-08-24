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

## Notebook chia làm hai phần — `MODE` chọn phiên này chạy phần nào

| | Làm gì | Khi nào chạy |
|---|---|---|
| **PHẦN A** | tạo dataset: ingest → generate → **kiểm tra + nghe thử** → đóng gói | chạy trước, xem dataset có ổn không |
| **PHẦN B** | huấn luyện: split → augment → WavLM → classifier → đánh giá | chỉ chạy khi dataset đã ưng ý |

Sinh fake bằng voice cloning mất nhiều giờ, nên hai phần thường **không** nằm cùng một
phiên. Công tắc **`MODE`** ở ô cài thư viện quyết định phiên này làm gì:

| `MODE` | Chạy | Khi nào dùng |
|---|---|---|
| `"dataset"` | chỉ phần A | dành cả phiên để sinh fake; corpus tự đẩy lên Dataset dọc đường |
| `"train"` | chỉ phần B | corpus đã có trong Dataset, chỉ muốn huấn luyện |
| `"both"` | A rồi B | chạy thử, hoặc quy mô nhỏ đủ gọn trong một phiên |

Ô nào không thuộc phần đang chạy thì tự bỏ qua và in ra lý do, nên **Save & Run All** ở
chế độ nào cũng đúng — không phải chọn tay từng ô.

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
#
# `optional=True` dành cho bước không bắt buộc (vd một engine sinh fake cần GPU
# hoặc cần quyền tải checkpoint): hỏng thì báo rồi đi tiếp, vì dữ liệu đã có từ
# các bước trước vẫn dùng được.
def run(*args, optional=False):
    import subprocess

    cmd = [sys.executable, "-m", "aidetector", *[str(a) for a in args], "-c", CFG]
    print("$ python -m aidetector " + " ".join(str(a) for a in args) + f" -c {{CFG}}\\n")
    if subprocess.run(cmd).returncode == 0:
        return True
    if optional:
        print(f"\\n⚠ Bước tuỳ chọn {{args[0]!r}} không chạy được — bỏ qua, đi tiếp.")
        return False
    raise SystemExit(f"✖ Stage {{args[0]!r}} thất bại — xem log ngay phía trên, "
                     f"đừng chạy tiếp các ô sau.")


print(f"Đã bung {{len(_raw) / 1024:.0f}} KB mã nguồn vào {{WORK}}")
if _stale:
    print(f"Đã gỡ {{len(_stale)}} module aidetector cũ khỏi bộ nhớ kernel")
"""),

md("""
Cài thư viện.

Hai công tắc của cả phiên nằm ở ô dưới, không ở A1, vì chính chúng quyết định phải cài
gói nào.

**`MODE`** — phiên này làm gì:

* **`"both"` (mặc định)** — chạy cả hai phần.
* **`"dataset"`** — chỉ phần A. Bỏ qua split/augment/train/evaluate.
* **`"train"`** — chỉ phần B, và **không cài engine sinh audio nào cả**: đỡ vài phút
  cài đặt, tránh hẳn màn giằng nhau về phiên bản `transformers` ở dưới. Corpus phải
  đến từ Input (xem A1b) — không có thì ô A1b dừng luôn thay vì train trên tay không.

**`TTS_ENGINES`** — engine giọng cố định, chỉ có nghĩa khi `MODE` còn tạo dataset:

* **rỗng (mặc định)** — chỉ voice cloning. Cài `transformers>=5.3` một lượt là xong.
* **`["piper", "kokoro"]`** — bật lại TTS. Kokoro cần `transformers` 4.x mà OmniVoice
  cần `>=5.3`; hai engine không sống chung trong một môi trường nên phải ghim 4.x ở đây
  rồi nâng lên 5.x ở A3b, tức sinh fake thành hai lượt.
"""),
code("""
# Phiên này làm gì: "dataset" (chỉ phần A) · "train" (chỉ phần B) · "both" (cả hai).
MODE = "both"

# Kho dữ liệu dùng chung cho MỌI chế độ: phần A đẩy corpus lên đây, mọi phiên sau nạp
# lại từ đây. Khai báo một chỗ duy nhất — A1b (nạp) và A2b (đẩy) đều đọc biến này, để
# không bao giờ có chuyện đẩy lên một dataset mà nạp về từ một dataset khác.
DATASET_ID = "sonpham12/vivos-fake-v2"

# Piper/Kokoro đang TẮT: giọng cố định, mô hình bắt ở EER 0.00% nên không dạy được gì,
# chỉ làm loãng dataset. Bật lại bằng: TTS_ENGINES = ["piper", "kokoro"]
TTS_ENGINES = []

if MODE not in ("dataset", "train", "both"):
    raise SystemExit(f'MODE={MODE!r} không hợp lệ — "dataset", "train" hoặc "both".')
MAKE_DATASET = MODE in ("dataset", "both")
DO_TRAIN = MODE in ("train", "both")

# Nói rõ vì sao một ô không làm gì: Run All mà im lặng thì log không đọc được.
def skipped(what):
    print(f"⏭ MODE={MODE!r} — bỏ qua {what}.")

!apt-get -qq install -y ffmpeg > /dev/null 2>&1 || true   # cần cho augment MP3/AAC

# subprocess chứ không `!pip`: magic của IPython không lồng vào `if` được.
import subprocess
import sys

def pip(*args, ok_to_fail=False):
    if subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args]).returncode:
        if not ok_to_fail:
            raise SystemExit(f"pip install {' '.join(args)} thất bại — xem log phía trên")
        print(f"⚠ bỏ qua: pip install {' '.join(args)}")

pip("-r", "requirements.txt")
# Image Kaggle đang có kaggle 2.0.2 (log phiên trước tự cảnh báo). Bản đó có thể chưa
# biết token kiểu mới `KGAT_`, mà đó lại là đường xác thực để đẩy dataset.
pip("-U", "kaggle", ok_to_fail=True)
if not MAKE_DATASET:
    # Không sinh audio thì không cần engine nào. WavLM chạy được trên cả hai nhánh
    # transformers nên cứ để bản Kaggle cài sẵn — đây là chế độ cài nhẹ nhất.
    print("Chỉ huấn luyện — không cài engine sinh audio.")
elif TTS_ENGINES:
    pip("piper-tts", ok_to_fail=True)
    pip("git+https://github.com/iamdinhthuan/Kokoro-Vietnamese.git", ok_to_fail=True)
    pip("transformers>=4.48,<5")
else:
    # Không có Kokoro thì bỏ được màn ghim-rồi-nâng transformers giữa phiên.
    pip("omnivoice", "transformers>=5.3")

import transformers, torch
print(f"MODE: {MODE} · phần A {'BẬT' if MAKE_DATASET else 'tắt'}"
      f" · phần B {'BẬT' if DO_TRAIN else 'tắt'}")
print(f"TTS: {TTS_ENGINES or 'tắt — chỉ voice cloning'}")
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

Ở `MODE = "train"` ô này chỉ đặt con số rồi thôi — nó **không** dò dataset giọng thật,
vì phiên chỉ-huấn-luyện mount corpus đã sinh sẵn chứ không mount VIVOS.
"""),
code("""
import logging
from pathlib import Path

from aidetector.ingest import detect_adapter
from aidetector.ingest.base import describe_directory

SMOKE = True        # ← True: chạy thử nhanh · False: chạy thật
RAW = None          # ← đặt tay nếu tự dò không đúng, vd "/kaggle/input/vivos"
# MODE và TTS_ENGINES đặt ở ô cài thư viện phía trên (chúng quyết định cài gói nào).
#
# `None` = KHÔNG áp trần nào. Ingest lấy mọi utterance đạt chuẩn của nguồn, và
# `generate` để `fake_to_real_ratio: 1.0` trong config tự tính ⇒ đúng một fake cho mỗi
# real. Không phải đoán con số nào, và không bao giờ lệch lớp.
#
# VIVOS đo thật: 12.420 file → 7.367 utterance đạt chuẩn (59,3%; phần bỏ là clip ngắn
# hơn min_seconds=3s), 65 speaker, ⇒ ~7,6 giờ sinh trên T4.
#
# PER_SPEAKER = None là quyết định có ý thức, không phải bỏ sót: trần 120 cho 5.395
# utterance và giữ mọi giọng ở mức xấp xỉ nhau, bỏ trần cho thêm 1.972 utterance nhưng
# chúng dồn vào những giọng nói nhiều (có giọng 250+, giọng khác ~20). Split là
# speaker-disjoint và test đo khả năng tổng quát sang GIỌNG MỚI, nên train lệch về vài
# giọng làm phép đo đó xấu đi. Đặt lại 120–200 nếu thấy EER trên test kém hơn val.
if SMOKE:
    N_REAL, PER_SPEAKER, N_FAKE_TTS, N_FAKE_CLONE = 60, 8, 30, 15
else:
    N_REAL, PER_SPEAKER, N_FAKE_TTS, N_FAKE_CLONE = None, None, None, None

# Dò dataset REAL chỉ khi phiên này thật sự sinh dữ liệu: MODE="train" mount corpus đã
# sinh sẵn chứ không mount VIVOS, nên đòi cho được một bộ giọng thật ở đây là dừng oan.
if not MAKE_DATASET:
    skipped("dò dataset REAL — corpus lấy từ Input ở ô A1b")
else:
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

    _muc = lambda n: "toàn bộ nguồn" if n is None else f"{n:,}"
    print(f"\\nNguồn REAL : {RAW}")
    print(f"Chế độ     : {'CHẠY THỬ' if SMOKE else 'CHẠY THẬT'}")
    print(f"Quy mô     : {_muc(N_REAL)} real · {_muc(N_FAKE_CLONE)} fake cloning"
          f" · {_muc(N_FAKE_TTS)} fake TTS (chỉ khi bật TTS_ENGINES)")
    print("Ước thời gian sinh: in ở ô A2 sau khi biết corpus có bao nhiêu real.")
"""),

md("""
### A1b. Nạp corpus của phiên trước

Bung `DATASET_ID` (khai báo ở ô setup) ra `/kaggle/working` để chạy tiếp. `ingest` và
`generate` đều idempotent theo `utt_id` nên chúng chỉ làm phần còn thiếu — không có bước
nào làm lại từ đầu.

Muốn nối lại thì phải **Add Input → Datasets → dataset đó**. Chưa add thì ô này vẫn hỏi
Kaggle xem dataset đang có gì (nếu đã cài token) rồi nhắc — chứ không im lặng bắt đầu lại
từ đầu và làm mất công phiên trước. Mount nhiều dataset thì ô này lấy **đúng** cái khớp
`DATASET_ID`, không phải cái đầu bảng chữ cái.

Ô này chạy ở **mọi** `MODE` — nó là đường duy nhất mang corpus vào phiên. Riêng
`MODE = "train"` thì corpus là điều kiện bắt buộc: không bung được gì, hoặc bung ra một
corpus thiếu hẳn một lớp, thì ô dừng ngay chứ không để phần B huấn luyện trên tay không.
"""),
code("""
import glob
import subprocess
from pathlib import Path

CORPUS = Path("/kaggle/working/corpus")

# Kaggle mount dataset ở /kaggle/input/<slug>. Tìm ĐÚNG dataset đã cấu hình trước rồi
# mới chấp nhận corpus.zip bất kỳ: mount nhiều dataset mà "lấy cái cuối theo abc" thì
# phiên này nối tiếp công của dataset nào là chuyện xổ số.
def _find(name):
    slug = DATASET_ID.split("/")[-1]
    return (sorted(glob.glob(f"/kaggle/input/{slug}/**/{name}", recursive=True))
            or sorted(glob.glob(f"/kaggle/input/**/{name}", recursive=True)))

_mounted = _find("corpus.zip")
_loose = _find("manifest.csv")

if CORPUS.joinpath("manifest.csv").exists():
    print("Corpus đã có sẵn trong /kaggle/working — không bung đè lên.")
    run("info")
elif _mounted:
    print(f"Bung corpus từ {_mounted[-1]}")
    run("unpack", _mounted[-1])
else:
    if _loose:
        # manifest để rời ngoài zip chính là để đọc tiến độ mà không phải tải cả GB.
        import csv

        with open(_loose[-1], encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        fakes = [r for r in rows if r.get("label") == "fake" and not r.get("augment")]
        print(f"Thấy manifest của dataset: {len(rows)} bản ghi · {len(fakes)} fake"
              f" · {len({r['speaker'] for r in fakes})} speaker đã có fake")
        print("Nhưng KHÔNG thấy corpus.zip — add đúng dataset vào Input rồi chạy lại ô này.")
    else:
        # Chưa mount thì vẫn hỏi API cho biết dataset đang có gì.
        r = subprocess.run(["kaggle", "datasets", "files", DATASET_ID],
                           capture_output=True, text=True)
        if r.returncode == 0:
            print("Dataset trên Kaggle đang có:")
            print(r.stdout.strip()[:800])
            print(f"\\n→ Add Input → Datasets → {DATASET_ID} rồi chạy lại ô này"
                  " để nối tiếp thay vì làm lại từ đầu.")
        else:
            print("Chưa nối được tới dataset (chưa add Input, chưa có token, hoặc dataset trống).")
            if MAKE_DATASET:
                print("Phiên này sẽ bắt đầu từ đầu.")

# Đã tới đâu rồi — con số này là mốc của cả phiên: phần A biết còn phải sinh bao nhiêu,
# phần B biết mình sắp huấn luyện trên cái gì.
if CORPUS.joinpath("manifest.csv").exists():
    from aidetector.corpus.manifest import Manifest

    _m = Manifest.load(CORPUS, required=True)
    _done = len({f.speaker for f in _m.fakes})
    print(f"\\nCorpus đang có: {len(_m.reals)} real · {len(_m.fakes)} fake"
          f" · {_done}/{len(_m.speakers('real'))} speaker đã có fake")

    # ĐÃ GEN ĐẾN ĐÂU so với đích "mỗi real đủ điều kiện có một fake". Đây là câu duy
    # nhất đáng hỏi trước khi bắt đầu một phiên nối tiếp, và nó đọc được từ chính
    # manifest — không cần nạp engine, không cần GPU.
    from aidetector.config import Config
    from aidetector.generate.texts import is_usable

    _c = Config.load(CFG)
    _pool = [r for r in _m.reals if not r.augment and r.text and is_usable(
        r.text, int(_c.get("generate.min_words", 6)), int(_c.get("generate.max_words", 40)))]
    _co_fake = {f.ref_utt_id for f in _m.fakes}
    _xong = sum(1 for r in _pool if r.utt_id in _co_fake)
    _con = len(_pool) - _xong
    print(f"Tiến độ gen   : {_xong}/{len(_pool)} real đủ điều kiện đã có fake"
          f" ({100 * _xong / max(len(_pool), 1):.0f}%) · còn {_con} mẫu"
          f" ≈ {_con * 3.7 / 3600:.1f} giờ trên T4")
    # Chỉ-huấn-luyện thì corpus không phải tiện lợi mà là điều kiện sống.
    if not MAKE_DATASET and not (_m.reals and _m.fakes):
        raise SystemExit(f"Corpus chỉ có một lớp (real={len(_m.reals)}, fake={len(_m.fakes)})"
                         " — phân loại real/fake cần cả hai.")
elif not MAKE_DATASET:
    raise SystemExit(
        f"MODE={MODE!r} nhưng không bung được corpus nào — không có gì để huấn luyện.\\n"
        f"Add Input → Datasets → {DATASET_ID} rồi chạy lại ô này."
    )
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

**`--limit` rải đều cho mọi speaker.** Adapter duyệt theo thư mục nên nó trả hết giọng
này mới sang giọng khác; cắt theo thứ tự đó là những giọng cuối bảng không có lấy một
utterance — trong khi chia tập là speaker-disjoint và **fake chỉ sinh được cho speaker đã
có real**. Nên `ingest` xếp lại nguồn theo vòng tròn qua speaker trước khi cắt: VIVOS 65
giọng với `N_REAL = 4000` ra ~61 utterance mỗi giọng, và fake phủ đủ 65 giọng đó.

`--limit` cũng là **tổng trong corpus**, không phải "thêm bao nhiêu lần này": phiên sau
chạy lại đúng lệnh đó thì ingest không làm gì (và đó không phải lỗi). Muốn thêm giọng
hoặc thêm câu thì nâng `N_REAL` — vòng tròn tự dồn phần thêm vào những giọng còn ít.
"""),
code("""
def _n_records():
    csv_path = CORPUS / "manifest.csv"
    return sum(1 for _ in csv_path.open(encoding="utf-8")) - 1 if csv_path.exists() else 0

# Cờ nào có trần thì truyền, không thì để trống — `--limit` vắng mặt nghĩa là lấy hết.
_tran = [*(["--limit", N_REAL] if N_REAL else []),
         *(["--per-speaker", PER_SPEAKER] if PER_SPEAKER else [])]

_before = _n_records()
if MAKE_DATASET:
    run("ingest", RAW, *_tran)
else:
    skipped("ingest — corpus đã bung ở A1b")

# Có thêm bản ghi thì mới có cái để đẩy. Không có thì bỏ lượt đẩy ở A2b: gói và tải cả
# GB dữ liệu y nguyên như trên dataset là đốt hàng chục phút của phiên vào việc vô ích.
INGEST_ADDED = _n_records() - _before
print(f"ingest thêm {INGEST_ADDED} bản ghi · corpus {_n_records()} bản ghi")
"""),
code("""
if MAKE_DATASET:
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
        # 3,7 giây/mẫu là số đo thật trên T4 (log phiên trước), không phải ước lượng suông.
    print(f"✔ dataset thật đủ điều kiện để sinh fake")
    print(f"  Sinh đủ 1 fake cho mỗi real ⇒ {n_real} mẫu ⇒ ~{n_real * 3.7 / 3600:.1f} giờ"
          f" trên T4 nếu bắt đầu từ 0. Phần đã có ở phiên trước không phải làm lại.")
else:
    skipped("kiểm tra dataset REAL — chỉ có nghĩa trước khi sinh fake")
"""),

md("""
## A2b. Đồng bộ lên Kaggle Dataset

Đích là `DATASET_ID` ở ô setup — **cùng một biến** mà ô A1b nạp về, nên không bao giờ có
chuyện đẩy lên một chỗ rồi phiên sau nạp từ chỗ khác. Mỗi lần đẩy gồm **toàn bộ**:
`corpus.zip` (real + fake + manifest) cộng một bản `manifest.csv` để rời bên ngoài — nhờ
đó A1b đọc được tiến độ mà không phải tải cả GB.

Mục này đặt **trước** bước sinh vì bước sinh gọi `sync_corpus.py`, file đó phải có sẵn.

#### Chu kỳ đẩy — ba mốc

| Mốc | Ở đâu | Bịt lỗ nào |
|---|---|---|
| **sau `ingest`** | ngay ô này, chỉ khi ingest thêm bản ghi | out lúc sinh giọng đầu — đúng lúc chưa có mốc nào được chốt |
| **xong MỖI speaker** | `generate --after-speaker`, chạy nền | out giữa lượt sinh nhiều giờ |
| **cuối phiên** | ô A5, `--force` — chặn, đợi lượt nền xong | phần lẻ sau mốc cuối |

Speaker là mốc dày nhất mà corpus có: trước ranh giới đó, phần đã xong chỉ là một nhúm
mẫu lẻ giữa chừng. 4000 mẫu trên ~46 speaker ⇒ mỗi giọng ~6 phút, nên out bất ngờ thì
mất tối đa cỡ **6 phút GPU**.

**Lượt đẩy chạy NỀN — đó là điều làm nhịp dày này khả thi.** Gói ~1 GB rồi upload mất cỡ
1–3 phút. Đẩy mà chặn dòng sinh thì 46 lượt cộng lại là hơn một giờ GPU đứng chờ, tức trả
hơn một giờ để rút cửa sổ mất mát từ 20 phút xuống 6 phút — lỗ. Chạy nền thì gói và upload
là việc của CPU với mạng, GPU sinh speaker tiếp, giá gần như bằng không.

Đổi lại phải giữ hai bất biến:

* **Không chồng lượt** — khoá theo PID. Speaker xong sớm hơn thời gian đẩy thì bỏ lượt đó,
  và không mất gì: mỗi lần đẩy là ảnh chụp **toàn bộ** corpus nên mốc sau gói cả phần vừa
  bỏ. Hai lượt cùng lúc thì lượt sau gói đè lên đúng file zip lượt trước đang tải.
* **Ảnh chụp nhất quán** — `pack` đọc manifest rồi zip đúng những file trong đó. Manifest
  ghi bằng `tmp` + `os.replace` nên bản đọc được luôn nguyên vẹn; audio sinh ra sau thời
  điểm đó chỉ đơn giản là chưa có trong ảnh này, lượt sau lấy.

`SYNC_EVERY_MINUTES = 0` là không chặn nhịp. Đặt > 0 nếu mạng chậm. `kaggle datasets
version` bị từ chối khi version trước còn đang xử lý — chuyện thường ở nhịp dày, và vô hại
vì lượt sau là ảnh chụp đầy đủ. Script chốt nhịp ngay khi bắt đầu chứ không đợi thành công,
nên hỏng thì chờ lượt sau thay vì gói-và-tải-lại liên tục.

**Số version là thứ duy nhất tăng theo nhịp mà không tự dọn.** Mỗi lượt đẩy là một version
~1 GB, nhịp theo speaker ⇒ vài chục version mỗi phiên. `KEEP_OLD_VERSIONS = False` thêm
`--delete-old-versions` để dataset chỉ giữ bản mới nhất — mất mát duy nhất là đường lùi,
vì bản mới nhất luôn là superset của mọi bản cũ. Mặc định vẫn `True` vì xoá version là
không lấy lại được; đổi khi dung lượng thành vấn đề.

Lượt đẩy nền không in được vào ô nào — xem bằng `sync_log()`; ô A5 tự in toàn bộ.

#### Cả ba chế độ dùng chung dataset này

| `MODE` | Nạp về | Đẩy lên |
|---|---|---|
| `"dataset"` | A1b bung corpus phiên trước | ba mốc ở trên |
| `"both"` | như trên | như trên |
| `"train"` | A1b bung corpus — **bắt buộc**, không có thì dừng ngay | không đẩy |

`"train"` không đẩy là có chủ ý, không phải bỏ sót: phần B chạy `augment`, nó ghi thêm
bản nhiễu/nén vào corpus. Đẩy sau đó là bơm dữ liệu phái sinh vào dataset, buộc mọi phiên
sau tải thêm phần mà một lệnh `augment` sinh lại được trong vài phút. Mô hình và báo cáo
đi đường Output — ô B4 gói `model.zip` và `reports_bundle.zip`.

Cài token một lần: [kaggle.com/settings](https://www.kaggle.com/settings) → Create New
Token → mở `kaggle.json`, rồi Add-ons → Secrets thêm `KAGGLE_USERNAME` và `KAGGLE_KEY`.
"""),
code("""
# DATASET_ID khai báo ở ô setup — cùng một biến với ô A1b nạp về.
#
# 0 = đẩy sau MỌI speaker. Làm được vì lượt đẩy chạy NỀN: gói + upload là việc của CPU và
# mạng, GPU vẫn sinh tiếp trong lúc đó. Đặt số > 0 nếu muốn thưa hơn — mạng chậm, hoặc
# muốn ít version trên dataset hơn.
SYNC_EVERY_MINUTES = 0

# Mỗi lượt đẩy tạo một version mới, và mỗi version là ảnh chụp TOÀN BỘ corpus. Nhịp theo
# speaker ⇒ vài chục version ~1 GB mỗi phiên. True = giữ hết (còn đường lùi nếu một bản
# đẩy ra rác); False = thêm `--delete-old-versions`, dataset chỉ giữ bản mới nhất.
#
# Giữ mặc định True: xoá version là không lấy lại được. Đổi sang False khi dung lượng
# dataset thành vấn đề — bản mới nhất luôn là superset của mọi bản cũ nên mất mát duy
# nhất là đường lùi.
KEEP_OLD_VERSIONS = True

import os
import subprocess
import sys
import textwrap
from pathlib import Path

# Lượt đẩy chạy nền nên không in được vào output của ô. Log ra file, xem bằng sync_log().
SYNC_LOG = Path("/kaggle/working/sync.log")

# Thử ĐÚNG công cụ sẽ dùng để đẩy, thay vì đoán qua biến môi trường.
#
# Bài học từ log phiên trước: `kaggle datasets files` ở ô A1b chạy được (liệt kê ra
# dataset thật), trong khi `UserSecretsClient` ném BackendError. Cổng cũ kiểm Secrets nên
# nó tắt đồng bộ suốt 4 giờ sinh — dù công cụ đẩy vốn xác thực được. Kiểm sai chỗ thì
# càng "an toàn" càng mất dữ liệu.
def kaggle_cli_ok():
    return subprocess.run(["kaggle", "datasets", "list", "-m", "--page-size", "1"],
                          capture_output=True).returncode == 0

# Kaggle có HAI kiểu credential và chúng không thay thế nhau được:
#
#   KAGGLE_API_TOKEN   token `KGAT_…` (Settings → API Tokens, kiểu mới, khuyến nghị)
#   KAGGLE_USERNAME + KAGGLE_KEY   cặp legacy trong kaggle.json
#
# Đặt secret nào cũng được — hàm dưới thử lần lượt. Token mới còn được ghi ra
# ~/.kaggle/access_token vì bản `kaggle` cài sẵn trên Kaggle có thể cũ hơn biến
# KAGGLE_API_TOKEN; đọc file thì client nào cũng biết đường.
def nap_credential():
    try:
        from kaggle_secrets import UserSecretsClient

        s = UserSecretsClient()
    except Exception as exc:
        print(f"Không mở được Kaggle Secrets ({type(exc).__name__}).")
        return []

    lay = []
    for ten in ("KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY"):
        try:
            os.environ[ten] = s.get_secret(ten)
            lay.append(ten)
        except Exception:
            pass          # secret không có là chuyện thường: chỉ cần MỘT kiểu là đủ

    if "KAGGLE_API_TOKEN" in lay:
        f = Path.home() / ".kaggle" / "access_token"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(os.environ["KAGGLE_API_TOKEN"])
        f.chmod(0o600)
        lay.append("~/.kaggle/access_token")
    print(f"Secrets đọc được: {lay or 'không có secret nào'}")
    return lay

def kaggle_ready():
    if kaggle_cli_ok():
        return True
    if nap_credential() and kaggle_cli_ok():
        return True
    print("`kaggle` CLI chưa xác thực được — sẽ không đẩy lên được. Cần MỘT trong hai:")
    print("  · Settings → API Tokens → Generate New Token, rồi Add-ons → Secrets thêm")
    print("    KAGGLE_API_TOKEN = KGAT_… (và tick attach cho notebook này)")
    print("  · hoặc Legacy API Key, thêm KAGGLE_USERNAME + KAGGLE_KEY")
    print("Không có thì dùng đường Output: Save Version, rồi phiên sau Add Input.")
    return False

# Script độc lập, để `generate --after-speaker` gọi được từ tiến trình con.
SYNC_SCRIPT = Path("/kaggle/working/sync_corpus.py")
SYNC_SCRIPT.write_text(textwrap.dedent(f'''
    import json, os, shutil, subprocess, sys, time
    from pathlib import Path

    DATASET_ID = {DATASET_ID!r}
    MIN_GAP = {SYNC_EVERY_MINUTES} * 60
    KEEP_OLD = {KEEP_OLD_VERSIONS!r}
    CORPUS = Path("/kaggle/working/corpus")
    STAGE = Path("/kaggle/working/dataset_upload")
    STAMP = Path("/kaggle/working/.last_sync")
    LOCK = Path("/kaggle/working/.sync_lock")
    FORCE = "--force" in sys.argv

    # PID của lượt đẩy đang chạy, hoặc None.
    def running():
        try:
            pid = int(LOCK.read_text())
            os.kill(pid, 0)          # chỉ hỏi còn sống không, không gửi tín hiệu thật
        except (OSError, ValueError):
            return None
        return pid

    # Hai lượt đẩy chồng nhau là cùng gói vào MỘT file zip mà lượt trước đang tải lên.
    # Speaker tới sớm hơn thời gian đẩy thì bỏ lượt — mốc sau gói cả phần vừa bỏ, vì
    # mỗi lần đẩy là một ảnh chụp TOÀN BỘ corpus chứ không phải phần tăng thêm.
    while running():
        if not FORCE:
            print(f"[{{time.strftime('%H:%M:%S')}}] bỏ lượt — pid {{running()}} còn đang đẩy")
            raise SystemExit(0)
        print(f"[{{time.strftime('%H:%M:%S')}}] đợi lượt đẩy nền (pid {{running()}}) xong…")
        time.sleep(15)

    # --force bỏ qua nhịp chặn: dùng khi vừa dừng tay và muốn lưu ngay.
    if not FORCE and MIN_GAP and STAMP.exists():
        waited = time.time() - STAMP.stat().st_mtime
        if waited < MIN_GAP:
            print(f"bỏ lượt — còn {{(MIN_GAP - waited) / 60:.0f}} phút tới nhịp sau")
            raise SystemExit(0)

    # Chốt nhịp NGAY khi bắt đầu, không đợi thành công. Kaggle từ chối vì version
    # trước còn đang xử lý là chuyện thường; nếu chỉ chốt khi thành công thì mỗi ranh
    # giới speaker lại gói và tải lại cả GB — hỏng liên tục thì đó là hammer, không
    # phải retry. Bản chốt cuối không mất: ô A5 đẩy bằng --force.
    STAMP.touch()
    LOCK.write_text(str(os.getpid()))
    started = time.time()

    try:
        STAGE.mkdir(exist_ok=True)
        # `pack` đọc manifest rồi zip đúng những file trong đó. Manifest được ghi bằng
        # tmp + os.replace nên bản đọc được luôn nguyên vẹn, và audio sinh ra SAU thời
        # điểm đó chỉ đơn giản là chưa có trong ảnh chụp này — lượt sau lấy.
        subprocess.run([sys.executable, "-m", "aidetector", "pack",
                        "--out", str(STAGE / "corpus.zip"), "-c", "configs/kaggle.yaml"],
                       check=True, cwd="/kaggle/working/ai-detector")
        # manifest để rời ngoài zip: A1b đọc tiến độ khỏi phải tải và giải nén cả GB.
        shutil.copy(CORPUS / "manifest.csv", STAGE / "manifest.csv")

        (STAGE / "dataset-metadata.json").write_text(json.dumps({{
            "title": "vivos fake v2",
            "id": DATASET_ID,
            "licenses": [{{"name": "CC0-1.0"}}],
        }}, ensure_ascii=False))

        note = (f"sau speaker {{os.environ.get('AIDETECTOR_SPEAKER', 'thủ công')}}"
                f" · {{os.environ.get('AIDETECTOR_KEPT', '?')}} mẫu")
        size = (STAGE / "corpus.zip").stat().st_size / 1024**3
        print(f"[{{time.strftime('%H:%M:%S')}}] gói xong {{size:.2f}} GB"
              f" trong {{time.time() - started:.0f}}s — {{note}}")

        add_version = ["datasets", "version", "-p", str(STAGE), "-m", note]
        if not KEEP_OLD:
            add_version.append("--delete-old-versions")

        # `version` cho dataset đã có, `create` cho lần đầu — thử lần lượt, đừng đoán.
        for argv, what in (
            (add_version, "thêm version"),
            (["datasets", "create", "-p", str(STAGE)], "tạo mới"),
        ):
            r = subprocess.run(["kaggle", *argv], capture_output=True, text=True)
            if r.returncode == 0:
                print(f"✔ {{what}} · cả lượt {{time.time() - started:.0f}}s"
                      f" — https://www.kaggle.com/datasets/{{DATASET_ID}}")
                break
            print(f"— {{what}} không xong: {{(r.stdout + r.stderr).strip()[-300:]}}")
        else:
            raise SystemExit(1)
    finally:
        LOCK.unlink(missing_ok=True)
'''))

def sync_now():
    # subprocess chứ không `!python`: magic của IPython không lồng vào `if` được.
    # Chạy CHẶN: --force đợi lượt nền đang dở rồi mới đẩy bản mới nhất.
    subprocess.run([sys.executable, str(SYNC_SCRIPT), "--force"])

def sync_log(n=40):
    # Lượt đẩy nền không in được vào ô nào, nên đây là cách duy nhất để xem nó đã làm gì.
    if SYNC_LOG.exists():
        print("\\n".join(SYNC_LOG.read_text().splitlines()[-n:]) or "(log rỗng)")
    else:
        print("Chưa có lượt đẩy nền nào.")

# Không sinh thêm gì thì không đẩy: dataset đã là bản mới nhất.
SYNC_READY = MAKE_DATASET and kaggle_ready()

# Hook dán vào MỌI lệnh generate, để lệnh nào cũng chốt tiến độ ở ranh giới speaker.
# Nó chạy NỀN, và cả ba thành phần của chuỗi đều bắt buộc:
#   nohup   — lượt đẩy sống tiếp khi tiến trình `generate` gọi nó đã kết thúc
#   >> log  — hook gọi bằng capture_output; con cháu còn giữ ống stdout thì nó VẪN đứng
#             chờ dù đã có `&`. Cắt ống mới thật sự không chặn.
#   &       — trả về ngay, GPU sinh speaker tiếp trong lúc gói + upload
SYNC_HOOK = ["--after-speaker",
             f"nohup {sys.executable} {SYNC_SCRIPT} >> {SYNC_LOG} 2>&1 &"] if SYNC_READY else []

_nhip = "sau MỖI speaker" if not SYNC_EVERY_MINUTES else f"tối đa {SYNC_EVERY_MINUTES} phút/lần"
_ver = "giữ mọi version" if KEEP_OLD_VERSIONS else "chỉ giữ version mới nhất"
print(f"Đồng bộ: {'BẬT' if SYNC_READY else 'TẮT'} · {DATASET_ID} · {_nhip} · chạy nền · {_ver}")
print(f"Xem lượt đẩy nền: sync_log()   ·   log ở {SYNC_LOG}")

# MỐC ĐẦU TIÊN: phần REAL vừa nạp. Không có nó thì bị out trong lúc sinh speaker đầu là
# mất luôn công ingest — mà đó lại đúng là lúc chưa có mốc nào được chốt.
if SYNC_READY and INGEST_ADDED:
    print(f"\\nChốt mốc sau ingest ({INGEST_ADDED} bản ghi mới)")
    sync_now()
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
if not MAKE_DATASET:
    skipped("sinh fake bằng TTS")
elif TTS_ENGINES:
    run("generate", "--engines", *TTS_ENGINES,
        *(["--count", N_FAKE_TTS] if N_FAKE_TTS else []), *SYNC_HOOK)
else:
    print("TTS đang tắt — chỉ sinh fake bằng voice cloning (xem TTS_ENGINES ở ô cài thư viện).")
"""),
md("""
### A3b. OmniVoice — voice cloning

Đây là engine **giá trị nhất về mặt dữ liệu**: nó clone thẳng giọng của chính
speaker thật, nên audio giả trùng với real **cả nội dung lẫn danh tính người nói**.
Piper và Kokoro chỉ có giọng cố định — nếu dataset chỉ có hai engine đó, mô hình rất
dễ học lối tắt *"nghe thấy mấy giọng này ⇒ fake"* thay vì học dấu vết tổng hợp.

Nhưng hai engine **không sống chung được trong một môi trường**:

| Engine | Cần |
|---|---|
| `kokoro` | `transformers <5` |
| `omnivoice` | `transformers >=5.3` |

Chỉ phải chạy hai lượt khi `TTS_ENGINES` còn bật; đang tắt nên `transformers>=5.3` đã
cài từ đầu phiên.

Đây là bước **dài nhất** của notebook (~4 giây/mẫu trên T4). Ô đầu báo còn thiếu bao
nhiêu để biết trước phải chạy bao lâu. Bị ngắt giữa chừng cũng không mất công: manifest
lưu sau mỗi 50 mẫu, corpus được đẩy lên dataset tại ranh giới mỗi speaker, và lượt sau
chỉ làm phần còn thiếu.

Checkpoint mặc định là **`splendor1811/omnivoice-vietnamese`** — fine-tune riêng cho
tiếng Việt và là repo công khai nên tải được ngay, không cần token.

**Nếu nghe thử ở A4 thấy giọng clone không giống người nói gốc**, xử lý theo thứ tự:

| Xem log | Nghĩa là | Làm gì |
|---|---|---|
| `Reference clone: trung bình N giây/mẫu` với N < 7 | mỗi speaker có quá ít bản ghi để ghép | tăng `PER_SPEAKER` ở ô A1 rồi chạy lại A2 |
| reference đủ dài nhưng vẫn "lệch người" | model bám prompt chưa đủ chặt | thêm `--set generate.options.omnivoice.guidance_scale=3.0` |
| phát âm chuẩn, danh tính sai hẳn | fine-tune một-ngôn-ngữ clone kém hơn bản gốc | đổi checkpoint sang `k2-fsa/OmniVoice` (đọc tiếng Việt kém hơn — đánh đổi) |

```python
run("generate", "--engines", "omnivoice", "--count", N_FAKE_CLONE,
    "--set", "generate.options.omnivoice.checkpoint=k2-fsa/OmniVoice",
    "--set", "generate.options.omnivoice.guidance_scale=3.0",
    "--overwrite", optional=True)
```

`--overwrite` là bắt buộc khi sinh lại: `generate` bỏ qua utt_id đã có, nên không có
cờ đó thì lượt chạy sau chỉ in `đã có N` và giữ nguyên audio cũ. Chỉ cần khi corpus
được nạp lại từ Kaggle Dataset của phiên trước — corpus mới trong `/kaggle/working`
thì không.

Reference được ghép từ nhiều utterance của cùng speaker cho tới ~12 giây, vì mỗi
utterance trong corpus chỉ 3–10 giây và 3 giây là quá ngắn để lấy ra danh tính một
người. Chi tiết: `TARGET_REF_SECONDS` trong `aidetector/generate/__init__.py`.
"""),
code("""
# Đã cài từ đầu phiên khi TTS tắt; chỉ phải nâng ở đây nếu Kokoro đã ghim 4.x.
if not MAKE_DATASET:
    skipped("cài omnivoice")
elif TTS_ENGINES:
    pip("omnivoice", "transformers>=5.3")
else:
    print("omnivoice + transformers>=5.3 đã cài từ đầu phiên — không phải nâng lại.")
"""),
code("""
if MAKE_DATASET:
    run("info")     # xác nhận omnivoice đã ✔ trước khi tốn thời gian sinh
else:
    skipped("kiểm tra engine sinh")
"""),
code("""
# CÒN BAO NHIÊU? `--dry-run` chạy đúng phép chọn của lượt sinh thật rồi đếm theo utt_id,
# không nạp model nên xong trong vài giây. Tiến độ theo speaker cũng in ra đây.
# `--count` vắng mặt ⇒ `fake_to_real_ratio: 1.0` trong config tự tính: đúng một fake
# cho mỗi real đủ điều kiện. Đây là định nghĩa "full" mà không phải gõ con số nào.
_soluong = ["--count", N_FAKE_CLONE] if N_FAKE_CLONE else []

if MAKE_DATASET:
    run("generate", "--engines", "omnivoice", *_soluong, "--dry-run")
else:
    skipped("đếm phần còn thiếu")
"""),
code("""
if MAKE_DATASET:
    # --after-speaker: xong mỗi giọng thì chốt manifest rồi gọi script đồng bộ. Script tự bỏ
    # qua nếu chưa tới nhịp, nên đây là "đẩy tại ranh giới speaker" chứ không phải "đẩy sau
    # TỪNG speaker" — lý do ở A2b.
    #
    # --overwrite ở chế độ thử: đang vòng lặp sửa-nghe-sửa nên cần audio MỚI mỗi lần. Lượt
    # chạy thật thì ngược lại, corpus cộng dồn và không đụng vào cái đã sinh.
    #
    # optional CHỈ khi còn engine khác gánh lớp fake. Tắt TTS rồi thì cloning là nguồn fake
    # DUY NHẤT: hỏng mà vẫn đi tiếp là kéo cả phần B vào corpus không có lớp fake nào.
    run("generate", "--engines", "omnivoice", *_soluong,
        *(["--overwrite"] if SMOKE else []), *SYNC_HOOK, optional=bool(TTS_ENGINES))
else:
    skipped("sinh fake bằng voice cloning")
"""),
md("""
### A3c. Xong chưa?

Đếm lại bằng đúng phép đếm ở đầu A3b. `còn 0 phải sinh` ⇒ corpus đã đủ, phiên sau đặt
`MODE = "train"`. Còn số dương ⇒ phiên hết giờ giữa đường: corpus đã được đẩy lên dataset
tại ranh giới mỗi speaker, nên phiên sau vào lại là tiếp đúng chỗ, không làm lại gì.
"""),
code("""
if MAKE_DATASET:
    run("generate", "--engines", "omnivoice", *_soluong, "--dry-run")
else:
    skipped("đếm lại phần còn thiếu")
"""),

code("""
# A/B CHECKPOINT — sinh thêm một lượt bằng bản đa ngữ gốc, trên ĐÚNG những câu vừa rồi.
#
# Fine-tune tiếng Việt đọc chuẩn hơn nhưng có dấu hiệu clone danh tính kém hơn; bản gốc
# thì ngược lại. Không có cách nào đoán được cái nào hợp dataset của anh — phải sinh cả
# hai rồi đo. Hai lượt mang tag khác nhau (`omnivoice` và `omnivoice:k2-fsa-omnivoice`)
# nên cùng tồn tại trong corpus, và ô đo ở A4 sẽ xếp chúng cạnh nhau.
#
# Chỉ chạy khi SMOKE: câu hỏi "checkpoint nào giống hơn" trả lời một lần trên 15 mẫu là
# đủ, không cần trả lời lại trên 800 mẫu của lượt chạy thật.
if not MAKE_DATASET:
    skipped("A/B checkpoint")
elif SMOKE:
    run("generate", "--engines", "omnivoice", *_soluong, "--overwrite",
        "--set", "generate.options.omnivoice.checkpoint=k2-fsa/OmniVoice", optional=True)
else:
    print("Bỏ qua A/B checkpoint — chỉ chạy ở chế độ thử (SMOKE = True).")
    print("Chốt được checkpoint rồi thì đặt nó vào configs/kaggle.yaml cho lượt chạy thật.")
"""),

md("""
## A4. Kiểm tra dataset

Ba việc: soi toàn corpus xem có file nào phạm chuẩn, xem thống kê, và **nghe thử**.

`validate`, thống kê và nghe thử chạy ở mọi `MODE` — ở `"train"` chúng chính là phép
kiểm bản corpus vừa bung ra. Hai ô đo bằng model (độ giống giọng, phát âm) thì chỉ chạy
khi phiên có sinh fake: chúng tải thêm model và mất vài phút, mà câu trả lời đã có sẵn
từ phiên sinh.
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
#
# Với engine cloning (omnivoice): bản REAL nghe ở đây là utterance CÙNG NỘI DUNG, KHÔNG
# phải đoạn audio đã dùng làm reference — reference được ghép từ các utterance khác của
# chính speaker đó. Nên chấm điểm "có giống người này không", đừng chấm "có khớp từng
# hơi thở của bản real này không".
from IPython.display import Audio, display

# Engine cloning lên trước: đó là engine duy nhất mà "có giống người gốc không" là
# câu hỏi có nghĩa. Piper/Kokoro giọng cố định, nghe chúng không nói lên điều gì về
# chất lượng clone — mà chúng lại đông hơn nên dễ chiếm hết ba chỗ.
from aidetector.generate.base import KIND_CLONE, available_generators

_clone_engines = {i for i, c in available_generators().items() if c.kind == KIND_CLONE}
pairs = []
for fake in sorted(manifest.fakes, key=lambda f: (f.engine not in _clone_engines, f.utt_id)):
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
if MAKE_DATASET:
    # ĐO ĐỘ GIỐNG GIỌNG của engine cloning — nghe vài mẫu bằng tai không kết luận được.
    #
    # Cosine giữa hai speaker embedding chỉ có nghĩa khi đặt cạnh MỐC: hai bản ghi khác
    # nhau của cùng một người cũng không bao giờ đạt 1.0, còn hai người khác nhau vẫn được
    # 0.5-0.6. Nên ô này đo cả ba: cùng-người (trần), khác-người (sàn), và clone-vs-người-gốc.
    import importlib.util
    import subprocess
    import sys

    # `!pip` không dùng được ở đây: nó là magic của IPython nên không lồng vào `if` được.
    if importlib.util.find_spec("resemblyzer") is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "resemblyzer"], check=True)

    from itertools import combinations

    import numpy as np
    from resemblyzer import VoiceEncoder, preprocess_wav

    encoder = VoiceEncoder(verbose=False)
    _cache = {}

    def embed(rec):
        if rec.utt_id not in _cache:
            try:
                _cache[rec.utt_id] = encoder.embed_utterance(
                    preprocess_wav(str(manifest.abs_path(rec)))
                )
            except Exception:      # file quá ngắn sau VAD ⇒ bỏ qua, đừng làm hỏng cả ô
                _cache[rec.utt_id] = None
        return _cache[rec.utt_id]

    def cosines(pairs, limit=80):
        out = []
        for a, b in pairs[:limit]:
            ea, eb = embed(a), embed(b)
            if ea is not None and eb is not None:
                out.append(float(ea @ eb))
        return np.array(out)

    rng = np.random.default_rng(0)
    reals = [r for r in manifest.reals if not r.augment]
    by_spk = {}
    for r in reals:
        by_spk.setdefault(r.speaker, []).append(r)

    # TRẦN: cùng người, khác bản ghi. Đây là mức cao nhất một bản clone có thể với tới.
    same = [p for recs in by_spk.values() for p in combinations(sorted(recs, key=lambda r: r.utt_id)[:4], 2)]
    # SÀN: hai người khác nhau — điểm quanh đây nghĩa là clone ra một người khác hẳn.
    spk = sorted(by_spk)
    diff = [(by_spk[spk[i]][0], by_spk[spk[j]][0]) for i, j in combinations(range(len(spk)), 2)]
    rng.shuffle(same); rng.shuffle(diff)

    ceiling, floor = cosines(same), cosines(diff)
    print(f"TRẦN  cùng người, khác câu : {np.median(ceiling):.3f}  (n={len(ceiling)})")
    print(f"SÀN   hai người khác nhau  : {np.median(floor):.3f}  (n={len(floor)})")
    print()

    from aidetector.generate.base import KIND_CLONE, available_generators

    _clone_engines = {i for i, c in available_generators().items() if c.kind == KIND_CLONE}

    # Engine cloning tách theo từng checkpoint — đó chính là thứ đang so. Engine TTS thì
    # gộp theo engine: chín giọng Kokoro tách thành chín dòng hai-ba mẫu là không đọc được gì.
    def group_of(rec):
        return rec.generator if rec.engine in _clone_engines else rec.engine

    for engine in sorted({group_of(f) for f in manifest.fakes if not f.augment}):
        pairs = []
        for fake in manifest.fakes:
            if fake.augment or group_of(fake) != engine:
                continue
            target = manifest.get(fake.ref_utt_id)
            if target is not None:
                pairs.append((fake, target))
        rng.shuffle(pairs)
        score = cosines(pairs)
        if not len(score):
            continue
        med = float(np.median(score))
        if med >= np.median(ceiling) - 0.05:
            verdict = "✔ giữ được danh tính người nói"
        elif med <= np.median(floor) + 0.05:
            verdict = "✖ ra giọng người khác hẳn"
        else:
            verdict = "~ ở giữa trần và sàn"
        print(f"{engine:<32} {med:.3f}  (n={len(score)})  {verdict}")

    print()
    print("Engine TTS giọng cố định (piper, kokoro) ĐÁNG LẼ phải nằm sát sàn — chúng đâu có")
    print("clone ai. Nếu chúng không sát sàn thì phép đo hỏng chứ không phải engine giỏi.")
else:
    skipped("đo độ giống giọng — đã đo ở phiên sinh")
"""),
code("""
if MAKE_DATASET:
    # ĐO PHÁT ÂM — engine có đọc đúng câu tiếng Việt được giao không?
    #
    # Ô trên đo GIỌNG CỦA AI, ô này đo ĐỌC CÁI GÌ. Hai trục khác nhau và một engine có thể
    # tốt trục này hỏng trục kia: clone đúng giọng nhưng nhả ra âm vô nghĩa thì audio đó vẫn
    # là rác đối với dataset.
    #
    # Cách đo: cho ASR nghe lại audio sinh ra rồi so với câu đã giao (WER). WER thô không đọc
    # được vì ASR cũng sai trên chính giọng thật — nên đo cả REAL làm SÀN LỖI.
    #
    # ASR chạy ở TIẾN TRÌNH RIÊNG, có lý do: ô A3b nâng transformers lên 5.x giữa phiên trong
    # khi kernel còn giữ bản cũ trong bộ nhớ. Import transformers thẳng ở đây là dính
    # ImportError do trộn hai phiên bản. Tiến trình con luôn nạp đúng thứ đang có trên đĩa.
    import json
    import re
    import subprocess
    import sys
    import tempfile
    from pathlib import Path

    _ASR_SCRIPT = "\\n".join([
        "import json, sys, torch",
        "from transformers import pipeline",
        "paths = json.load(open(sys.argv[1]))",
        'asr = pipeline("automatic-speech-recognition", model="vinai/PhoWhisper-small",',
        "               device=0 if torch.cuda.is_available() else -1)",
        'out = asr(paths, batch_size=8, generate_kwargs={"language": "vi", "task": "transcribe"})',
        'json.dump([o["text"] for o in out], open(sys.argv[2], "w"))',
    ])

    def transcribe(paths):
        if not paths:
            return []
        work = Path(tempfile.mkdtemp())
        (work / "asr.py").write_text(_ASR_SCRIPT)
        (work / "in.json").write_text(json.dumps([str(p) for p in paths]))
        done = subprocess.run([sys.executable, str(work / "asr.py"),
                               str(work / "in.json"), str(work / "out.json")],
                              capture_output=True, text=True)
        if done.returncode != 0:
            print("ASR hỏng — bỏ qua phép đo phát âm. Cuối log lỗi:")
            print(done.stderr.strip()[-800:])
            return None
        return json.loads((work / "out.json").read_text())

    def _words(text):
        return re.sub(r"[^\\w\\s]", " ", text.lower()).split()

    def wer(reference, hypothesis):
        # Levenshtein mức TỪ, viết tay 8 dòng — đỡ thêm một phụ thuộc chỉ dùng một lần.
        ref, hyp = _words(reference), _words(hypothesis)
        if not ref:
            return None
        prev = list(range(len(hyp) + 1))
        for i, r in enumerate(ref, 1):
            cur = [i]
            for j, h in enumerate(hyp, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
            prev = cur
        return prev[-1] / len(ref)

    # Gom hết bản ghi cần đo rồi phiên âm MỘT LƯỢT: model chỉ phải nạp một lần cho cả bảng.
    rng = np.random.default_rng(0)

    def sample(recs, limit):
        recs = [r for r in recs if r.text.strip()]
        rng.shuffle(recs)
        return recs[:limit]

    groups = {"(real)": sample([r for r in manifest.reals if not r.augment], 20)}
    for engine in sorted({group_of(f) for f in manifest.fakes if not f.augment}):
        groups[engine] = sample([f for f in manifest.fakes
                                 if not f.augment and group_of(f) == engine], 15)

    flat = [r for recs in groups.values() for r in recs]
    hyps = transcribe([manifest.abs_path(r) for r in flat])

    if hyps is not None:
        scored, at = {}, 0
        for name, recs in groups.items():
            rows = [(w, r, h) for r, h in ((r, hyps[at + k]) for k, r in enumerate(recs))
                    if (w := wer(r.text, h)) is not None]
            at += len(recs)
            scored[name] = rows

        floor = float(np.median([w for w, _, _ in scored["(real)"]])) if scored["(real)"] else 0.0
        print(f"SÀN LỖI  ASR nghe chính giọng thật : WER {floor:.1%}  (n={len(scored['(real)'])})")
        print()
        for name, rows in scored.items():
            if name == "(real)" or not rows:
                continue
            med = float(np.median([w for w, _, _ in rows]))
            if med <= floor + 0.10:
                verdict = "✔ đọc đúng"
            elif med <= floor + 0.30:
                verdict = "~ sai lác đác"
            else:
                verdict = "✖ ĐỌC HỎNG — audio này là rác cho dataset"
            print(f"{name:<32} WER {med:6.1%}  (n={len(rows)})  {verdict}")

        worst = max((row for name, rows in scored.items() if name != "(real)" for row in rows),
                    key=lambda row: row[0], default=None)
        if worst:
            score, rec, hyp = worst
            print()
            print(f"Mẫu tệ nhất — {rec.generator} · WER {score:.0%}")
            print(f"  giao   : {rec.text.lower()}")
            print(f"  đọc ra : {hyp.strip()}")
            display(Audio(str(manifest.abs_path(rec))))
else:
    skipped("đo phát âm — đã đo ở phiên sinh")
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
## A5. Đẩy bản cuối lên dataset

Trong lúc sinh, corpus đã được đẩy tại ranh giới các speaker. Chạy xong thì đẩy nốt
phần còn lại — lần này ép đẩy, bỏ qua nhịp chặn 20 phút.
"""),
code("""
if not MAKE_DATASET:
    skipped("đẩy corpus — phiên này không sinh thêm gì")
elif SYNC_READY:
    sync_now()          # chặn: đợi lượt nền đang dở, rồi đẩy bản mới nhất
    print()
    sync_log()          # toàn bộ các lượt đẩy nền trong phiên
else:
    run("pack", "--out", "/kaggle/working/corpus.zip")
    print("Chưa có token — dùng Save Version → Save & Run All để giữ /kaggle/working.")
"""),

md("""
> ### Dừng lại ở đây nếu chỉ cần dataset
>
> Xem lại A4: hai lớp có cân bằng không, engine nào sinh được bao nhiêu, nghe thử
> thấy hợp lý chưa. Nếu đang ở `SMOKE = True` thì giờ đặt `SMOKE = False` ở ô A1 và
> chạy lại A2–A5 để làm thật. Ưng rồi mới sang phần B.
>
> Đặt `MODE = "dataset"` thì mọi ô của phần B dưới đây tự bỏ qua, không phải chọn tay —
> rồi phiên sau `MODE = "train"` huấn luyện trên đúng corpus vừa đẩy lên.
"""),

# ─────────────────────────────────────────────────────────── PHẦN B
md("""
---
# PHẦN B — Huấn luyện

Chạy khi dataset đã ưng, tức `MODE` là `"train"` hoặc `"both"`. Corpus được nạp ở ô
**A1b** — ô đó chạy ở mọi chế độ nên phần B không phải bung lại gì. Muốn lấy corpus từ
một dataset khác thì `run("unpack", "/kaggle/input/<tên-dataset>/corpus.zip")`.
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
if DO_TRAIN:
    run("split")
    run("augment", "--copies", 1)
else:
    skipped("split + augment")
"""),

md("""
## B2. WavLM → Classifier

Embedding cache theo `utt_id` nên chạy lại chỉ trích phần mới. Đổi backbone chỉ cần
`--set features.backbone.name=wav2vec2` — cache tách riêng, không đè lên nhau.
"""),
code("""
if DO_TRAIN:
    run("features")
    run("train")
    run("evaluate")
else:
    skipped("features + train + evaluate")
"""),

md("## B3. Kết quả"),
code("""
if DO_TRAIN:
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
else:
    skipped("xem kết quả — phiên này chưa huấn luyện")
"""),

md("## B4. Thử trên file bất kỳ + lưu mô hình"),
code("""
import glob

if DO_TRAIN:
    # Bất kỳ engine nào có trong corpus — cứng nhắc "piper" là rỗng khi TTS tắt.
    mau = sorted(glob.glob("/kaggle/working/corpus/audio/fake/*/*/*.wav"))[:5]
    mau += sorted(glob.glob("/kaggle/working/corpus/audio/real/*/*/*.wav"))[:5]
    run("detect", *mau)
else:
    skipped("thử detect — phiên này chưa huấn luyện mô hình nào")
"""),
code("""
import shutil
from pathlib import Path

# `!ls` là magic của IPython nên không lồng vào `if` được — liệt kê bằng Python.
if DO_TRAIN:
    shutil.make_archive("/kaggle/working/model",          "zip", "/kaggle/working/checkpoints")
    shutil.make_archive("/kaggle/working/reports_bundle", "zip", "/kaggle/working/reports")
    for _zip in sorted(Path("/kaggle/working").glob("*.zip")):
        print(f"{_zip.stat().st_size / 1024**2:8.1f} MB  {_zip}")
else:
    skipped("đóng gói mô hình")
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
