"""Schema của `corpus/<bộ>/metadata.csv` — nguồn sự thật duy nhất về dữ liệu.

Tên cột theo chuẩn corpus Việt (`id`, `audio`, `label` 0/1, `speaker_id`…), nên một
bảng dựng sẵn ngoài dự án nạp thẳng được, không cần bước map. Ngoài phần chuẩn đó,
`Record` còn giữ sáu cột RIÊNG của pipeline này — `ref_id`, `language`, `augment`,
`parent_id`, `checked`, `schema_version` — vì chúng chịu lực: `augment`/`parent_id`
giữ bất biến "bản augment nằm cùng split với bản gốc", `checked` cho `validate` soi
tăng dần thay vì đọc lại cả corpus mỗi phiên.

Nhóm cột SỐ ĐO (`rms_db`, `speech_ratio`, `sha256_*`, `original_*`…) là tuỳ chọn:
giữ nguyên khi nạp từ bảng ngoài, để trống khi `ingest`/`generate` tự sinh bản ghi.
Rỗng ⇒ `None`, phân biệt được với một giá trị 0 đo thật.

Schema cũ (`utt_id`, `path`, `speaker`, `label` real/fake, split `val`) vẫn ĐỌC được
— xem `LEGACY_COLUMNS`. Lượt `save()` kế tiếp ghi ra chuẩn mới.
"""

from __future__ import annotations

from dataclasses import MISSING, asdict, dataclass, fields
from pathlib import Path

from .. import CORPUS_SCHEMA_VERSION
from ..utils import slugify, stable_id

# 0 = real, 1 = fake (nhất quán toàn dự án, và trùng luôn nhãn số dùng lúc train).
LABEL_REAL = 0
LABEL_FAKE = 1
LABELS = (LABEL_REAL, LABEL_FAKE)

#: Tên chữ của nhãn — dùng cho thư mục, log và báo cáo, KHÔNG dùng trong manifest.
LABEL_NAME = {LABEL_REAL: "real", LABEL_FAKE: "fake"}
NAME_TO_LABEL = {v: k for k, v in LABEL_NAME.items()}

#: Cột của schema cũ → cột hiện hành. Nhờ bảng này, mọi `corpus.zip` đã đẩy lên
#: Kaggle vẫn nạp được nguyên vẹn thay vì thành rác.
LEGACY_COLUMNS = {
    "utt_id": "id",
    "path": "audio",
    "speaker": "speaker_id",
    "ref_utt_id": "ref_id",
    "parent_utt_id": "parent_id",
}

#: Giá trị `split` cũ → hiện hành. Chuẩn mới viết đủ chữ `validation`.
LEGACY_SPLITS = {"val": "validation", "dev": "validation"}


@dataclass
class Record:
    """Một utterance trong corpus."""

    # --- định danh & nội dung ------------------------------------------------
    id: str                       # khoá chính, ổn định giữa các lần chạy
    audio: str                    # đường dẫn tương đối so với gốc corpus
    label: int                    # 0 = real, 1 = fake
    speaker_id: str = ""          # id speaker (real) hoặc giọng được clone (fake)
    speaker_known: bool = False   # False khi nguồn không công bố danh tính speaker
    source: str = ""              # tên bộ dữ liệu gốc: vivos, common_voice, folder:xyz
    generator: str = ""           # "" nếu real; vd: piper:vi_VN-vais1000-medium
    recording_id: str = ""        # id bản thu ở nguồn, trước khi cắt đoạn
    duration: float = 0.0         # giây
    sample_rate: int = 16_000
    channels: int = 1
    split: str = ""               # train | validation | test | ""
    text: str = ""                # transcript, "" nếu không có
    gender: str = ""

    # --- số đo: nạp từ bảng ngoài thì giữ, tự sinh thì để trống ---------------
    speech_ratio: float | None = None
    rms_db: float | None = None
    clipping_ratio: float | None = None
    original_sample_rate: int | None = None
    original_channels: int | None = None
    original_split: str = ""
    original_duration: float | None = None
    norm_gain: float | None = None
    norm_peak: float | None = None
    sha256_raw: str = ""
    sha256_norm: str = ""
    #: Shard của NGUỒN (vd `real_shard_0000`) — chỉ là vết tích xuất xứ. Không liên
    #: quan tới `shard_name()` bên dưới, thứ quyết định bản ghi nằm ở thư mục bộ nào.
    shard: str = ""
    bytes: int | None = None
    extra: str = ""               # JSON tự do của nguồn

    # --- riêng của pipeline này ----------------------------------------------
    ref_id: str = ""              # utt real dùng làm reference/nguồn text khi sinh fake
    language: str = "vi"
    augment: str = ""             # "" = bản clean gốc; vd: "noise_snr15+mp3_64k"
    parent_id: str = ""           # utt gốc nếu đây là bản augment
    # Vân tay chuẩn audio mà bản ghi này đã được `validate` soi qua và đạt. Rỗng = chưa
    # soi. Nhờ cột này, phiên sau chỉ soi phần MỚI thay vì đọc lại cả corpus.
    checked: str = ""
    schema_version: int = CORPUS_SCHEMA_VERSION

    # ------------------------------------------------------------------ helpers
    @property
    def is_fake(self) -> bool:
        return self.label == LABEL_FAKE

    @property
    def label_name(self) -> str:
        """`"real"` / `"fake"` — cho thư mục, log, báo cáo."""
        return LABEL_NAME.get(self.label, str(self.label))

    @property
    def engine(self) -> str:
        """Phần engine của `generator` (bỏ tên voice): `piper:vi_VN-x` → `piper`."""
        return self.generator.split(":", 1)[0] if self.generator else ""

    def to_row(self) -> dict[str, object]:
        # `None` ra CSV thành chuỗi "None" nếu để mặc định — ghi rỗng để lượt nạp
        # sau đọc lại đúng là "chưa đo".
        return {k: ("" if v is None else v) for k, v in asdict(self).items()}

    @classmethod
    def from_row(cls, row: dict[str, str]) -> "Record":
        # Cột cũ được đổi tên TRƯỚC, và chỉ khi cột mới chưa có mặt — bảng lai
        # (đã có `id`, còn sót `utt_id`) thì cột mới thắng.
        data = dict(row)
        for cu, moi in LEGACY_COLUMNS.items():
            if cu in data and not data.get(moi):
                data[moi] = data.pop(cu)

        kwargs: dict[str, object] = {}
        for f in fields(cls):
            kwargs[f.name] = _doc_o(f, data.get(f.name))
        return cls(**kwargs)  # type: ignore[arg-type]

    def validate(self) -> list[str]:
        errs = []
        if not self.id:
            errs.append("id rỗng")
        if self.label not in LABELS:
            errs.append(f"label không hợp lệ: {self.label!r} (chờ 0 hoặc 1)")
        if self.label == LABEL_FAKE and not self.generator:
            errs.append("bản ghi fake nhưng thiếu generator")
        if self.label == LABEL_REAL and self.generator:
            errs.append("bản ghi real nhưng lại có generator")
        if not self.audio:
            errs.append("audio rỗng")
        return errs


def _doc_o(f, raw: object) -> object:
    """Một ô CSV → giá trị đúng kiểu của field `f`.

    Thiếu cột và ô rỗng đều rơi về DEFAULT của field, nên một bảng chỉ có phần cột
    chuẩn vẫn ra `language="vi"`, `sample_rate=16000` thay vì rỗng/0 — và cột SỐ ĐO
    (khai `X | None`, default `None`) ra `None` = "chưa đo", phân biệt được với một
    giá trị 0 đo thật.
    """
    kieu = str(f.type)
    s = "" if raw is None else str(raw).strip()

    if s == "":
        if f.default is not MISSING:
            return f.default
        return 0 if f.name == "label" else ""      # id, audio, label: không có default

    if f.name == "label":
        # Schema cũ ghi chữ (`real`/`fake`); chuẩn mới ghi số.
        return int(s) if s in ("0", "1") else NAME_TO_LABEL.get(s, s)
    if f.name == "split":
        return LEGACY_SPLITS.get(s, s)
    if kieu.startswith("bool"):
        return s.lower() in ("true", "1", "yes")
    if kieu.startswith("int"):
        return int(float(s))
    if kieu.startswith("float"):
        return float(s)
    return s


COLUMNS: tuple[str, ...] = tuple(f.name for f in fields(Record))

#: Cột SỐ ĐO — gắn với đúng một file audio. Bản sao có audio khác (augment chẳng hạn)
#: phải xoá sạch nhóm này thay vì thừa hưởng: một `sha256_norm` hay `rms_db` chép lại
#: từ bản gốc là số đo SAI, và sai một cách im lặng.
MEASURED_COLUMNS: tuple[str, ...] = (
    "speech_ratio", "rms_db", "clipping_ratio",
    "original_sample_rate", "original_channels", "original_split", "original_duration",
    "norm_gain", "norm_peak", "sha256_raw", "sha256_norm", "shard", "bytes",
)


def clear_measurements(rec: Record) -> Record:
    """Đưa mọi cột số đo của `rec` về mặc định (tại chỗ), rồi trả lại chính nó."""
    mac_dinh = {f.name: f.default for f in fields(Record)}
    for ten in MEASURED_COLUMNS:
        setattr(rec, ten, mac_dinh[ten])
    return rec


# --------------------------------------------------------------------- đặt tên
def make_id(source: str, speaker: str, key: str, chunk: int = 0) -> str:
    """ID ổn định: cùng đầu vào ⇒ cùng id ⇒ ingest lại là idempotent."""
    suffix = f"-{chunk}" if chunk else ""
    return f"{slugify(source, 20)}-{stable_id(source, speaker, key)}{suffix}"


def shard_name(source: str) -> str:
    """Tên thư mục của một BỘ DỮ LIỆU trong corpus — tầng ngoài cùng của cây.

    Cùng phép chuẩn hoá tên với `audio_folder`, và `Manifest` dùng nó để biết bản ghi
    này thuộc `metadata.csv` nào. Fake thừa hưởng `source` của real gốc nên nó tự về
    đúng thư mục bộ dữ liệu đã sinh ra nó.

    Không liên quan tới CỘT `shard` của `Record` — cột đó chỉ ghi lại shard ở nguồn.
    """
    return slugify(source or "unknown")


def audio_folder(rec: Record) -> str:
    """Thư mục chuẩn của một bản ghi, tính từ gốc corpus.

        <bộ>/real/<speaker>/
        <bộ>/fake/<speaker>/
        <bộ>/augment/<speaker>/

    Tầng ngoài cùng là BỘ DỮ LIỆU, để mỗi bộ là một thư mục tự chứa: `real/`, `fake/`
    và `metadata.csv` của riêng nó. Thêm hay bỏ một bộ là thêm hay bỏ một thư mục.

    Tầng cuối luôn là speaker — kể cả fake, vì fake mang đúng speaker của real gốc. Nhờ
    vậy đứng ở một giọng là thấy ngay cả hai lớp của giọng đó cạnh nhau.

    Đường dẫn tính từ gốc corpus (không phải từ thư mục bộ) nên `audio` là khoá tra cứu
    duy nhất, dùng được ở mọi chỗ mà không cần biết bản ghi thuộc bộ nào — mà thư mục bộ
    vẫn dời được sang corpus khác miễn giữ nguyên tên.

    ENGINE KHÔNG vào đường dẫn. Một quy tắc duy nhất cho cả ba lớp —
    `<bộ>/<lớp>/<speaker>/` — nên `fake/` đối xứng với `real/`: đứng ở một giọng là thấy
    hai lớp của giọng đó cùng độ sâu, đối chiếu được ngay. Tách thư mục theo engine thì
    cây phình ra một tầng mà không ai duyệt theo chiều đó, và đổi engine là đổi đường dẫn
    của cùng một mẫu. Engine (kèm cả tên voice: `piper:vi_VN-vais1000`) vẫn nằm đủ trong
    cột `generator`, lọc theo nó là một phép trên manifest chứ không phải trên cây thư mục.
    """
    tang = [shard_name(rec.source), "augment" if rec.augment else rec.label_name,
            slugify(rec.speaker_id or "unknown")]
    return "/".join(tang)


def audio_name(index: int) -> str:
    """Tên file: số thứ tự trong thư mục, 4 chữ số.

    Số này được cấp MỘT lần rồi nằm luôn trong cột `audio`, nên chạy lại không đánh số
    lại — đó là điều kiện để ingest/generate còn idempotent. Xem `Manifest.allocate_path`.
    """
    return f"{index:04d}.wav"
