"""Schema của `corpus/manifest.csv` — nguồn sự thật duy nhất về dữ liệu."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path

from .. import CORPUS_SCHEMA_VERSION
from ..utils import slugify, stable_id

LABEL_REAL = "real"
LABEL_FAKE = "fake"
LABELS = (LABEL_REAL, LABEL_FAKE)

# 0 = real, 1 = fake (nhất quán toàn dự án).
LABEL_TO_INT = {LABEL_REAL: 0, LABEL_FAKE: 1}


@dataclass
class Record:
    """Một utterance trong corpus."""

    utt_id: str                  # khoá chính, ổn định giữa các lần chạy
    path: str                    # đường dẫn tương đối so với gốc corpus
    label: str                   # real | fake
    source: str = ""             # tên bộ dữ liệu gốc: vivos, common_voice, folder:xyz
    speaker: str = ""            # id speaker (real) hoặc giọng/speaker được clone (fake)
    text: str = ""               # transcript, "" nếu không có
    generator: str = ""          # "" nếu real; vd: piper:vi_VN-vais1000-medium
    ref_utt_id: str = ""         # utt real dùng làm reference/nguồn text khi sinh fake
    language: str = "vi"
    duration: float = 0.0        # giây
    sample_rate: int = 16_000
    channels: int = 1
    augment: str = ""            # "" = bản clean gốc; vd: "noise_snr15+mp3_64k"
    parent_utt_id: str = ""      # utt gốc nếu đây là bản augment
    split: str = ""              # train | val | test | ""
    # Vân tay chuẩn audio mà bản ghi này đã được `validate` soi qua và đạt. Rỗng = chưa
    # soi. Nhờ cột này, phiên sau chỉ soi phần MỚI thay vì đọc lại cả corpus.
    checked: str = ""
    schema_version: int = CORPUS_SCHEMA_VERSION

    # ------------------------------------------------------------------ helpers
    @property
    def is_fake(self) -> bool:
        return self.label == LABEL_FAKE

    @property
    def label_int(self) -> int:
        return LABEL_TO_INT[self.label]

    @property
    def engine(self) -> str:
        """Phần engine của `generator` (bỏ tên voice): `piper:vi_VN-x` → `piper`."""
        return self.generator.split(":", 1)[0] if self.generator else ""

    def to_row(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_row(cls, row: dict[str, str]) -> "Record":
        kwargs: dict[str, object] = {}
        for f in fields(cls):
            raw = row.get(f.name, "")
            if raw is None:
                raw = ""
            if f.type in ("float", float):
                kwargs[f.name] = float(raw or 0)
            elif f.type in ("int", int):
                kwargs[f.name] = int(float(raw or 0))
            else:
                kwargs[f.name] = str(raw)
        return cls(**kwargs)  # type: ignore[arg-type]

    def validate(self) -> list[str]:
        errs = []
        if not self.utt_id:
            errs.append("utt_id rỗng")
        if self.label not in LABELS:
            errs.append(f"label không hợp lệ: {self.label!r}")
        if self.label == LABEL_FAKE and not self.generator:
            errs.append("bản ghi fake nhưng thiếu generator")
        if self.label == LABEL_REAL and self.generator:
            errs.append("bản ghi real nhưng lại có generator")
        if not self.path:
            errs.append("path rỗng")
        return errs


COLUMNS: tuple[str, ...] = tuple(f.name for f in fields(Record))


# --------------------------------------------------------------------- đặt tên
def make_utt_id(source: str, speaker: str, key: str, chunk: int = 0) -> str:
    """ID ổn định: cùng đầu vào ⇒ cùng id ⇒ ingest lại là idempotent."""
    suffix = f"-{chunk}" if chunk else ""
    return f"{slugify(source, 20)}-{stable_id(source, speaker, key)}{suffix}"


def audio_folder(rec: Record) -> str:
    """Thư mục chuẩn của một bản ghi — ba tầng, giống nhau cho mọi nhãn.

        real/<source>/<speaker>/
        fake/<engine>/<speaker>/
        augment/<engine|source>/<speaker>/

    Tầng giữa là NGUỒN của audio: bộ dữ liệu với real, engine với fake. Tầng ba luôn là
    speaker — kể cả fake, vì fake mang đúng speaker của real gốc. Nhờ vậy đứng ở một
    giọng là thấy ngay cả hai lớp của giọng đó cạnh nhau.

    Tên voice KHÔNG vào path (`piper:vi_VN-vais1000` chỉ lấy `piper`): một engine nhiều
    giọng mà tách thư mục thì cây phình ra mà không ai duyệt theo chiều đó — voice vẫn
    nằm đủ trong cột `generator`.
    """
    nguon = rec.generator.partition(":")[0] if rec.generator else rec.source
    tang1 = "augment" if rec.augment else rec.label
    return "/".join((tang1, slugify(nguon or "unknown"),
                     slugify(rec.speaker or "unknown")))


def audio_name(index: int) -> str:
    """Tên file: số thứ tự trong thư mục, 4 chữ số.

    Số này được cấp MỘT lần rồi nằm luôn trong cột `path`, nên chạy lại không đánh số
    lại — đó là điều kiện để ingest/generate còn idempotent. Xem `Manifest.allocate_path`.
    """
    return f"{index:04d}.wav"
