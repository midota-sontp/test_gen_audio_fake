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


def relative_audio_path(rec: Record) -> str:
    """Vị trí chuẩn của file audio bên trong corpus.

        audio/real/<source>/<speaker>/<utt_id>.wav
        audio/fake/<engine>/<voice>/<utt_id>.wav
        audio/aug/<label>/<utt_id>.wav
    """
    if rec.augment:
        return str(Path("audio/aug") / rec.label / f"{rec.utt_id}.wav")
    if rec.label == LABEL_FAKE:
        engine, _, voice = rec.generator.partition(":")
        return str(
            Path("audio/fake") / slugify(engine) / slugify(voice or "default") / f"{rec.utt_id}.wav"
        )
    return str(
        Path("audio/real") / slugify(rec.source) / slugify(rec.speaker or "unknown") / f"{rec.utt_id}.wav"
    )
