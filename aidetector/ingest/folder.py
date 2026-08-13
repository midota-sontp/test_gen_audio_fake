"""Adapter tổng quát cho thư mục audio bất kỳ — lưới an toàn cuối cùng.

`FolderAdapter` nhận mọi thư mục có file audio. Speaker suy ra từ tên thư mục con,
transcript được dò theo nhiều quy ước phổ biến (metadata.csv kiểu LJSpeech, file
.txt/.lab cùng tên, hoặc prompts.txt dạng `<id> <text>`).

`LabeledFolderAdapter` dành cho dataset đã chia sẵn real/fake (real/ + fake/, hoặc
bonafide/ + spoof/), nhận nhãn trực tiếp từ tên thư mục.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from ..corpus.spec import iter_audio_files
from ..corpus.schema import LABEL_FAKE, LABEL_REAL
from ..utils import get_logger
from .base import SourceAdapter, SourceItem, register

log = get_logger("aidetector.ingest.folder")

_REAL_DIRS = {"real", "bonafide", "genuine", "human", "that", "goc"}
_FAKE_DIRS = {"fake", "spoof", "synthetic", "tts", "deepfake", "gia", "clone"}


# --------------------------------------------------------------------- transcript
def _load_transcripts(root: Path) -> dict[str, str]:
    """Gom transcript từ mọi quy ước quen thuộc; khoá theo stem của file audio."""
    table: dict[str, str] = {}

    # 1) metadata.csv / metadata.txt kiểu LJSpeech: `id|text` hoặc `id|raw|normalized`
    for name in ("metadata.csv", "metadata.txt", "metadata.tsv"):
        path = root / name
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            delim = "|" if "|" in line else ("\t" if "\t" in line else ",")
            parts = [p.strip() for p in line.split(delim)]
            if len(parts) >= 2:
                table.setdefault(Path(parts[0]).stem, parts[-1])

    # 2) prompts.txt / transcripts.txt: `<id> <text>`
    for name in ("prompts.txt", "transcript.txt", "transcripts.txt"):
        path = root / name
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            utt, _, text = line.strip().partition(" ")
            if utt and text:
                table.setdefault(Path(utt).stem, text.strip())

    # 3) file .txt/.lab đặt cạnh từng file audio
    for side in list(root.rglob("*.txt")) + list(root.rglob("*.lab")):
        if side.name in ("prompts.txt", "transcript.txt", "transcripts.txt", "metadata.txt"):
            continue
        try:
            table.setdefault(side.stem, side.read_text(encoding="utf-8", errors="replace").strip())
        except OSError:
            continue
    return table


def _speaker_from_path(audio: Path, root: Path) -> str:
    """Thư mục chứa file thường là speaker; file nằm phẳng ngay dưới root ⇒ unknown.

    Lấy thư mục *gần file nhất* chứ không phải tầng đầu tiên dưới root: bố cục hay
    gặp là `<bộ dữ liệu>/<split>/<speaker>/x.wav`, dùng tầng đầu sẽ gom tất cả về
    một speaker duy nhất và làm hỏng bước chia tập speaker-disjoint.
    """
    return audio.parent.name if audio.parent != root else "unknown"


@register
class FolderAdapter(SourceAdapter):
    name = "folder"
    description = "Thư mục audio bất kỳ (speaker = thư mục con, transcript tự dò)"

    @classmethod
    def probe(cls, root: Path) -> float:
        # Điểm thấp có chủ đích: chỉ thắng khi không adapter chuyên biệt nào nhận.
        return 0.15 if next(iter_audio_files(root), None) else 0.0

    def count_hint(self, root: Path) -> int | None:
        return sum(1 for _ in iter_audio_files(root)) or None

    def iter_items(self, root: Path) -> Iterator[SourceItem]:
        transcripts = _load_transcripts(root)
        if transcripts:
            log.info("Tìm thấy %d transcript", len(transcripts))
        for audio in iter_audio_files(root):
            yield SourceItem(
                key=str(audio.relative_to(root)),
                audio_path=audio,
                speaker=_speaker_from_path(audio, root),
                text=transcripts.get(audio.stem, ""),
            )


@register
class LabeledFolderAdapter(SourceAdapter):
    name = "labeled_folder"
    description = "Dataset đã chia sẵn real/ và fake/ (hoặc bonafide/ và spoof/)"

    @classmethod
    def probe(cls, root: Path) -> float:
        names = {p.name.lower() for p in root.iterdir() if p.is_dir()} if root.is_dir() else set()
        has_real = bool(names & _REAL_DIRS)
        has_fake = bool(names & _FAKE_DIRS)
        if has_real and has_fake:
            return 0.9
        return 0.0

    def iter_items(self, root: Path) -> Iterator[SourceItem]:
        transcripts = _load_transcripts(root)
        for sub in sorted(p for p in root.iterdir() if p.is_dir()):
            key = sub.name.lower()
            if key in _REAL_DIRS:
                label = LABEL_REAL
            elif key in _FAKE_DIRS:
                label = LABEL_FAKE
            else:
                continue
            for audio in iter_audio_files(sub):
                rel = audio.relative_to(sub)
                yield SourceItem(
                    key=str(audio.relative_to(root)),
                    audio_path=audio,
                    speaker=rel.parts[0] if len(rel.parts) > 1 else "unknown",
                    text=transcripts.get(audio.stem, ""),
                    # Nhãn + tên generator (nếu tầng thư mục thứ hai là tên engine)
                    # được chuyển qua meta để runner đọc.
                    meta={
                        "label": label,
                        "generator": (rel.parts[0] if label == LABEL_FAKE and len(rel.parts) > 1
                                      else ("unknown" if label == LABEL_FAKE else "")),
                    },
                )
