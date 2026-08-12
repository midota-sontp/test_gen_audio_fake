"""Đặc tả kỹ thuật của audio trong corpus — "chuẩn riêng" của dự án.

| Thuộc tính       | Chuẩn                                                   |
|------------------|---------------------------------------------------------|
| Sample rate      | 16 000 Hz                                                |
| Channels         | Mono (1)                                                 |
| Format           | WAV                                                      |
| Bit depth        | 16-bit PCM                                               |
| Duration         | 3–10 giây / audio                                        |
| Peak / RMS       | RMS ≈ -23 dBFS, peak trần -1 dBFS                        |
| Silence          | Cắt silence dài ở đầu/cuối                               |
| Clipping         | Không được clipping                                      |
| NaN / Inf        | Không được có                                            |
| Background noise | Cho phép, nhưng phải có CẢ bản clean lẫn bản noisy       |
| Compression      | MP3/AAC sinh thêm ở tầng augmentation, không ở corpus gốc |

`normalize()` là hàm DUY NHẤT được phép tạo file audio trong corpus — ingest và
generate đều đi qua nó, nên real và fake luôn cùng một tiền xử lý (tránh việc mô
hình học "mẹo" theo định dạng thay vì theo dấu vết giả mạo).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from ..utils import get_logger

log = get_logger("aidetector.corpus.spec")


@dataclass(frozen=True)
class AudioSpec:
    sample_rate: int = 16_000
    channels: int = 1
    subtype: str = "PCM_16"          # 16-bit PCM
    container: str = "WAV"
    min_seconds: float = 3.0
    max_seconds: float = 10.0
    target_rms_dbfs: float = -23.0   # mức RMS chuẩn hoá
    peak_ceiling_dbfs: float = -1.0  # trần peak, chừa headroom chống clipping
    trim_silence: bool = True
    trim_top_db: float = 35.0        # ngưỡng dò silence (dB dưới đỉnh)
    keep_silence_ms: float = 60.0    # chừa lại một chút silence hai đầu cho tự nhiên
    # File dài hơn max_seconds: "split" cắt thành nhiều đoạn, "crop" lấy 1 đoạn giữa,
    # "drop" bỏ hẳn.
    long_policy: str = "split"
    # File ngắn hơn min_seconds: "drop" bỏ, "pad" đệm im lặng cho đủ.
    short_policy: str = "drop"
    clipping_threshold: float = 0.999
    clipping_max_ratio: float = 0.001  # >0.1% mẫu chạm trần ⇒ coi là clipping

    @property
    def min_samples(self) -> int:
        return int(self.min_seconds * self.sample_rate)

    @property
    def max_samples(self) -> int:
        return int(self.max_seconds * self.sample_rate)

    @classmethod
    def from_config(cls, data: dict | None) -> "AudioSpec":
        if not data:
            return cls()
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        unknown = set(data) - known
        if unknown:
            log.warning("Bỏ qua khoá lạ trong audio spec: %s", ", ".join(sorted(unknown)))
        return cls(**{k: v for k, v in data.items() if k in known})

    def describe(self) -> str:
        return (
            f"{self.sample_rate} Hz · mono · {self.container}/{self.subtype} · "
            f"{self.min_seconds:g}-{self.max_seconds:g}s · RMS {self.target_rms_dbfs:g} dBFS"
        )


DEFAULT_SPEC = AudioSpec()


# --------------------------------------------------------------------------- I/O
def load_audio(path: str | Path, sample_rate: int) -> np.ndarray:
    """Đọc file bất kỳ (wav/mp3/flac/m4a/ogg) → mono float32 @ sample_rate."""
    import librosa

    audio, _ = librosa.load(str(path), sr=sample_rate, mono=True)
    return np.asarray(audio, dtype=np.float32)


def save_audio(path: str | Path, audio: np.ndarray, spec: AudioSpec = DEFAULT_SPEC) -> None:
    """Ghi WAV đúng chuẩn (16 kHz, mono, PCM_16)."""
    import soundfile as sf

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(audio, dtype=np.float32), spec.sample_rate, subtype=spec.subtype)


# --------------------------------------------------------------- các bước chuẩn hoá
def sanitize(audio: np.ndarray) -> np.ndarray:
    """Loại NaN/Inf và ép về float32 1 chiều."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if not np.all(np.isfinite(audio)):
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    return audio


def trim_edges(audio: np.ndarray, spec: AudioSpec) -> np.ndarray:
    """Cắt silence dài ở đầu/cuối, vẫn chừa lại `keep_silence_ms`."""
    if not spec.trim_silence or audio.size == 0:
        return audio
    import librosa

    _, index = librosa.effects.trim(audio, top_db=spec.trim_top_db)
    pad = int(spec.keep_silence_ms / 1000 * spec.sample_rate)
    start = max(0, int(index[0]) - pad)
    end = min(audio.size, int(index[1]) + pad)
    trimmed = audio[start:end]
    return trimmed if trimmed.size else audio


def normalize_level(audio: np.ndarray, spec: AudioSpec) -> np.ndarray:
    """Chuẩn hoá RMS về mức mục tiêu, rồi hạ gain nếu peak vượt trần.

    Chuẩn hoá mức âm lượng cho cả real lẫn fake là bắt buộc: nếu không, mô hình
    có thể phân biệt hai lớp chỉ nhờ độ to trung bình khác nhau.
    """
    if audio.size == 0:
        return audio
    rms = float(np.sqrt(np.mean(audio**2)))
    if rms > 1e-8:
        audio = audio * (10 ** (spec.target_rms_dbfs / 20) / rms)
    peak = float(np.max(np.abs(audio)))
    ceiling = 10 ** (spec.peak_ceiling_dbfs / 20)
    if peak > ceiling:
        audio = audio * (ceiling / peak)
    return np.clip(audio, -1.0, 1.0).astype(np.float32)


def fit_duration(audio: np.ndarray, spec: AudioSpec) -> list[np.ndarray]:
    """Ép độ dài về khoảng [min_seconds, max_seconds]; có thể trả nhiều đoạn."""
    n = audio.size
    if n < spec.min_samples:
        if spec.short_policy == "pad":
            return [np.pad(audio, (0, spec.min_samples - n))]
        return []
    if n <= spec.max_samples:
        return [audio]

    if spec.long_policy == "drop":
        return []
    if spec.long_policy == "crop":
        start = (n - spec.max_samples) // 2
        return [audio[start : start + spec.max_samples]]

    # "split": cắt liên tiếp, đoạn cuối quá ngắn thì bỏ.
    chunks = [audio[i : i + spec.max_samples] for i in range(0, n, spec.max_samples)]
    return [c for c in chunks if c.size >= spec.min_samples]


def normalize(audio: np.ndarray, spec: AudioSpec = DEFAULT_SPEC) -> list[np.ndarray]:
    """Toàn bộ chuỗi chuẩn hoá: sanitize → trim → cắt độ dài → chuẩn mức.

    Trả về DANH SÁCH đoạn audio hợp lệ (rỗng nếu file không dùng được).
    """
    audio = sanitize(audio)
    if audio.size == 0:
        return []
    audio = trim_edges(audio, spec)
    return [normalize_level(chunk, spec) for chunk in fit_duration(audio, spec)]


def normalize_file(
    src: str | Path, spec: AudioSpec = DEFAULT_SPEC
) -> list[np.ndarray]:
    """Đọc file nguồn rồi chuẩn hoá. Lỗi đọc ⇒ trả danh sách rỗng (không ném)."""
    try:
        audio = load_audio(src, spec.sample_rate)
    except Exception as exc:  # noqa: BLE001 — nguồn ngoài, đủ kiểu hỏng
        log.warning("Không đọc được %s: %s", src, exc)
        return []
    return normalize(audio, spec)


# ------------------------------------------------------------------ kiểm định
@dataclass
class QualityIssue:
    path: str
    code: str
    detail: str = ""


def check_quality(
    audio: np.ndarray, spec: AudioSpec, path: str = ""
) -> list[QualityIssue]:
    """Soi một mảng audio xem có vi phạm chuẩn không."""
    issues: list[QualityIssue] = []
    if audio.size == 0:
        issues.append(QualityIssue(path, "empty", "audio rỗng"))
        return issues
    if not np.all(np.isfinite(audio)):
        issues.append(QualityIssue(path, "nan_inf", "chứa NaN hoặc Inf"))
    duration = audio.size / spec.sample_rate
    if duration < spec.min_seconds - 1e-3:
        issues.append(QualityIssue(path, "too_short", f"{duration:.2f}s < {spec.min_seconds:g}s"))
    if duration > spec.max_seconds + 1e-3:
        issues.append(QualityIssue(path, "too_long", f"{duration:.2f}s > {spec.max_seconds:g}s"))
    clipped = float(np.mean(np.abs(audio) >= spec.clipping_threshold))
    if clipped > spec.clipping_max_ratio:
        issues.append(QualityIssue(path, "clipping", f"{clipped:.2%} mẫu chạm trần"))
    rms = float(np.sqrt(np.mean(audio**2)))
    if rms < 1e-5:
        issues.append(QualityIssue(path, "silent", "gần như im lặng hoàn toàn"))
    return issues


def iter_audio_files(root: str | Path, extensions: tuple[str, ...] = (
    ".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus", ".aac", ".wma",
)) -> Iterator[Path]:
    """Duyệt đệ quy mọi file audio dưới `root` (bỏ file ẩn)."""
    root = Path(root)
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in extensions and not path.name.startswith("."):
            yield path
