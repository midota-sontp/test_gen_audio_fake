"""Các phép augmentation ở mức sóng âm.

Nguyên tắc: mọi phép ở đây phải **áp dụng như nhau cho cả real lẫn fake**. Nếu chỉ
augment một lớp, chính phép augment trở thành manh mối và mô hình sẽ học nhầm.

Mỗi hàm nhận (audio float32 mono, sample_rate, rng) và trả (audio, tên-mô-tả).
"""

from __future__ import annotations

import random
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from ..utils import get_logger

log = get_logger("aidetector.augment.ops")

_FFMPEG = shutil.which("ffmpeg")


def has_ffmpeg() -> bool:
    return _FFMPEG is not None


def _rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(audio**2)) + 1e-12)


def _mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Trộn nhiễu vào tín hiệu sạch ở đúng mức SNR yêu cầu."""
    if noise.size < clean.size:
        noise = np.tile(noise, int(np.ceil(clean.size / max(noise.size, 1))))
    noise = noise[: clean.size]
    scale = _rms(clean) / (_rms(noise) * (10 ** (snr_db / 20)))
    return clean + noise * scale


# --------------------------------------------------------------------- nhiễu
def gaussian_noise(audio, sr, rng: random.Random, snr_range=(10.0, 25.0)):
    snr = rng.uniform(*snr_range)
    noise = np.random.default_rng(rng.randrange(2**31)).standard_normal(audio.size).astype(np.float32)
    return _mix_at_snr(audio, noise, snr), f"gauss{snr:.0f}db"


def background_noise(audio, sr, rng: random.Random, noise_files: list[Path] | None = None,
                     snr_range=(5.0, 20.0)):
    """Trộn nhiễu nền thật lấy từ thư mục người dùng cung cấp (MUSAN, WHAM, ...)."""
    if not noise_files:
        return gaussian_noise(audio, sr, rng, snr_range)
    import librosa

    path = noise_files[rng.randrange(len(noise_files))]
    snr = rng.uniform(*snr_range)
    try:
        noise, _ = librosa.load(str(path), sr=sr, mono=True)
    except Exception as exc:  # noqa: BLE001
        log.debug("Bỏ qua file nhiễu %s: %s", path, exc)
        return audio, ""
    if noise.size > audio.size:
        start = rng.randrange(noise.size - audio.size)
        noise = noise[start : start + audio.size]
    return _mix_at_snr(audio, noise.astype(np.float32), snr), f"bg{snr:.0f}db"


# ------------------------------------------------------------------ vang/phòng
def reverb(audio, sr, rng: random.Random, rir_files: list[Path] | None = None,
           rt60_range=(0.15, 0.6)):
    """Vang phòng: dùng RIR thật nếu có, không thì tổng hợp bằng nhiễu suy giảm mũ."""
    if rir_files:
        import librosa

        path = rir_files[rng.randrange(len(rir_files))]
        try:
            rir, _ = librosa.load(str(path), sr=sr, mono=True)
            rir = rir / (np.max(np.abs(rir)) + 1e-9)
            wet = np.convolve(audio, rir)[: audio.size]
            return wet.astype(np.float32), f"rir:{path.stem[:12]}"
        except Exception as exc:  # noqa: BLE001
            log.debug("Bỏ qua RIR %s: %s", path, exc)

    rt60 = rng.uniform(*rt60_range)
    length = int(rt60 * sr)
    gen = np.random.default_rng(rng.randrange(2**31))
    decay = np.exp(-6.9 * np.arange(length) / max(length, 1))
    rir = (gen.standard_normal(length) * decay).astype(np.float32)
    rir[0] = 1.0
    rir /= np.max(np.abs(rir)) + 1e-9
    wet = np.convolve(audio, rir)[: audio.size]
    return wet.astype(np.float32), f"rev{rt60:.2f}"


# ----------------------------------------------------------------- băng thông
def band_limit(audio, sr, rng: random.Random, cutoffs=(3400, 4000, 5000, 6000)):
    """Giả lập kênh thoại/băng hẹp bằng cách hạ rồi nâng lại tần số lấy mẫu."""
    import librosa

    cutoff = cutoffs[rng.randrange(len(cutoffs))]
    low_sr = int(cutoff * 2)
    down = librosa.resample(audio, orig_sr=sr, target_sr=low_sr)
    up = librosa.resample(down, orig_sr=low_sr, target_sr=sr)
    return up[: audio.size].astype(np.float32), f"band{cutoff}"


def speed_perturb(audio, sr, rng: random.Random, factors=(0.9, 0.95, 1.05, 1.1)):
    """Đổi tốc độ (kéo theo cao độ) — biến thể chuẩn của Kaldi."""
    import librosa

    factor = factors[rng.randrange(len(factors))]
    out = librosa.resample(audio, orig_sr=sr, target_sr=int(sr / factor))
    return out.astype(np.float32), f"speed{factor:g}"


def gain(audio, sr, rng: random.Random, db_range=(-6.0, 6.0)):
    db = rng.uniform(*db_range)
    out = audio * (10 ** (db / 20))
    return np.clip(out, -1.0, 1.0).astype(np.float32), f"gain{db:+.0f}db"


# --------------------------------------------------------------------- nén
_CODEC_ARGS = {
    "mp3": (["-codec:a", "libmp3lame", "-b:a", "{br}k"], "mp3"),
    "aac": (["-codec:a", "aac", "-b:a", "{br}k"], "m4a"),
    "opus": (["-codec:a", "libopus", "-b:a", "{br}k"], "ogg"),
}


def codec(audio, sr, rng: random.Random, codecs=("mp3", "aac"), bitrates=(32, 48, 64, 96)):
    """Nén mất dữ liệu rồi giải nén — mô phỏng audio đi qua mạng xã hội/điện thoại.

    Đây là phép augment quan trọng nhất cho bài toán này: hầu hết audio deepfake
    ngoài đời đều đã qua ít nhất một vòng nén.
    """
    if not has_ffmpeg():
        return audio, ""
    import soundfile as sf

    name = codecs[rng.randrange(len(codecs))]
    bitrate = bitrates[rng.randrange(len(bitrates))]
    args, ext = _CODEC_ARGS[name]
    args = [a.format(br=bitrate) for a in args]

    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "in.wav"
        mid = Path(tmp) / f"mid.{ext}"
        dst = Path(tmp) / "out.wav"
        sf.write(str(src), audio, sr, subtype="PCM_16")
        base = [_FFMPEG, "-hide_banner", "-loglevel", "error", "-y"]
        try:
            subprocess.run(base + ["-i", str(src), *args, str(mid)], check=True, timeout=120)
            subprocess.run(
                base + ["-i", str(mid), "-ar", str(sr), "-ac", "1", str(dst)],
                check=True, timeout=120,
            )
            out, _ = sf.read(str(dst), dtype="float32")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            log.debug("ffmpeg lỗi (%s %dk): %s", name, bitrate, exc)
            return audio, ""
    out = np.asarray(out, dtype=np.float32).reshape(-1)
    if out.size < audio.size:                       # codec có thể thêm/bớt vài mẫu
        out = np.pad(out, (0, audio.size - out.size))
    return out[: audio.size], f"{name}{bitrate}k"


#: tên phép → hàm. Config chỉ cần nhắc tên, thêm phép mới chỉ cần bổ sung vào đây.
OPS = {
    "gaussian_noise": gaussian_noise,
    "background_noise": background_noise,
    "reverb": reverb,
    "band_limit": band_limit,
    "speed_perturb": speed_perturb,
    "gain": gain,
    "codec": codec,
}
