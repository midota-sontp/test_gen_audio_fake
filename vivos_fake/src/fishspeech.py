"""Step 5: fake-speech generators. Default backend = Fish Speech S2 (fishaudio/s2-pro).

The S2 inference interface is isolated behind `Generator` so a different backend
(another local TTS, or the fish.audio cloud API) is a drop-in: implement the two
methods and register it in `get_generator`.

Fish S2 cloning follows the documented 3-step CLI (speech.fish.audio/inference):
  1. encode reference wav -> VQ prompt tokens (.npy)   [once per speaker, cached]
  2. text2semantic: target text + prompt tokens/text -> semantic codes (.npy)
  3. decode semantic codes -> waveform (.wav)

Weights + package are installed by setup_fish.sh, not here.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

log = logging.getLogger(__name__)

_DAC = "fish_speech/models/dac/inference.py"
_T2S = "fish_speech/models/text2semantic/inference.py"


def pick_device(pref: str = "auto") -> str:
    if pref and pref != "auto":
        return pref
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


class Generator:
    """Backend interface. `prepare_speaker` returns an opaque handle passed to
    `generate` for every utterance of that speaker."""

    name = "base"

    def ensure_ready(self) -> None: ...

    def prepare_speaker(self, speaker: str, ref_wav: Path, ref_text: str): ...

    def generate(self, handle, target_text: str, out_wav: Path, seed: int | None = None) -> Path:
        raise NotImplementedError


class FishSpeechS2Generator(Generator):
    name = "FishSpeechS2"
    folder = "fishspeech"          # dataset/fake/<folder>/<spk>/

    def __init__(self, cfg: dict, cache_dir: str | Path):
        self.repo_dir = Path(cfg.get("repo_dir", "third_party/fish-speech")).resolve()
        self.ckpt = Path(cfg.get("checkpoint_dir",
                                 "third_party/fish-speech/checkpoints/s2-pro")).resolve()
        self.codec = self.ckpt / "codec.pth"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.device = pick_device(cfg.get("device", "auto"))
        self.temperature = float(cfg.get("temperature", 0.7))
        self.top_p = float(cfg.get("top_p", 0.9))
        self.top_k = int(cfg.get("top_k", 30))
        self.max_new_tokens = int(cfg.get("max_new_tokens", 0))
        self.seed = int(cfg.get("seed", 42))
        self.step_timeout = int(cfg.get("step_timeout", 900))
        self._checked = False

    # -- setup validation -------------------------------------------------
    def ensure_ready(self) -> None:
        if self._checked:
            return
        missing = []
        if not (self.repo_dir / _DAC).exists():
            missing.append(f"fish-speech repo at {self.repo_dir} (missing {_DAC})")
        if not self.codec.exists():
            missing.append(f"codec weights at {self.codec}")
        if not any(self.ckpt.glob("*.safetensors")):
            missing.append(f"S2 model weights (*.safetensors) in {self.ckpt}")
        if missing:
            raise RuntimeError(
                "Fish Speech S2 is not set up:\n  - " + "\n  - ".join(missing)
                + "\nRun `bash setup_fish.sh` first."
            )
        log.info("Fish S2 ready | repo=%s device=%s", self.repo_dir, self.device)
        self._checked = True

    # -- subprocess helper ------------------------------------------------
    def _run(self, script: str, args: list[str], cwd: Path) -> None:
        cmd = [sys.executable, str(self.repo_dir / script), *args]
        proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True,
                              text=True, timeout=self.step_timeout)
        if proc.returncode != 0:
            tail = "\n".join((proc.stderr or proc.stdout or "").strip().splitlines()[-25:])
            raise RuntimeError(f"{script} failed (exit {proc.returncode}, device={self.device}):\n{tail}")

    # -- step 1: encode reference (cached per speaker) --------------------
    def prepare_speaker(self, speaker: str, ref_wav: Path, ref_text: str):
        self.ensure_ready()
        npy = self.cache_dir / f"prompt_{speaker}.npy"
        if not npy.exists():
            with tempfile.TemporaryDirectory(dir=self.cache_dir) as td:
                tdp = Path(td)
                self._run(_DAC, [
                    "-i", str(Path(ref_wav).resolve()),
                    "-o", str(tdp / "ref.wav"),
                    "--checkpoint-path", str(self.codec),
                    "--device", self.device,
                ], cwd=tdp)
                produced = sorted(tdp.glob("*.npy"))
                if not produced:
                    raise RuntimeError(f"encode produced no .npy for speaker {speaker}")
                shutil.move(str(produced[0]), str(npy))
            log.info("encoded reference for %s", speaker)
        return {"speaker": speaker, "prompt_npy": npy, "ref_text": ref_text}

    # -- steps 2+3: synthesize target text in the cloned voice ------------
    def generate(self, handle, target_text: str, out_wav: Path, seed: int | None = None) -> Path:
        self.ensure_ready()
        out_wav = Path(out_wav)
        if out_wav.exists():
            return out_wav
        out_wav.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=self.cache_dir) as td:
            tdp = Path(td)
            t2s = [
                "--text", target_text,
                "--prompt-text", handle["ref_text"],
                "--prompt-tokens", str(handle["prompt_npy"]),
                "--checkpoint-path", str(self.ckpt),
                "--device", self.device,
                "--temperature", str(self.temperature),
                "--top-p", str(self.top_p),
                "--top-k", str(self.top_k),
                "--seed", str(self.seed if seed is None else seed),
                "--output-dir", str(tdp),
            ]
            if self.max_new_tokens > 0:
                t2s += ["--max-new-tokens", str(self.max_new_tokens)]
            self._run(_T2S, t2s, cwd=tdp)
            codes = sorted(tdp.glob("codes_*.npy")) or sorted(tdp.glob("*.npy"))
            if not codes:
                raise RuntimeError("text2semantic produced no codes")
            self._run(_DAC, [
                "-i", str(codes[0]),
                "-o", str(out_wav),
                "--checkpoint-path", str(self.codec),
                "--device", self.device,
            ], cwd=tdp)
        if not out_wav.exists():
            raise RuntimeError(f"decode produced no wav at {out_wav}")
        return out_wav


_REGISTRY = {"FishSpeechS2": FishSpeechS2Generator}


def get_generator(name: str, cfg: dict, cache_dir: str | Path) -> Generator:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown generator '{name}'. Available: {list(_REGISTRY)}")
    return _REGISTRY[name](cfg, cache_dir)
