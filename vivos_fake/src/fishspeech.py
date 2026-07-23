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

import json
import logging
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

log = logging.getLogger(__name__)

# invoked as installed modules (default — works in the official image and after
# `pip install -e`), or as repo script files (invoke="script")
_DAC_MOD = "fish_speech.models.dac.inference"
_T2S_MOD = "fish_speech.models.text2semantic.inference"
_DAC = "fish_speech/models/dac/inference.py"
_T2S = "fish_speech/models/text2semantic/inference.py"


def pick_device(pref: str = "auto") -> str:
    # env override wins (Docker: no MPS -> FISH_DEVICE=cpu; Linux+NVIDIA -> cuda)
    env = os.environ.get("FISH_DEVICE")
    if env:
        return env
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


class _FishWorker:
    """Drives a persistent fish_worker.py subprocess (model loaded once). A
    background thread drains the worker's stdout so its pipe never blocks; only
    lines tagged with SENT are treated as protocol responses (everything else,
    e.g. loguru/torch logs on a stray stdout write, is ignored)."""

    SENT = "@@FWRESP@@"

    def __init__(self, cmd: list[str], cwd: Path, env: dict,
                 load_timeout: int, req_timeout: int):
        self.req_timeout = req_timeout
        self.proc = subprocess.Popen(
            cmd, cwd=str(cwd), env=env,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, bufsize=1,
        )
        self._q: queue.Queue = queue.Queue()
        threading.Thread(target=self._reader, daemon=True).start()
        try:
            ev = self._q.get(timeout=load_timeout)
        except queue.Empty:
            raise RuntimeError(f"fish worker did not become ready within {load_timeout}s")
        if ev.get("event") != "ready":
            raise RuntimeError(f"fish worker sent unexpected first message: {ev}")

    def _reader(self) -> None:
        for line in self.proc.stdout:  # inherits worker stdout until EOF
            line = line.strip()
            if line.startswith(self.SENT):
                try:
                    self._q.put(json.loads(line[len(self.SENT):]))
                except Exception:
                    pass
        self._q.put({"event": "eof"})

    def request(self, obj: dict) -> dict:
        if self.proc.poll() is not None:
            raise RuntimeError(f"fish worker already exited (code {self.proc.returncode})")
        self.proc.stdin.write(json.dumps(obj) + "\n")
        self.proc.stdin.flush()
        try:
            r = self._q.get(timeout=self.req_timeout)
        except queue.Empty:
            raise RuntimeError(f"fish worker request timed out after {self.req_timeout}s")
        if r.get("event") == "eof":
            raise RuntimeError("fish worker process died")
        if not r.get("ok"):
            raise RuntimeError(r.get("error", "fish worker error"))
        return r

    def close(self) -> None:
        try:
            self.proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
            self.proc.stdin.flush()
            self.proc.wait(timeout=10)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass


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
        # "module"/"script": one CLI subprocess per step (reloads the model each
        # clip). "worker": one persistent process, model loaded once (much faster).
        self.invoke = cfg.get("invoke", "module")
        self.use_worker = self.invoke == "worker"
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
        self.half = bool(cfg.get("half", False))  # fp16 -> ~half the RAM (helps CPU OOM)
        self.step_timeout = int(cfg.get("step_timeout", 900))
        # worker-only knobs
        self.compile = bool(cfg.get("compile", True))       # torch.compile decode (fast after warmup)
        self.chunk_length = int(cfg.get("chunk_length", 300))
        self.max_seq_len = int(cfg.get("max_seq_len", 2048))  # caps KV cache (T4 16GB safe)
        self.load_timeout = int(cfg.get("load_timeout", 600))
        self._checked = False
        self._worker: _FishWorker | None = None
        self._req_id = 0

    # -- setup validation -------------------------------------------------
    def ensure_ready(self) -> None:
        if self._checked:
            return
        missing = []
        if self.invoke == "module":
            import importlib.util
            if importlib.util.find_spec("fish_speech") is None:
                missing.append("python package `fish_speech` not importable (use the Docker image "
                               "or run setup_fish.sh)")
        elif not (self.repo_dir / _DAC).exists():
            missing.append(f"fish-speech repo at {self.repo_dir} (missing {_DAC})")
        if not self.codec.exists():
            missing.append(f"codec weights at {self.codec}")
        if not any(self.ckpt.glob("*.safetensors")):
            missing.append(f"S2 model weights (*.safetensors) in {self.ckpt}")
        if missing:
            raise RuntimeError(
                "Fish Speech S2 is not set up:\n  - " + "\n  - ".join(missing)
                + "\nRun in the Docker image, or `bash setup_fish.sh` for a native install."
            )
        log.info("Fish S2 ready | invoke=%s device=%s ckpt=%s", self.invoke, self.device, self.ckpt)
        self._checked = True
        if self.use_worker:
            self._start_worker()

    # -- persistent worker ------------------------------------------------
    def _next_id(self) -> int:
        self._req_id += 1
        return self._req_id

    def _start_worker(self) -> None:
        if self._worker is not None:
            return
        worker_py = Path(__file__).with_name("fish_worker.py")
        cmd = [
            sys.executable, str(worker_py),
            "--checkpoint-path", str(self.ckpt),
            "--device", self.device,
            "--temperature", str(self.temperature),
            "--top-p", str(self.top_p),
            "--top-k", str(self.top_k),
            "--max-new-tokens", str(self.max_new_tokens if self.max_new_tokens > 0 else 512),
            "--chunk-length", str(self.chunk_length),
            "--max-seq-len", str(self.max_seq_len),
        ]
        if self.half:
            cmd.append("--half")
        if self.compile:
            cmd.append("--compile")
        env = dict(os.environ)
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        log.info("starting persistent Fish worker (compile=%s half=%s) — loading model once ...",
                 self.compile, self.half)
        # first generate() triggers torch.compile warmup, so requests get step_timeout
        self._worker = _FishWorker(cmd, cwd=Path.cwd(), env=env,
                                   load_timeout=self.load_timeout, req_timeout=self.step_timeout)
        log.info("Fish worker ready")

    # -- subprocess helper ------------------------------------------------
    def _base_cmd(self, which: str) -> list[str]:
        if self.invoke == "module":
            return [sys.executable, "-m", (_DAC_MOD if which == "dac" else _T2S_MOD)]
        return [sys.executable, str(self.repo_dir / (_DAC if which == "dac" else _T2S))]

    def _run(self, which: str, args: list[str], cwd: Path) -> None:
        cmd = [*self._base_cmd(which), *args]
        # reduce CUDA fragmentation — on tight cards (e.g. T4 16GB) the t2s KV
        # cache leaves only a few hundred MB free; expandable_segments recovers it
        env = dict(os.environ)
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True,
                              text=True, timeout=self.step_timeout, env=env)
        if proc.returncode != 0:
            tail = "\n".join((proc.stderr or proc.stdout or "").strip().splitlines()[-25:])
            raise RuntimeError(f"{which} step failed (exit {proc.returncode}, device={self.device}):\n{tail}")

    # -- step 1: encode reference (cached per speaker) --------------------
    def prepare_speaker(self, speaker: str, ref_wav: Path, ref_text: str):
        self.ensure_ready()
        if self.use_worker:
            self._worker.request({
                "cmd": "prepare", "id": self._next_id(), "speaker": speaker,
                "ref_wav": str(Path(ref_wav).resolve()), "ref_text": ref_text,
            })
            return {"speaker": speaker}
        npy = self.cache_dir / f"prompt_{speaker}.npy"
        if not npy.exists():
            with tempfile.TemporaryDirectory(dir=self.cache_dir) as td:
                tdp = Path(td)
                self._run("dac", [
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
        if self.use_worker:
            self._worker.request({
                "cmd": "gen", "id": self._next_id(), "speaker": handle["speaker"],
                "text": target_text, "out_wav": str(out_wav.resolve()),
                "seed": self.seed if seed is None else seed,
            })
            if not out_wav.exists():
                raise RuntimeError(f"worker produced no wav at {out_wav}")
            return out_wav
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
            if self.half:
                t2s += ["--half"]
            self._run("t2s", t2s, cwd=tdp)
            codes = sorted(tdp.glob("codes_*.npy")) or sorted(tdp.glob("*.npy"))
            if not codes:
                raise RuntimeError("text2semantic produced no codes")
            self._run("dac", [
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
