"""Persistent Fish Speech S2 worker: load the 4B LM + codec ONCE, then serve
many (encode-reference / generate text->wav) requests over a stdin/stdout JSON
protocol. This amortizes the ~40s model load + optional torch.compile warmup
across the whole run instead of paying it per utterance (the subprocess-per-clip
path in fishspeech.py reloads everything each time).

Protocol — one JSON object per line:
  stdin  <- {"cmd":"prepare","id":N,"speaker":S,"ref_wav":P,"ref_text":T}
            {"cmd":"gen","id":N,"speaker":S,"text":T,"out_wav":P,"seed":K}
            {"cmd":"quit"}
  stdout -> lines prefixed with SENT, JSON:
            {"event":"ready"}
            {"id":N,"ok":true,"out":P} | {"id":N,"ok":false,"error":E}

All model/library logging goes to stderr (inherited from the parent), so stdout
stays clean for the protocol. The parent filters on the SENT prefix anyway.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np  # noqa: F401  (imported by fish_speech paths; kept explicit)
import soundfile as sf
import torch

from fish_speech.models.text2semantic.inference import (
    decode_to_audio,
    encode_audio,
    generate_long,
    init_model,
    load_codec_model,
)

SENT = "@@FWRESP@@"


def emit(obj: dict) -> None:
    sys.stdout.write(SENT + json.dumps(obj) + "\n")
    sys.stdout.flush()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-path", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=30)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--chunk-length", type=int, default=300)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    a = ap.parse_args()

    device = a.device
    precision = torch.half if a.half else torch.bfloat16
    ckpt = Path(a.checkpoint_path)

    # --- load once ---
    model, decode_one_token = init_model(ckpt, device, precision, compile=a.compile)
    with torch.device(device):
        model.setup_caches(
            max_batch_size=1,
            max_seq_len=min(model.config.max_seq_len, a.max_seq_len),
            dtype=next(model.parameters()).dtype,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    codec = load_codec_model(ckpt / "codec.pth", device, precision)

    prompts: dict[str, tuple[str, torch.Tensor]] = {}  # speaker -> (ref_text, tokens_cpu)

    def do_gen(text: str, ref_text: str, ptoks: torch.Tensor, out_wav: str, seed: int) -> None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        gen = generate_long(
            model=model,
            device=device,
            decode_one_token=decode_one_token,
            text=text,
            num_samples=1,
            max_new_tokens=a.max_new_tokens,
            top_p=a.top_p,
            top_k=a.top_k,
            temperature=a.temperature,
            compile=a.compile,
            iterative_prompt=True,
            chunk_length=a.chunk_length,
            prompt_text=[ref_text],
            prompt_tokens=[ptoks],
        )
        codes = []
        for r in gen:
            if r.action == "sample":
                codes.append(r.codes)
            elif r.action == "next":  # end of sample 0 (num_samples=1)
                break
        if not codes:
            raise RuntimeError("generate_long produced no codes")
        merged = torch.cat(codes, dim=1)
        audio = decode_to_audio(merged.to(device), codec)
        Path(out_wav).parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_wav), audio.cpu().float().numpy(), codec.sample_rate)

    emit({"event": "ready"})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception:
            continue
        cmd = req.get("cmd")
        if cmd == "quit":
            break
        rid = req.get("id")
        try:
            if cmd == "prepare":
                idx = encode_audio(req["ref_wav"], codec, device).cpu()
                prompts[req["speaker"]] = (req["ref_text"], idx)
                emit({"id": rid, "ok": True})
            elif cmd == "gen":
                spk = req["speaker"]
                if spk not in prompts:
                    raise RuntimeError(f"speaker {spk} not prepared")
                ref_text, ptoks = prompts[spk]
                do_gen(req["text"], ref_text, ptoks, req["out_wav"], int(req.get("seed", 42)))
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                emit({"id": rid, "ok": True, "out": req["out_wav"]})
            else:
                emit({"id": rid, "ok": False, "error": f"unknown cmd {cmd}"})
        except Exception as e:  # keep the worker alive across a single bad request
            traceback.print_exc()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            emit({"id": rid, "ok": False, "error": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    main()
