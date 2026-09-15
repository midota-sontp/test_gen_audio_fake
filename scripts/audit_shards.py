#!/usr/bin/env python3
"""Kiểm tra độc lập audio ĐÃ GHI trong shard, trước/sau khi đẩy lên Kaggle.

Không tin vào metadata: mở từng file WAV trong tar ra đo lại, rồi mới đối chiếu.
Trả về mã thoát 1 nếu có bất kỳ sai lệch nào.

    python scripts/audit_shards.py --out /data/out
    python scripts/audit_shards.py --out /data/out --pushed-only   # chỉ shard đã đẩy
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import soundfile as sf                                   # noqa: E402

from corpus import qc                                    # noqa: E402
from corpus.state import State                           # noqa: E402


def audit(out: Path, pushed_only: bool, sample: int | None,
          min_dur: float, max_dur: float, min_speech: float,
          max_clip: float) -> int:
    st = State(out / "state" / "corpus.sqlite", readonly=True)
    shards = {r["name"]: r for r in st.shards()}
    st.db.close()

    rows = []
    for f in sorted((out / "dist" / "metadata").glob("*.jsonl")):
        with open(f, encoding="utf-8") as fh:
            rows += [json.loads(l) for l in fh if l.strip()]
    by_shard: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in rows:
        by_shard[r["shard"]][r["audio"]] = r

    targets = [s for s in sorted(by_shard)
               if not pushed_only or (shards.get(s) or {})["pushed_at"]]
    if not targets:
        print("Không có shard nào để kiểm.", file=sys.stderr)
        return 1

    fail = Counter()
    seen_norm, seen_raw, seen_id = {}, {}, {}
    checked = 0

    def bad(kind: str, msg: str) -> None:
        fail[kind] += 1
        if fail[kind] <= 5:
            print(f"  ✗ [{kind}] {msg}", file=sys.stderr)

    for shard in targets:
        tar_path = out / "dist" / "audio" / f"{shard}.tar"
        if not tar_path.exists():
            bad("thiếu_tar", shard)
            continue
        want = dict(by_shard[shard])
        pushed = (shards.get(shard) or {})["pushed_at"]
        n_in_tar = 0
        with tarfile.open(tar_path, "r:") as tf:
            for m in tf:
                if not m.isfile():
                    continue
                n_in_tar += 1
                meta = want.pop(m.name, None)
                if meta is None:
                    bad("thừa_trong_tar", f"{shard}:{m.name} không có dòng metadata")
                    continue
                if sample and checked >= sample:
                    continue
                data = tf.extractfile(m).read()
                checked += 1

                # --- đo lại từ chính file đã ghi ---
                try:
                    info = sf.info(io.BytesIO(data))
                    pcm, sr = sf.read(io.BytesIO(data), dtype="int16", always_2d=False)
                except Exception as e:
                    bad("không_đọc_được", f"{meta['id']}: {e}")
                    continue
                if sr != 16000:
                    bad("sai_sample_rate", f"{meta['id']}: {sr}")
                if info.channels != 1:
                    bad("sai_channels", f"{meta['id']}: {info.channels}")
                if info.subtype != "PCM_16":
                    bad("sai_encoding", f"{meta['id']}: {info.subtype}")
                if pcm.ndim != 1:
                    bad("sai_channels", f"{meta['id']}: ndim={pcm.ndim}")
                    continue

                dur = len(pcm) / sr
                if not (min_dur <= dur <= max_dur):
                    bad("duration_ngoài_ngưỡng", f"{meta['id']}: {dur:.3f}s")
                if abs(dur - meta["duration"]) > 0.001:
                    bad("duration_lệch_metadata",
                        f"{meta['id']}: file {dur:.3f} vs metadata {meta['duration']}")

                h = hashlib.sha256(pcm.tobytes()).hexdigest()
                if h != meta["sha256_norm"]:
                    bad("sha256_không_khớp", meta["id"])

                x = pcm.astype("float32") / 32768.0
                speech = qc.speech_ratio(x)
                if speech < min_speech:
                    bad("speech_dưới_ngưỡng", f"{meta['id']}: {speech:.3f}")
                if meta["speech_ratio"] < min_speech:
                    bad("metadata_ghi_speech_xấu", f"{meta['id']}: {meta['speech_ratio']}")
                if meta["clipping_ratio"] >= max_clip:
                    bad("metadata_ghi_clipping_xấu", f"{meta['id']}: {meta['clipping_ratio']}")

                if meta["label"] == 1 and not meta.get("generator"):
                    bad("fake_thiếu_generator", meta["id"])
                if meta["label"] == 0 and meta.get("generator"):
                    bad("real_có_generator", meta["id"])
                if not meta.get("speaker_id") or not meta.get("recording_id"):
                    bad("thiếu_speaker_recording", meta["id"])

                for key, store, tag in ((h, seen_norm, "trùng_sha_norm"),
                                        (meta["sha256_raw"], seen_raw, "trùng_sha_raw"),
                                        (meta["id"], seen_id, "trùng_id")):
                    if key in store:
                        bad(tag, f"{meta['id']} ↔ {store[key]}")
                    store[key] = meta["id"]

        for leftover in want:
            bad("thiếu_trong_tar", f"{shard}: metadata có {leftover} nhưng tar không có")
        flag = "đã đẩy" if pushed else "chưa đẩy"
        print(f"  {shard:<20} {n_in_tar:>6,} file trong tar  ({flag})")

    total = sum(len(v) for k, v in by_shard.items() if k in targets)
    print(f"\nShard kiểm  : {len(targets)}")
    print(f"Metadata    : {total:,} dòng")
    print(f"Mở & đo lại : {checked:,} file")
    if not fail:
        print("KẾT LUẬN    : ✓ mọi file đều đạt chuẩn 16 kHz mono PCM_16, "
              "đúng duration/speech/clipping, hash khớp, không trùng lặp.")
        return 0
    print("KẾT LUẬN    : ✗ có sai lệch:", file=sys.stderr)
    for k, v in fail.most_common():
        print(f"   {k:<28} {v:,}", file=sys.stderr)
    return 1


def main() -> int:
    p = argparse.ArgumentParser(description="Audit audio trong shard")
    p.add_argument("--out", default="/data/out")
    p.add_argument("--pushed-only", action="store_true",
                   help="chỉ kiểm các shard đã đẩy lên Kaggle")
    p.add_argument("--sample", type=int, default=None,
                   help="chỉ mở & đo lại N file đầu (kiểm nhanh)")
    p.add_argument("--min-duration", type=float, default=2.0)
    p.add_argument("--max-duration", type=float, default=10.0)
    p.add_argument("--min-speech-ratio", type=float, default=0.50)
    p.add_argument("--max-clipping-ratio", type=float, default=0.01)
    a = p.parse_args()
    return audit(Path(a.out), a.pushed_only, a.sample, a.min_duration,
                 a.max_duration, a.min_speech_ratio, a.max_clipping_ratio)


if __name__ == "__main__":
    raise SystemExit(main())
