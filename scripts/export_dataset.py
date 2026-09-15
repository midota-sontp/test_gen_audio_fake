#!/usr/bin/env python3
"""Bung shard tar thành cây dataset cuối theo spec Demo v1 §12.

    vietnamese-audio-deepfake-demo/
    ├── audio/real/{common_voice,vietmed,vivos}/*.wav
    ├── audio/fake/<generator>/*.wav
    ├── metadata/{metadata.csv,metadata.parquet,metadata.jsonl}
    └── README.md

Tar shard chỉ là dạng vận chuyển (Kaggle upload 30k file rời rất chậm). Bước này
tách khỏi pipeline để chạy lại bao nhiêu lần cũng được mà không đụng tới tiến độ.

    python scripts/export_dataset.py --out /data/out --dest /data/dataset
    python scripts/export_dataset.py --out /data/out --dest /data/dataset --verify
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import shutil
import sys
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

DEFAULT_NAME = "vietnamese-audio-deepfake-demo"

# Thứ tự cột theo spec §11: bắt buộc trước, bổ sung sau.
COLUMNS = [
    "id", "audio", "label", "speaker_id", "source", "generator", "recording_id",
    "duration", "sample_rate", "channels", "split",
    "text", "gender", "speech_ratio", "rms_db", "clipping_ratio",
    "original_sample_rate", "original_channels", "original_split", "original_duration",
    "speaker_known", "norm_gain", "norm_peak", "sha256_raw", "sha256_norm",
    "shard", "bytes", "extra",
]

SOURCE_CREDIT = {
    "common_voice": "Common Voice Corpus 20 — hataphu/common-voice-corpus-20 (HuggingFace)",
    "vietmed": "VietMed — leduckhai/VietMed (HuggingFace)",
    "vivos": "VIVOS — AILAB-VNUHCM/vivos (HuggingFace)",
    "vieneu_tts": "VieNeu-TTS-140h — pnnbao-ump/VieNeu-TTS-140h (HuggingFace, CC BY-NC 4.0)",
}


def load_metadata(out: Path) -> list[dict]:
    rows = []
    for f in sorted((out / "dist" / "metadata").glob("*.jsonl")):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def extract(out: Path, dest: Path, rows: list[dict], verify: bool,
            overwrite: bool) -> tuple[int, list[str]]:
    """Bung từng shard một, khớp theo arcname trong metadata."""
    want: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in rows:
        want[r["shard"]][r["audio"]] = r

    n, problems = 0, []
    for shard in sorted(want):
        tar_path = out / "dist" / "audio" / f"{shard}.tar"
        if not tar_path.exists():
            problems.append(f"thiếu shard {shard}.tar ({len(want[shard])} audio)")
            continue
        seen = set()
        with tarfile.open(tar_path, "r:") as tf:
            for member in tf:
                meta = want[shard].get(member.name)
                if meta is None:
                    continue
                target = dest / member.name
                seen.add(member.name)
                if target.exists() and not overwrite and target.stat().st_size == meta["bytes"]:
                    n += 1
                    continue
                fh = tf.extractfile(member)
                if fh is None:
                    problems.append(f"{shard}: không đọc được {member.name}")
                    continue
                data = fh.read()
                if verify:
                    import soundfile as sf
                    pcm, _ = sf.read(io.BytesIO(data), dtype="int16", always_2d=False)
                    h = hashlib.sha256(pcm.tobytes()).hexdigest()
                    if h != meta["sha256_norm"]:
                        problems.append(f"{meta['id']}: sha256_norm không khớp")
                        continue
                target.parent.mkdir(parents=True, exist_ok=True)
                tmp = target.with_suffix(".wav.tmp")
                tmp.write_bytes(data)
                tmp.replace(target)
                n += 1
        for missing in set(want[shard]) - seen:
            problems.append(f"{shard}: metadata có {missing} nhưng tar không có")
    return n, problems


def flatten(r: dict) -> dict:
    row = {c: r.get(c) for c in COLUMNS}
    row["extra"] = json.dumps(r.get("extra") or {}, ensure_ascii=False)
    return row


def write_metadata(dest: Path, rows: list[dict]) -> None:
    md = dest / "metadata"
    md.mkdir(parents=True, exist_ok=True)
    flat = [flatten(r) for r in rows]

    with open(md / "metadata.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(flat)

    with open(md / "metadata.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    import pyarrow as pa
    import pyarrow.parquet as pq
    pq.write_table(pa.Table.from_pylist(flat), md / "metadata.parquet",
                   compression="zstd")


def write_readme(dest: Path, rows: list[dict]) -> None:
    by_source = Counter(r["source"] for r in rows)
    by_label = Counter(r["label"] for r in rows)
    hours = sum(r["duration"] for r in rows) / 3600
    buckets = Counter()
    for r in rows:
        d = r["duration"]
        buckets["2-4s" if d < 4 else "4-6s" if d < 6 else "6-8s" if d < 8 else "8-10s"] += 1
    speakers = len({r["speaker_id"] for r in rows})

    lines = [
        f"# {dest.name}", "",
        "Corpus tiếng Việt cho bài toán phát hiện audio deepfake (REAL / FAKE).", "",
        "## Chuẩn audio", "",
        "```", "Sample rate : 16,000 Hz", "Channels    : Mono",
        "Format      : WAV", "Encoding    : PCM 16-bit", "Duration    : 2–10 giây", "```", "",
        "## Bộ lọc chất lượng", "",
        "| Điều kiện | Ngưỡng |", "|---|---|",
        "| Duration | 2.0 – 10.0 s |",
        "| Speech ratio | ≥ 0.50 (WebRTC VAD 30 ms) |",
        "| Clipping ratio | < 0.01, đo trên audio gốc |",
        "| Duplicate | loại theo SHA256 audio gốc và SHA256 sau chuẩn hoá |",
        "| Corrupt | loại file không decode được / rỗng / sample rate lạ |", "",
        "## Thống kê", "",
        f"- Tổng: **{len(rows):,}** audio, **{hours:.2f}** giờ, **{speakers:,}** speaker",
        f"- REAL: {by_label.get(0, 0):,} | FAKE: {by_label.get(1, 0):,}", "",
        "| Nguồn | Số audio | Nguồn gốc |", "|---|---|---|",
    ]
    for s, n in sorted(by_source.items()):
        lines.append(f"| `{s}` | {n:,} | {SOURCE_CREDIT.get(s, '—')} |")
    lines += ["", "| Duration | Số audio | Tỷ lệ |", "|---|---|---|"]
    for b in ("2-4s", "4-6s", "6-8s", "8-10s"):
        n = buckets.get(b, 0)
        lines.append(f"| {b} | {n:,} | {n / max(len(rows), 1) * 100:.1f}% |")

    unknown = sum(1 for r in rows if r.get("speaker_known") is False)
    lines += [
        "", "## Cấu trúc", "", "```",
        f"{dest.name}/", "├── audio/",
        "│   ├── real/{common_voice,vietmed,vivos}/*.wav",
        "│   └── fake/<generator>/*.wav",
        "├── metadata/", "│   ├── metadata.csv",
        "│   ├── metadata.parquet", "│   └── metadata.jsonl",
        "└── README.md", "```", "",
        "Split quản lý bằng cột `split` trong metadata, không chia theo thư mục —",
        "đổi cách chia không phải di chuyển hàng chục nghìn file.", "",
        "## Lưu ý", "",
    ]
    if unknown:
        lines.append(
            f"- **{unknown:,} audio có `speaker_known = false`** (mirror Common Voice 20 "
            "không kèm `client_id`): mỗi clip được coi là một speaker riêng, nên split "
            "speaker-disjoint KHÔNG bảo đảm với phần này.")
    if any(r.get("split") is None for r in rows):
        lines.append("- Cột `split` đang trống — chưa chạy bước chia speaker-disjoint.")
    lines += [
        "- Mỗi dataset nguồn có license riêng, xem bảng trên. Tải về được không đồng "
        "nghĩa với được re-upload công khai; giữ attribution khi phân phối lại.", "",
    ]
    (dest / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description="Bung shard thành cây dataset cuối")
    p.add_argument("--out", default="/data/out", help="thư mục output của build_corpus")
    p.add_argument("--dest", default="/data/dataset", help="thư mục cha của dataset")
    p.add_argument("--name", default=DEFAULT_NAME)
    p.add_argument("--verify", action="store_true",
                   help="đối chiếu sha256_norm từng file (chậm hơn)")
    p.add_argument("--overwrite", action="store_true", help="ghi đè file đã có")
    p.add_argument("--metadata-only", action="store_true", help="chỉ dựng lại metadata")
    a = p.parse_args()

    out = Path(a.out)
    dest = Path(a.dest) / a.name
    rows = load_metadata(out)
    if not rows:
        print(f"Không tìm thấy metadata nào trong {out / 'dist' / 'metadata'}", file=sys.stderr)
        return 1
    dest.mkdir(parents=True, exist_ok=True)

    print(f"Metadata : {len(rows):,} audio")
    if not a.metadata_only:
        need = sum(r["bytes"] for r in rows)
        free = shutil.disk_usage(dest).free
        print(f"Cần      : {need / 1e9:.2f} GB (còn trống {free / 1e9:.2f} GB)")
        if need > free:
            print("LỖI: không đủ dung lượng để bung.", file=sys.stderr)
            return 1
        n, problems = extract(out, dest, rows, a.verify, a.overwrite)
        print(f"Đã bung  : {n:,} file WAV")
        for msg in problems[:20]:
            print(f"  ! {msg}", file=sys.stderr)
        if problems:
            print(f"  ({len(problems)} vấn đề)", file=sys.stderr)

    write_metadata(dest, rows)
    write_readme(dest, rows)
    print(f"Xong     : {dest}")
    for f in ("metadata/metadata.csv", "metadata/metadata.parquet",
              "metadata/metadata.jsonl", "README.md"):
        print(f"           {f}  {(dest / f).stat().st_size / 1e6:.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
