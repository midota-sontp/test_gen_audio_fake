"""Download a balanced audio subset from a HF dataset and split it speaker-disjoint.

Works with two gated Vietnamese anti-spoofing datasets out of the box:
  * hustep-lab/VSASV-Dataset  (gated: auto — accept terms + token, instant)
  * Jack-ppkdczgx/SEA-Spoof   (gated: manual — request access, wait for approval)
Provide a token via the HF_TOKEN env var (or `huggingface-cli login`).

Column names vary, so audio/label/speaker/file/language columns are auto-detected
from the feature schema and can be overridden in configs/mvp.yaml (dataset.columns).
When no speaker column exists (e.g. VSASV), the speaker id is derived from the
file-name column.
"""
from __future__ import annotations

import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from ..monitoring.status import RunStatus, _stamp
from ..utils.config import Config, resolve
from ..utils.logging import get_logger

log = get_logger("dataset")

_AUDIO_HINTS = ["audio", "wav", "speech", "waveform"]
_LABEL_HINTS = ["label", "class", "spoof", "bonafide", "target", "is_spoof", "attack"]
_SPEAKER_HINTS = ["speaker", "spk", "spkr", "subject", "talker"]
_FILE_HINTS = ["file", "filename", "filepath", "path", "name", "utt", "id"]
_LANG_HINTS = ["language", "lang", "locale"]


def _pick_column(features: dict, hints: list[str], override: str | None) -> str | None:
    if override:
        return override
    names = list(features.keys())
    lower = {n.lower(): n for n in names}
    for h in hints:                       # exact-ish match first
        if h in lower:
            return lower[h]
    for h in hints:                       # then substring match
        for ln, orig in lower.items():
            if h in ln:
                return orig
    return None


def _to_binary_label(value: Any, cfg: Config) -> int | None:
    """Return 1 for fake/spoof, 0 for real/bonafide, None if unknown."""
    fake = {str(v).lower() for v in cfg.dataset.fake_label_values}
    real = {str(v).lower() for v in cfg.dataset.real_label_values}
    s = str(value).strip().lower()
    if s in fake:
        return 1
    if s in real:
        return 0
    return None


def _speaker_from_file(name: str) -> str:
    """Derive a speaker id from a file path/name (first alphanumeric token)."""
    base = Path(str(name)).stem
    tokens = re.split(r"[^A-Za-z0-9]+", base)
    return tokens[0] if tokens and tokens[0] else base


def _read_audio(value: Any) -> tuple[np.ndarray, int]:
    """Handle both an HF Audio-decoded dict and a plain {array, sampling_rate} struct."""
    if isinstance(value, dict) and "array" in value and "sampling_rate" in value:
        return np.asarray(value["array"], dtype=np.float32), int(value["sampling_rate"])
    raise ValueError(f"Unrecognized audio value of type {type(value)}")


def _hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None


def download_and_split(cfg: Config) -> Path:
    dcfg = cfg.dataset
    rng = random.Random(cfg.seed)
    raw_dir = resolve(cfg.paths.raw_dir)
    manifest_dir = resolve(cfg.paths.manifest_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    # download is the first stage → start a fresh run id for this pipeline run
    run_id = cfg.get("_run_id") or _stamp()
    cfg["_run_id"] = run_id
    status = RunStatus(resolve(cfg.monitoring.status_file), run_id, reset=True)
    status.start_stage("download", detail=dcfg.hf_id)

    token = _hf_token()
    if token is None:
        log.warning("No HF token (HF_TOKEN unset). Gated datasets will fail to download.")

    want = {0: int(dcfg.n_real), 1: int(dcfg.n_fake)}
    writer = _RawWriter(raw_dir)  # writes each wav to disk immediately (bounds RAM)
    rows, got, seen_labels = _collect_from_shards(cfg, want, token, status, writer)

    log.info("Collected real=%d fake=%d (target real=%d fake=%d)",
             got[0], got[1], want[0], want[1])
    if got[0] + got[1] == 0:
        raise RuntimeError(
            "No samples matched the configured labels. Observed label values: "
            f"{dict(seen_labels.most_common(10))}. Update dataset.real/fake_label_values."
        )
    if got[0] == 0 or got[1] == 0:
        log.warning("One class is empty — observed labels: %s", dict(seen_labels.most_common(10)))

    _assign_splits(rows, cfg, rng)  # sets r["split"] on each row (no audio held in memory)

    manifest_path = manifest_dir / "raw.csv"
    lines = ["path,label,speaker,language,split"]
    for r in rows:
        lines.append(f"{r['path']},{r['label']},{r['speaker']},{r['language']},{r['split']}")
    manifest_path.write_text("\n".join(lines) + "\n")
    log.info("Wrote %d raw clips + manifest %s", writer.n, manifest_path)
    _log_split_stats(rows)
    status.finish_stage("download", real=got[0], fake=got[1], total=len(rows))
    return manifest_path


def _detect_schema(sample, cfg):
    """Value-based column detection from a sample of rows.

    The class (real/fake) column is chosen by which column's VALUES match the
    configured real/fake vocab — not by name — so VSASV's `utt_type` is picked
    over the same-dataset `label` (speaker-id) column. Logs every string column's
    value distribution so the actual vocab is visible on the first shard.
    """
    names = list(sample[0].keys())
    feats = {n: None for n in names}
    over = cfg.dataset.get_path("columns", {}) or {}
    audio_col = _pick_column(feats, _AUDIO_HINTS, over.get("audio"))
    file_col = _pick_column(feats, _FILE_HINTS, over.get("file"))
    lang_col = _pick_column(feats, _LANG_HINTS, over.get("language"))

    str_cols = [c for c in names if isinstance(sample[0].get(c), (str, int, bool))]
    dist = {c: Counter(str(r[c]) for r in sample) for c in str_cols}
    log.info("Sampled string columns (%d rows):", len(sample))
    for c in str_cols:
        log.info("  %-12s distinct=%-4d top=%s", c, len(dist[c]), dist[c].most_common(6))

    # class column: config override, else best value-match among low-cardinality columns
    class_col = over.get("label")
    if not class_col:
        best, best_score = None, 0.0
        for c in str_cols:
            total = sum(dist[c].values()) or 1
            match = sum(n for v, n in dist[c].items() if _to_binary_label(v, cfg) is not None)
            score = match / total
            if len(dist[c]) <= 8 and score > best_score:
                best, best_score = c, score
        class_col = best or _pick_column(feats, _LABEL_HINTS, None)

    # speaker: override, else a speaker-named column, else the highest-cardinality
    # string column that is neither class nor file (an id like VSASV's `label`);
    # falls back to file-derived speaker in _RawWriter when None.
    speaker_col = over.get("speaker") or _pick_column(feats, _SPEAKER_HINTS, None)
    if not speaker_col:
        cand = [c for c in str_cols if c not in (class_col, file_col)]
        speaker_col = max(cand, key=lambda c: len(dist[c])) if cand else None

    if not audio_col or not class_col:
        raise RuntimeError(
            "Could not detect audio/class columns. Observed string columns: "
            + "; ".join(f"{c}={dist[c].most_common(4)}" for c in str_cols)
            + ". Set dataset.columns.{audio,label} and real/fake_label_values in mvp.yaml."
        )

    # Resolve value -> {0,1} for the class column. Infer the odd one out when the
    # column is binary and only one value is recognised (e.g. genuine/spoofed).
    class_vals = list(dist.get(class_col, {}))
    label_map = {v: _to_binary_label(v, cfg) for v in class_vals}
    label_map = {v: b for v, b in label_map.items() if b is not None}
    if len(class_vals) == 2 and len(label_map) == 1:
        known_v, known_b = next(iter(label_map.items()))
        other = next(v for v in class_vals if v != known_v)
        label_map[other] = 1 - known_b
        log.info("Inferred binary class: %r=%d (other=%r=%d)",
                 known_v, known_b, other, 1 - known_b)

    log.info("Detected -> audio=%s class=%s speaker=%s file=%s language=%s | label_map=%s",
             audio_col, class_col, speaker_col, file_col, lang_col, label_map)
    return audio_col, class_col, speaker_col, file_col, lang_col, label_map


def _row_speaker(ex, cols):
    speaker_col, file_col = cols[2], cols[3]
    if speaker_col and ex.get(speaker_col) not in (None, ""):
        return _safe(str(ex[speaker_col]))
    if file_col and ex.get(file_col):
        return _safe(_speaker_from_file(ex[file_col]))
    return None


class _RawWriter:
    """Writes each clip's wav to disk as it is collected, so the collector holds
    only lightweight metadata rows — RAM stays flat regardless of sample count
    (avoids the OOM from buffering thousands of float audio arrays)."""

    def __init__(self, raw_dir):
        self.raw_dir = resolve(raw_dir)  # absolute, so relative_to(repo root) is safe
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.n = 0

    def write(self, ex, y, cols):
        audio_col, lang_col = cols[0], cols[4]
        spk = _row_speaker(ex, cols) or f"unk{self.n}"
        arr, sr = _read_audio(ex[audio_col])
        lab = "fake" if y == 1 else "real"
        out = self.raw_dir / f"{lab}_{spk}_{self.n:06d}.wav"
        self.n += 1
        sf.write(str(out), arr, sr)
        rel = out.relative_to(resolve(".")).as_posix()
        return {"path": rel, "label": y, "speaker": spk,
                "language": str(ex.get(lang_col, "")) if lang_col else ""}


def _spread_indices(n, k):
    """Up to `k` shard indices spread across [0,n), ordered front-and-back."""
    base = list(range(n)) if n <= k else sorted(
        {round(i * (n - 1) / (k - 1)) for i in range(k)})
    order, i, j = [], 0, len(base) - 1
    while i <= j:
        order.append(base[i])
        if i != j:
            order.append(base[j])
        i += 1
        j -= 1
    return order


def _resolve_label(v, label_map, cfg):
    y = label_map.get(v)
    return y if y is not None else _to_binary_label(v, cfg)


def _find_boundary(n, class_of):
    """Index of the last shard whose class is 0 (bonafide), assuming shards are
    class-ordered 0..0,1..1. Returns -1 if shard 0 isn't class 0, n-1 if all class 0.
    Binary search => ~log2(n) shard reads."""
    if class_of(0) != 0:
        return -1
    lo, hi = 0, n - 1
    if class_of(hi) == 0:
        return hi
    while hi - lo > 1:                     # invariant: class_of(lo)==0, class_of(hi)==1
        mid = (lo + hi) // 2
        if class_of(mid) == 0:
            lo = mid
        else:
            hi = mid
    return lo


def _collect_from_shards(cfg, want, token, status, writer):
    """Download parquet shards and read them locally (fast, resumable, memory-safe).

    Dispatches to shared-speaker collection (confound-free) or per-class fill.
    Falls back to streaming if pyarrow/huggingface_hub are unavailable.
    """
    dcfg = cfg.dataset
    try:
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download, list_repo_files
    except Exception as e:
        log.warning("pyarrow/huggingface_hub unavailable (%s) — falling back to streaming", e)
        return _collect_streaming(cfg, want, token, writer)

    all_files = list_repo_files(dcfg.hf_id, repo_type="dataset", token=token)
    split = dcfg.source_split
    shards = sorted(f for f in all_files
                    if f.endswith(".parquet") and f"/{split}" in f + "/" and f.startswith("data/"))
    if not shards:
        shards = sorted(f for f in all_files if f.endswith(".parquet") and split in f)
    if not shards:
        log.warning("No parquet shards found for split '%s' — falling back to streaming", split)
        return _collect_streaming(cfg, want, token)

    def dl(i):
        return hf_hub_download(dcfg.hf_id, shards[i], repo_type="dataset", token=token)

    log.info("Found %d parquet shards for '%s'", len(shards), split)
    cols = _detect_schema(next(pq.ParquetFile(dl(0)).iter_batches(batch_size=400)).to_pylist(), cfg)

    if bool(dcfg.get_path("shared_speakers", True)):
        try:
            return _collect_shared_speakers(cfg, want, shards, dl, cols, pq, status, writer)
        except Exception as e:
            log.warning("Shared-speaker collection failed (%s) — falling back to per-class fill", e)
    return _collect_independent(cfg, want, shards, dl, cols, pq, status, writer)


def _collect_independent(cfg, want, shards, dl, cols, pq, status, writer):
    """Per-class fill from shards spread across the range (real & fake may come
    from different speakers — kept as a fallback)."""
    max_shards = int(cfg.dataset.get_path("max_shards", 15))
    class_col, label_map = cols[1], cols[5]
    rows: list[dict] = []
    got = {0: 0, 1: 0}
    seen: Counter = Counter()
    order = _spread_indices(len(shards), max_shards)
    for k, i in enumerate(order):
        if got[0] >= want[0] and got[1] >= want[1]:
            break
        log.info("Downloading shard %d/%d (idx %d): %s", k + 1, len(order), i, shards[i])
        pf = pq.ParquetFile(dl(i))
        for batch in pf.iter_batches(batch_size=128):
            if got[0] >= want[0] and got[1] >= want[1]:
                break
            for ex in batch.to_pylist():
                if got[0] >= want[0] and got[1] >= want[1]:
                    break
                v = str(ex[class_col])
                seen[v] += 1
                y = _resolve_label(v, label_map, cfg)
                if y is None or got[y] >= want[y]:
                    continue
                rows.append(writer.write(ex, y, cols))
                got[y] += 1
        log.info("  after shard %d: real=%d fake=%d", k + 1, got[0], got[1])
        status.update_stage("download", processed=got[0] + got[1],
                            total=want[0] + want[1], detail=f"shard {k + 1}")
    return rows, got, seen


def _collect_shared_speakers(cfg, want, shards, dl, cols, pq, status, writer):
    """Collect bonafide & spoof from the SAME speakers (removes the speaker/
    class confound). Detects the bonafide->spoof boundary, then pairs
    front-bonafide with just-after-boundary-spoof shards (same speaker-id range),
    keeps only speakers present in both classes, and reads their audio."""
    dcfg = cfg.dataset
    n = len(shards)
    budget = int(dcfg.get_path("max_shards", 40))
    _, class_col, speaker_col, _, _, label_map = cols

    def class_of(i):  # majority class of shard i, reading only the class column
        vals = pq.ParquetFile(dl(i)).read(columns=[class_col]).column(0).to_pylist()
        ys = [y for y in (_resolve_label(str(v), label_map, cfg) for v in vals) if y is not None]
        return round(sum(ys) / len(ys)) if ys else None

    b = _find_boundary(n, class_of)
    if b < 0 or b >= n - 1:
        raise RuntimeError(f"no bonafide->spoof boundary found (b={b})")
    log.info("Boundary: bonafide shards <=%d, spoof shards >=%d", b, b + 1)
    bona_idx = list(range(0, b + 1))
    spoof_idx = list(range(b + 1, n))

    # Pass A (light): read only the speaker column from paired bona/spoof shards.
    spk = defaultdict(lambda: {0: 0, 1: 0})
    used = {0: [], 1: []}

    def read_speakers(i, y):
        for s in pq.ParquetFile(dl(i)).read(columns=[speaker_col]).column(0).to_pylist():
            spk[_safe(str(s))][y] += 1
        used[y].append(i)

    bi = si = 0
    while len(used[0]) + len(used[1]) < budget and (bi < len(bona_idx) or si < len(spoof_idx)):
        if bi < len(bona_idx):
            read_speakers(bona_idx[bi], 0); bi += 1
        if si < len(spoof_idx):
            read_speakers(spoof_idx[si], 1); si += 1
        shared = [s for s, c in spk.items() if c[0] > 0 and c[1] > 0]
        a0 = sum(spk[s][0] for s in shared)
        a1 = sum(spk[s][1] for s in shared)
        log.info("  bona=%d spoof=%d shards read | shared speakers=%d | avail real=%d fake=%d",
                 len(used[0]), len(used[1]), len(shared), a0, a1)
        status.update_stage("download", processed=min(a0, want[0]) + min(a1, want[1]),
                            total=want[0] + want[1], detail=f"scan shared={len(shared)}")
        if a0 >= want[0] and a1 >= want[1]:
            break

    shared = [s for s, c in spk.items() if c[0] > 0 and c[1] > 0]
    if not shared:
        raise RuntimeError("no speaker appears in both classes among sampled shards")

    rng = random.Random(cfg.seed)
    rng.shuffle(shared)
    chosen, acc = set(), {0: 0, 1: 0}
    for s in shared:
        if acc[0] >= want[0] and acc[1] >= want[1]:
            break
        chosen.add(s)
        acc[0] += spk[s][0]
        acc[1] += spk[s][1]
    log.info("Chosen %d shared speakers (avail real=%d fake=%d of target %d/%d)",
             len(chosen), acc[0], acc[1], want[0], want[1])

    # Pass B: read audio only for chosen speakers, capped per class.
    rows: list[dict] = []
    got = {0: 0, 1: 0}
    seen: Counter = Counter()
    for y, idxs in ((0, used[0]), (1, used[1])):
        for i in idxs:
            if got[y] >= want[y]:
                break
            for batch in pq.ParquetFile(dl(i)).iter_batches(batch_size=128):
                if got[y] >= want[y]:
                    break
                for ex in batch.to_pylist():
                    if got[y] >= want[y]:
                        break
                    if _safe(str(ex[speaker_col])) not in chosen:
                        continue
                    v = str(ex[class_col])
                    seen[v] += 1
                    if _resolve_label(v, label_map, cfg) != y:
                        continue
                    rows.append(writer.write(ex, y, cols))
                    got[y] += 1
            status.update_stage("download", processed=got[0] + got[1],
                                total=want[0] + want[1], detail=f"read class{y}")
    log.info("Shared-speaker collection: real=%d fake=%d from %d speakers",
             got[0], got[1], len(chosen))
    return rows, got, seen


def _collect_streaming(cfg, want, token, writer):
    """Fallback: stream row-by-row (slower). Kept for non-parquet datasets."""
    import itertools

    from datasets import load_dataset

    dcfg = cfg.dataset
    log.info("Streaming %s [%s] ...", dcfg.hf_id, dcfg.source_split)
    ds = load_dataset(dcfg.hf_id, split=dcfg.source_split, streaming=True, token=token)

    it = iter(ds)
    peek = list(itertools.islice(it, 400))  # buffer for value-based detection
    if not peek:
        return [], {0: 0, 1: 0}, Counter()
    cols = _detect_schema(peek, cfg)
    class_col, label_map = cols[1], cols[5]

    rows: list[dict] = []
    got = {0: 0, 1: 0}
    seen: Counter = Counter()
    scanned = 0
    for ex in itertools.chain(peek, it):
        if got[0] >= want[0] and got[1] >= want[1]:
            break
        scanned += 1
        seen[str(ex[class_col])] += 1
        y = _resolve_label(str(ex[class_col]), label_map, cfg)
        if y is None or got[y] >= want[y]:
            continue
        rows.append(writer.write(ex, y, cols))
        got[y] += 1
        if scanned % 200 == 0:
            log.info("scanned=%d collected real=%d fake=%d", scanned, got[0], got[1])
    return rows, got, seen


def _splits_balanced(rows) -> bool:
    """True iff every split is non-empty and contains both classes."""
    per = defaultdict(Counter)
    for r in rows:
        per[r["split"]][r["label"]] += 1
    return all(per[s][0] > 0 and per[s][1] > 0 for s in ("train", "val", "test"))


def _stratified_split(rows, ratios, rng) -> None:
    """Per-SAMPLE stratified split (per class). Guarantees balanced non-empty splits
    but a speaker can land in multiple splits (speaker leakage) — last-resort only."""
    by_label: dict[int, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_label[r["label"]].append(i)
    r_train, r_val, _ = ratios
    for _, idxs in by_label.items():
        rng.shuffle(idxs)
        n = len(idxs)
        n_tr = int(round(n * r_train))
        n_val = max(1, int(round(n * r_val))) if n >= 3 else 0
        for j, i in enumerate(idxs):
            rows[i]["split"] = "train" if j < n_tr else ("val" if j < n_tr + n_val else "test")


def _speaker_disjoint_stratified(rows, ratios, rng) -> None:
    """Split SPEAKERS (not samples) within each class so the resulting per-SAMPLE
    counts hit ~ratios (e.g. 70/15/15), while keeping every speaker in exactly one
    split (no speaker-identity leakage).

    Speakers have very unequal clip counts, so a naive split-by-speaker-count badly
    skews sample counts (VSASV val landed at ~5% instead of 15%). Instead we greedily
    pack speakers — largest first — into whichever split is most below its sample
    quota. Result: speaker-disjoint AND sample-balanced, with both classes in every
    split. Correct anti-spoofing protocol: test speakers are unseen, so the model
    must learn spoof cues, not who is talking.
    """
    r_train, r_val, r_test = ratios
    splits = ("train", "val", "test")
    spk_rows: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        spk_rows[r["speaker"]].append(i)
    by_class: dict[int, list[str]] = defaultdict(list)
    for spk, idxs in spk_rows.items():
        cls = Counter(rows[i]["label"] for i in idxs).most_common(1)[0][0]
        by_class[cls].append(spk)

    assign: dict[str, str] = {}
    for _, spks in by_class.items():
        rng.shuffle(spks)                        # reproducible tie-break
        total = sum(len(spk_rows[s]) for s in spks)
        target = {"train": total * r_train, "val": total * r_val, "test": total * r_test}
        filled = {s: 0 for s in splits}
        # largest speakers first so the small ones fine-tune the balance
        for spk in sorted(spks, key=lambda s: len(spk_rows[s]), reverse=True):
            choice = max(splits, key=lambda s: target[s] - filled[s])
            assign[spk] = choice
            filled[choice] += len(spk_rows[spk])
    for r in rows:
        r["split"] = assign[r["speaker"]]


def _assign_splits(rows, cfg, rng) -> None:
    """Speaker-disjoint stratified split (no leakage); last-resort per-sample
    stratified only if a split ends up empty/one-class (too few speakers)."""
    _speaker_disjoint_stratified(rows, cfg.dataset.split_ratios, rng)
    if _splits_balanced(rows):
        log.info("Split: speaker-disjoint stratified (test speakers unseen in training)")
        return
    log.warning("Too few speakers for a balanced speaker-disjoint split — falling back to "
                "per-sample stratified (WARNING: speaker leakage → metrics may be optimistic).")
    _stratified_split(rows, cfg.dataset.split_ratios, rng)


def _log_split_stats(rows) -> None:
    per = defaultdict(Counter)
    spk = defaultdict(set)
    for r in rows:
        per[r["split"]][r["label"]] += 1
        spk[r["split"]].add(r["speaker"])
    for s in ("train", "val", "test"):
        c = per[s]
        log.info("  %-5s: real=%d fake=%d total=%d | speakers=%d",
                 s, c[0], c[1], c[0] + c[1], len(spk[s]))
    # confirm no speaker leakage across splits
    tr, va, te = spk["train"], spk["val"], spk["test"]
    leak = (tr & va) | (tr & te) | (va & te)
    log.info("  speaker leakage across splits: %d %s", len(leak), "(OK)" if not leak else "(!)")


def _safe(name: str) -> str:
    return "".join(ch if ch.isalnum() else "-" for ch in str(name))[:40] or "x"
