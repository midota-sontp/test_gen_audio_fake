"""Orchestrator: scan -> real clips + references (parallel) -> fakes (sequential) -> metadata.

Resume-safe: an audio file that already exists on disk and is valid is skipped, so
an interrupted run continues where it stopped. Fake generation is serialized on the
accelerator (running several 4B S2 processes on one GPU/MPS would OOM); num_workers
parallelizes only the I/O-bound real-audio normalization and reference building.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .fishspeech import get_generator
from .metadata import MetadataWriter
from .parser import Utterance, group_by_speaker, scan_dataset
from .preprocess import normalize_file, validate_audio
from .reference_builder import Reference, build_reference

log = logging.getLogger(__name__)

try:
    from tqdm import tqdm
except Exception:  # tqdm optional — fall back to a no-op wrapper
    def tqdm(it, **kw):
        return it


class DatasetGenerator:
    def __init__(self, config: dict):
        self.cfg = config
        self.dataset_root = Path(config["dataset_root"])
        self.out = Path(config["output_root"])
        self.splits = config.get("splits", ["train", "test"])
        self.sr = int(config.get("sample_rate", 16000))
        self.mono = bool(config.get("mono", True))
        self.peak = bool(config.get("peak_normalize", True))
        self.ref_seconds = float(config.get("reference_seconds", 15))
        self.num_workers = max(1, int(config.get("num_workers", 4)))
        self.overwrite = bool(config.get("overwrite", False))
        self.limit = config.get("limit")                    # None = all
        self.max_per_speaker = config.get("max_per_speaker")  # None = no cap
        self.seed = int(config.get("fishspeech", {}).get("seed", 42))

        self.real_dir = self.out / "real"
        self.ref_dir = self.out / "reference"
        self.cache_dir = self.out / ".cache"
        self.meta = MetadataWriter(self.out / "metadata" / "metadata.csv")

        self.gen = get_generator(
            config.get("generator", "FishSpeechS2"),
            config.get("fishspeech", {}),
            self.cache_dir,
        )
        self.fake_dir = self.out / "fake" / getattr(self.gen, "folder", self.gen.name.lower())

    # -- path helpers -----------------------------------------------------
    def _rel(self, p: Path) -> str:
        return p.resolve().relative_to(self.out.resolve()).as_posix()

    def _real_path(self, u: Utterance) -> Path:
        return self.real_dir / u.speaker / f"{u.audio_id}.wav"

    def _fake_path(self, u: Utterance) -> Path:
        return self.fake_dir / u.speaker / f"{u.audio_id}_fake.wav"

    def _done(self, dst: Path, rel: str) -> bool:
        return (not self.overwrite) and validate_audio(dst) and self.meta.has(rel)

    # -- phase A: real clips + references (parallel, I/O bound) -----------
    def _process_real(self, u: Utterance) -> str:
        dst = self._real_path(u)
        rel = self._rel(dst)
        if self._done(dst, rel):
            return "skip"
        if self.overwrite and dst.exists():
            dst.unlink()
        if not normalize_file(u.wav_path, dst, self.sr, self.mono, self.peak):
            return "corrupt"
        if not validate_audio(dst):
            return "invalid"
        self.meta.add(rel, 0, u.speaker, u.text, "real", u.split)
        return "ok"

    def _phase_real(self, records: list[Utterance]) -> None:
        log.info("Phase A: normalizing %d real clips (workers=%d)", len(records), self.num_workers)
        counts = {"ok": 0, "skip": 0, "corrupt": 0, "invalid": 0}
        with ThreadPoolExecutor(max_workers=self.num_workers) as ex:
            for res in tqdm(ex.map(self._process_real, records),
                            total=len(records), desc="real", unit="clip"):
                counts[res] += 1
        log.info("Phase A done: %s", counts)

    def _build_refs(self, groups: dict[str, list[Utterance]]) -> dict[str, Reference]:
        log.info("Building references for %d speakers", len(groups))
        refs: dict[str, Reference] = {}

        def one(item):
            spk, utts = item
            try:
                return spk, build_reference(spk, utts, self.ref_seconds, self.sr,
                                            self.ref_dir, self.overwrite)
            except Exception as e:
                log.error("reference failed for %s: %s", spk, e)
                return spk, None

        with ThreadPoolExecutor(max_workers=self.num_workers) as ex:
            for spk, ref in tqdm(ex.map(one, groups.items()),
                                 total=len(groups), desc="refs", unit="spk"):
                if ref is not None:
                    refs[spk] = ref
        return refs

    # -- phase B: fakes (sequential on the accelerator) -------------------
    def _phase_fakes(self, groups: dict[str, list[Utterance]], refs: dict[str, Reference]) -> None:
        # Pair each fake with a REAL clip that actually exists (a corrupt/skipped
        # source has no real clip, so it gets no fake). Then drop those already done (resume).
        def needs(u: Utterance) -> bool:
            if not validate_audio(self._real_path(u)):
                return False
            fp = self._fake_path(u)
            return not self._done(fp, self._rel(fp))

        pending = {spk: [u for u in utts if needs(u)] for spk, utts in groups.items()}
        total_pending = sum(len(v) for v in pending.values())
        total = sum(len(v) for v in groups.values())
        log.info("Phase B: %d/%d fakes to generate (rest already done)", total_pending, total)
        if total_pending == 0:
            return
        self.gen.ensure_ready()  # fail fast if S2 isn't set up

        made = errors = 0
        bar = tqdm(total=total_pending, desc="fake", unit="clip")
        for spk in sorted(groups):
            pend = pending[spk]
            if not pend:
                continue
            if spk not in refs:
                log.warning("no reference for %s — skipping %d fakes", spk, len(pend))
                continue
            try:
                handle = self.gen.prepare_speaker(spk, refs[spk].wav_path, refs[spk].text)
            except Exception as e:
                log.error("prepare_speaker failed for %s (%d fakes skipped): %s", spk, len(pend), e)
                bar.update(len(pend))
                continue
            for i, u in enumerate(pend):
                dst = self._fake_path(u)
                rel = self._rel(dst)
                if self.overwrite and dst.exists():
                    dst.unlink()
                try:
                    self.gen.generate(handle, u.text, dst, seed=self.seed + i)
                except Exception as e:
                    log.error("generate failed %s: %s", u.audio_id, e)
                    errors += 1
                    bar.update(1)
                    continue
                # enforce 16k/mono/PCM-16 on the S2 output, then validate before metadata
                if not normalize_file(dst, dst, self.sr, self.mono, self.peak) or not validate_audio(dst):
                    log.error("generated audio invalid, dropping: %s", rel)
                    if dst.exists():
                        dst.unlink()
                    errors += 1
                    bar.update(1)
                    continue
                self.meta.add(rel, 1, u.speaker, u.text, self.gen.name, u.split)
                made += 1
                bar.update(1)
        bar.close()
        log.info("Phase B done: generated=%d errors=%d", made, errors)

    def _select(self, records: list[Utterance]) -> list[Utterance]:
        """Apply max_per_speaker + limit. Round-robin across speakers so a `limit`
        stays balanced across voices rather than draining one speaker first."""
        if self.max_per_speaker is None and self.limit is None:
            return records
        by_spk = group_by_speaker(records)
        cap = self.max_per_speaker
        queues = {s: (u[: int(cap)] if cap is not None else list(u)) for s, u in by_spk.items()}
        selected: list[Utterance] = []
        limit = None if self.limit is None else int(self.limit)
        order = sorted(queues)
        i = 0
        while any(queues.values()) and (limit is None or len(selected) < limit):
            q = queues[order[i % len(order)]]
            i += 1
            if q:
                selected.append(q.pop(0))
            if i % len(order) == 0 and not any(queues.values()):
                break
        log.info("Selected %d/%d utterances (limit=%s, max_per_speaker=%s)",
                 len(selected), len(records), self.limit, self.max_per_speaker)
        return selected

    # -- entry ------------------------------------------------------------
    def run(self) -> None:
        records = scan_dataset(self.dataset_root, self.splits)
        records = self._select(records)
        groups = group_by_speaker(records)
        log.info("Processing %d utterances across %d speakers", len(records), len(groups))
        self._phase_real(records)
        refs = self._build_refs(groups)
        self._phase_fakes(groups, refs)
        log.info("Complete. metadata rows=%d -> %s", len(self.meta), self.meta.path)
