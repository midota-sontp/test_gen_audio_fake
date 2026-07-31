"""Checkpoint the generated dataset to a private Kaggle dataset, and pull it back on start.

Why not "sync after every clip": a Kaggle dataset is a *versioned artifact*, not a
filesystem. Every sync creates a new version and re-uploads every file in the payload
(kagglehub has no delta upload). So we do the nearest workable thing:

  * audio is packed into append-only **shard zips** (`shard-001.zip`, ...), each shard
    published as its own Kaggle dataset `<handle>-001`, `<handle>-002`, ...
  * a checkpoint re-uploads only the **open** shard, so cost stays ~O(dataset size)
    over the whole run instead of O(n^2) if everything lived in one dataset;
  * when the open shard passes `shard_mb` it is sealed (never re-uploaded) and the
    next one opens;
  * `metadata.csv` (whole file, small) rides along in every shard, so the newest shard
    always carries the full "what is done" list.

What goes up is `include` (default real+fake+reference) filtered by `exclude_patterns`,
and only files already committed to `metadata.csv` — a row is written after the audio is
validated, so a half-written wav can never reach Kaggle. `metadata.csv` (and the log file
when `logs` is included) travel loose next to the zip and are refreshed on every push,
since they keep changing; everything else is packed once.

Pushes run on a background thread (generation keeps going) and are coalesced: if one
is still uploading, the next tick is skipped. `flush()` is blocking and runs from the
`finally` of the pipeline, so Ctrl-C still checkpoints.

Resume across sessions = `pull()` on start: download every shard, extract into
output_root, merge metadata. Combined with the existing skip logic, a run started on a
fresh Kaggle session picks up exactly where the previous one stopped.

Auth: KAGGLE_USERNAME/KAGGLE_KEY, or ~/.kaggle/kaggle.json. Checked up front (whoami)
so a bad token fails in seconds rather than after hours of generation.
"""
from __future__ import annotations

import csv
import json
import logging
import shutil
import tempfile
import threading
import time
import zipfile
from fnmatch import fnmatch
from pathlib import Path

from .metadata import FIELDS

log = logging.getLogger(__name__)

_UPLOAD_DIR = "_upload"
_STATE = "state.json"


class NullSync:
    """No-op used when kaggle_sync is disabled — keeps the generator free of `if`s."""

    enabled = False

    def preflight(self) -> None: ...
    def pull(self) -> None: ...
    def tick(self, n: int = 1) -> None: ...
    def flush(self) -> None: ...


class KaggleSync:
    enabled = True

    def __init__(self, cfg: dict, output_root: str | Path, metadata_path: str | Path):
        self.out = Path(output_root)
        self.meta_path = Path(metadata_path)
        self.handle = str(cfg.get("handle", "")).strip().strip("/")
        if self.handle.count("/") != 1:
            raise SystemExit(
                f"kaggle_sync.handle must look like '<username>/<dataset-slug>' (got {self.handle!r})"
            )
        self.every_clips = int(cfg.get("every_clips", 200) or 0)
        self.every_minutes = float(cfg.get("every_minutes", 20) or 0)
        self.shard_mb = float(cfg.get("shard_mb", 400))
        # single_dataset: everything lives in ONE Kaggle dataset at `handle` (no -001
        # suffix, no sealing). Every push re-uploads the whole payload, so cost grows
        # with the dataset — that is inherent to Kaggle versioning, not a bug here.
        self.single = bool(cfg.get("single_dataset", False))
        self.pull_on_start = bool(cfg.get("pull_on_start", True))
        self.include = _parse_include(cfg)
        self.exclude = list(cfg.get("exclude_patterns") or [])

        self.sync_dir = self.out / ".sync"
        self.sync_dir.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()      # serializes push() (background + flush)
        self._thread: threading.Thread | None = None
        self._since = 0
        self._last_push = time.monotonic()
        self._state = self._load_state()
        self._assigned: set[str] = set(self._state.get("assigned", []))
        # persisted: a crash between "zipped" and "uploaded" must still upload on restart
        self._pending = bool(self._state.get("pending", False))

        was = self._state.get("include")
        if was is not None and set(was) != self.include:
            log.warning(
                "kaggle_sync.include changed %s -> %s: the payload already published still "
                "holds the old selection. Delete %s and the Kaggle version to rebuild it.",
                sorted(was), sorted(self.include), self.sync_dir)
        self._state["include"] = sorted(self.include)

    # -- state ------------------------------------------------------------
    def _load_state(self) -> dict:
        p = self.sync_dir / _STATE
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception as e:
                log.warning("unreadable %s (%s) — starting a fresh shard state", p, e)
        return {"open": 1, "assigned": []}

    def _save_state(self) -> None:
        self._state["assigned"] = sorted(self._assigned)
        self._state["pending"] = self._pending
        tmp = self.sync_dir / (_STATE + ".tmp")
        tmp.write_text(json.dumps(self._state), encoding="utf-8")
        tmp.replace(self.sync_dir / _STATE)   # atomic: never leave a truncated state file

    def _shard_handle(self, index: int) -> str:
        return self.handle if self.single else f"{self.handle}-{index:03d}"

    def _zip_path(self, index: int) -> Path:
        name = "vivos-fake-data.zip" if self.single else f"shard-{index:03d}.zip"
        return self.sync_dir / name

    # -- auth -------------------------------------------------------------
    def preflight(self) -> None:
        try:
            import kagglehub
            who = kagglehub.whoami()
        except Exception as e:
            raise SystemExit(
                f"Kaggle auth failed ({e}).\n"
                "Set KAGGLE_USERNAME + KAGGLE_KEY, or put a token at ~/.kaggle/kaggle.json.\n"
                "On a Kaggle notebook: Add-ons -> Secrets, then expose them as those env vars.\n"
                "Or disable syncing with kaggle_sync.enabled: false / --no-kaggle-sync."
            )
        log.info("Kaggle sync -> %s (auth ok: %s) | uploading: %s + metadata.csv%s",
                 self.handle if self.single else f"{self.handle}-NNN",
                 who.get("username", "?"), sorted(self.include) or "nothing",
                 f" | excluding {self.exclude}" if self.exclude else "")

    # -- what is safe to upload -------------------------------------------
    def _committed_files(self) -> list[str]:
        """What goes into the payload: audio that metadata.csv says is done and that the
        `include` selection asks for. Nothing else is ever uploaded — a metadata row is
        written only after the audio validates, so a half-written wav cannot leak out.

        metadata.csv itself always rides along outside the zip; it IS the resume state.
        """
        rels: list[str] = []
        if self.meta_path.exists() and self.include & {"real", "fake"}:
            with open(self.meta_path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    rel = (row.get("audio_path") or "").strip()
                    if rel and rel.split("/", 1)[0] in self.include:
                        rels.append(rel)
        if "reference" in self.include:     # not in metadata, but needed to reproduce fakes
            rels += [p.relative_to(self.out).as_posix()
                     for p in sorted((self.out / "reference").glob("*.wav"))]
        if self.exclude:
            rels = [r for r in rels if not any(fnmatch(r, pat) for pat in self.exclude)]
        return [r for r in rels if (self.out / r).is_file()]

    # -- push -------------------------------------------------------------
    def _due(self) -> bool:
        if self.every_clips and self._since >= self.every_clips:
            return True
        if self.every_minutes and (time.monotonic() - self._last_push) >= self.every_minutes * 60:
            return self._since > 0
        return False

    def tick(self, n: int = 1) -> None:
        self._since += n
        if not self._due():
            return
        if self._thread is not None and self._thread.is_alive():
            return                      # a push is still uploading — coalesce, try again later
        self._since = 0
        self._last_push = time.monotonic()
        self._thread = threading.Thread(target=self._push_guarded, name="kaggle-sync", daemon=True)
        self._thread.start()

    def _push_guarded(self) -> None:
        try:
            self.push()
        except Exception as e:                      # never let a sync failure kill generation
            log.warning("Kaggle checkpoint failed (will retry at next checkpoint): %s", e)

    def push(self, final: bool = False) -> None:
        with self._lock:
            index = int(self._state.get("open", 1))
            zpath = self._zip_path(index)
            new = [r for r in self._committed_files() if r not in self._assigned]
            if not new and not self._pending:
                return          # nothing added since the last successful upload

            added = 0
            with zipfile.ZipFile(zpath, "a", compression=zipfile.ZIP_STORED) as z:
                for rel in new:
                    z.write(self.out / rel, arcname=rel)
                    added += 1
            self._assigned.update(new)
            self._pending = True
            self._save_state()

            up = self.sync_dir / _UPLOAD_DIR
            shutil.rmtree(up, ignore_errors=True)
            up.mkdir(parents=True)
            try:
                _link_or_copy(zpath, up / zpath.name)
                # loose (not zipped): files that keep changing, so every push refreshes them
                if self.meta_path.exists():         # full CSV -> the payload is self-describing
                    shutil.copy2(self.meta_path, up / "metadata.csv")
                if "logs" in self.include:
                    for lg in sorted((self.out / "logs").glob("*.log")):
                        (up / "logs").mkdir(exist_ok=True)
                        shutil.copy2(lg, up / "logs" / lg.name)
                size_mb = zpath.stat().st_size / 1e6
                log.info("Kaggle checkpoint: +%d files -> %s (%.0f MB)%s",
                         added, self._shard_handle(index), size_mb, " [final]" if final else "")
                import kagglehub
                kagglehub.dataset_upload(
                    self._shard_handle(index), str(up),
                    version_notes=f"{len(self._assigned)} files, +{added} new",
                )
            finally:
                shutil.rmtree(up, ignore_errors=True)
            self._pending = False
            self._save_state()

            if not self.single and zpath.stat().st_size / 1e6 >= self.shard_mb and not final:
                self._state["open"] = index + 1     # seal: this shard is never re-uploaded
                self._save_state()
                zpath.unlink()                      # ...so the local copy is dead weight
                log.info("shard %03d sealed (>= %.0f MB) — next checkpoint opens %03d",
                         index, self.shard_mb, index + 1)

    def flush(self) -> None:
        t = self._thread
        if t is not None and t.is_alive():
            log.info("waiting for the in-flight Kaggle checkpoint ...")
            t.join()
        try:
            self.push(final=True)
        except Exception as e:
            log.error("final Kaggle push failed — local dataset is intact at %s: %s", self.out, e)

    # -- pull -------------------------------------------------------------
    def pull(self) -> None:
        """Restore state from Kaggle: extract every shard into output_root, merge metadata.

        Existing local files win (they are already validated), so pulling on top of a
        partially-populated output dir is safe.
        """
        if not self.pull_on_start:
            return
        import kagglehub

        index, restored = 1, 0
        while True:
            h = self._shard_handle(index)
            tmp = Path(tempfile.mkdtemp(prefix=f"kgl-{index:03d}-", dir=str(self.sync_dir)))
            try:
                path = Path(kagglehub.dataset_download(h, output_dir=str(tmp)))
            except Exception as e:
                shutil.rmtree(tmp, ignore_errors=True)
                if _is_missing(e):
                    break
                raise SystemExit(f"Kaggle pull failed on {h}: {e}")
            try:
                for csv_file in path.rglob("metadata.csv"):
                    self._merge_metadata(csv_file)
                zips = sorted(path.rglob("*.zip"))   # our payload, or a zip from an older flow
                keep = self._zip_path(index)
                for zf in zips:
                    restored += self._extract(zf, ours=(zf.name == keep.name))
                # keep the payload we append to (ours if present, else start from theirs)
                mine = [z for z in zips if z.name == keep.name]
                if mine and not keep.exists():
                    shutil.copy2(mine[0], keep)
                prev = self._zip_path(index - 1)
                if not self.single and prev.exists():
                    prev.unlink()
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
            if self.single:
                index += 1
                break                               # one dataset holds everything
            index += 1

        if index == 1:
            log.info("Kaggle: nothing at %s yet — a new dataset will be created on the first push",
                     self._shard_handle(1))
            return
        self._state["open"] = index - 1             # append to the newest shard; it seals on size
        self._save_state()
        log.info("Kaggle pull: %d dataset(s), %d file(s) restored, %d tracked",
                 index - 1, restored, len(self._assigned))

    def _extract(self, zf: Path, ours: bool) -> int:
        """Unpack a payload zip into output_root.

        `ours` = this is the zip we append to, so its entries are already published and
        must not be packed again. A foreign zip (e.g. one from an older manual flow) is
        deliberately left unassigned: the next push re-packs those files into our payload,
        otherwise the new dataset version would silently drop them.
        """
        n = 0
        meta_rel = self.meta_path.relative_to(self.out).as_posix()
        with zipfile.ZipFile(zf) as z:
            names = [x for x in z.namelist() if not x.endswith("/")]
            strip = _common_prefix(names)   # tolerates a zip made with `zip -r x.zip dataset`
            for name in names:
                rel = name[len(strip):] if strip else name
                if not rel or rel.startswith(".sync/"):
                    continue
                if rel == meta_rel:                 # merge rows, never overwrite the CSV
                    tmpf = self.sync_dir / "_meta_in.csv"
                    with z.open(name) as src, open(tmpf, "wb") as out:
                        shutil.copyfileobj(src, out)
                    self._merge_metadata(tmpf)
                    tmpf.unlink()
                    continue
                if ours:
                    self._assigned.add(rel)
                dst = self.out / rel
                if dst.exists() and dst.stat().st_size > 0:
                    continue                        # local copy already there — keep it
                dst.parent.mkdir(parents=True, exist_ok=True)
                with z.open(name) as src, open(dst, "wb") as out:
                    shutil.copyfileobj(src, out)
                n += 1
        self._save_state()
        return n

    def _merge_metadata(self, remote_csv: Path) -> None:
        """Union remote rows into the local metadata.csv (dedup on audio_path)."""
        seen: set[str] = set()
        if self.meta_path.exists():
            with open(self.meta_path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if row.get("audio_path"):
                        seen.add(row["audio_path"])
        else:
            self.meta_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.meta_path, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=FIELDS).writeheader()

        added = 0
        with open(remote_csv, newline="", encoding="utf-8") as f, \
                open(self.meta_path, "a", newline="", encoding="utf-8") as out:
            w = csv.DictWriter(out, fieldnames=FIELDS)
            for row in csv.DictReader(f):
                rel = (row.get("audio_path") or "").strip()
                if not rel or rel in seen:
                    continue
                w.writerow({k: row.get(k, "") for k in FIELDS})
                seen.add(rel)
                added += 1
        if added:
            log.info("metadata merged from Kaggle: +%d rows (total %d)", added, len(seen))


_ROOTS = {"real", "fake", "reference", "metadata", "logs"}
_PARTS = {"real", "fake", "reference", "logs"}      # what `include` may name
_DEFAULT_INCLUDE = {"real", "fake", "reference"}


def _parse_include(cfg: dict) -> set[str]:
    """`include: [fake, reference]` -> what lands in the Kaggle payload.

    metadata.csv is not listed: it is the resume state and always travels with the
    payload. `include_real: false` is still honoured as a shorthand for dropping 'real'.
    """
    raw = cfg.get("include")
    if raw is None:
        inc = set(_DEFAULT_INCLUDE)
    else:
        inc = {str(x).strip().strip("/").lower() for x in raw}
        inc.discard("metadata")                     # always sent; listing it is harmless
        unknown = inc - _PARTS
        if unknown:
            raise SystemExit(
                f"kaggle_sync.include has unknown entries {sorted(unknown)}; "
                f"valid: {sorted(_PARTS)} (metadata.csv is always included)"
            )
    if cfg.get("include_real") is False:
        inc.discard("real")
    if not inc:
        log.warning("kaggle_sync.include is empty — only metadata.csv will be uploaded")
    return inc


def _common_prefix(names: list[str]) -> str:
    """'dataset/' for a zip built as `zip -r dataset.zip dataset`, '' for ours.

    Lets a payload from an older/manual flow be restored into output_root unchanged.
    """
    tops = {n.split("/", 1)[0] for n in names if "/" in n}
    if len(tops) == 1:
        top = tops.pop()
        if top not in _ROOTS:
            return top + "/"
    return ""


def _link_or_copy(src: Path, dst: Path) -> None:
    """Hardlink the shard into the upload dir (a multi-GB copy per checkpoint is
    pointless); fall back to copying across filesystems."""
    try:
        dst.hardlink_to(src)
    except Exception:
        shutil.copy2(src, dst)


def _is_missing(e: Exception) -> bool:
    """True for 'this shard does not exist yet' — anything else (401/403/network) is fatal."""
    s = str(e).lower()
    return "404" in s or "not found" in s or "does not exist" in s


def make_sync(cfg: dict | None, output_root: str | Path, metadata_path: str | Path):
    if not cfg or not cfg.get("enabled") or not cfg.get("handle"):
        return NullSync()
    return KaggleSync(cfg, output_root, metadata_path)
