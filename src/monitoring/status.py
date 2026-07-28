"""Live pipeline status → a JSON file the Streamlit dashboard polls.

Read-modify-write on every update so it works whether stages run in one
orchestrator process or as separate script invocations. All failures are
swallowed — monitoring must never break the pipeline.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .resources import resource_snapshot

STAGES = ["download", "preprocess", "extract", "train", "evaluate"]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def run_id_from_status(path: str | Path) -> str | None:
    try:
        return json.loads(Path(path).read_text()).get("run_id")
    except Exception:
        return None


def resolve_run_id(cfg) -> str:
    """Consistent run id across stages: config override -> status file -> timestamp."""
    rid = cfg.get("_run_id") if hasattr(cfg, "get") else None
    if rid:
        return rid
    from ..utils.config import resolve
    rid = run_id_from_status(resolve(cfg.monitoring.status_file))
    return rid or _stamp()


class RunStatus:
    def __init__(self, path: str | Path, run_id: str | None = None, reset: bool = False) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.state = {} if reset else self._load()
        if run_id:
            self.state["run_id"] = run_id
        self.state.setdefault("run_id", _stamp())
        self.state.setdefault("stages", {})
        self._starts: dict[str, float] = {}

    def _load(self) -> dict[str, Any]:
        try:
            return json.loads(self.path.read_text())
        except Exception:
            return {}

    def _write(self) -> None:
        try:
            self.state["updated_at"] = _now_iso()
            self.state["resources"] = resource_snapshot(self.path.parent)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.state, indent=2, default=str))
            os.replace(tmp, self.path)
        except Exception:
            pass

    # -- stage lifecycle --------------------------------------------------
    def start_stage(self, name: str, total: int | None = None, detail: str = "") -> None:
        self._starts[name] = time.time()
        self.state["current_stage"] = name
        self.state["stages"][name] = {
            "status": "running", "progress": 0.0, "processed": 0,
            "total": total, "detail": detail, "eta_sec": None,
        }
        self._write()

    def update_stage(self, name: str, processed: int | None = None, **extra: Any) -> None:
        st = self.state["stages"].setdefault(name, {"status": "running"})
        if processed is not None:
            st["processed"] = processed
            total = st.get("total")
            if total:
                st["progress"] = round(processed / total, 4)
                elapsed = time.time() - self._starts.get(name, time.time())
                if processed > 0:
                    st["eta_sec"] = round(elapsed / processed * (total - processed), 1)
        st.update(extra)
        self._write()

    def finish_stage(self, name: str, **extra: Any) -> None:
        st = self.state["stages"].setdefault(name, {})
        if st.get("total"):                       # show total/total, not 0/total, when done
            st["processed"] = st["total"]
        st.update({"status": "done", "progress": 1.0, "eta_sec": 0}, **extra)
        self._write()

    def set_early_stopping(self, info: dict) -> None:
        self.state["early_stopping"] = info
        self._write()

    def set(self, key: str, value: Any) -> None:
        self.state[key] = value
        self._write()
