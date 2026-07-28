"""Experiment tracking wrapper: TensorBoard + MLflow.

Both back-ends are optional and guarded — if a dependency is missing or a
back-end is disabled in config, tracking degrades to a no-op and the pipeline
keeps running (metrics are always in reports/history.csv regardless).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..utils.config import Config, resolve
from ..utils.logging import get_logger

log = get_logger("tracking")


class ExperimentTracker:
    def __init__(self, cfg: Config, run_id: str) -> None:
        self.run_id = run_id
        self._tb = None
        self._mlflow = None
        tcfg = cfg.get_path("tracking", {}) or {}

        tb = tcfg.get("tensorboard", {}) or {}
        if tb.get("enable", False):
            try:
                from torch.utils.tensorboard import SummaryWriter

                log_dir = resolve(tb.get("log_dir", "reports/tensorboard")) / run_id
                self._tb = SummaryWriter(log_dir=str(log_dir))
                log.info("TensorBoard logging -> %s", log_dir)
            except Exception as e:
                log.warning("TensorBoard disabled: %s", e)

        ml = tcfg.get("mlflow", {}) or {}
        if ml.get("enable", False):
            try:
                import mlflow

                uri = ml.get("tracking_uri", "sqlite:///reports/mlflow.db")
                if "://" not in uri:                       # bare path -> file store
                    p = resolve(uri); p.mkdir(parents=True, exist_ok=True)
                    uri = f"file:{p}"
                elif uri.startswith("sqlite:///"):         # ensure the db's dir exists
                    resolve(uri[len("sqlite:///"):]).parent.mkdir(parents=True, exist_ok=True)
                mlflow.set_tracking_uri(uri)

                exp_name = ml.get("experiment", "wavlm-mvp")
                if mlflow.get_experiment_by_name(exp_name) is None:
                    art = resolve(ml.get("artifact_dir", "reports/mlartifacts"))
                    art.mkdir(parents=True, exist_ok=True)
                    mlflow.create_experiment(exp_name, artifact_location=art.as_uri())
                mlflow.set_experiment(exp_name)
                mlflow.start_run(run_name=run_id)
                self._mlflow = mlflow
                log.info("MLflow logging -> %s", uri)
            except Exception as e:
                log.warning("MLflow disabled: %s", e)

    def log_params(self, params: dict[str, Any]) -> None:
        if self._mlflow:
            try:
                self._mlflow.log_params({k: str(v) for k, v in params.items()})
            except Exception:
                pass

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        if self._tb:
            for k, v in metrics.items():
                try:
                    self._tb.add_scalar(k, float(v), step)
                except Exception:
                    pass
        if self._mlflow:
            try:
                self._mlflow.log_metrics(
                    {k: float(v) for k, v in metrics.items() if v == v}, step=step  # skip NaN
                )
            except Exception:
                pass

    def log_artifacts(self, path: str | Path) -> None:
        if self._mlflow:
            try:
                p = Path(path)
                if p.is_dir():
                    self._mlflow.log_artifacts(str(p))
                elif p.exists():
                    self._mlflow.log_artifact(str(p))
            except Exception:
                pass

    def close(self) -> None:
        if self._tb:
            try:
                self._tb.flush(); self._tb.close()
            except Exception:
                pass
        if self._mlflow:
            try:
                self._mlflow.end_run()
            except Exception:
                pass
