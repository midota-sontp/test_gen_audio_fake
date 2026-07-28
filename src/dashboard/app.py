"""Bảng điều khiển realtime cho pipeline phát hiện audio thật/giả (WavLM).

Chỉ đọc các artifact pipeline ghi ra (run_status.json, stats.json, history.csv,
metrics.json, predictions.csv, registry.jsonl, plots) nên chạy được cả native
lẫn trong Docker. Khởi động:

    streamlit run src/dashboard/app.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.monitoring.overfitting import analyze  # noqa: E402
from src.utils.config import load_config, resolve  # noqa: E402

STAGES = ["download", "preprocess", "extract", "train", "evaluate"]
STAGE_VI = {"download": "Tải dữ liệu", "preprocess": "Tiền xử lý", "extract": "Trích xuất",
            "train": "Huấn luyện", "evaluate": "Đánh giá"}
LEVEL_VI = {"EXCELLENT": "🟢 Rất tốt", "GOOD": "🟢 Tốt", "WARNING": "🟠 Cảnh báo",
            "CRITICAL": "🔴 Nghiêm trọng", "UNKNOWN": "⚪ Chưa rõ"}


def _read_json(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _read_csv(p: Path):
    try:
        return pd.read_csv(p)
    except Exception:
        return None


def _fmt_eta(sec):
    if sec is None:
        return "—"
    sec = int(sec)
    return f"{sec // 60}m{sec % 60:02d}s" if sec >= 60 else f"{sec}s"


def main() -> None:
    st.set_page_config(page_title="Bảng điều khiển WavLM Detector", page_icon="🎙️", layout="wide")
    cfg = load_config("configs/mvp.yaml")

    status_p = resolve(cfg.monitoring.status_file)
    stats_p = resolve(cfg.paths.manifest_dir) / "stats.json"
    history_p = resolve(cfg.paths.report_dir) / "history.csv"
    metrics_p = resolve(cfg.paths.report_dir) / "metrics.json"
    preds_p = resolve(cfg.paths.report_dir) / "predictions.csv"
    registry_p = resolve(cfg.tracking.registry)
    report_dir = resolve(cfg.paths.report_dir)

    st.sidebar.title("🎙️ WavLM Detector")
    auto = st.sidebar.checkbox("Tự động làm mới", value=True, key="auto_refresh")
    interval = st.sidebar.number_input("Chu kỳ (giây)", 2, 60,
                                       int(cfg.dashboard.get_path("refresh_seconds", 5)),
                                       key="refresh_interval")
    if st.sidebar.button("Làm mới ngay"):
        st.rerun()

    status = _read_json(status_p) or {}
    st.sidebar.markdown(f"**Lần chạy:** `{status.get('run_id', '—')}`")
    st.sidebar.markdown(f"**Cập nhật:** {status.get('updated_at', '—')}")
    cur = status.get("current_stage", "—")
    st.sidebar.markdown(f"**Giai đoạn hiện tại:** `{STAGE_VI.get(cur, cur)}`")

    st.title("Giám sát pipeline")

    _section_pipeline(status)
    c1, c2 = st.columns(2)
    with c1:
        _section_dataset(_read_json(stats_p))
        _section_extract(status)
    with c2:
        _section_resources(status.get("resources", {}))
        _section_early_stopping(status.get("early_stopping", {}))

    _section_training(_read_csv(history_p), status, dict(cfg.monitoring.overfitting))
    _section_test(_read_json(metrics_p), report_dir)
    _section_predictions(_read_csv(preds_p))
    _section_experiments(registry_p)

    if auto:
        time.sleep(int(interval))
        st.rerun()


def _section_pipeline(status: dict) -> None:
    st.subheader("Tiến độ pipeline")
    stages = status.get("stages", {})
    cols = st.columns(len(STAGES))
    for col, name in zip(cols, STAGES):
        s = stages.get(name, {})
        state = s.get("status", "pending")
        icon = {"done": "✅", "running": "⏳", "pending": "⬜"}.get(state, "⬜")
        col.markdown(f"**{icon} {STAGE_VI.get(name, name)}**")
        prog = s.get("progress")
        if prog is not None:
            col.progress(min(1.0, float(prog)))
        if state == "running" and s.get("eta_sec") is not None:
            col.caption(f"Còn lại {_fmt_eta(s.get('eta_sec'))}")
        elif s.get("total"):
            col.caption(f"{s.get('processed', 0)}/{s.get('total')}")


def _section_dataset(stats: dict | None) -> None:
    st.subheader("Phân bố dữ liệu")
    if not stats:
        st.info("Chưa có stats.json (hãy chạy tiền xử lý).")
        return
    per = stats.get("per_split", {})
    if per:
        df = pd.DataFrame(per).T[["real", "fake", "total"]]
        st.bar_chart(df[["real", "fake"]])
        st.dataframe(df, use_container_width=True)
    st.caption(
        f"giữ lại={stats.get('kept')} · lỗi={stats.get('errors')} · "
        f"bỏ (ngắn)={stats.get('dropped_short')} · bỏ (im lặng)={stats.get('dropped_silent')}"
    )
    durs = stats.get("durations_sec") or []
    if durs:
        st.caption(f"Thời lượng: {min(durs):.1f}–{max(durs):.1f}s (n={len(durs)})")


def _section_extract(status: dict) -> None:
    st.subheader("Trích xuất WavLM")
    s = status.get("stages", {}).get("extract", {})
    if not s:
        st.info("Chưa bắt đầu.")
        return
    st.metric("Embedding", f"{s.get('processed', 0)}/{s.get('total', '?')}")
    st.caption(f"thiết bị={s.get('detail', '')} · split={s.get('current_split', '—')} · "
               f"còn lại {_fmt_eta(s.get('eta_sec'))}")


def _section_resources(res: dict) -> None:
    st.subheader("Tài nguyên")
    if not res:
        st.info("Chưa có dữ liệu tài nguyên.")
        return
    a, b, c = st.columns(3)
    a.metric("CPU %", res.get("cpu_percent", "—"))
    b.metric("RAM %", res.get("ram_percent", "—"),
             f"{res.get('ram_used_gb','?')}/{res.get('ram_total_gb','?')} GB")
    c.metric("Đĩa %", res.get("disk_percent", "—"))
    dev = res.get("device", "cpu")
    mem = res.get("mps_mem_alloc_gb") or res.get("gpu_mem_alloc_gb")
    st.caption(f"thiết bị={dev}" + (f" · bộ nhớ tăng tốc {mem} GB" if mem else ""))


def _section_early_stopping(es: dict) -> None:
    st.subheader("Dừng sớm (Early stopping)")
    if not es:
        st.info("Chưa huấn luyện.")
        return
    a, b, c = st.columns(3)
    a.metric("Val loss tốt nhất", es.get("best_val_loss", "—"))
    b.metric("Val loss hiện tại", es.get("current_val_loss", "—"))
    c.metric("Patience còn lại", es.get("patience_left", "—"))
    st.caption(f"Epoch tốt nhất: {es.get('best_epoch', '—')} / patience {es.get('patience', '—')}")


def _section_training(history, status: dict, of_cfg: dict) -> None:
    st.subheader("Huấn luyện — đường học (learning curves)")
    if history is None or len(history) == 0:
        st.info("Chưa có history.csv.")
        return
    losses = [c for c in ["train_loss", "val_loss"] if c in history]
    if losses:
        st.caption("Loss (train vs val)")
        st.line_chart(history.set_index("epoch")[losses])
    metric_cols = [c for c in ["accuracy", "f1", "roc_auc", "eer"] if c in history]
    if metric_cols:
        st.caption("Chỉ số trên tập validation")
        st.line_chart(history.set_index("epoch")[metric_cols])

    of_res = status.get("stages", {}).get("train", {}).get("overfitting")
    if not of_res:
        of_res = analyze(history.to_dict("records"), of_cfg)
    level = of_res.get("level", "UNKNOWN")
    st.markdown(f"**Quá khớp (Overfitting): {LEVEL_VI.get(level, level)}** "
                f"(chênh lệch val−train = {of_res.get('gap')})")
    for m in of_res.get("messages", []):
        st.warning(m)


def _section_test(metrics: dict | None, report_dir: Path) -> None:
    st.subheader("Đánh giá trên tập test")
    if not metrics:
        st.info("Chưa có metrics.json (hãy chạy đánh giá).")
        return
    labels = {"accuracy": "Độ chính xác", "precision": "Precision", "recall": "Recall",
              "f1": "F1", "roc_auc": "ROC-AUC", "eer": "EER"}
    cols = st.columns(6)
    for col, k in zip(cols, ["accuracy", "precision", "recall", "f1", "roc_auc", "eer"]):
        v = metrics.get(k)
        col.metric(labels[k], f"{v:.3f}" if isinstance(v, (int, float)) else "—")
    imgs = [(p, t) for p, t in [("roc.png", "Đường ROC"), ("pr.png", "Precision-Recall"),
                                ("confusion_matrix.png", "Ma trận nhầm lẫn")]
            if (report_dir / p).exists()]
    if imgs:
        for col, (name, title) in zip(st.columns(len(imgs)), imgs):
            col.image(str(report_dir / name), caption=title)


def _section_predictions(preds) -> None:
    st.subheader("Mẫu dự đoán")
    if preds is None or len(preds) == 0:
        st.info("Chưa có predictions.csv.")
        return
    vi = {"real": "thật", "fake": "giả"}
    for _, row in preds.head(8).iterrows():
        c1, c2 = st.columns([3, 2])
        wav = resolve(row["path"])
        with c1:
            if wav.exists():
                st.audio(str(wav))
            else:
                st.caption(str(row["path"]))
        ok = row["prediction"] == row["ground_truth"]
        c2.markdown(f"{'✅' if ok else '❌'} dự đoán **{vi.get(row['prediction'], row['prediction'])}** "
                    f"(p={row['probability']}) · thực tế **{vi.get(row['ground_truth'], row['ground_truth'])}**")


def _section_experiments(registry_p: Path) -> None:
    st.subheader("Lịch sử thí nghiệm")
    try:
        rows = [json.loads(l) for l in registry_p.read_text().splitlines() if l.strip()]
    except Exception:
        rows = []
    if not rows:
        st.info("Chưa có lần chạy nào được ghi nhận.")
        return
    flat = []
    for r in rows:
        tm = r.get("test_metrics", {})
        flat.append({
            "lần chạy": r.get("run_id"), "thời điểm": r.get("timestamp"),
            "eer": tm.get("eer"), "roc_auc": tm.get("roc_auc"),
            "f1": tm.get("f1"), "accuracy": tm.get("accuracy"),
            "lr": r.get("params", {}).get("lr"),
            "hidden_dim": r.get("params", {}).get("hidden_dim"),
        })
    st.dataframe(pd.DataFrame(flat), use_container_width=True)


if __name__ == "__main__":
    main()
