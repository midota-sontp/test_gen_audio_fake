#!/usr/bin/env python3
"""Dashboard HTML theo dõi tiến độ dựng corpus.

    python scripts/serve_monitor.py --out /data/out --port 8000
    # rồi mở http://<máy-docker>:8000

Server chỉ ĐỌC: sqlite mở ở chế độ read-only, không đụng vào tiến độ của job.
"""
from __future__ import annotations

import argparse
import json
import sys
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from corpus.dashboard import build                       # noqa: E402

PAGE = ROOT / "dashboard" / "index.html"


class Handler(BaseHTTPRequestHandler):
    def __init__(self, *args, out: Path, **kw):
        self.out = out
        super().__init__(*args, **kw)

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            if not PAGE.exists():
                self._send(500, b"thieu dashboard/index.html", "text/plain")
                return
            self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        elif path == "/api/progress":
            try:
                payload = build(self.out)
            except Exception as e:                       # dashboard không được làm sập job
                payload = {"ready": False, "message": f"{type(e).__name__}: {e}"}
            self._send(200, json.dumps(payload, ensure_ascii=False, default=str)
                       .encode("utf-8"), "application/json; charset=utf-8")
        else:
            self._send(404, b"not found", "text/plain")

    def log_message(self, fmt, *args) -> None:
        pass                                             # khỏi spam log mỗi 5 giây


def main() -> int:
    p = argparse.ArgumentParser(description="Dashboard theo dõi tiến độ")
    p.add_argument("--out", default="/data/out")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    a = p.parse_args()
    srv = ThreadingHTTPServer((a.host, a.port),
                              partial(Handler, out=Path(a.out)))
    print(f"Dashboard: http://{a.host}:{a.port}  (đọc {a.out})", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nĐã dừng dashboard.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
