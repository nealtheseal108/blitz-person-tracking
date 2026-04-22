"""
Embedded dashboard server.

Stdlib-only HTTP server (no Flask, no FastAPI) that the engine pushes
into. Runs in a daemon thread so the main per-frame loop stays
single-threaded and predictable.

Endpoints
─────────
GET  /                  -> dashboard.html (the SPA)
GET  /stream.mjpg       -> multipart MJPEG stream of the latest frame
GET  /api/summary       -> JSON funnel summary (current snapshot)
GET  /api/events        -> JSON list of last N events
GET  /api/tracks        -> JSON per-track rows (for the table)
POST /api/click         -> increment clicked counter (for testing)

Usage
─────
    dash = Dashboard(funnel)
    dash.start(host="0.0.0.0", port=8000)
    ...
    dash.publish_frame(annotated_bgr)         # after each frame
    dash.publish_event({"event": "click"})    # for ticker

The server holds the latest frame as JPEG bytes in memory. MJPEG
clients receive every new frame as it's published; if the engine is
faster than the browser, browsers naturally drop frames. There is no
per-client queue, so memory is bounded.
"""
from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Deque, Optional

import cv2
import numpy as np


_DASHBOARD_HTML_PATH = os.path.join(os.path.dirname(__file__), "dashboard.html")


class Dashboard:
    def __init__(self, funnel=None, max_events: int = 200, jpeg_quality: int = 80):
        self.funnel = funnel  # FunnelTracker or None; used for /api/* snapshots
        self.jpeg_quality = jpeg_quality

        # Latest annotated frame as JPEG bytes + a condition var so the
        # MJPEG endpoint can wake on each new frame instead of polling.
        self._frame_jpeg: Optional[bytes] = None
        self._frame_index: int = 0
        self._frame_cv = threading.Condition()

        # Recent events (for the ticker)
        self._events: Deque[dict] = deque(maxlen=max_events)
        self._events_lock = threading.Lock()

        # Optional click sink (e.g. funnel.note_click) for the test button
        self._on_click: Optional[Callable[[str], None]] = None

        self._httpd: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._started_at = time.time()

    # ── Public API the engine calls ─────────────────────────────────────────

    def publish_frame(self, frame_bgr: np.ndarray) -> None:
        ok, buf = cv2.imencode(".jpg", frame_bgr,
                               [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        if not ok:
            return
        with self._frame_cv:
            self._frame_jpeg = buf.tobytes()
            self._frame_index += 1
            self._frame_cv.notify_all()

    def publish_event(self, event: dict) -> None:
        with self._events_lock:
            self._events.append(event)

    def set_click_handler(self, fn: Callable[[str], None]) -> None:
        self._on_click = fn

    def event_sink(self) -> Callable[[dict], None]:
        """Convenience: returns a callable suitable for FunnelTracker(event_sink=...)."""
        return self.publish_event

    # ── Server lifecycle ────────────────────────────────────────────────────

    def start(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        handler_cls = _make_handler(self)
        self._httpd = ThreadingHTTPServer((host, port), handler_cls)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        print(f"[dashboard] http://{host}:{port}/")

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None

    # ── Internal: snapshot accessors used by the handler ────────────────────

    def _snapshot_summary(self) -> dict:
        if self.funnel is None:
            return {
                "passed_by": 0, "looked_at": 0, "approached": 0,
                "clicked": 0, "purchased": 0,
                "look_rate": 0.0, "approach_rate": 0.0,
                "click_rate": 0.0, "purchase_rate": 0.0,
                "overall_conversion": 0.0,
            }
        s = dict(self.funnel.summary())
        s["uptime_sec"] = round(time.time() - self._started_at, 1)
        return s

    def _snapshot_events(self, limit: int = 50) -> list:
        with self._events_lock:
            return list(self._events)[-limit:]

    def _snapshot_tracks(self) -> list:
        if self.funnel is None:
            return []
        return self.funnel.per_track_rows()

    def _await_frame(self, last_index: int, timeout: float = 5.0) -> Optional[bytes]:
        with self._frame_cv:
            if self._frame_index == last_index:
                self._frame_cv.wait(timeout=timeout)
            if self._frame_jpeg is None:
                return None
            return self._frame_jpeg

    def _current_frame_index(self) -> int:
        with self._frame_cv:
            return self._frame_index

    def _trigger_click(self, source: str) -> None:
        if self._on_click is not None:
            self._on_click(source)


def _make_handler(dashboard: Dashboard):
    """Build a request handler class bound to the given Dashboard instance.
    We define the class inline so it captures `dashboard` in a closure
    without resorting to globals."""

    class Handler(BaseHTTPRequestHandler):
        # Quiet default access logging -- it spams the engine console
        def log_message(self, fmt, *args):
            return

        # ── Helpers ─────────────────────────────────────────────────────────
        def _send_json(self, payload, code: int = 200) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def _send_text(self, text: str, code: int = 200, ctype: str = "text/plain") -> None:
            body = text.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype + "; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        # ── Routes ──────────────────────────────────────────────────────────
        def do_GET(self):  # noqa: N802
            try:
                if self.path in ("/", "/index.html"):
                    return self._serve_dashboard()
                if self.path == "/api/summary":
                    return self._send_json(dashboard._snapshot_summary())
                if self.path.startswith("/api/events"):
                    return self._send_json(dashboard._snapshot_events())
                if self.path == "/api/tracks":
                    return self._send_json(dashboard._snapshot_tracks())
                if self.path in ("/stream.mjpg", "/stream"):
                    return self._serve_mjpeg()
                if self.path == "/frame.jpg":
                    return self._serve_single_frame()
                self.send_error(404, "not found")
            except (BrokenPipeError, ConnectionResetError):
                # Browser tab closed mid-stream; not interesting
                pass

        def do_POST(self):  # noqa: N802
            if self.path == "/api/click":
                length = int(self.headers.get("content-length", 0) or 0)
                body = self.rfile.read(length).decode("utf-8", "ignore") if length else ""
                source = body[:64] if body else "dashboard_button"
                dashboard._trigger_click(source)
                return self._send_json({"ok": True, "source": source})
            self.send_error(404, "not found")

        # ── Dashboard HTML ──────────────────────────────────────────────────
        def _serve_dashboard(self):
            try:
                with open(_DASHBOARD_HTML_PATH, "r", encoding="utf-8") as f:
                    html = f.read()
            except FileNotFoundError:
                return self._send_text(
                    "dashboard.html not found next to dashboard_server.py",
                    code=500,
                )
            return self._send_text(html, ctype="text/html")

        # ── Single-frame fallback (for browsers that hate MJPEG) ────────────
        def _serve_single_frame(self):
            with dashboard._frame_cv:
                jpg = dashboard._frame_jpeg
            if jpg is None:
                return self._send_text("no frame yet", code=503)
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(jpg)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(jpg)

        # ── MJPEG stream (multipart/x-mixed-replace) ────────────────────────
        def _serve_mjpeg(self):
            boundary = b"--frame"
            self.send_response(200)
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Connection", "close")
            self.send_header(
                "Content-Type",
                "multipart/x-mixed-replace; boundary=frame",
            )
            self.end_headers()

            last = -1
            while True:
                jpg = dashboard._await_frame(last_index=last, timeout=5.0)
                if jpg is None:
                    # Send a heartbeat so proxies don't kill the connection
                    try:
                        self.wfile.write(b"\r\n")
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        return
                    continue
                last = dashboard._current_frame_index()
                try:
                    self.wfile.write(boundary + b"\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(jpg)}\r\n\r\n".encode())
                    self.wfile.write(jpg)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    return

    return Handler
