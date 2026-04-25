#!/usr/bin/env python3

import argparse
import json
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib import request

from capture_mira_lib import CaptureConfig, capture_raw_frame, connect_ssh
from mira_capture_app import AppConfig, start_remote_preview, stop_remote_preview


INDEX_HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Mira 220 Capture</title>
  <style>
    :root {
      color-scheme: light;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f4f6f7;
      color: #1d2a31;
    }
    body {
      margin: 0;
    }
    header {
      display: flex;
      gap: 10px;
      align-items: end;
      padding: 14px;
      background: #edf2f3;
      border-bottom: 1px solid #c9d2d6;
      flex-wrap: wrap;
    }
    label {
      display: grid;
      gap: 4px;
      font-size: 12px;
      font-weight: 600;
    }
    input {
      width: 150px;
      padding: 7px 8px;
      font-size: 14px;
      border: 1px solid #aebbc0;
      border-radius: 4px;
      background: white;
    }
    button {
      height: 36px;
      padding: 0 14px;
      border: 0;
      border-radius: 4px;
      color: white;
      font-size: 14px;
      font-weight: 700;
      cursor: pointer;
    }
    button:disabled {
      opacity: .55;
      cursor: wait;
    }
    #start { background: #1f7a4d; }
    #capture { background: #255f85; }
    #stop { background: #6d3640; }
    main {
      padding: 14px;
      display: grid;
      gap: 10px;
    }
    #previewWrap {
      min-height: 560px;
      background: #101820;
      border: 1px solid #27343b;
      display: grid;
      place-items: center;
      overflow: hidden;
    }
    #preview {
      display: none;
      max-width: 100%;
      max-height: calc(100vh - 210px);
      object-fit: contain;
    }
    #placeholder {
      color: #d9e3e8;
      font-size: 20px;
      text-align: center;
      padding: 24px;
    }
    #log {
      height: 132px;
      overflow: auto;
      white-space: pre-wrap;
      background: white;
      border: 1px solid #c9d2d6;
      padding: 10px;
      font: 12px Menlo, monospace;
    }
  </style>
</head>
<body>
  <header>
    <label>Pi host <input id="host" value="192.168.1.239"></label>
    <label>User <input id="user" value="pi"></label>
    <label>Password <input id="password" value="pi" type="password"></label>
    <button id="start">Start Preview</button>
    <button id="capture">Capture Raw</button>
    <button id="stop">Stop Preview</button>
  </header>
  <main>
    <div id="previewWrap">
      <img id="preview" alt="Mira camera preview">
      <div id="placeholder">Click Start Preview to connect to the Mira camera.</div>
    </div>
    <div id="log">Ready.</div>
  </main>
<script>
const logEl = document.getElementById("log");
const preview = document.getElementById("preview");
const placeholder = document.getElementById("placeholder");
const buttons = [...document.querySelectorAll("button")];

function log(message) {
  const ts = new Date().toLocaleTimeString();
  logEl.textContent += `\\n${ts}  ${message}`;
  logEl.scrollTop = logEl.scrollHeight;
}

function config() {
  return {
    host: document.getElementById("host").value.trim(),
    user: document.getElementById("user").value.trim(),
    password: document.getElementById("password").value
  };
}

async function post(path) {
  buttons.forEach(button => button.disabled = true);
  try {
    const response = await fetch(path, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(config())
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || response.statusText);
    log(payload.message);
    return payload;
  } catch (error) {
    log(`ERROR: ${error.message}`);
  } finally {
    buttons.forEach(button => button.disabled = false);
  }
}

document.getElementById("start").onclick = async () => {
  const payload = await post("/api/start-preview");
  if (payload && payload.ok) {
    preview.src = `/preview.mjpg?cacheBust=${Date.now()}`;
    preview.style.display = "block";
    placeholder.style.display = "none";
    log("Preview stream attached.");
  }
};

document.getElementById("stop").onclick = async () => {
  await post("/api/stop-preview");
  preview.removeAttribute("src");
  preview.style.display = "none";
  placeholder.style.display = "block";
  placeholder.textContent = "Preview stopped.";
};

document.getElementById("capture").onclick = async () => {
  preview.removeAttribute("src");
  preview.style.display = "none";
  placeholder.style.display = "block";
  placeholder.textContent = "Capturing raw frame...";
  const payload = await post("/api/capture");
  if (payload && payload.ok) {
    preview.src = `/preview.mjpg?cacheBust=${Date.now()}`;
    preview.style.display = "block";
    placeholder.style.display = "none";
  }
};
</script>
</body>
</html>
"""


class MiraWebState:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.ssh = None
        self.lock = threading.Lock()

    def connect(self, cfg: AppConfig):
        if self.ssh is not None:
            self.ssh.close()
        self.ssh = connect_ssh(cfg.host, cfg.user, cfg.password)
        self.cfg = cfg

    def ensure_connected(self, cfg: AppConfig):
        if self.ssh is None or self.cfg.host != cfg.host or self.cfg.user != cfg.user:
            self.connect(cfg)
        else:
            self.cfg = cfg

    def close(self):
        if self.ssh is not None:
            try:
                stop_remote_preview(self.ssh)
            finally:
                self.ssh.close()
                self.ssh = None


def app_config_from_payload(payload: dict, current: AppConfig) -> AppConfig:
    return AppConfig(
        host=payload.get("host") or current.host,
        user=payload.get("user") or current.user,
        password=payload.get("password") or current.password,
        preview_port=current.preview_port,
        preview_width=current.preview_width,
        preview_height=current.preview_height,
        capture_width=current.capture_width,
        capture_height=current.capture_height,
        capture_ms=current.capture_ms,
        bit_depth=current.bit_depth,
        remote_dir=current.remote_dir,
        local_root=current.local_root,
    )


def make_handler(state: MiraWebState):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            return

        def do_GET(self):
            if self.path == "/" or self.path.startswith("/?"):
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(INDEX_HTML.encode("utf-8"))
                return

            if self.path.startswith("/preview.mjpg"):
                self.proxy_preview()
                return

            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length) or b"{}")
                cfg = app_config_from_payload(payload, state.cfg)

                if self.path == "/api/start-preview":
                    with state.lock:
                        state.ensure_connected(cfg)
                        pid = start_remote_preview(state.ssh, cfg)
                    self.reply({"ok": True, "message": f"Preview server started on Pi, pid {pid}."})
                    return

                if self.path == "/api/stop-preview":
                    with state.lock:
                        state.ensure_connected(cfg)
                        stop_remote_preview(state.ssh)
                    self.reply({"ok": True, "message": "Preview stopped."})
                    return

                if self.path == "/api/capture":
                    with state.lock:
                        state.ensure_connected(cfg)
                        stop_remote_preview(state.ssh)
                        script_dir = Path(__file__).resolve().parent
                        capture_dir = capture_raw_frame(
                            CaptureConfig(
                                host=cfg.host,
                                user=cfg.user,
                                password=cfg.password,
                                width=cfg.capture_width,
                                height=cfg.capture_height,
                                capture_ms=cfg.capture_ms,
                                bit_depth=cfg.bit_depth,
                                remote_dir=cfg.remote_dir,
                                local_root=script_dir / cfg.local_root,
                            ),
                            ssh=state.ssh,
                        )
                        pid = start_remote_preview(state.ssh, cfg)
                    self.reply({
                        "ok": True,
                        "message": f"Capture saved to {capture_dir}. Preview restarted, pid {pid}.",
                    })
                    return

                self.send_error(HTTPStatus.NOT_FOUND)
            except Exception as exc:
                self.reply({"ok": False, "error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

        def proxy_preview(self):
            cfg = state.cfg
            upstream_url = f"http://{cfg.host}:{cfg.preview_port}/stream.mjpg"
            try:
                upstream = request.urlopen(upstream_url, timeout=10)
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                while True:
                    chunk = upstream.read(8192)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
            except Exception:
                return

        def reply(self, payload: dict, status=HTTPStatus.OK):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def main():
    parser = argparse.ArgumentParser(description="Browser-based Mac control app for Mira 220 capture.")
    parser.add_argument("--host", default="192.168.1.239")
    parser.add_argument("--user", default="pi")
    parser.add_argument("--password", default="pi")
    parser.add_argument("--port", type=int, default=8765, help="Local Mac web app port.")
    parser.add_argument("--preview-port", type=int, default=8081, help="Raspberry Pi preview stream port.")
    args = parser.parse_args()

    state = MiraWebState(
        AppConfig(
            host=args.host,
            user=args.user,
            password=args.password,
            preview_port=args.preview_port,
        )
    )
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(state))
    print(f"Open http://127.0.0.1:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        state.close()
        server.server_close()


if __name__ == "__main__":
    main()
