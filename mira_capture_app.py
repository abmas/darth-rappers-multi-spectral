#!/usr/bin/env python3

import argparse
import io
import queue
import shlex
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from urllib import request

import paramiko
from PIL import Image, ImageTk

from capture_mira_lib import CaptureConfig, capture_raw_frame, connect_ssh, run_remote_command


REMOTE_PREVIEW_SCRIPT = r"""#!/usr/bin/env python3
import argparse
import io
import logging
import socketserver
from http import server
from threading import Condition

from picamera2 import Picamera2
from picamera2.encoders import MJPEGEncoder
from picamera2.outputs import FileOutput


class StreamingOutput(io.BufferedIOBase):
    def __init__(self):
        self.frame = None
        self.condition = Condition()

    def write(self, buf):
        with self.condition:
            self.frame = bytes(buf)
            self.condition.notify_all()
        return len(buf)


class StreamingHandler(server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path not in ("/", "/stream.mjpg"):
            self.send_error(404)
            self.end_headers()
            return

        self.send_response(200)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
        self.end_headers()

        try:
            while True:
                with output.condition:
                    output.condition.wait()
                    frame = output.frame
                self.wfile.write(b"--FRAME\r\n")
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", len(frame))
                self.end_headers()
                self.wfile.write(frame)
                self.wfile.write(b"\r\n")
        except Exception as exc:
            logging.warning("Removed streaming client: %s", exc)


class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True


parser = argparse.ArgumentParser()
parser.add_argument("--host", default="0.0.0.0")
parser.add_argument("--port", type=int, default=8081)
parser.add_argument("--width", type=int, default=800)
parser.add_argument("--height", type=int, default=700)
args = parser.parse_args()

picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": (args.width, args.height)})
picam2.configure(config)
output = StreamingOutput()

picam2.start_recording(MJPEGEncoder(), FileOutput(output))
try:
    address = (args.host, args.port)
    StreamingServer(address, StreamingHandler).serve_forever()
finally:
    picam2.stop_recording()
"""


@dataclass
class AppConfig:
    host: str = "192.168.1.239"
    user: str = "pi"
    password: str = "pi"
    capture_width: int = 1600
    capture_height: int = 1400
    preview_width: int = 800
    preview_height: int = 700
    preview_port: int = 8081
    capture_ms: int = 2000
    bit_depth: int = 12
    remote_dir: str = "/tmp/mira-capture"
    local_root: str = "captures"


def quote(value: str) -> str:
    return shlex.quote(value)


def upload_preview_server(ssh: paramiko.SSHClient, remote_path: str = "/tmp/mira_preview_server.py") -> str:
    with ssh.open_sftp() as sftp:
        with sftp.file(remote_path, "wb") as remote_file:
            remote_file.write(REMOTE_PREVIEW_SCRIPT.encode("utf-8"))
        sftp.chmod(remote_path, 0o755)
    return remote_path


def start_remote_preview(ssh: paramiko.SSHClient, cfg: AppConfig) -> str:
    remote_script = upload_preview_server(ssh)
    run_remote_command(ssh, "pkill -f '[m]ira_preview_server.py' || true")
    command = (
        f"nohup python3 {quote(remote_script)} "
        f"--port {cfg.preview_port} "
        f"--width {cfg.preview_width} "
        f"--height {cfg.preview_height} "
        f"> /tmp/mira_preview.log 2>&1 < /dev/null & echo $!"
    )
    pid = run_remote_command(ssh, command).strip()
    time.sleep(2.0)
    return pid


def stop_remote_preview(ssh: paramiko.SSHClient) -> None:
    run_remote_command(ssh, "pkill -f '[m]ira_preview_server.py' || true")


class MjpegReader(threading.Thread):
    def __init__(self, url: str, frames: "queue.Queue[Image.Image]", events: "queue.Queue[str]", stop_event: threading.Event):
        super().__init__(daemon=True)
        self.url = url
        self.frames = frames
        self.events = events
        self.stop_event = stop_event

    def run(self) -> None:
        while not self.stop_event.is_set():
            try:
                with request.urlopen(self.url, timeout=8) as stream:
                    data = b""
                    self.events.put("Preview connected.")
                    while not self.stop_event.is_set():
                        chunk = stream.read(4096)
                        if not chunk:
                            raise RuntimeError("Preview stream ended.")
                        data += chunk
                        start = data.find(b"\xff\xd8")
                        end = data.find(b"\xff\xd9", start + 2)
                        if start >= 0 and end >= 0:
                            jpg = data[start : end + 2]
                            data = data[end + 2 :]
                            image = Image.open(io.BytesIO(jpg)).convert("RGB")
                            self._put_latest(image)
            except Exception as exc:
                if not self.stop_event.is_set():
                    self.events.put(f"Preview waiting: {exc}")
                    time.sleep(2.0)

    def _put_latest(self, image: Image.Image) -> None:
        try:
            while True:
                self.frames.get_nowait()
        except queue.Empty:
            pass
        self.frames.put(image)


class MiraCaptureApp(tk.Tk):
    def __init__(self, cfg: AppConfig):
        super().__init__()
        self.title("Mira 220 Capture")
        self.geometry("1080x860")
        self.minsize(860, 690)
        self.configure(bg="#f5f7f8")

        self.cfg = cfg
        self.ssh = None
        self.preview_reader = None
        self.preview_stop = threading.Event()
        self.frame_queue: "queue.Queue[Image.Image]" = queue.Queue(maxsize=2)
        self.event_queue: "queue.Queue[str]" = queue.Queue()
        self.preview_photo = None
        self.latest_frame = None
        self.busy = False

        self._build_ui()
        self.update_idletasks()
        self.after(80, self._poll_queues)
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_ui(self) -> None:
        outer = tk.Frame(self, bg="#f5f7f8", padx=14, pady=14)
        outer.pack(fill="both", expand=True)

        top = tk.Frame(outer, bg="#eef2f3", padx=12, pady=10, highlightthickness=1, highlightbackground="#c9d2d6")
        top.pack(fill="x")

        self.host_var = tk.StringVar(value=self.cfg.host)
        self.user_var = tk.StringVar(value=self.cfg.user)
        self.password_var = tk.StringVar(value=self.cfg.password)

        label_opts = {"bg": "#eef2f3", "fg": "#223037", "font": ("Helvetica", 12)}
        entry_opts = {"font": ("Helvetica", 14), "relief": "solid", "bd": 1}
        button_opts = {"font": ("Helvetica", 14), "relief": "raised", "bd": 2, "padx": 14, "pady": 6}

        tk.Label(top, text="Pi host", **label_opts).grid(row=0, column=0, sticky="w")
        tk.Entry(top, textvariable=self.host_var, width=18, **entry_opts).grid(row=1, column=0, sticky="ew", padx=(0, 10))
        tk.Label(top, text="User", **label_opts).grid(row=0, column=1, sticky="w")
        tk.Entry(top, textvariable=self.user_var, width=10, **entry_opts).grid(row=1, column=1, sticky="ew", padx=(0, 10))
        tk.Label(top, text="Password", **label_opts).grid(row=0, column=2, sticky="w")
        tk.Entry(top, textvariable=self.password_var, show="*", width=12, **entry_opts).grid(row=1, column=2, sticky="ew", padx=(0, 10))

        self.connect_button = tk.Button(
            top,
            text="Start Preview",
            command=self.start_preview,
            bg="#1f7a4d",
            fg="white",
            activebackground="#17613d",
            activeforeground="white",
            **button_opts,
        )
        self.connect_button.grid(row=1, column=3, sticky="ew", padx=(0, 10))

        self.capture_button = tk.Button(
            top,
            text="Capture Raw",
            command=self.capture,
            state="disabled",
            bg="#255f85",
            fg="white",
            activebackground="#1c4966",
            activeforeground="white",
            disabledforeground="#72808a",
            **button_opts,
        )
        self.capture_button.grid(row=1, column=4, sticky="ew")

        top.columnconfigure(0, weight=1)

        self.preview_canvas = tk.Canvas(
            outer,
            bg="#101820",
            highlightthickness=1,
            highlightbackground="#27343b",
        )
        self.preview_canvas.pack(fill="both", expand=True, pady=(14, 8))
        self.preview_canvas.bind("<Configure>", lambda _event: self._redraw_preview())
        self._draw_canvas_message("Click Start Preview to connect to the Mira camera.")

        self.log_box = tk.Text(
            outer,
            height=5,
            bg="#ffffff",
            fg="#223037",
            relief="solid",
            bd=1,
            font=("Menlo", 12),
            wrap="word",
        )
        self.log_box.pack(fill="x", pady=(0, 8))
        self.log_box.insert("end", "Ready.\n")
        self.log_box.configure(state="disabled")

        bottom = tk.Frame(outer, bg="#f5f7f8")
        bottom.pack(fill="x")
        self.status_var = tk.StringVar(value="Ready.")
        tk.Label(
            bottom,
            textvariable=self.status_var,
            bg="#f5f7f8",
            fg="#223037",
            anchor="w",
            font=("Helvetica", 13),
        ).pack(side="left", fill="x", expand=True)
        tk.Button(
            bottom,
            text="Stop Preview",
            command=self.stop_preview,
            font=("Helvetica", 13),
            padx=10,
            pady=4,
        ).pack(side="right")

    def current_config(self) -> AppConfig:
        return AppConfig(
            host=self.host_var.get().strip(),
            user=self.user_var.get().strip(),
            password=self.password_var.get(),
            capture_width=self.cfg.capture_width,
            capture_height=self.cfg.capture_height,
            preview_width=self.cfg.preview_width,
            preview_height=self.cfg.preview_height,
            preview_port=self.cfg.preview_port,
            capture_ms=self.cfg.capture_ms,
            bit_depth=self.cfg.bit_depth,
            remote_dir=self.cfg.remote_dir,
            local_root=self.cfg.local_root,
        )

    def set_status(self, message: str) -> None:
        self.status_var.set(message)
        self._append_log(message)

    def _append_log(self, message: str) -> None:
        self.log_box.configure(state="normal")
        self.log_box.insert("end", f"{time.strftime('%H:%M:%S')}  {message}\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def set_busy(self, busy: bool) -> None:
        self.busy = busy
        state = "disabled" if busy else "normal"
        self.connect_button.configure(state=state)
        self.capture_button.configure(state="disabled" if busy or self.ssh is None else "normal")

    def start_preview(self) -> None:
        if self.busy:
            return
        cfg = self.current_config()
        self._draw_canvas_message("Connecting to Raspberry Pi camera...")
        self.set_busy(True)
        threading.Thread(target=self._start_preview_worker, args=(cfg,), daemon=True).start()

    def _start_preview_worker(self, cfg: AppConfig) -> None:
        try:
            self.event_queue.put(f"Connecting to {cfg.user}@{cfg.host} ...")
            if self.ssh is not None:
                self.ssh.close()
            self.ssh = connect_ssh(cfg.host, cfg.user, cfg.password)
            pid = start_remote_preview(self.ssh, cfg)
            self.event_queue.put(f"Preview server started on Pi, pid {pid}.")

            self.preview_stop.set()
            self.preview_stop = threading.Event()
            url = f"http://{cfg.host}:{cfg.preview_port}/stream.mjpg"
            self.preview_reader = MjpegReader(url, self.frame_queue, self.event_queue, self.preview_stop)
            self.preview_reader.start()
        except Exception as exc:
            self.event_queue.put(f"Could not start preview: {exc}")
            self.ssh = None
        finally:
            self.event_queue.put("__not_busy__")

    def stop_preview(self) -> None:
        self.preview_stop.set()
        if self.ssh is not None:
            self.set_busy(True)
            threading.Thread(target=self._stop_preview_worker, daemon=True).start()
        else:
            self.capture_button.configure(state="disabled")
            self.set_status("Preview stopped.")

    def _stop_preview_worker(self) -> None:
        try:
            stop_remote_preview(self.ssh)
            self.event_queue.put("Preview stopped.")
        except Exception as exc:
            self.event_queue.put(f"Could not stop preview: {exc}")
        finally:
            self.event_queue.put("__not_busy__")

    def capture(self) -> None:
        if self.busy:
            return
        cfg = self.current_config()
        self.set_busy(True)
        threading.Thread(target=self._capture_worker, args=(cfg,), daemon=True).start()

    def _capture_worker(self, cfg: AppConfig) -> None:
        try:
            if self.ssh is None:
                self.ssh = connect_ssh(cfg.host, cfg.user, cfg.password)

            self.event_queue.put("Pausing preview for raw capture ...")
            self.preview_stop.set()
            stop_remote_preview(self.ssh)

            script_dir = Path(__file__).resolve().parent
            capture_cfg = CaptureConfig(
                host=cfg.host,
                user=cfg.user,
                password=cfg.password,
                width=cfg.capture_width,
                height=cfg.capture_height,
                capture_ms=cfg.capture_ms,
                bit_depth=cfg.bit_depth,
                remote_dir=cfg.remote_dir,
                local_root=script_dir / cfg.local_root,
            )
            capture_dir = capture_raw_frame(capture_cfg, ssh=self.ssh, log=self.event_queue.put)
            self.event_queue.put(f"Capture saved: {capture_dir}")

            self.event_queue.put("Restarting preview ...")
            start_remote_preview(self.ssh, cfg)
            self.preview_stop = threading.Event()
            url = f"http://{cfg.host}:{cfg.preview_port}/stream.mjpg"
            self.preview_reader = MjpegReader(url, self.frame_queue, self.event_queue, self.preview_stop)
            self.preview_reader.start()
        except Exception as exc:
            self.event_queue.put(f"Capture failed: {exc}")
        finally:
            self.event_queue.put("__not_busy__")

    def _poll_queues(self) -> None:
        try:
            while True:
                event = self.event_queue.get_nowait()
                if event == "__not_busy__":
                    self.set_busy(False)
                else:
                    self.set_status(event)
        except queue.Empty:
            pass

        try:
            image = self.frame_queue.get_nowait()
            self._show_frame(image)
        except queue.Empty:
            pass

        self.after(80, self._poll_queues)

    def _show_frame(self, image: Image.Image) -> None:
        self.latest_frame = image
        self._redraw_preview()

    def _redraw_preview(self) -> None:
        if self.latest_frame is None:
            return

        width = max(320, self.preview_canvas.winfo_width())
        height = max(240, self.preview_canvas.winfo_height())
        image = self.latest_frame.copy()
        image.thumbnail((width, height), Image.Resampling.LANCZOS)
        self.preview_photo = ImageTk.PhotoImage(image)

        self.preview_canvas.delete("all")
        x = width // 2
        y = height // 2
        self.preview_canvas.create_image(x, y, image=self.preview_photo, anchor="center")

    def _draw_canvas_message(self, message: str) -> None:
        self.latest_frame = None
        width = max(320, self.preview_canvas.winfo_width())
        height = max(240, self.preview_canvas.winfo_height())
        self.preview_canvas.delete("all")
        self.preview_canvas.create_rectangle(0, 0, width, height, fill="#101820", outline="")
        self.preview_canvas.create_text(
            width // 2,
            height // 2,
            text=message,
            fill="#d9e3e8",
            font=("Helvetica", 20),
            width=max(280, width - 80),
        )

    def on_close(self) -> None:
        self.preview_stop.set()
        try:
            if self.ssh is not None:
                stop_remote_preview(self.ssh)
                self.ssh.close()
        except Exception:
            pass
        self.destroy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Mac desktop preview and capture app for the Mira 220 Raspberry Pi setup.")
    parser.add_argument("--host", default="192.168.1.239")
    parser.add_argument("--user", default="pi")
    parser.add_argument("--password", default="pi")
    parser.add_argument("--preview-port", type=int, default=8081)
    args = parser.parse_args()

    app = MiraCaptureApp(
        AppConfig(
            host=args.host,
            user=args.user,
            password=args.password,
            preview_port=args.preview_port,
        )
    )
    app.mainloop()


if __name__ == "__main__":
    main()
