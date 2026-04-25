#!/usr/bin/env python3

import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import paramiko


LogFn = Callable[[str], None]


@dataclass
class CaptureConfig:
    host: str = "192.168.1.239"
    user: str = "pi"
    password: str = "pi"
    width: int = 1600
    height: int = 1400
    capture_ms: int = 2000
    bit_depth: int = 12
    remote_dir: str = "/tmp/mira-capture"
    local_root: Path = Path("captures")
    stop_camera_processes: bool = False


def quote(value: str) -> str:
    return shlex.quote(value)


def default_log(message: str = "") -> None:
    print(message, flush=True)


def run_remote_command(ssh: paramiko.SSHClient, command: str) -> str:
    stdin, stdout, stderr = ssh.exec_command(command)
    exit_code = stdout.channel.recv_exit_status()
    out = stdout.read().decode("utf-8", errors="replace").strip()
    err = stderr.read().decode("utf-8", errors="replace").strip()
    if exit_code != 0:
        details = err or out or command
        raise RuntimeError(f"Remote command failed with exit code {exit_code}: {details}")
    return out


def connect_ssh(host: str, user: str, password: str) -> paramiko.SSHClient:
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(
        host,
        username=user,
        password=password,
        timeout=15,
        look_for_keys=False,
        allow_agent=False,
    )
    return ssh


def download_file(ssh: paramiko.SSHClient, remote_path: str, local_path: Path) -> None:
    with ssh.open_sftp() as sftp:
        sftp.get(remote_path, str(local_path))


def convert_raw(
    converter_path: Path,
    input_path: Path,
    width: int,
    height: int,
    zero_extended_path: Path,
    left_shifted_path: Path,
) -> None:
    subprocess.run(
        [
            sys.executable,
            str(converter_path),
            "--input",
            str(input_path),
            "--width",
            str(width),
            "--height",
            str(height),
            "--output-zero-extended",
            str(zero_extended_path),
            "--output-left-shifted",
            str(left_shifted_path),
        ],
        check=True,
    )


def capture_raw_frame(
    cfg: CaptureConfig,
    ssh: Optional[paramiko.SSHClient] = None,
    log: LogFn = default_log,
) -> Path:
    script_dir = Path(__file__).resolve().parent
    converter_path = script_dir / "raw-codex.py"
    if not converter_path.exists():
        raise FileNotFoundError(f"Converter script not found: {converter_path}")

    cfg.local_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    capture_dir = cfg.local_root / timestamp
    capture_dir.mkdir(parents=True, exist_ok=True)

    remote_pattern = f"{cfg.remote_dir}/frame_{timestamp}_%04d.raw"
    remote_cleanup_pattern = f"{quote(cfg.remote_dir)}/frame_{timestamp}_*.raw"
    local_raw = capture_dir / "image.raw"
    local_zero_extended = capture_dir / "image12in16.raw"
    local_scaled = capture_dir / "image_scaled_to_16bit.raw"

    owns_ssh = ssh is None
    if ssh is None:
        log(f"Connecting to {cfg.user}@{cfg.host} ...")
        ssh = connect_ssh(cfg.host, cfg.user, cfg.password)

    try:
        if cfg.stop_camera_processes:
            log("Stopping known camera-holding processes on Raspberry Pi ...")
            run_remote_command(
                ssh,
                "pkill -f '/home/pi/ams_rpi_software/common/app_full_ams.py' || true",
            )

        capture_cmd = (
            f"mkdir -p {quote(cfg.remote_dir)} && "
            f"rm -f {remote_cleanup_pattern} && "
            f"libcamera-raw --rawfull --mode {cfg.width}:{cfg.height}:{cfg.bit_depth}:P "
            f"--frames 1 -t {cfg.capture_ms} -n -o {quote(remote_pattern)}"
        )
        log("Capturing raw frame on Raspberry Pi ...")
        run_remote_command(ssh, capture_cmd)

        remote_raw = run_remote_command(
            ssh,
            f"ls -1 {quote(cfg.remote_dir)}/frame_{timestamp}_*.raw | sort | tail -n 1",
        )
        if not remote_raw:
            raise RuntimeError("Capture completed, but no raw frame was found on the Raspberry Pi.")

        log(f"Downloading {remote_raw} ...")
        download_file(ssh, remote_raw, local_raw)

        log("Converting packed 12-bit raw to 16-bit outputs ...")
        convert_raw(
            converter_path=converter_path,
            input_path=local_raw,
            width=cfg.width,
            height=cfg.height,
            zero_extended_path=local_zero_extended,
            left_shifted_path=local_scaled,
        )

        log(f"Capture folder: {capture_dir}")
        return capture_dir
    finally:
        if owns_ssh:
            ssh.close()
