#!/usr/bin/env python3

import argparse
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import paramiko


def quote(value: str) -> str:
    return shlex.quote(value)


def log(message: str = "") -> None:
    print(message, flush=True)


def run_remote_command(ssh, command: str) -> str:
    stdin, stdout, stderr = ssh.exec_command(command)
    exit_code = stdout.channel.recv_exit_status()
    out = stdout.read().decode("utf-8", errors="replace").strip()
    err = stderr.read().decode("utf-8", errors="replace").strip()
    if out:
        log(out)
    if err:
        log(err)
    if exit_code != 0:
        raise RuntimeError(f"Remote command failed with exit code {exit_code}: {command}")
    return out


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


def connect_ssh(args):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(
        args.host,
        username=args.user,
        password=args.password,
        timeout=15,
        look_for_keys=False,
        allow_agent=False,
    )
    return ssh


def download_file(ssh, remote_path: str, local_path: Path) -> None:
    with ssh.open_sftp() as sftp:
        sftp.get(remote_path, str(local_path))


def main_cli() -> None:
    parser = argparse.ArgumentParser(description="Capture a raw frame from a Raspberry Pi and convert it locally.")
    parser.add_argument("--host", default="192.168.1.239", help="Raspberry Pi IP or hostname.")
    parser.add_argument("--user", default="pi", help="SSH username.")
    parser.add_argument("--password", default="pi", help="SSH password.")
    parser.add_argument("--width", type=int, default=1600, help="Image width in pixels.")
    parser.add_argument("--height", type=int, default=1400, help="Image height in pixels.")
    parser.add_argument("--capture-ms", type=int, default=2000, help="Capture duration in milliseconds.")
    parser.add_argument("--bit-depth", type=int, default=12, help="Raw sensor bit depth.")
    parser.add_argument("--remote-dir", default="/tmp/mira-capture", help="Remote directory for temporary raw files.")
    parser.add_argument("--local-root", default="captures", help="Local directory where captures are stored.")
    parser.add_argument(
        "--stop-camera-processes",
        action="store_true",
        help="Stop known camera-holding processes on the Pi before capture.",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    converter_path = script_dir / "raw-codex.py"
    if not converter_path.exists():
        raise FileNotFoundError(f"Converter script not found: {converter_path}")

    local_root = script_dir / args.local_root
    local_root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    capture_dir = local_root / timestamp
    capture_dir.mkdir(parents=True, exist_ok=True)

    remote_pattern = f"{args.remote_dir}/frame_{timestamp}_%04d.raw"
    remote_cleanup_pattern = f"{quote(args.remote_dir)}/frame_{timestamp}_*.raw"
    local_raw = capture_dir / "image.raw"
    local_zero_extended = capture_dir / "image12in16.raw"
    local_scaled = capture_dir / "image_scaled_to_16bit.raw"

    log(f"Connecting to {args.user}@{args.host} ...")
    ssh = connect_ssh(args)
    try:
        if args.stop_camera_processes:
            log("Stopping known camera-holding processes on Raspberry Pi ...")
            run_remote_command(
                ssh,
                "pkill -f '/home/pi/ams_rpi_software/common/app_full_ams.py' || true",
            )

        capture_cmd = (
            f"mkdir -p {quote(args.remote_dir)} && "
            f"rm -f {remote_cleanup_pattern} && "
            f"libcamera-raw --rawfull --mode {args.width}:{args.height}:{args.bit_depth}:P "
            f"--frames 1 -t {args.capture_ms} -n -o {quote(remote_pattern)}"
        )
        log("Capturing raw frame on Raspberry Pi ...")
        run_remote_command(ssh, capture_cmd)

        remote_raw = run_remote_command(
            ssh,
            f"ls -1 {quote(args.remote_dir)}/frame_{timestamp}_*.raw | sort | tail -n 1",
        )
        if not remote_raw:
            raise RuntimeError("Capture completed, but no raw frame was found on the Raspberry Pi.")

        log(f"Downloading {remote_raw} ...")
        download_file(ssh, remote_raw, local_raw)

        log("Converting packed 12-bit raw to 16-bit outputs ...")
        convert_raw(
            converter_path=converter_path,
            input_path=local_raw,
            width=args.width,
            height=args.height,
            zero_extended_path=local_zero_extended,
            left_shifted_path=local_scaled,
        )

        log()
        log("Capture complete.")
        log(f"Capture folder: {capture_dir}")
        log(f"Raw:            {local_raw}")
        log(f"Zero-extended:  {local_zero_extended}")
        log(f"Scaled 16-bit:  {local_scaled}")
    finally:
        ssh.close()


if __name__ == "__main__":
    main_cli()
