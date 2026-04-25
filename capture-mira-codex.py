#!/usr/bin/env python3

import argparse
from pathlib import Path

from capture_mira_lib import CaptureConfig, capture_raw_frame


def log(message: str = "") -> None:
    print(message, flush=True)


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
    capture_dir = capture_raw_frame(
        CaptureConfig(
            host=args.host,
            user=args.user,
            password=args.password,
            width=args.width,
            height=args.height,
            capture_ms=args.capture_ms,
            bit_depth=args.bit_depth,
            remote_dir=args.remote_dir,
            local_root=script_dir / args.local_root,
            stop_camera_processes=args.stop_camera_processes,
        ),
        log=log,
    )

    log()
    log("Capture complete.")
    log(f"Capture folder: {capture_dir}")
    log(f"Raw:            {capture_dir / 'image.raw'}")
    log(f"Zero-extended:  {capture_dir / 'image12in16.raw'}")
    log(f"Scaled 16-bit:  {capture_dir / 'image_scaled_to_16bit.raw'}")


if __name__ == "__main__":
    main_cli()
