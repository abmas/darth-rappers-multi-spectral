#!/usr/bin/env python3

import argparse
from pathlib import Path
from array import array


def unpack_packed12_little_endian(data: bytes) -> array:
    """Unpack 2x 12-bit pixels stored in 3 bytes."""
    byte_count = len(data)
    if byte_count % 3 != 0:
        raise ValueError(
            f"Packed 12-bit data length must be divisible by 3, got {byte_count} bytes."
        )

    pixels = array("H")
    for index in range(0, byte_count, 3):
        byte0 = data[index]
        byte1 = data[index + 1]
        byte2 = data[index + 2]

        first = byte0 | ((byte1 & 0x0F) << 8)
        second = (byte1 >> 4) | (byte2 << 4)

        pixels.append(first)
        pixels.append(second)

    return pixels


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a packed 12-bit raw image into 16-bit-per-pixel output. "
            "Assumes little-endian 12-bit packing: two pixels in three bytes."
        )
    )
    parser.add_argument(
        "--input",
        default="image.raw",
        help="Path to the packed 12-bit raw input file.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1600,
        help="Image width in pixels.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1400,
        help="Image height in pixels.",
    )
    parser.add_argument(
        "--output-zero-extended",
        default="image12in16.raw",
        help="Output path for 12-bit values stored in uint16.",
    )
    parser.add_argument(
        "--output-left-shifted",
        default="image_scaled_to_16bit.raw",
        help="Output path for 12-bit values shifted left by 4 bits.",
    )
    parser.add_argument(
        "--frame-index",
        type=int,
        default=-1,
        help="Frame index to extract when the input contains multiple packed frames. Defaults to the last frame.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    expected_pixels = args.width * args.height
    expected_input_bytes = expected_pixels * 12 // 8

    raw_bytes = input_path.read_bytes()
    if len(raw_bytes) == expected_input_bytes:
        frame_count = 1
        selected_frame = 0
    elif len(raw_bytes) % expected_input_bytes == 0:
        frame_count = len(raw_bytes) // expected_input_bytes
        selected_frame = args.frame_index if args.frame_index >= 0 else frame_count - 1
        if selected_frame < 0 or selected_frame >= frame_count:
            raise ValueError(
                f"Frame index {args.frame_index} is out of range for {frame_count} frames."
            )
        start = selected_frame * expected_input_bytes
        end = start + expected_input_bytes
        raw_bytes = raw_bytes[start:end]
        print(
            f"Input contains {frame_count} frames; extracting frame {selected_frame} "
            f"({expected_input_bytes} bytes)."
        )
    else:
        raise ValueError(
            "Unexpected input size for packed 12-bit data: "
            f"expected {expected_input_bytes} bytes for "
            f"{args.width}x{args.height}, got {len(raw_bytes)} bytes."
        )

    pixels12 = unpack_packed12_little_endian(raw_bytes)
    if len(pixels12) != expected_pixels:
        raise ValueError(
            f"Unexpected pixel count after unpacking: expected {expected_pixels}, got {len(pixels12)}."
        )

    zero_extended = pixels12
    left_shifted = array("H", ((value << 4) for value in pixels12))

    Path(args.output_zero_extended).write_bytes(zero_extended.tobytes())
    Path(args.output_left_shifted).write_bytes(left_shifted.tobytes())

    print(f"Input: {input_path} ({input_path.stat().st_size} bytes)")
    print(f"Frames in input: {frame_count}")
    print(f"Pixels unpacked: {len(pixels12)}")
    print(f"Zero-extended output: {args.output_zero_extended} ({len(zero_extended) * 2} bytes)")
    print(f"Left-shifted output: {args.output_left_shifted} ({len(left_shifted) * 2} bytes)")


if __name__ == "__main__":
    main()
