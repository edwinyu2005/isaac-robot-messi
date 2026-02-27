"""
Optimized CLI tool for processing Isaac Sim recordings into compact GIFs.
Uses scaling and frame rate reduction to minimize file size.
"""

import argparse
import subprocess
import os
import sys


def parse_args():
    """Parses command line arguments with visibility for optimization params."""
    parser = argparse.ArgumentParser(
        description="Messi Project Video Processor (Optimized)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--input", required=True, help="Path to input webm file")
    parser.add_argument("--output", default="milestone1_optimized.gif", help="Output name")
    parser.add_argument("--start", default="00:00:16", help="Start time HH:MM:SS")
    parser.add_argument("--end", default="00:00:20", help="End time HH:MM:SS")
    parser.add_argument("--speed", type=float, default=0.5, help="Speed factor")

    # Cropping parameters
    parser.add_argument("--crop_x", type=int, default=150, help="Crop X offset")
    parser.add_argument("--crop_y", type=int, default=100, help="Crop Y offset")
    parser.add_argument("--crop_w", type=int, default=1280, help="Crop width")
    parser.add_argument("--crop_h", type=int, default=720, help="Crop height")

    # Optimization parameters
    parser.add_argument("--target_w", type=int, default=640, help="Target width for scaling")
    parser.add_argument("--fps", type=int, default=15, help="Target frames per second")

    return parser.parse_args()


def main():
    """Executes an optimized FFmpeg pipeline for small file sizes."""
    args = parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] Input file not found: {args.input}")
        sys.exit(1)

    pts_factor = 1.0 / args.speed

    # Filter Chain:
    # 1. Crop original window
    # 2. Adjust speed
    # 3. Downscale for size (e.g., 640px width)
    # 4. Reduce FPS to 15 (standard for compact tech demos)
    # 5. Generate high-quality limited palette
    vf = (
        f"crop={args.crop_w}:{args.crop_h}:{args.crop_x}:{args.crop_y},"
        f"setpts={pts_factor}*PTS,"
        f"scale={args.target_w}:-1:flags=lanczos,"
        f"fps={args.fps},"
        f"split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse"
    )

    cmd = [
        "ffmpeg", "-y",
        "-ss", args.start,
        "-to", args.end,
        "-i", args.input,
        "-vf", vf,
        "-loop", "0",
        args.output
    ]

    print(f"[INFO] Compressing {args.input} -> {args.output}...")
    try:
        subprocess.run(cmd, check=True)
        size_mb = os.path.getsize(args.output) / (1024 * 1024)
        print(f"[SUCCESS] GIF generated: {args.output} ({size_mb:.2f} MB)")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] FFmpeg failed with exit code {e.returncode}")


if __name__ == "__main__":
    main()
