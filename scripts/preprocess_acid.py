#!/usr/bin/env python3
"""Convert ACID scene metadata to RegGS sample_data format.

Input txt format (ACID):
- first line: YouTube URL
- remaining lines:
  timestamp fx fy cx cy k1 k2 r11 r12 r13 t1 r21 r22 r23 t2 r31 r32 r33 t3

Output structure:
- <out_root>/<scene_id>/images/00000.png ...
- <out_root>/<scene_id>/intrinsics.json
- <out_root>/<scene_id>/cameras.json
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import cv2

    HAS_CV2 = True
except ModuleNotFoundError:
    HAS_CV2 = False

try:
    from PIL import Image

    HAS_PIL = True
except ModuleNotFoundError:
    HAS_PIL = False

try:
    import torch
    from roma import roma

    HAS_ROMA = True
except ModuleNotFoundError:
    HAS_ROMA = False


@dataclass
class CameraRecord:
    timestamp: int
    fx: float
    fy: float
    cx: float
    cy: float
    k1: float
    k2: float
    rotation_w2c: np.ndarray
    translation_w2c: np.ndarray


def parse_scene_txt(scene_txt: Path) -> tuple[str, list[CameraRecord]]:
    with scene_txt.open("r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if len(lines) < 2:
        raise ValueError(f"Invalid scene file: {scene_txt}")

    youtube_url = lines[0]
    records: list[CameraRecord] = []

    for idx, line in enumerate(lines[1:], start=2):
        parts = line.split()
        if len(parts) != 19:
            raise ValueError(
                f"Line {idx} has {len(parts)} fields, expected 19 in {scene_txt}"
            )

        vals = [float(x) for x in parts]
        timestamp = int(vals[0])
        fx, fy, cx, cy, k1, k2 = vals[1:7]
        mat = np.array(vals[7:], dtype=np.float64).reshape(3, 4)

        records.append(
            CameraRecord(
                timestamp=timestamp,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                k1=k1,
                k2=k2,
                rotation_w2c=mat[:, :3],
                translation_w2c=mat[:, 3],
            )
        )

    return youtube_url, records


def _fallback_rotmat_to_xyzw(rotation_c2w: np.ndarray) -> list[float]:
    m = rotation_c2w
    trace = float(np.trace(m))

    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s

    q = np.array([x, y, z, w], dtype=np.float64)
    q /= np.linalg.norm(q)
    if q[3] < 0:
        q = -q
    return q.tolist()


def rotation_w2c_to_quaternion_xyzw(rotation_w2c: np.ndarray) -> list[float]:
    rotation_c2w = rotation_w2c.T
    if HAS_ROMA:
        quat = roma.rotmat_to_unitquat(
            torch.as_tensor(rotation_c2w, dtype=torch.float32).unsqueeze(0)
        )[0].cpu().numpy()
        if quat[3] < 0:
            quat = -quat
        return quat.tolist()
    return _fallback_rotmat_to_xyzw(rotation_c2w)


def maybe_download_video(youtube_url: str, video_path: Path, yt_dlp_bin: str) -> None:
    video_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [yt_dlp_bin, "-f", "best[ext=mp4]/best", "-o", str(video_path), youtube_url]
    print("Downloading video with yt-dlp...")
    subprocess.run(cmd, check=True)


def select_records(records: list[CameraRecord], max_frames: int) -> list[CameraRecord]:
    if max_frames <= 0 or max_frames >= len(records):
        return records
    return records[:max_frames]


def save_json(path: Path, data: object) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _extract_frames_cv2(records: Iterable[CameraRecord], video_path: Path, image_dir: Path) -> list[str]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    image_dir.mkdir(parents=True, exist_ok=True)
    image_names: list[str] = []

    for i, rec in enumerate(records):
        time_ms = rec.timestamp / 1000.0
        cap.set(cv2.CAP_PROP_POS_MSEC, time_ms)
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(
                f"Failed to decode frame at timestamp={rec.timestamp} (index={i})"
            )

        image_name = f"{i:05d}.png"
        out_path = image_dir / image_name
        if not cv2.imwrite(str(out_path), frame):
            raise RuntimeError(f"Failed to write image: {out_path}")
        image_names.append(image_name)

    cap.release()
    return image_names


def _extract_frames_ffmpeg(records: Iterable[CameraRecord], video_path: Path, image_dir: Path) -> list[str]:
    image_dir.mkdir(parents=True, exist_ok=True)
    image_names: list[str] = []

    for i, rec in enumerate(records):
        time_sec = rec.timestamp / 1_000_000.0
        cmd = [
            "ffmpeg",
            "-ss",
            f"{time_sec}",
            "-i",
            str(video_path),
            "-vframes",
            "1",
            "-q:v",
            "5",
            "-f",
            "image2pipe",
            "-c:v",
            "png",
            "-",
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, check=True, timeout=10)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            raise RuntimeError(
                f"Failed to extract frame at timestamp={rec.timestamp} (index={i}): {e}"
            )

        img = Image.open(io.BytesIO(result.stdout))
        image_name = f"{i:05d}.png"
        out_path = image_dir / image_name
        img.save(str(out_path))
        image_names.append(image_name)

    return image_names


def extract_frames(records: Iterable[CameraRecord], video_path: Path, image_dir: Path) -> list[str]:
    if HAS_CV2:
        return _extract_frames_cv2(records, video_path, image_dir)
    if HAS_PIL:
        return _extract_frames_ffmpeg(records, video_path, image_dir)
    raise RuntimeError("Neither cv2 nor PIL available. Install opencv-python or Pillow.")


def build_cameras_json(
    records: list[CameraRecord],
    image_names: list[str],
    original_width: int | None = 455,
    target_width: int = 256,
    crop_offset_x: int = 0,
) -> list[dict]:
    if crop_offset_x != 0:
        print("Warning: --crop-offset-x is currently ignored.")

    fx_scale = 1.0 if original_width in (None, target_width) else original_width / target_width

    cameras = []
    for i, (rec, image_name) in enumerate(zip(records, image_names)):
        cam_trans_c2w = -rec.rotation_w2c.T @ rec.translation_w2c
        cameras.append(
            {
                "cam_id": i,
                "cam_quat": rotation_w2c_to_quaternion_xyzw(rec.rotation_w2c),
                "cam_trans": cam_trans_c2w.astype(float).tolist(),
                "cx": float(rec.cx),
                "cy": float(rec.cy),
                "fx": float(rec.fx) * fx_scale,
                "fy": float(rec.fy),
                "image_name": image_name,
                "timestamp": int(rec.timestamp),
            }
        )
    return cameras


def build_intrinsics_json(
    records: list[CameraRecord],
    original_width: int | None = 455,
    target_width: int = 256,
    crop_offset_x: int = 0,
) -> dict:
    if crop_offset_x != 0:
        print("Warning: --crop-offset-x is currently ignored.")

    first = records[0]
    fx_scale = 1.0 if original_width in (None, target_width) else original_width / target_width

    intrinsics = {
        "cx": float(first.cx),
        "cy": float(first.cy),
        "fx": float(first.fx) * fx_scale,
        "fy": float(first.fy),
    }

    arr = np.array([[r.fx, r.fy, r.cx, r.cy] for r in records], dtype=np.float64)
    if not np.allclose(arr, arr[0], atol=1e-6, rtol=0):
        print("Warning: intrinsics vary across frames; using first-frame intrinsics.")

    return intrinsics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert ACID scene to RegGS sample_data format")
    parser.add_argument(
        "--scene-txt",
        type=Path,
        required=True,
        help="Path to an ACID scene txt file (e.g., datasets/acid/test/<scene>.txt)",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("sample_data"),
        help="Output root directory (default: sample_data)",
    )
    parser.add_argument(
        "--scene-name",
        type=str,
        default=None,
        help="Output scene name (default: stem of --scene-txt)",
    )
    parser.add_argument(
        "--video-path",
        type=Path,
        default=None,
        help="Path to local source video (mp4). If not provided, use --download-video",
    )
    parser.add_argument(
        "--download-video",
        action="store_true",
        help="Download video using yt-dlp when --video-path is missing",
    )
    parser.add_argument(
        "--yt-dlp-bin",
        type=str,
        default="yt-dlp",
        help="yt-dlp binary path/name (default: yt-dlp)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=-1,
        help="Maximum number of frames to process; <=0 means all",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output scene directory",
    )
    parser.add_argument(
        "--original-width",
        type=int,
        default=455,
        help="Original frame width before 256 preprocessing; default 455 for ACID",
    )
    parser.add_argument(
        "--target-width",
        type=int,
        default=256,
        help="Target frame width after preprocessing (default: 256)",
    )
    parser.add_argument(
        "--crop-offset-x",
        type=int,
        default=0,
        help="Reserved for compatibility (currently ignored)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.scene_txt.exists():
        print(f"Error: scene txt not found: {args.scene_txt}", file=sys.stderr)
        return 1

    scene_name = args.scene_name or args.scene_txt.stem
    out_scene_dir = args.out_root / scene_name
    image_dir = out_scene_dir / "images"

    if out_scene_dir.exists() and not args.overwrite:
        print(
            f"Error: output scene directory already exists: {out_scene_dir}\n"
            "Use --overwrite to continue.",
            file=sys.stderr,
        )
        return 1

    if out_scene_dir.exists() and args.overwrite:
        shutil.rmtree(out_scene_dir)

    out_scene_dir.mkdir(parents=True, exist_ok=True)

    youtube_url, all_records = parse_scene_txt(args.scene_txt)
    records = select_records(all_records, args.max_frames)

    if not records:
        print("Error: no records to process", file=sys.stderr)
        return 1

    video_path = args.video_path
    if video_path is None:
        if not args.download_video:
            print("Error: provide --video-path, or enable --download-video.", file=sys.stderr)
            return 1
        video_path = out_scene_dir / "video.mp4"
        maybe_download_video(youtube_url, video_path, args.yt_dlp_bin)

    if not video_path.exists():
        print(f"Error: video file not found: {video_path}", file=sys.stderr)
        return 1

    print(f"Processing ACID scene: {scene_name}")
    print(f"Records: {len(records)}")
    print(f"Video: {video_path}")

    image_names = extract_frames(records, video_path, image_dir)
    cameras_json = build_cameras_json(
        records,
        image_names,
        original_width=args.original_width,
        target_width=args.target_width,
        crop_offset_x=args.crop_offset_x,
    )
    intrinsics_json = build_intrinsics_json(
        records,
        original_width=args.original_width,
        target_width=args.target_width,
        crop_offset_x=args.crop_offset_x,
    )

    save_json(out_scene_dir / "cameras.json", cameras_json)
    save_json(out_scene_dir / "intrinsics.json", intrinsics_json)

    print(f"Done. Output: {out_scene_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
