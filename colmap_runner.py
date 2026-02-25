import json
import math
import subprocess
import threading
import queue
import time
import re
from pathlib import Path

import numpy as np

FFMPEG = Path.home() / "bin" / "ffmpeg"
FFPROBE = Path.home() / "bin" / "ffprobe"


def extract_frames(video_path: str, output_dir: str) -> dict:
    """
    Extract every single frame from video_path using FFmpeg.

    Command:
      ffmpeg -i {video_path} -q:v 2 -vsync 0 {output_dir}/frame_%06d.png

    Uses %06d (6 digits) to handle shots up to 7200 frames (4 min at 30 fps).
    Runs ffprobe first to get fps/duration/dimensions.

    Returns:
        {
            "frame_count": int,
            "fps": float,
            "duration_sec": float,
            "width": int,
            "height": int,
            "output_dir": str,
        }
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- ffprobe: get video stream metadata ---
    probe_result = subprocess.run(
        [str(FFPROBE), "-v", "quiet", "-print_format", "json", "-show_streams", str(video_path)],
        capture_output=True, text=True
    )
    if probe_result.returncode != 0:
        raise RuntimeError(f"ffprobe failed:\n{probe_result.stderr}")

    streams = json.loads(probe_result.stdout).get("streams", [])
    video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    if not video_stream:
        raise RuntimeError(f"No video stream found in {video_path}")

    width = int(video_stream["width"])
    height = int(video_stream["height"])

    # r_frame_rate is a fraction string like "24/1" or "30000/1001"
    fps_num, fps_den = video_stream["r_frame_rate"].split("/")
    fps = float(fps_num) / float(fps_den)

    duration_sec = float(video_stream.get("duration", 0.0))

    # --- FFmpeg: extract every frame ---
    cmd = [
        str(FFMPEG), "-y", "-i", str(video_path),
        "-q:v", "2",
        "-vsync", "0",
        str(output_dir / "frame_%06d.png")
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed:\n{result.stderr}")

    frame_count = len(sorted(output_dir.glob("frame_*.png")))

    return {
        "frame_count": frame_count,
        "fps": fps,
        "duration_sec": duration_sec,
        "width": width,
        "height": height,
        "output_dir": str(output_dir),
    }


def run_colmap_solve(frames_dir: str, workspace_dir: str, log_queue: queue.Queue = None) -> dict:
    """
    Run COLMAP feature extraction → matching → mapping → PLY export.
    Streams stdout/stderr to log_queue if provided.
    Returns dict with ply_path, solve_error, camera_poses_path, success.
    """
    frames_dir = Path(frames_dir)
    workspace_dir = Path(workspace_dir)
    workspace_dir.mkdir(parents=True, exist_ok=True)

    db_path = workspace_dir / "db.db"
    sparse_dir = workspace_dir / "sparse"
    sparse_dir.mkdir(exist_ok=True)
    sparse_0 = sparse_dir / "0"
    log_path = workspace_dir.parent / "colmap.log"

    def run_step(cmd, step_name):
        if log_queue:
            log_queue.put(f"\n--- {step_name} ---\n")
        with open(log_path, "a") as log:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1
            )
            for line in proc.stdout:
                log.write(line)
                if log_queue:
                    log_queue.put(line)
            proc.wait()
        return proc.returncode

    steps = [
        (["colmap", "feature_extractor",
          "--database_path", str(db_path),
          "--image_path", str(frames_dir),
          "--ImageReader.single_camera", "1",
          "--SiftExtraction.use_gpu", "1"],
         "Feature extraction"),

        (["colmap", "sequential_matcher",
          "--database_path", str(db_path),
          "--SequentialMatching.overlap", "10",
          "--SequentialMatching.loop_detection", "0",
          "--SiftMatching.use_gpu", "1"],
         "Feature matching"),

        (["colmap", "mapper",
          "--database_path", str(db_path),
          "--image_path", str(frames_dir),
          "--output_path", str(sparse_dir)],
         "Sparse mapping"),
    ]

    for cmd, name in steps:
        rc = run_step(cmd, name)
        if rc != 0:
            if log_queue:
                log_queue.put(f"ERROR: {name} failed (exit {rc})\n")
            return {"success": False, "solve_error": None, "ply_path": None, "camera_poses_path": None}

    # Check mapper produced output
    if not sparse_0.exists():
        if log_queue:
            log_queue.put("ERROR: mapper produced no output in sparse/0\n")
        return {"success": False, "solve_error": None, "ply_path": None, "camera_poses_path": None}

    # Convert to PLY — output_path must be the full file path for PLY format
    ply_path = sparse_0 / "fused.ply"
    rc = run_step(
        ["colmap", "model_converter",
         "--input_path", str(sparse_0),
         "--output_path", str(ply_path),
         "--output_type", "PLY"],
        "PLY export"
    )
    if rc != 0 or not ply_path.exists():
        if log_queue:
            log_queue.put("WARNING: PLY export may have failed\n")

    # Export TXT format (needed by parse_camera_path / parse_camera_intrinsics)
    run_step(
        ["colmap", "model_converter",
         "--input_path", str(sparse_0),
         "--output_path", str(sparse_0),
         "--output_type", "TXT"],
        "TXT export"
    )

    # Parse solve error from log
    solve_error = _parse_solve_error(log_path)

    # Parse camera path and intrinsics, write JSON files
    shot_dir = workspace_dir.parent
    total_frames = len(sorted(frames_dir.glob("frame_*.png")))
    try:
        intrinsics = parse_camera_intrinsics(str(sparse_0))
        camera_path = parse_camera_path(str(sparse_0), total_frames)

        # Write intrinsics.json
        intrinsics_out = {
            "model": intrinsics["model"],
            "width": intrinsics["width"],
            "height": intrinsics["height"],
            "focal_length_x": intrinsics["focal_length"],
            "focal_length_y": intrinsics["focal_length"],
            "cx": intrinsics["cx"],
            "cy": intrinsics["cy"],
            "fov_y_deg": intrinsics["fov_y_deg"],
            "aspect_ratio": intrinsics["width"] / intrinsics["height"],
        }
        (shot_dir / "intrinsics.json").write_text(
            json.dumps(intrinsics_out, indent=2)
        )

        # Write camera_path.json
        camera_path_out = {
            "shot_name": shot_dir.name,
            "fps": 0.0,          # filled in by app.py from extract_frames result
            "total_frames": total_frames,
            "solve_error": solve_error,
            "frames": camera_path["frames"],
        }
        (shot_dir / "camera_path.json").write_text(
            json.dumps(camera_path_out, indent=2)
        )

        if log_queue:
            reg = camera_path["total_registered"]
            pct = camera_path["registration_pct"]
            log_queue.put(
                f"\nRegistered {reg}/{total_frames} frames ({pct:.1f}%)\n"
            )
    except Exception as e:
        if log_queue:
            log_queue.put(f"WARNING: camera path parsing failed: {e}\n")

    camera_poses_path = sparse_0 / "cameras.bin"
    if not camera_poses_path.exists():
        camera_poses_path = sparse_0 / "cameras.txt"

    if log_queue:
        log_queue.put(f"\nSolve complete. Error: {solve_error}\n")

    return {
        "success": True,
        "ply_path": str(ply_path) if ply_path.exists() else None,
        "solve_error": solve_error,
        "camera_poses_path": str(camera_poses_path) if camera_poses_path.exists() else None,
    }


def _parse_solve_error(log_path: Path) -> float:
    """Parse mean reprojection error from COLMAP mapper log."""
    if not log_path.exists():
        return 0.0
    text = log_path.read_text(errors="ignore")
    # COLMAP prints: "Mean reprojection error: 0.512"
    matches = re.findall(r"mean reprojection error[^\d]*([\d.]+)", text, re.IGNORECASE)
    if matches:
        return float(matches[-1])
    return 0.0


def _quat_to_rotation_matrix(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """COLMAP quaternion [qw, qx, qy, qz] → 3×3 rotation matrix (camera-from-world)."""
    return np.array([
        [1 - 2*(qy**2 + qz**2),  2*(qx*qy - qz*qw),   2*(qx*qz + qy*qw)  ],
        [2*(qx*qy + qz*qw),      1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw) ],
        [2*(qx*qz - qy*qw),      2*(qy*qz + qx*qw),    1 - 2*(qx**2 + qy**2)],
    ])


def parse_camera_intrinsics(sparse_dir: str) -> dict:
    """
    Parse COLMAP sparse/0/cameras.txt.

    cameras.txt format:
      CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]
    SIMPLE_RADIAL: PARAMS = [f, cx, cy, k]
    PINHOLE:       PARAMS = [fx, fy, cx, cy]

    Returns:
        {model, width, height, focal_length, cx, cy, fov_y_deg}
    """
    cameras_txt = Path(sparse_dir) / "cameras.txt"
    if not cameras_txt.exists():
        raise FileNotFoundError(f"cameras.txt not found in {sparse_dir}")

    for line in cameras_txt.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        model = parts[1]
        width = int(parts[2])
        height = int(parts[3])
        params = [float(p) for p in parts[4:]]

        if model in ("SIMPLE_RADIAL", "SIMPLE_PINHOLE"):
            focal_length = params[0]
            cx, cy = params[1], params[2]
        elif model in ("PINHOLE", "RADIAL"):
            focal_length = params[0]   # fx; fy = params[1]
            cx, cy = params[2], params[3]
        else:
            # Fallback: first param is focal length
            focal_length = params[0]
            cx = width / 2.0
            cy = height / 2.0

        fov_y_deg = math.degrees(2.0 * math.atan(height / 2.0 / focal_length))

        return {
            "model": model,
            "width": width,
            "height": height,
            "focal_length": focal_length,
            "cx": cx,
            "cy": cy,
            "fov_y_deg": fov_y_deg,
        }

    raise RuntimeError("No camera entries found in cameras.txt")


def parse_camera_path(sparse_dir: str, total_frames: int = None) -> dict:
    """
    Parse COLMAP sparse/0/images.txt into an ordered list of camera poses.

    images.txt has two lines per image:
      IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME   ← pose line
      POINTS2D ...                                     ← keypoints (skipped)

    NAME is the frame filename (e.g. frame_000001.png). Sort by the numeric
    part to get temporal order — IMAGE_ID order is not guaranteed temporal.

    Camera position in world = -R^T @ t  (COLMAP stores camera-from-world).

    If total_frames is provided, the returned 'frames' list covers every
    frame index 0..total_frames-1. Unregistered frames get registered=False
    and positions interpolated linearly between adjacent registered frames.

    Returns:
        {
            "frames": [...],
            "total_registered": int,
            "total_frames": int,
            "registration_pct": float,
        }
    """
    images_txt = Path(sparse_dir) / "images.txt"
    if not images_txt.exists():
        raise FileNotFoundError(f"images.txt not found in {sparse_dir}")

    lines = [l.strip() for l in images_txt.read_text().splitlines()
             if not l.startswith("#") and l.strip()]

    # Pose lines are every other line starting at index 0
    pose_lines = lines[0::2]

    registered: dict[int, dict] = {}   # frame_index (0-based) → frame dict
    for line in pose_lines:
        parts = line.split()
        image_id = int(parts[0])
        qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
        name = parts[9]

        # Frame index from filename (0-based)
        match = re.search(r"(\d+)", name)
        frame_index = int(match.group(1)) - 1 if match else image_id - 1

        R = _quat_to_rotation_matrix(qw, qx, qy, qz)
        t = np.array([tx, ty, tz])
        position = (-R.T @ t).tolist()

        registered[frame_index] = {
            "frame_index": frame_index,
            "image_name": name,
            "image_id": image_id,
            "registered": True,
            "interpolated": False,
            "position": position,
            "rotation_matrix": R.T.tolist(),     # world-from-camera (R^T)
            "quaternion": [qw, qx, qy, qz],
            "translation": [tx, ty, tz],
        }

    total_registered = len(registered)

    if total_frames is None:
        # Return only registered frames, sorted by frame_index
        frames = sorted(registered.values(), key=lambda f: f["frame_index"])
        return {
            "frames": frames,
            "total_registered": total_registered,
            "total_frames": total_registered,
            "registration_pct": 100.0,
        }

    # Build full frame list with interpolation for unregistered frames
    sorted_reg = sorted(registered.keys())
    frames = []

    for i in range(total_frames):
        if i in registered:
            frames.append(registered[i])
        else:
            # Interpolate: find nearest registered frames before and after
            prev_keys = [k for k in sorted_reg if k < i]
            next_keys = [k for k in sorted_reg if k > i]

            if prev_keys and next_keys:
                p_idx = prev_keys[-1]
                n_idx = next_keys[0]
                alpha = (i - p_idx) / (n_idx - p_idx)
                p_pos = np.array(registered[p_idx]["position"])
                n_pos = np.array(registered[n_idx]["position"])
                pos = (p_pos + alpha * (n_pos - p_pos)).tolist()
            elif prev_keys:
                pos = registered[prev_keys[-1]]["position"]
            elif next_keys:
                pos = registered[next_keys[0]]["position"]
            else:
                pos = [0.0, 0.0, 0.0]

            frames.append({
                "frame_index": i,
                "image_name": f"frame_{i+1:06d}.png",
                "image_id": None,
                "registered": False,
                "interpolated": True,
                "position": pos,
                "rotation_matrix": None,
                "quaternion": None,
                "translation": None,
            })

    return {
        "frames": frames,
        "total_registered": total_registered,
        "total_frames": total_frames,
        "registration_pct": total_registered / total_frames * 100.0,
    }


def export_for_comfyui(
    shot_name: str,
    shot_dir: str,
    camera_path: dict,
    intrinsics: dict,
) -> dict:
    """
    Write a ComfyUI-ready export to {shot_dir}/comfyui_export/:
        camera_poses.json     camera-to-world matrices per frame
        intrinsics.json       camera intrinsics copy
        sparse_ply_path.txt   absolute path to fused.ply
        export_manifest.json  summary + usage notes

    camera_poses.json uses the standard c2w (camera-to-world) 4×4 format
    expected by KJNodes CameraPath nodes and nerfstudio pipelines.

    For registered frames, c2w upper-left 3×3 = R^T (rotation_matrix from
    camera_path.json) and c2w[:3,3] = position (camera centre in world).
    For unregistered frames, rotation is identity and position is interpolated.

    Returns the manifest dict.
    """
    from datetime import datetime as _dt

    shot_dir = Path(shot_dir)
    export_dir = shot_dir / "comfyui_export"
    export_dir.mkdir(exist_ok=True)

    ply_path = shot_dir / "colmap" / "sparse" / "0" / "fused.ply"

    # ── camera_poses.json ────────────────────────────────────────────────────
    frames_out = []
    for frame in camera_path.get("frames", []):
        if frame.get("registered") and frame.get("rotation_matrix") is not None:
            R_c2w = np.array(frame["rotation_matrix"])   # already R^T (3×3)
            t_c2w = np.array(frame["position"])           # already -R^T @ t
        else:
            R_c2w = np.eye(3)
            t_c2w = np.array(frame["position"] if frame.get("position") else [0.0, 0.0, 0.0])

        c2w = np.eye(4)
        c2w[:3, :3] = R_c2w
        c2w[:3, 3]  = t_c2w

        frames_out.append({
            "frame_index":  frame["frame_index"],
            "c2w":          c2w.tolist(),
            "registered":   bool(frame.get("registered", False)),
            "interpolated": bool(frame.get("interpolated", False)),
        })

    camera_poses = {
        "camera_model":  "PINHOLE",
        "width":         intrinsics.get("width",  1920),
        "height":        intrinsics.get("height", 1080),
        "focal_length":  intrinsics.get("focal_length_x", intrinsics.get("focal_length", 0.0)),
        "cx":            intrinsics.get("cx", 0.0),
        "cy":            intrinsics.get("cy", 0.0),
        "frames":        frames_out,
    }
    camera_poses_path = export_dir / "camera_poses.json"
    camera_poses_path.write_text(json.dumps(camera_poses, indent=2))

    # ── intrinsics.json ──────────────────────────────────────────────────────
    intrinsics_path = export_dir / "intrinsics.json"
    intrinsics_path.write_text(json.dumps(intrinsics, indent=2))

    # ── sparse_ply_path.txt ──────────────────────────────────────────────────
    ply_txt_path = export_dir / "sparse_ply_path.txt"
    ply_txt_path.write_text(str(ply_path.resolve()) + "\n")

    # ── export_manifest.json ─────────────────────────────────────────────────
    registered_count = sum(1 for f in camera_path.get("frames", []) if f.get("registered"))
    manifest = {
        "shot_name":           shot_name,
        "export_timestamp":    _dt.now().isoformat(timespec="seconds"),
        "ply_path":            str(ply_path.resolve()),
        "camera_poses_path":   str(camera_poses_path.resolve()),
        "intrinsics_path":     str(intrinsics_path.resolve()),
        "total_frames":        camera_path.get("total_frames", len(frames_out)),
        "registered_frames":   registered_count,
        "solve_error":         camera_path.get("solve_error"),
        "comfyui_usage": {
            "load_camera_path": "KJNodes > Load Camera Path JSON → camera_poses.json",
            "load_ply":         "ComfyUI-3D-Pack > LoadPly → path in sparse_ply_path.txt",
            "note":             "All paths are WSL2 absolute paths. Windows ComfyUI: /mnt/c/ → C:\\",
        },
    }
    manifest_path = export_dir / "export_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    return manifest


def render_pointcloud_preview(ply_path: str, output_image_path: str) -> str:
    """
    Load .ply with open3d, render to PNG using matplotlib (no display required).
    Returns path to saved PNG.
    """
    import open3d as o3d
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend, no display needed
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    pcd = o3d.io.read_point_cloud(str(ply_path))

    if len(pcd.points) == 0:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.text(0.5, 0.5, "No points in cloud", ha="center", va="center",
                transform=ax.transAxes, color="white", fontsize=14)
        ax.set_facecolor("black")
        fig.patch.set_facecolor("black")
        fig.savefig(output_image_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        return output_image_path

    # Remove statistical outliers
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    pts = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else None

    # Subsample for display performance
    max_pts = 20000
    if len(pts) > max_pts:
        idx = np.random.choice(len(pts), max_pts, replace=False)
        pts = pts[idx]
        if colors is not None:
            colors = colors[idx]

    fig = plt.figure(figsize=(10, 5.6), facecolor="black")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("black")

    c = colors if colors is not None else "cyan"
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=c, s=0.5, linewidths=0)

    ax.set_axis_off()
    ax.set_box_aspect([1, 1, 1])

    # Remove margins
    fig.tight_layout(pad=0)
    fig.savefig(output_image_path, dpi=120, bbox_inches="tight",
                facecolor="black", edgecolor="none")
    plt.close(fig)

    return output_image_path


if __name__ == "__main__":
    import sys
    import tempfile

    if len(sys.argv) < 2:
        print("Usage: python colmap_runner.py <video_path>")
        sys.exit(1)

    video = sys.argv[1]

    with tempfile.TemporaryDirectory() as tmp:
        frames_dir = Path(tmp) / "frames"
        workspace = Path(tmp) / "colmap"

        print("Extracting all frames...")
        info = extract_frames(video, str(frames_dir))
        print(f"Extracted {info['frame_count']} frames  "
              f"({info['fps']:.3f} fps, {info['duration_sec']:.3f}s, "
              f"{info['width']}x{info['height']})")

        log_q = queue.Queue()

        def print_logs():
            while True:
                try:
                    line = log_q.get(timeout=1)
                    if line is None:
                        break
                    print(line, end="")
                except queue.Empty:
                    pass

        t = threading.Thread(target=print_logs, daemon=True)
        t.start()

        result = run_colmap_solve(str(frames_dir), str(workspace), log_q)
        log_q.put(None)
        t.join()

        print(f"\nResult: {result}")
