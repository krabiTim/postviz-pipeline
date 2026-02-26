"""
PostViz Pipeline — Gradio UI Definition

This module defines build_ui() which returns a gr.Blocks instance.
It is imported by main.py and mounted onto the FastAPI app via gr.mount_gradio_app().

Do NOT call app.launch() here — main.py owns the uvicorn process.

When run directly (python app.py) it falls back to standalone mode
and starts the viewer server thread for local development.
"""

import gradio as gr
import json
import queue
import threading
import time
import requests
from datetime import datetime
from pathlib import Path

from colmap_runner import extract_frames, run_colmap_solve, export_for_comfyui, export_per_frame_plys
from comfyui_api import (
    run_sam2_mask, generate_backgrounds,
    apply_relight, composite_frame, get_available_checkpoints
)
from workflow_registry import get_workflows
from shot_manifest import create_manifest, update_stage, update_meta, pipeline_log
from camera_import import import_camera, SUPPORTED_EXTENSIONS

COMFYUI_URL = "http://localhost:8188"
PROJECTS_DIR = Path(__file__).parent / "projects"
DELIVERY_DIR = Path(__file__).parent / "delivery"
PROJECTS_DIR.mkdir(exist_ok=True)
DELIVERY_DIR.mkdir(exist_ok=True)

# Unified server — viewer lives on same port as Gradio
VIEWER_PORT = 7860

STYLE_SUFFIXES = {
    "Natural light":  ", natural daylight, soft shadows, photorealistic",
    "Golden hour":    ", golden hour lighting, warm tones, long shadows, cinematic",
    "Overcast":       ", overcast sky, diffused light, flat shadows, muted tones",
    "Night":          ", night scene, artificial lighting, dramatic contrast, cinematic",
    "Studio":         ", studio lighting, clean background, professional photography",
    "Custom":         "",
}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _workflow_choices(stage: str) -> list:
    """Return [(display_name, workflow_id), ...] for a given stage."""
    workflows = get_workflows(stage=stage)
    if not workflows:
        return [("(no workflows installed)", "__none__")]
    return [(w.name, w.id) for w in workflows]


def get_viewer_html(shot_name: str) -> str:
    url = f"http://localhost:{VIEWER_PORT}/viewer?shot={shot_name}"
    return (
        f'<div style="font-family:monospace;padding:10px 0">'
        f'<a href="{url}" target="_blank" '
        f'style="display:inline-block;padding:8px 18px;background:#0d2a0d;'
        f'border:1px solid #3a7a3a;color:#8f8;text-decoration:none;'
        f'border-radius:3px;font-size:13px">&#9654; Open Camera Path Viewer</a>'
        f'&nbsp;&nbsp;<span style="color:#555;font-size:11px">{url}</span>'
        f'</div>'
    )


def check_comfyui() -> tuple[bool, str]:
    try:
        r = requests.get(f"{COMFYUI_URL}/system_stats", timeout=3)
        r.raise_for_status()
        v = r.json()["system"]["comfyui_version"]
        return True, f"ComfyUI {v} running at {COMFYUI_URL}"
    except Exception:
        return False, f"ComfyUI not detected at {COMFYUI_URL} — run ~/launch-postviz.sh first"


def make_shot_dir(video_path: str) -> Path:
    stem = Path(video_path).stem
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    shot = PROJECTS_DIR / f"{stem}_{ts}"
    (shot / "frames").mkdir(parents=True)
    (shot / "colmap").mkdir()
    (shot / "masks").mkdir()
    (shot / "backgrounds").mkdir()
    (shot / "composites").mkdir()
    return shot


# ─── TAB 1: TRACK ─────────────────────────────────────────────────────────────

def do_extract(video_file, state):
    if video_file is None:
        return "No video uploaded.", state, gr.update(), gr.update(visible=False)

    shot = make_shot_dir(video_file)
    state["shot_dir"] = str(shot)
    frames_dir = shot / "frames"

    try:
        info = extract_frames(video_file, str(frames_dir))
    except Exception as e:
        return f"Frame extraction failed: {e}", state, gr.update(), gr.update(visible=False)

    state["frames_dir"] = str(frames_dir)
    state["fps"] = info["fps"]

    # Write shot manifest
    create_manifest(
        shot,
        video_source=Path(video_file).name,
        fps=info["fps"],
        frame_count=info["frame_count"],
        width=info["width"],
        height=info["height"],
    )
    update_stage(shot, "frames", done=True)
    pipeline_log(shot, "frames", f"Extracted {info['frame_count']} frames at {info['fps']:.3f} fps")

    frame_files = sorted(frames_dir.glob("frame_*.png"))
    first = str(frame_files[0]) if frame_files else None

    return (
        f"Extracted {info['frame_count']} frames  "
        f"({info['fps']:.3f} fps, {info['width']}x{info['height']}, "
        f"{info['duration_sec']:.2f}s)",
        state,
        gr.update(value=first, visible=True),
        gr.update(visible=True),
    )


def do_import_camera(camera_file, fps_override, state):
    """Import camera path from FBX / USD / camera_path.json."""
    if camera_file is None:
        return "No camera file uploaded.", state, gr.update(visible=False)

    if not state.get("shot_dir"):
        # No shot dir yet — create one from the camera filename
        shot = make_shot_dir(camera_file)
        state["shot_dir"] = str(shot)
        create_manifest(shot, video_source="", fps=float(fps_override) if fps_override else 24.0)

    shot = Path(state["shot_dir"])
    fps = float(fps_override) if fps_override else state.get("fps", 24.0)

    try:
        result = import_camera(camera_file, fps=fps)
    except Exception as e:
        return f"Camera import failed: {e}", state, gr.update(visible=False)

    # Write camera_path.json and intrinsics.json
    cp_path = shot / "camera_path.json"
    cp_path.write_text(json.dumps(result, indent=2))

    intrinsics = {
        "focal_length_px": result.get("focal_length_px", 0),
        "sensor_width_px": result.get("width", 0),
        "sensor_height_px": result.get("height", 0),
        "cx": result.get("cx", 0),
        "cy": result.get("cy", 0),
    }
    (shot / "intrinsics.json").write_text(json.dumps(intrinsics, indent=2))

    source = result.get("source", "unknown")
    n_frames = result.get("total_frames", 0)
    state["camera_poses_path"] = str(cp_path)
    state["colmap_result"] = {"success": True, "solve_error": 0.0}

    update_stage(shot, "camera", done=True, source=source, solve_error=0.0)
    pipeline_log(shot, "camera", f"Imported {n_frames} frames from {source} file")

    viewer_html = get_viewer_html(shot.name)
    msg = f"Imported {n_frames} frames from {Path(camera_file).name} (source: {source})"
    return msg, state, gr.update(value=viewer_html, visible=True)


def do_colmap_solve(state, progress=gr.Progress()):
    if not state.get("frames_dir"):
        yield "Run frame extraction first.", state, gr.update(), gr.update(), gr.update()
        return

    shot = Path(state["shot_dir"])
    workspace = shot / "colmap"
    log_q = queue.Queue()
    log_lines = []
    result_holder = {}

    def run():
        result_holder["result"] = run_colmap_solve(
            state["frames_dir"], str(workspace), log_q
        )
        log_q.put(None)

    t = threading.Thread(target=run, daemon=True)
    t.start()

    STEP_PROGRESS = {
        "Feature extraction": (0.05, "Extracting SIFT features..."),
        "Feature matching":   (0.30, "Matching features..."),
        "Sparse mapping":     (0.55, "Sparse mapping (this takes a while)..."),
        "PLY export":         (0.88, "Exporting point cloud..."),
        "TXT export":         (0.92, "Exporting camera data..."),
    }
    step_label = "Starting COLMAP..."
    last_yield = 0.0
    progress(0, desc="Starting COLMAP...")

    while True:
        try:
            line = log_q.get(timeout=0.1)
            if line is None:
                break
            log_lines.append(line.rstrip())

            stripped = line.strip()
            if stripped.startswith("---") and stripped.endswith("---"):
                for key, (pct, label) in STEP_PROGRESS.items():
                    if key.lower() in stripped.lower():
                        step_label = label
                        progress(pct, desc=label)
                        break

            now = time.time()
            if now - last_yield > 0.2:
                last_yield = now
                yield (
                    "\n".join(log_lines[-200:]),
                    state,
                    gr.update(value=""),
                    gr.update(value=f"  {step_label}"),
                    gr.update(visible=False),
                )
        except queue.Empty:
            pass

    t.join()
    progress(1.0, desc="Done")
    result = result_holder.get("result", {})
    state["colmap_result"] = result

    if not result.get("success"):
        pipeline_log(shot, "camera", "COLMAP solve FAILED")
        yield (
            "\n".join(log_lines[-200:]),
            state,
            gr.update(value=""),
            gr.update(value="SOLVE FAILED — check log"),
            gr.update(visible=False),
        )
        return

    err = result.get("solve_error") or 0.0
    state["ply_path"] = result.get("ply_path")
    state["camera_poses_path"] = result.get("camera_poses_path")

    if err < 1.0:
        quality = f"Solve error: {err:.3f}px (GOOD)"
    elif err < 2.0:
        quality = f"Solve error: {err:.3f}px (ACCEPTABLE)"
    else:
        quality = f"Solve error: {err:.3f}px (POOR)"

    # Patch fps into camera_path.json
    fps = state.get("fps", 0.0)
    cp_path = shot / "camera_path.json"
    if cp_path.exists() and fps > 0:
        try:
            cp = json.loads(cp_path.read_text())
            cp["fps"] = fps
            cp_path.write_text(json.dumps(cp, indent=2))
        except Exception as e:
            log_lines.append(f"fps patch failed: {e}")

    # Generate per-frame PLY files for viewer temporal scrubbing
    try:
        n_plys = export_per_frame_plys(str(shot))
        log_lines.append(f"Exported {n_plys} per-frame PLY files")
    except Exception as e:
        log_lines.append(f"Per-frame PLY export skipped: {e}")

    update_stage(shot, "camera", done=True, source="colmap", solve_error=err)
    pipeline_log(shot, "camera", f"COLMAP solve OK — error {err:.3f}px")

    yield (
        "\n".join(log_lines[-200:]),
        state,
        gr.update(value=get_viewer_html(shot.name)),
        gr.update(value=quality),
        gr.update(visible=True),
    )


def do_accept_solve(state):
    if not state.get("colmap_result", {}).get("success"):
        return state, "No accepted solve yet."
    return state, "Solve accepted. Go to 02 MASK tab."


def do_export_comfyui(state):
    if not state.get("colmap_result", {}).get("success"):
        return "<p style='color:#f88'>Run and accept a camera solve first.</p>"

    shot_dir = Path(state["shot_dir"])
    shot_name = shot_dir.name
    cp_path = shot_dir / "camera_path.json"
    intr_path = shot_dir / "intrinsics.json"

    if not cp_path.exists() or not intr_path.exists():
        return "<p style='color:#f88'>camera_path.json or intrinsics.json not found.</p>"

    try:
        camera_path = json.loads(cp_path.read_text())
        intrinsics  = json.loads(intr_path.read_text())
        manifest = export_for_comfyui(shot_name, str(shot_dir), camera_path, intrinsics)
    except Exception as e:
        return f"<p style='color:#f88'>Export failed: {e}</p>"

    export_dir = shot_dir / "comfyui_export"
    reg = manifest["registered_frames"]
    tot = manifest["total_frames"]
    err = manifest.get("solve_error")
    err_str = f"{err:.3f} px" if err is not None else "-"

    return (
        f"<div style='font-family:monospace;font-size:12px;color:#ccc;padding:8px'>"
        f"<p style='color:#8f8;font-size:13px'>ComfyUI Export Ready</p>"
        f"<p><b>Written to:</b> <code>{export_dir}</code></p>"
        f"<p>Registered {reg}/{tot} frames &nbsp;|&nbsp; Solve error: {err_str}</p>"
        f"<hr style='border-color:#333;margin:8px 0'>"
        f"<p><b>In ComfyUI:</b></p>"
        f"<ul style='margin:4px 0 4px 16px'>"
        f"<li>Load <code>camera_poses.json</code> via KJNodes &gt; Load Camera Path JSON</li>"
        f"<li>PLY path via <code>sparse_ply_path.txt</code></li>"
        f"</ul>"
        f"</div>"
    )


# ─── TAB 2: MASK ──────────────────────────────────────────────────────────────

def load_frame(frame_idx, state):
    frames_dir = state.get("frames_dir")
    if not frames_dir:
        return None
    frames = sorted(Path(frames_dir).glob("frame_*.png"))
    if not frames:
        return None
    idx = max(0, min(int(frame_idx), len(frames) - 1))
    return str(frames[idx])


def get_frame_count(state):
    frames_dir = state.get("frames_dir")
    if not frames_dir:
        return 0
    return max(0, len(list(Path(frames_dir).glob("frame_*.png"))) - 1)


def do_run_mask(workflow_id, state, progress=gr.Progress()):
    if not state.get("frames_dir"):
        return "Run camera solve first.", state, gr.update(), gr.update()

    shot = Path(state["shot_dir"])
    masks_dir = shot / "masks"

    try:
        result = run_sam2_mask(
            frames_dir=state["frames_dir"],
            click_x=0.5,
            click_y=0.5,
            output_dir=str(masks_dir),
        )
    except Exception as e:
        return f"Masking failed: {e}", state, gr.update(), gr.update()

    state["masks_dir"] = str(masks_dir)
    coverage = result.get("coverage_pct", 0.0)

    if coverage < 5.0:
        quality = f"Coverage {coverage:.1f}% — too low (wrong subject?)"
    elif coverage > 95.0:
        quality = f"Coverage {coverage:.1f}% — too high (too much masked?)"
    else:
        quality = f"Coverage {coverage:.1f}%"

    mask_files = result.get("mask_frames", [])
    preview = mask_files[0] if mask_files else None

    update_stage(shot, "mask", done=True, workflow=workflow_id)
    pipeline_log(shot, "mask", f"Mask propagated via {workflow_id} — coverage {coverage:.1f}%")

    return (
        quality,
        state,
        gr.update(value=preview, visible=preview is not None),
        gr.update(visible=True),
    )


def do_accept_mask(state):
    if not state.get("masks_dir"):
        return state, "No mask accepted yet."
    return state, "Mask accepted. Go to 03 GENERATE tab."


# ─── TAB 3: GENERATE + COMPOSITE ──────────────────────────────────────────────

def do_generate(workflow_id, prompt, style, state, progress=gr.Progress()):
    if not state.get("masks_dir"):
        return "Run masking first.", state, None, None, None

    shot = Path(state["shot_dir"])
    frames = sorted(Path(state["frames_dir"]).glob("frame_*.png"))
    masks = sorted(Path(state["masks_dir"]).glob("*.png"))

    if not frames or not masks:
        return "No frames or masks found.", state, None, None, None

    key_frame = str(frames[0])
    key_mask = str(masks[0])

    full_prompt = prompt + STYLE_SUFFIXES.get(style, "")

    try:
        bg_paths = generate_backgrounds(key_frame, key_mask, full_prompt, num_variants=3)
    except Exception as e:
        return f"Generation failed: {e}", state, None, None, None

    state["bg_paths"] = bg_paths
    state["selected_bg"] = bg_paths[0] if bg_paths else None

    update_stage(shot, "background", done=False, workflow=workflow_id)
    pipeline_log(shot, "background", f"Generated 3 variants via {workflow_id}")

    imgs = bg_paths[:3] + [None] * (3 - len(bg_paths))
    return "Generated 3 variants. Click one to select.", state, imgs[0], imgs[1], imgs[2]


def select_bg(img_path, state):
    state["selected_bg"] = img_path
    return state, f"Selected: {Path(img_path).name if img_path else 'none'}"


def do_composite(use_relight, denoise, state, progress=gr.Progress()):
    if not state.get("selected_bg"):
        return "Select a background variant first.", state, gr.update()

    shot = Path(state["shot_dir"])
    frames = sorted(Path(state["frames_dir"]).glob("frame_*.png"))
    masks = sorted(Path(state["masks_dir"]).glob("*.png"))
    composites_dir = shot / "composites"

    if not frames or not masks:
        return "Missing frames or masks.", state, gr.update()

    key_frame = str(frames[0])
    key_mask = str(masks[0])
    bg_path = state["selected_bg"]

    try:
        if use_relight:
            fg_path = apply_relight(key_frame, key_mask, bg_path, float(denoise))
        else:
            fg_path = key_frame

        out_path = str(composites_dir / "composite_preview.png")
        composite_frame(fg_path, key_mask, bg_path, out_path)
    except Exception as e:
        return f"Composite failed: {e}", state, gr.update()

    state["composite_path"] = out_path
    update_stage(shot, "composite", done=True, relight=use_relight)
    pipeline_log(shot, "composite", f"Composite rendered (relight={use_relight})")

    return "Composite complete.", state, gr.update(value=out_path, visible=True)


def do_save_delivery(state):
    comp = state.get("composite_path")
    if not comp or not Path(comp).exists():
        return "No composite to save."
    shot = Path(state["shot_dir"])
    dest = DELIVERY_DIR / Path(comp).name
    dest.write_bytes(Path(comp).read_bytes())
    update_stage(shot, "delivery", done=True, path=str(dest))
    pipeline_log(shot, "delivery", f"Saved to {dest}")
    return f"Saved to {dest}"


# ─── BUILD UI ─────────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    """Build and return the Gradio Blocks interface. Called by main.py."""

    comfyui_ok, comfyui_msg = check_comfyui()
    mask_choices = _workflow_choices("mask")
    bg_choices = _workflow_choices("background")
    cam_extensions = ", ".join(SUPPORTED_EXTENSIONS)

    with gr.Blocks(title="PostViz Pipeline") as demo:

        state = gr.State({})

        # ComfyUI status banner
        with gr.Row():
            gr.Markdown(
                f"{'OK' if comfyui_ok else 'WARN'} {comfyui_msg}"
            )

        # ── Tab 1: TRACK ──────────────────────────────────────────────────────
        with gr.Tab("01 · TRACK"):
            gr.Markdown("## Camera Solve / Import")

            camera_source = gr.Radio(
                choices=["COLMAP Solve", "Import Camera File"],
                value="COLMAP Solve",
                label="Camera source",
            )

            with gr.Row():
                with gr.Column(scale=1):
                    video_input = gr.File(
                        label="Upload video (.mp4 .mov .mxf .avi)",
                        file_types=[".mp4", ".mov", ".mxf", ".avi"]
                    )
                    extract_btn = gr.Button("Extract Frames", variant="primary")
                    extract_status = gr.Textbox(label="Status", interactive=False, lines=2)
                    frame_preview = gr.Image(label="First frame", visible=False)

                    # COLMAP group
                    with gr.Group(visible=True) as colmap_group:
                        solve_btn = gr.Button("Run Camera Solve", variant="primary", visible=False)

                    # Import camera file group
                    with gr.Group(visible=False) as import_group:
                        camera_file_input = gr.File(
                            label=f"Camera file ({cam_extensions})",
                            file_types=SUPPORTED_EXTENSIONS,
                        )
                        fps_input = gr.Number(label="FPS (leave 0 to use video fps)", value=0, precision=3)
                        import_btn = gr.Button("Import Camera File", variant="primary")
                        import_status = gr.Textbox(label="Import status", interactive=False, lines=2)

                with gr.Column(scale=1):
                    with gr.Accordion("COLMAP log", open=False):
                        solve_log = gr.Textbox(
                            label="",
                            lines=12,
                            interactive=False,
                            max_lines=20,
                            show_label=False,
                        )
                    solve_quality = gr.Markdown(value="", visible=True)
                    viewer_html = gr.HTML(value="")
                    accept_solve_btn = gr.Button("Accept Solve ->", variant="secondary", visible=False)
                    accept_solve_status = gr.Textbox(label="", interactive=False, visible=True, lines=1)
                    export_btn = gr.Button("Export to ComfyUI", variant="secondary", visible=False)
                    export_status = gr.HTML(value="")

            # Toggle COLMAP / Import groups based on camera_source radio
            def _toggle_camera_source(choice):
                colmap_vis = (choice == "COLMAP Solve")
                return gr.update(visible=colmap_vis), gr.update(visible=not colmap_vis)

            camera_source.change(
                _toggle_camera_source,
                inputs=[camera_source],
                outputs=[colmap_group, import_group],
            )

            extract_btn.click(
                do_extract,
                inputs=[video_input, state],
                outputs=[extract_status, state, frame_preview, solve_btn]
            )
            solve_btn.click(
                do_colmap_solve,
                inputs=[state],
                outputs=[solve_log, state, viewer_html, solve_quality, accept_solve_btn]
            )
            import_btn.click(
                do_import_camera,
                inputs=[camera_file_input, fps_input, state],
                outputs=[import_status, state, viewer_html],
            )
            accept_solve_btn.click(
                do_accept_solve,
                inputs=[state],
                outputs=[state, accept_solve_status]
            )
            accept_solve_btn.click(
                lambda s: gr.update(visible=s.get("colmap_result", {}).get("success", False)),
                inputs=[state],
                outputs=[export_btn]
            )
            export_btn.click(
                do_export_comfyui,
                inputs=[state],
                outputs=[export_status]
            )

        # ── Tab 2: MASK ───────────────────────────────────────────────────────
        with gr.Tab("02 · MASK"):
            gr.Markdown("## Foreground Masking")

            mask_workflow_dropdown = gr.Dropdown(
                choices=mask_choices,
                value=mask_choices[0][1] if mask_choices else "__none__",
                label="Masking workflow",
            )

            with gr.Row():
                with gr.Column(scale=1):
                    frame_slider = gr.Slider(0, 100, value=0, step=1, label="Frame scrubber")
                    frame_display = gr.Image(label="Frame viewer", interactive=False)
                    mask_btn = gr.Button("Propagate Mask", variant="primary")
                    mask_status = gr.Textbox(label="Mask quality", interactive=False, lines=2)

                with gr.Column(scale=1):
                    mask_preview = gr.Image(label="Mask preview", visible=False)
                    accept_mask_btn = gr.Button("Accept Mask ->", variant="secondary", visible=False)
                    accept_mask_status = gr.Textbox(label="", interactive=False, lines=1)

            frame_slider.change(load_frame, inputs=[frame_slider, state], outputs=[frame_display])
            mask_btn.click(
                do_run_mask,
                inputs=[mask_workflow_dropdown, state],
                outputs=[mask_status, state, mask_preview, accept_mask_btn]
            )
            accept_mask_btn.click(
                do_accept_mask,
                inputs=[state],
                outputs=[state, accept_mask_status]
            )

        # ── Tab 3: GENERATE + COMPOSITE ───────────────────────────────────────
        with gr.Tab("03 · GENERATE"):
            gr.Markdown("## Background Generation + Composite")

            bg_workflow_dropdown = gr.Dropdown(
                choices=bg_choices,
                value=bg_choices[0][1] if bg_choices else "__none__",
                label="Background workflow",
            )

            with gr.Row():
                with gr.Column(scale=1):
                    prompt_box = gr.Textbox(label="Background description", lines=3,
                                            placeholder="e.g. vast stormy coastline with crashing waves...")
                    style_dropdown = gr.Dropdown(
                        choices=list(STYLE_SUFFIXES.keys()),
                        value="Natural light",
                        label="Lighting style"
                    )
                    generate_btn = gr.Button("Generate Backgrounds (3 variants)", variant="primary")
                    generate_status = gr.Textbox(label="Status", interactive=False, lines=2)

                    gr.Markdown("### Select variant")
                    with gr.Row():
                        bg_img_0 = gr.Image(label="Variant 1", interactive=True, height=160)
                        bg_img_1 = gr.Image(label="Variant 2", interactive=True, height=160)
                        bg_img_2 = gr.Image(label="Variant 3", interactive=True, height=160)
                    selected_bg_label = gr.Textbox(label="Selected", interactive=False, lines=1)

                with gr.Column(scale=1):
                    use_relight = gr.Checkbox(label="Apply IC-Light relighting", value=False)
                    denoise_slider = gr.Slider(0.2, 0.8, value=0.45, step=0.05,
                                               label="Relight denoise strength")
                    composite_btn = gr.Button("Composite", variant="primary")
                    composite_status = gr.Textbox(label="Status", interactive=False, lines=2)
                    composite_preview = gr.Image(label="Composite result", visible=False)
                    save_btn = gr.Button("Save to delivery folder")
                    save_status = gr.Textbox(label="", interactive=False, lines=1)

            generate_btn.click(
                do_generate,
                inputs=[bg_workflow_dropdown, prompt_box, style_dropdown, state],
                outputs=[generate_status, state, bg_img_0, bg_img_1, bg_img_2]
            )
            bg_img_0.select(lambda s: select_bg(s, state), inputs=[bg_img_0], outputs=[state, selected_bg_label])
            bg_img_1.select(lambda s: select_bg(s, state), inputs=[bg_img_1], outputs=[state, selected_bg_label])
            bg_img_2.select(lambda s: select_bg(s, state), inputs=[bg_img_2], outputs=[state, selected_bg_label])

            composite_btn.click(
                do_composite,
                inputs=[use_relight, denoise_slider, state],
                outputs=[composite_status, state, composite_preview]
            )
            save_btn.click(do_save_delivery, inputs=[state], outputs=[save_status])

    return demo


# ─── Standalone (dev) mode ────────────────────────────────────────────────────

if __name__ == "__main__":
    # Only import viewer thread when running standalone (not via main.py)
    from viewer_server import start_viewer_server_thread
    start_viewer_server_thread()

    ui = build_ui()
    ui.queue()
    ui.launch(server_name="0.0.0.0", server_port=7860, share=False, theme=gr.themes.Base())
