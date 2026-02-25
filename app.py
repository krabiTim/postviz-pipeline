import gradio as gr
import json
import queue
import threading
import time
import requests
from datetime import datetime
from pathlib import Path

from colmap_runner import extract_frames, run_colmap_solve, export_for_comfyui
from comfyui_api import (
    run_sam2_mask, generate_backgrounds,
    apply_relight, composite_frame, get_available_checkpoints
)
from viewer_server import start_viewer_server_thread

COMFYUI_URL = "http://localhost:8188"
PROJECTS_DIR = Path(__file__).parent / "projects"
DELIVERY_DIR = Path(__file__).parent / "delivery"
PROJECTS_DIR.mkdir(exist_ok=True)
DELIVERY_DIR.mkdir(exist_ok=True)

STYLE_SUFFIXES = {
    "Natural light":  ", natural daylight, soft shadows, photorealistic",
    "Golden hour":    ", golden hour lighting, warm tones, long shadows, cinematic",
    "Overcast":       ", overcast sky, diffused light, flat shadows, muted tones",
    "Night":          ", night scene, artificial lighting, dramatic contrast, cinematic",
    "Studio":         ", studio lighting, clean background, professional photography",
    "Custom":         "",
}


def get_viewer_html(shot_name: str) -> str:
    url = f"http://localhost:7861/viewer?shot={shot_name}"
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
        return False, f"⚠ ComfyUI not detected at {COMFYUI_URL} — run ~/launch-postviz.sh first"


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


# ─── TAB 1: TRACK ────────────────────────────────────────────────────────────

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

    # Map --- Step name --- log markers to progress fractions
    STEP_PROGRESS = {
        "Feature extraction": (0.05, "Extracting SIFT features…"),
        "Feature matching":   (0.30, "Matching features…"),
        "Sparse mapping":     (0.55, "Sparse mapping (this takes a while)…"),
        "PLY export":         (0.88, "Exporting point cloud…"),
        "TXT export":         (0.92, "Exporting camera data…"),
    }
    step_label = "Starting COLMAP…"
    last_yield = 0.0
    progress(0, desc="Starting COLMAP…")

    while True:
        try:
            line = log_q.get(timeout=0.1)
            if line is None:
                break
            log_lines.append(line.rstrip())

            # Detect step transitions from --- markers
            stripped = line.strip()
            if stripped.startswith("---") and stripped.endswith("---"):
                for key, (pct, label) in STEP_PROGRESS.items():
                    if key.lower() in stripped.lower():
                        step_label = label
                        progress(pct, desc=label)
                        break

            # Stream log to UI ~5× per second
            now = time.time()
            if now - last_yield > 0.2:
                last_yield = now
                yield (
                    "\n".join(log_lines[-200:]),
                    state,
                    gr.update(value=""),
                    gr.update(value=f"⏳ {step_label}", visible=True),
                    gr.update(visible=False),
                )
        except queue.Empty:
            pass

    t.join()
    progress(1.0, desc="Done")
    result = result_holder.get("result", {})
    state["colmap_result"] = result

    if not result.get("success"):
        yield (
            "\n".join(log_lines[-200:]),
            state,
            gr.update(value=""),
            gr.update(value="❌ SOLVE FAILED", visible=True),
            gr.update(visible=False),
        )
        return

    err = result.get("solve_error") or 0.0
    state["ply_path"] = result.get("ply_path")
    state["camera_poses_path"] = result.get("camera_poses_path")

    if err < 1.0:
        quality = f"🟢 Solve error: {err:.3f}px (GOOD)"
    elif err < 2.0:
        quality = f"🟡 Solve error: {err:.3f}px (ACCEPTABLE)"
    else:
        quality = f"🔴 Solve error: {err:.3f}px (POOR)"

    # Patch fps into camera_path.json (colmap_runner writes 0.0 as placeholder)
    fps = state.get("fps", 0.0)
    cp_path = shot / "camera_path.json"
    if cp_path.exists() and fps > 0:
        try:
            cp = json.loads(cp_path.read_text())
            cp["fps"] = fps
            cp_path.write_text(json.dumps(cp, indent=2))
        except Exception as e:
            log_lines.append(f"fps patch failed: {e}")

    yield (
        "\n".join(log_lines[-200:]),
        state,
        gr.update(value=get_viewer_html(shot.name)),
        gr.update(value=quality, visible=True),
        gr.update(visible=True),
    )


def do_accept_solve(state):
    if not state.get("colmap_result", {}).get("success"):
        return state, "No accepted solve yet."
    return state, "Solve accepted. Go to 02 · MASK tab."


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
    err_str = f"{err:.3f} px" if err is not None else "—"

    return (
        f"<div style='font-family:monospace;font-size:12px;color:#ccc;padding:8px'>"
        f"<p style='color:#8f8;font-size:13px'>✓ ComfyUI Export Ready</p>"
        f"<p><b>Written to:</b> <code>{export_dir}</code></p>"
        f"<p>Registered {reg}/{tot} frames &nbsp;|&nbsp; Solve error: {err_str}</p>"
        f"<hr style='border-color:#333;margin:8px 0'>"
        f"<p><b>In ComfyUI:</b></p>"
        f"<ul style='margin:4px 0 4px 16px'>"
        f"<li>Load <code>camera_poses.json</code> → KJNodes &gt; Load Camera Path JSON</li>"
        f"<li>PLY path → <code>sparse_ply_path.txt</code></li>"
        f"<li>All positions in COLMAP world space</li>"
        f"</ul>"
        f"</div>"
    )


# ─── TAB 2: MASK ─────────────────────────────────────────────────────────────

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


def do_run_mask(state, progress=gr.Progress()):
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
        quality = f"⚠ Coverage {coverage:.1f}% — too low (wrong subject?)"
    elif coverage > 95.0:
        quality = f"⚠ Coverage {coverage:.1f}% — too high (too much masked?)"
    else:
        quality = f"✓ Coverage {coverage:.1f}%"

    mask_files = result.get("mask_frames", [])
    preview = mask_files[0] if mask_files else None

    return (
        quality,
        state,
        gr.update(value=preview, visible=preview is not None),
        gr.update(visible=True),
    )


def do_accept_mask(state):
    if not state.get("masks_dir"):
        return state, "No mask accepted yet."
    return state, "Mask accepted. Go to 03 · GENERATE tab."


# ─── TAB 3: GENERATE + COMPOSITE ─────────────────────────────────────────────

def do_generate(prompt, style, state, progress=gr.Progress()):
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
    return "Composite complete.", state, gr.update(value=out_path, visible=True)


def do_save_delivery(state):
    comp = state.get("composite_path")
    if not comp or not Path(comp).exists():
        return "No composite to save."
    dest = DELIVERY_DIR / Path(comp).name
    dest.write_bytes(Path(comp).read_bytes())
    return f"Saved to {dest}"


# ─── BUILD UI ─────────────────────────────────────────────────────────────────

comfyui_ok, comfyui_msg = check_comfyui()

with gr.Blocks(title="PostViz Pipeline") as app:

    state = gr.State({})

    # ComfyUI status banner
    with gr.Row():
        status_banner = gr.Markdown(
            f"{'✓' if comfyui_ok else '⚠'} {comfyui_msg}"
        )

    # ── Tab 1: TRACK ─────────────────────────────────────────────────────────
    with gr.Tab("01 · TRACK"):
        gr.Markdown("## Camera Solve")

        with gr.Row():
            with gr.Column(scale=1):
                video_input = gr.File(
                    label="Upload video (.mp4 .mov .mxf .avi)",
                    file_types=[".mp4", ".mov", ".mxf", ".avi"]
                )
                extract_btn = gr.Button("Extract Frames", variant="primary")
                extract_status = gr.Textbox(label="Status", interactive=False, lines=2)
                frame_preview = gr.Image(label="First frame", visible=False)
                solve_btn = gr.Button("Run Camera Solve", variant="primary", visible=False)

            with gr.Column(scale=1):
                solve_log = gr.Textbox(label="COLMAP log", lines=10, interactive=False, max_lines=10)
                solve_quality = gr.Textbox(label="Solve quality", interactive=False, visible=False)
                viewer_html = gr.HTML(value="")
                accept_solve_btn = gr.Button("Accept Solve →", variant="secondary", visible=False)
                accept_solve_status = gr.Textbox(label="", interactive=False, visible=True, lines=1)
                export_btn = gr.Button("Export to ComfyUI", variant="secondary", visible=False)
                export_status = gr.HTML(value="")

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
        accept_solve_btn.click(
            do_accept_solve,
            inputs=[state],
            outputs=[state, accept_solve_status]
        )
        # Show export button once solve is accepted
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

    # ── Tab 2: MASK ──────────────────────────────────────────────────────────
    with gr.Tab("02 · MASK"):
        gr.Markdown("## Foreground Masking\n_Uses SAM2 + GroundingDINO with 'person' prompt. Suitable for greenscreen/bluescreen subjects._")

        with gr.Row():
            with gr.Column(scale=1):
                frame_slider = gr.Slider(0, 100, value=0, step=1, label="Frame scrubber")
                frame_display = gr.Image(label="Frame viewer", interactive=False)
                mask_btn = gr.Button("Propagate Mask", variant="primary")
                mask_status = gr.Textbox(label="Mask quality", interactive=False, lines=2)

            with gr.Column(scale=1):
                mask_preview = gr.Image(label="Mask preview", visible=False)
                accept_mask_btn = gr.Button("Accept Mask →", variant="secondary", visible=False)
                accept_mask_status = gr.Textbox(label="", interactive=False, lines=1)

        frame_slider.change(load_frame, inputs=[frame_slider, state], outputs=[frame_display])
        mask_btn.click(
            do_run_mask,
            inputs=[state],
            outputs=[mask_status, state, mask_preview, accept_mask_btn]
        )
        accept_mask_btn.click(
            do_accept_mask,
            inputs=[state],
            outputs=[state, accept_mask_status]
        )

    # ── Tab 3: GENERATE + COMPOSITE ──────────────────────────────────────────
    with gr.Tab("03 · GENERATE"):
        gr.Markdown("## Background Generation + Composite")

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
            inputs=[prompt_box, style_dropdown, state],
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


if __name__ == "__main__":
    start_viewer_server_thread()
    app.queue()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False, theme=gr.themes.Base())
