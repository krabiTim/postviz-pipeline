# PostViz Pipeline — Developer Reference

## What This Is

A VFX dailies review pipeline for greenscreen/bluescreen HD footage. Upload a video clip,
extract frames, solve camera motion (COLMAP or imported FBX/USD), mask the foreground with
SAM2, generate AI replacement backgrounds with ComfyUI, relight with IC-Light, composite,
and deliver.

**Stack:** Python 3.12 · FastAPI · Gradio · uvicorn · COLMAP · Three.js · ComfyUI HTTP API
**Runtime:** WSL2 Ubuntu 24.04, RTX 4090, CUDA 12.x
**Entry point:** `python main.py` (port 7860)

---

## Core Principles

1. **Browser is the UI** — WSL2 + browser is the stack. No GUI frameworks. No Electron.
   The browser IS the app. Runs on `http://localhost:7860`.

2. **FastAPI is the backbone** — Gradio mounts ON FastAPI (`gr.mount_gradio_app`), not the
   reverse. All API routes live on the FastAPI app. A React frontend can be mounted later
   at `/app` without touching the Python pipeline code.

3. **One shot = one folder** — all data for a shot lives under `projects/{shot_name}/`.
   Nothing is stored in a database. Everything is a file. Shots can be moved, backed up,
   or shared by copying the folder.

4. **`camera_path.json` is the universal camera contract** — whether the camera comes from a
   COLMAP solve, an imported FBX, a USD file, or a hand-crafted JSON, the output is always
   this one schema. Everything downstream (viewer, masking, compositing) reads only this file.

5. **Workflows are JSON configuration** — a new ComfyUI workflow means dropping two files into
   `workflows/`: `name.json` (the ComfyUI graph) and `name.meta.json` (metadata). No Python
   changes. The UI rebuilds workflow dropdowns at startup from the registry scan.

6. **Camera importers are plugins** — add a module to `camera_import/` that implements
   `import_camera(file_path, fps) -> dict` and register it in `camera_import/__init__.py`
   `IMPORTERS` dict. No changes to `app.py`.

7. **Stages are independent** — each pipeline stage (frames, camera, mask, background,
   composite, delivery) reads its inputs from the shot folder and writes its outputs there.
   Stages do not depend on other stages having been run in this session.

8. **Install via git** — `git clone https://github.com/krabiTim/postviz-pipeline.git` then
   `bash install.sh` produces a working environment. No manual steps.

9. **Log everything** — every stage appends structured lines to
   `projects/{shot_name}/pipeline.log`. Format: `[YYYY-MM-DD HH:MM:SS] [STAGE] message`

10. **Extend, don't rewrite** — new features add to the pipeline. Old features are never
    broken by additions. No regressions.

---

## Project Layout

```
postviz-pipeline/
├── main.py                      ← entry point: uvicorn + FastAPI + Gradio
├── app.py                       ← Gradio blocks definition only (no launch call)
├── colmap_runner.py             ← COLMAP subprocess wrapper + frame extraction
├── comfyui_api.py               ← ComfyUI HTTP+WebSocket client
├── comfyui_nodes.py             ← /object_info query, node search, workflow validation
├── workflow_registry.py         ← scans workflows/, returns WorkflowMeta list
├── shot_manifest.py             ← read/write shot.json per shot
│
├── api/
│   ├── __init__.py
│   └── viewer_routes.py         ← FastAPI router: /viewer, /api/shots, /api/scene/...
│
├── camera_import/
│   ├── __init__.py              ← IMPORTERS dict, import_camera() dispatcher
│   ├── fbx.py                   ← FBX via Blender headless subprocess
│   ├── usd.py                   ← USD (stub, future: pxr / Blender)
│   ├── json_passthrough.py      ← accept an existing camera_path.json directly
│   └── blender_scripts/
│       └── extract_camera.py    ← runs inside `blender --background --python`
│
├── workflows/
│   ├── sam2_mask.json           ← ComfyUI workflow for foreground masking
│   ├── sam2_mask.meta.json
│   ├── bg_flux_inpaint.json     ← FLUX.1 Fill inpainting background
│   ├── bg_flux_inpaint.meta.json
│   ├── bg_generate.json         ← SDXL inpainting background (legacy name)
│   ├── bg_generate.meta.json
│   ├── relight_iclight.json     ← IC-Light relighting
│   ├── relight_iclight.meta.json
│   ├── bg_controlnet_depth.json ← ControlNet depth-guided background
│   ├── bg_controlnet_depth.meta.json
│   ├── wan_video_bg.json        ← Wan 2.2 animated background
│   ├── wan_video_bg.meta.json
│   ├── upscale_esrgan.json      ← ESRGAN delivery upscaling
│   └── upscale_esrgan.meta.json
│
├── static/
│   ├── viewer.html              ← Three.js r128 camera path + point cloud viewer
│   ├── three.min.js
│   ├── OrbitControls.js
│   └── PLYLoader.js
│
├── projects/                    ← shot data (one subfolder per shot)
│   └── {shot_name}/
│       ├── shot.json            ← shot manifest: stage status, camera source, metadata
│       ├── pipeline.log         ← append-only log for all stages
│       ├── camera_path.json     ← universal camera contract
│       ├── intrinsics.json
│       ├── frames/              ← extracted PNGs (frame_000000.png ...)
│       ├── colmap/              ← COLMAP workspace
│       ├── masks/               ← SAM2 mask PNGs
│       ├── backgrounds/         ← generated background images/videos
│       ├── composites/          ← composited output frames
│       ├── frames_ply/          ← per-frame sparse PLY files
│       └── delivery/            ← final exported files
│
├── docs/
│   └── COMFYUI_LLM_ONBOARDING.md  ← LLM onboarding prompt for workflow generation
│
├── .cache/                      ← auto-generated: ComfyUI node info cache
│   └── object_info.json         ← cached /object_info response (auto-refreshed)
│
├── delivery/                    ← global delivery output folder
├── install.sh                   ← team install script
└── requirements.txt
```

---

## camera_path.json Schema

All camera importers and the COLMAP runner output this exact schema:

```json
{
  "total_frames": 240,
  "fps": 24.0,
  "solve_error": 0.42,
  "source": "colmap",
  "frames": [
    {
      "frame_index": 0,
      "registered": true,
      "position": [x, y, z],
      "rotation_matrix": [
        [r00, r01, r02],
        [r10, r11, r12],
        [r20, r21, r22]
      ]
    }
  ]
}
```

- `source`: `"colmap"` | `"fbx"` | `"usd"` | `"json"`
- `registered`: false for frames COLMAP could not register
- `rotation_matrix`: R^T (world-from-camera, COLMAP convention, 3×3 row-major)
  - Row 0 = camera right vector
  - Row 1 = camera down vector
  - Row 2 = camera forward vector
- `solve_error`: mean reprojection error in pixels (0.0 for imported cameras)

---

## COLMAP Coordinate System

| | COLMAP | Three.js |
|---|---|---|
| Y axis | down | up |
| Z axis | forward (into scene) | toward viewer |
| Convention | OpenCV | OpenGL |

**Fix applied in viewer:** `sceneGroup.scale.set(1, -1, -1)` — flips all COLMAP data Y and Z.

**Follow Camera math (COLMAP → Three.js):**
```javascript
camera.position.set(pos[0], -pos[1], -pos[2])
camera.up.set(-rotT[1][0], rotT[1][1], rotT[1][2])
controls.target.set(pos[0] + rotT[2][0], -pos[1] - rotT[2][1], -pos[2] - rotT[2][2])
```

---

## How To: Add a New ComfyUI Workflow

**Step 0 — Discover real node names** (never guess):
```bash
# Fetch + cache all installed nodes from live ComfyUI
python comfyui_nodes.py fetch

# Search for nodes by keyword
python comfyui_nodes.py search sam
python comfyui_nodes.py search inpaint

# Inspect a specific node's exact interface
python comfyui_nodes.py inspect ExactNodeName
```

See `docs/COMFYUI_LLM_ONBOARDING.md` for the full methodology.

1. Export the workflow from ComfyUI as JSON (Save → Export API format)
2. Replace hardcoded values with template tokens:
   - `"__FRAME__"` — path to source frame (uploaded by comfyui_api.py)
   - `"__MASK__"` — path to the mask PNG
   - `"__BG__"` — path to the background image
   - `"__PROMPT__"` — text prompt injected at generation time
3. Save as `workflows/your_workflow.json`
4. Create `workflows/your_workflow.meta.json`:
   ```json
   {
     "name": "My Workflow",
     "stage": "background",
     "description": "What this does in one sentence.",
     "required_models": ["model_name.safetensors"],
     "inputs": ["frame", "mask", "prompt"],
     "outputs": ["image"]
   }
   ```
5. **Validate against ComfyUI** before committing:
   ```bash
   python comfyui_nodes.py validate workflows/your_workflow.json
   # Or validate all workflows at once:
   python workflow_registry.py --validate
   ```
6. Restart the app — the workflow appears in the dropdown immediately.

Valid `stage` values: `"mask"` | `"background"` | `"relight"` | `"upscale"` | `"video_bg"`

---

## How To: Add a New Camera Format Importer

1. Create `camera_import/yourformat.py` with:
   ```python
   def import_camera(file_path: str, fps: float) -> dict:
       """Return a camera_path.json-compatible dict."""
       ...
       return camera_path_dict
   ```
2. Register in `camera_import/__init__.py`:
   ```python
   from .yourformat import import_camera as _import_yourformat
   IMPORTERS[".ext"] = _import_yourformat
   ```
3. The extension appears automatically in the camera file picker.

---

## How To: Deploy to a New Machine (Team Install)

```bash
# Prerequisites: WSL2 Ubuntu 24.04, NVIDIA GPU with CUDA, ComfyUI on port 8188

git clone https://github.com/krabiTim/postviz-pipeline.git ~/postviz-pipeline
cd ~/postviz-pipeline
bash install.sh

# Start:
source ~/comfyui-postviz/venv/bin/activate
python main.py
```

Access at `http://localhost:7860` (or the machine's LAN IP for team use).

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Gradio UI |
| GET | `/viewer` | Three.js camera path viewer |
| GET | `/api/shots` | List solved shots |
| GET | `/api/scene/{shot_name}` | Scene JSON (camera_path + intrinsics + PLY info) |
| GET | `/api/ply/{shot_name}` | Binary PLY (global sparse cloud) |
| GET | `/api/frame_ply/{shot_name}/{frame_idx}` | Per-frame sparse PLY |
| GET | `/api/workflows` | List all workflows (optionally ?stage=background) |
| GET | `/api/shots/{shot_name}/manifest` | Shot manifest JSON |

---

## Environment

| Thing | Detail |
|-------|--------|
| OS | WSL2 Ubuntu 24.04 |
| Python | 3.12.3 (system) |
| Venv | `~/comfyui-postviz/venv/` (NOT in project dir) |
| GPU | RTX 4090, CUDA passthrough |
| COLMAP | 3.9.1 via `apt` |
| FFmpeg | `~/bin/ffmpeg` v7.0.2 (static binary, full path) |
| ComfyUI (WSL) | `~/comfyui-postviz/` port 8188 |
| ComfyUI (Windows) | `C:\aiProjects\ComfyUI\` — READ ONLY, do not touch |
| Blender | `/usr/bin/blender` (for FBX/USD camera import) |

**Key gotchas:**
- `sudo apt` requires interactive password — do it in a WSL terminal, not from Claude Code
- Shell scripts written via Windows path get CRLF — always run `sed -i 's/\r//' <file>`
- `~/bin` not in PATH for `bash -c "..."` — use full paths like `/home/krabiboy/bin/ffmpeg`
- Background processes: use `run_in_background=True` on the Bash tool (not `&`)
- `gr.mount_gradio_app` requires Gradio blocks to be fully defined before mounting

---

## Known Issues / Decisions

| Issue | Decision |
|-------|----------|
| `THREE.PLYLoader.load()` stalls on binary over localhost (XHR bug) | `fetch()` + `PLYLoader.parse()` + 15s timeout |
| `sceneGroup.removeFromParent()` missing in Three.js r128 | `scene.remove(sceneGroup)` |
| `colmap_runner.py` writes `focal_length_x`, viewer expected `focal_length` | `const fl = intr.focal_length || intr.focal_length_x` |
| Follow Cam disabled OrbitControls | Use `controls.target` instead of `controls.enabled = false` |
| Gradio `gr.Textbox(visible=False)` for solve_quality didn't persist | `gr.Markdown(value="", visible=True)` |
