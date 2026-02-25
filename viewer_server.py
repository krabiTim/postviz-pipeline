"""
Lightweight FastAPI server serving the Three.js viewer at port 7861.
Runs in a background thread when app.py starts.

Endpoints:
  GET /viewer                  → serves static/viewer.html
  GET /api/shots               → lists available shots in projects/
  GET /api/scene/{shot_name}   → returns scene data JSON for that shot
  GET /api/ply/{shot_name}     → serves the sparse PLY file for Three.js PLYLoader
"""

import json
import socket
import threading
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

PROJECTS_DIR = Path.home() / "postviz-pipeline" / "projects"
STATIC_DIR = Path(__file__).parent / "static"
VIEWER_PORT = 7861

app_viewer = FastAPI()

app_viewer.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


# ── Endpoints ────────────────────────────────────────────────────────────────

@app_viewer.get("/viewer", response_class=HTMLResponse)
async def get_viewer():
    html_path = STATIC_DIR / "viewer.html"
    if not html_path.exists():
        return HTMLResponse(
            "<html><body style='background:#111;color:#aaa;font-family:monospace;"
            "padding:2rem'><h2>Viewer not yet built</h2>"
            "<p>viewer.html will be added in Step 4.</p></body></html>"
        )
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app_viewer.get("/api/shots")
async def list_shots():
    if not PROJECTS_DIR.exists():
        return JSONResponse({"shots": []})
    shots = sorted(
        [d.name for d in PROJECTS_DIR.iterdir()
         if d.is_dir() and (d / "camera_path.json").exists()],
        reverse=True,
    )
    return JSONResponse({"shots": shots})


@app_viewer.get("/api/scene/{shot_name}")
async def get_scene(shot_name: str):
    shot_dir = PROJECTS_DIR / shot_name

    cp_path = shot_dir / "camera_path.json"
    intr_path = shot_dir / "intrinsics.json"

    if not cp_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"camera_path.json not found for shot '{shot_name}'. "
                   "Run camera solve first."
        )
    if not intr_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"intrinsics.json not found for shot '{shot_name}'. "
                   "Run camera solve first."
        )

    camera_path = json.loads(cp_path.read_text())
    intrinsics = json.loads(intr_path.read_text())

    ply_path = shot_dir / "colmap" / "sparse" / "0" / "fused.ply"

    return JSONResponse({
        "shot_name": shot_name,
        "camera_path": camera_path["frames"],
        "intrinsics": intrinsics,
        "ply_url": f"/api/ply/{shot_name}" if ply_path.exists() else None,
        "total_frames": camera_path["total_frames"],
        "fps": camera_path.get("fps", 24.0),
        "solve_error": camera_path.get("solve_error"),
        "total_registered": sum(
            1 for f in camera_path["frames"] if f.get("registered")
        ),
    })


@app_viewer.get("/api/ply/{shot_name}")
async def get_ply(shot_name: str):
    ply_path = (
        PROJECTS_DIR / shot_name / "colmap" / "sparse" / "0" / "fused.ply"
    )
    if not ply_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"PLY not found for shot '{shot_name}'"
        )
    return FileResponse(
        str(ply_path),
        media_type="application/octet-stream",
        filename="fused.ply",
    )


# ── Server lifecycle ─────────────────────────────────────────────────────────

def port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


def run_viewer_server():
    uvicorn.run(app_viewer, host="0.0.0.0", port=VIEWER_PORT, log_level="error")


def start_viewer_server_thread():
    if port_in_use(VIEWER_PORT):
        print(f"[viewer_server] port {VIEWER_PORT} already in use — skipping start")
        return
    t = threading.Thread(target=run_viewer_server, daemon=True)
    t.start()
    print(f"[viewer_server] started on http://localhost:{VIEWER_PORT}")


if __name__ == "__main__":
    print(f"[viewer_server] running standalone on http://0.0.0.0:{VIEWER_PORT}")
    uvicorn.run(app_viewer, host="0.0.0.0", port=VIEWER_PORT, log_level="info")
