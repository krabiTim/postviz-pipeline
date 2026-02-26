"""
PostViz Pipeline — Main Entry Point

Single uvicorn process serving:
  - FastAPI on port 7860
  - Gradio UI mounted at /
  - Viewer API routes at /viewer and /api/*

Usage:
    source ~/comfyui-postviz/venv/bin/activate
    python main.py

Then open http://localhost:7860 in your browser.
"""

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

import gradio as gr

# ── Import sub-modules ────────────────────────────────────────────────────────

from api.viewer_routes import router as viewer_router
from workflow_registry import reload as registry_reload

# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="PostViz Pipeline",
    description="VFX dailies review pipeline — COLMAP solve, SAM2 mask, AI background, IC-Light relight",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static assets (Three.js, PLYLoader, OrbitControls, viewer.html)
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Register viewer API routes (/viewer, /api/shots, /api/scene/*, etc.)
app.include_router(viewer_router)

# ── Gradio UI ─────────────────────────────────────────────────────────────────

# Import the Gradio blocks (defined in app.py, no launch call)
from app import build_ui

gradio_app = build_ui()

# Mount Gradio onto the FastAPI app at root path
app = gr.mount_gradio_app(app, gradio_app, path="/")

# ── Startup ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Reload workflow registry on startup
    registry_reload()

    print("\n" + "=" * 60)
    print("  PostViz Pipeline v2.0")
    print("  http://localhost:7860")
    print("  http://localhost:7860/viewer")
    print("=" * 60 + "\n")

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=7860,
        log_level="warning",
        reload=False,
    )
