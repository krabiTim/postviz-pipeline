# PostViz Pipeline

GenAI-forward postviz pipeline for VFX: camera tracking → foreground masking →
background generation → composite. Built for greenscreen/bluescreen dailies review.

## Stack
- COLMAP 3.9.1 — camera tracking (sequential SfM)
- ComfyUI — SAM2 masking, background generation
- Three.js viewer — interactive camera path + point cloud review
- Gradio — operator UI
- DirectorsConsole — CPE prompt engineering + storyboard review

## Services
| Service | Port | Purpose |
|---------|------|---------|
| Gradio app | 7860 | Track + Mask + Generate tabs |
| Viewer server | 7861 | Three.js PLY + camera path viewer |
| ComfyUI | 8188 | AI generation backend |
| DirectorsConsole | 5173 | Storyboard canvas + CPE |

## Launch
```bash
~/launch-all.sh
```

## Current state
- Track tab: ✅ COLMAP solve, camera path viewer, ComfyUI export
- Mask tab: ⚠️ SAM2 single-frame only (propagation not yet wired)
- Generate tab: ⚠️ scaffold (inpainting workflow not yet wired)
