"""
USD Camera Importer (stub)

Future implementation: parse USD/USDA/USDC/USDZ camera animation using
the OpenUSD Python bindings (pxr) or Blender headless USD import.

Planned sources:
  - Unreal Engine USD export
  - DCC tools (Houdini, Maya, Blender USD)
  - SynthEyes / 3DEqualizer USD export

TODO — Option A (pxr OpenUSD):
    from pxr import Usd, UsdGeom, Gf
    stage = Usd.Stage.Open(file_path)
    # Find UsdGeom.Camera prim, extract xformOp:transform per frame

TODO — Option B (Blender headless):
    Same approach as fbx.py — call blender --background with a USD import script.
    Blender 4.x has native USD read/write support.
"""

from __future__ import annotations


def import_usd(file_path: str, fps: float = 24.0) -> dict:
    """USD camera import — not yet implemented."""
    raise NotImplementedError(
        "USD camera import is not yet implemented. "
        "See camera_import/usd.py for planned implementation options."
    )
