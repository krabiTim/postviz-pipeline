"""
Workflow Registry — scans the workflows/ directory at startup and exposes
WorkflowMeta objects so the UI can build dynamic dropdowns without hardcoded names.

Includes validation via comfyui_nodes.py which queries ComfyUI's /object_info
to verify that workflow JSON references only real, installed nodes.

Usage:
    from workflow_registry import get_workflows, load_workflow, validate_workflow

    # Populate a Gradio dropdown:
    choices = [(w.name, w.id) for w in get_workflows(stage="background")]

    # Load the ComfyUI JSON for a selected workflow:
    wf = load_workflow("bg_flux_inpaint")

    # Validate a workflow against live ComfyUI:
    errors = validate_workflow("bg_flux_inpaint")
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

WORKFLOWS_DIR = Path(__file__).parent / "workflows"


@dataclass
class WorkflowMeta:
    id: str                          # filename stem (e.g. "sam2_mask")
    name: str                        # display name (e.g. "SAM2 Mask")
    stage: str                       # "mask"|"background"|"relight"|"upscale"|"video_bg"
    description: str = ""
    required_models: list[str] = field(default_factory=list)
    inputs: list[str] = field(default_factory=list)   # e.g. ["frame", "mask", "prompt"]
    outputs: list[str] = field(default_factory=list)  # e.g. ["image"]
    json_path: Optional[Path] = None

    @property
    def has_json(self) -> bool:
        return self.json_path is not None and self.json_path.exists()


def _load_all() -> list[WorkflowMeta]:
    """Scan workflows/ and return all WorkflowMeta objects that have a .meta.json."""
    if not WORKFLOWS_DIR.exists():
        return []

    results: list[WorkflowMeta] = []
    for meta_file in sorted(WORKFLOWS_DIR.glob("*.meta.json")):
        wf_id = meta_file.stem.replace(".meta", "")
        try:
            raw = json.loads(meta_file.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[workflow_registry] Could not parse {meta_file.name}: {e}")
            continue

        json_path = WORKFLOWS_DIR / f"{wf_id}.json"
        results.append(WorkflowMeta(
            id=wf_id,
            name=raw.get("name", wf_id),
            stage=raw.get("stage", ""),
            description=raw.get("description", ""),
            required_models=raw.get("required_models", []),
            inputs=raw.get("inputs", []),
            outputs=raw.get("outputs", []),
            json_path=json_path if json_path.exists() else None,
        ))

    return results


# Module-level cache — rebuilt on each import (i.e. each app start)
_REGISTRY: list[WorkflowMeta] = _load_all()


def get_workflows(stage: Optional[str] = None) -> list[WorkflowMeta]:
    """Return all registered workflows, optionally filtered by stage."""
    if stage is None:
        return list(_REGISTRY)
    return [w for w in _REGISTRY if w.stage == stage]


def load_workflow(workflow_id: str) -> dict:
    """
    Load and return the ComfyUI JSON dict for the given workflow id.
    Raises FileNotFoundError if the .json file is missing.
    Raises ValueError if the workflow_id is not in the registry.
    """
    matches = [w for w in _REGISTRY if w.id == workflow_id]
    if not matches:
        raise ValueError(
            f"Workflow '{workflow_id}' not in registry. "
            f"Available: {[w.id for w in _REGISTRY]}"
        )
    meta = matches[0]
    if not meta.has_json:
        raise FileNotFoundError(
            f"Workflow JSON missing for '{workflow_id}'. "
            f"Expected: {WORKFLOWS_DIR / workflow_id}.json"
        )
    return json.loads(meta.json_path.read_text(encoding="utf-8"))


def validate_workflow(workflow_id: str) -> list[str]:
    """
    Validate a workflow's JSON against ComfyUI's installed nodes.
    Uses comfyui_nodes.py to query /object_info and check every node type.

    Returns list of error strings. Empty = valid.
    Raises ValueError if workflow_id not in registry.
    Raises FileNotFoundError if the .json file is missing.
    """
    wf = load_workflow(workflow_id)

    from comfyui_nodes import validate_workflow as _validate_ui
    from comfyui_nodes import validate_workflow_api_format as _validate_api

    # Detect format: API format has string keys with class_type
    if isinstance(wf, dict) and any(
        isinstance(v, dict) and "class_type" in v
        for v in wf.values()
        if isinstance(v, dict)
    ):
        return _validate_api(wf)
    return _validate_ui(wf)


def reload() -> None:
    """Rescan workflows/ directory and update the in-memory registry."""
    global _REGISTRY
    _REGISTRY = _load_all()
    print(f"[workflow_registry] Loaded {len(_REGISTRY)} workflows: "
          f"{[w.id for w in _REGISTRY]}")


if __name__ == "__main__":
    import sys

    reload()
    for w in _REGISTRY:
        status = "OK" if w.has_json else "NO JSON"
        print(f"  [{w.stage:12s}] {w.name:35s} [{status}]")

    # If --validate flag, validate all workflows with JSON against live ComfyUI
    if "--validate" in sys.argv:
        print("\nValidating workflows against ComfyUI...")
        for w in _REGISTRY:
            if not w.has_json:
                continue
            errors = validate_workflow(w.id)
            if errors:
                print(f"  {w.id}: INVALID")
                for e in errors:
                    print(f"    {e}")
            else:
                print(f"  {w.id}: VALID")
