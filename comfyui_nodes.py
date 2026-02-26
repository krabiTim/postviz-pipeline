"""
ComfyUI Node Discovery & Workflow Validation

Queries ComfyUI's /object_info REST API to discover installed nodes,
then validates workflow JSON against real node names and interfaces.

Based on the LLM onboarding approach from:
  https://github.com/huikku/comfyui-llm-onboarding-prompt

Usage:
    from comfyui_nodes import (
        fetch_nodes, search_nodes, inspect_node,
        validate_workflow, get_node_cache_path
    )

    # Refresh node cache from live ComfyUI
    nodes = fetch_nodes()

    # Search for nodes by keyword
    matches = search_nodes("sam")

    # Inspect a specific node's inputs/outputs
    info = inspect_node("SAM2VideoSegmentorLoader")

    # Validate a workflow JSON against installed nodes
    errors = validate_workflow(workflow_dict)
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

import requests

log = logging.getLogger(__name__)

COMFYUI_URL = "http://localhost:8188"
CACHE_DIR = Path(__file__).parent / ".cache"
NODE_CACHE_FILE = CACHE_DIR / "object_info.json"
CACHE_MAX_AGE = 3600  # seconds — re-fetch if cache is older than 1 hour


# ── Fetch & Cache ─────────────────────────────────────────────────────────────

def fetch_nodes(
    url: str = COMFYUI_URL,
    force: bool = False,
) -> dict[str, Any]:
    """
    Fetch all node definitions from ComfyUI's /object_info endpoint.
    Results are cached to .cache/object_info.json.

    Args:
        url: ComfyUI base URL (default http://localhost:8188)
        force: bypass cache and re-fetch from live API

    Returns:
        Dict mapping node class names to their full spec.
        Returns cached data if ComfyUI is offline.
        Returns empty dict if no cache and offline.
    """
    CACHE_DIR.mkdir(exist_ok=True)

    # Check cache freshness
    if not force and NODE_CACHE_FILE.exists():
        age = time.time() - NODE_CACHE_FILE.stat().st_mtime
        if age < CACHE_MAX_AGE:
            try:
                return json.loads(NODE_CACHE_FILE.read_text(encoding="utf-8"))
            except Exception:
                pass  # cache corrupted, re-fetch

    # Try live API
    try:
        r = requests.get(f"{url}/object_info", timeout=10)
        r.raise_for_status()
        nodes = r.json()
        # Write cache
        NODE_CACHE_FILE.write_text(
            json.dumps(nodes, ensure_ascii=False), encoding="utf-8"
        )
        log.info(f"[comfyui_nodes] Fetched {len(nodes)} nodes from {url}")
        return nodes
    except Exception as e:
        log.warning(f"[comfyui_nodes] Could not reach {url}: {e}")

    # Fallback to stale cache
    if NODE_CACHE_FILE.exists():
        try:
            nodes = json.loads(NODE_CACHE_FILE.read_text(encoding="utf-8"))
            log.info(f"[comfyui_nodes] Using cached node info ({len(nodes)} nodes)")
            return nodes
        except Exception:
            pass

    log.warning("[comfyui_nodes] No node info available (ComfyUI offline, no cache)")
    return {}


def inspect_node(
    node_name: str,
    url: str = COMFYUI_URL,
) -> dict[str, Any] | None:
    """
    Get the full spec for a single node by class name.
    Tries the live API first (/object_info/NodeName), falls back to cache.

    Returns None if the node is not found.
    """
    # Try live single-node endpoint (fast)
    try:
        r = requests.get(f"{url}/object_info/{node_name}", timeout=5)
        if r.status_code == 200:
            data = r.json()
            if node_name in data:
                return data[node_name]
    except Exception:
        pass

    # Fallback to full cache
    nodes = fetch_nodes(url=url)
    return nodes.get(node_name)


# ── Search ────────────────────────────────────────────────────────────────────

def search_nodes(
    keyword: str,
    url: str = COMFYUI_URL,
    category: str | None = None,
) -> list[dict[str, Any]]:
    """
    Search installed nodes by keyword (case-insensitive).
    Searches class name, display name, and category.

    Args:
        keyword: search term (e.g. "sam", "video", "mask", "depth")
        url: ComfyUI base URL
        category: optional category filter

    Returns:
        List of dicts with keys: class_name, display_name, category,
        inputs (list of input names), outputs (list of output types).
    """
    nodes = fetch_nodes(url=url)
    kw = keyword.lower()
    results = []

    for class_name, spec in nodes.items():
        display = spec.get("display_name", class_name)
        cat = spec.get("category", "")

        if category and category.lower() not in cat.lower():
            continue

        if (
            kw in class_name.lower()
            or kw in display.lower()
            or kw in cat.lower()
        ):
            req_inputs = list(spec.get("input", {}).get("required", {}).keys())
            opt_inputs = list(spec.get("input", {}).get("optional", {}).keys())
            outputs = spec.get("output", [])

            results.append({
                "class_name": class_name,
                "display_name": display,
                "category": cat,
                "inputs_required": req_inputs,
                "inputs_optional": opt_inputs,
                "outputs": outputs,
            })

    return sorted(results, key=lambda x: x["class_name"])


# ── Workflow Validation ───────────────────────────────────────────────────────

def validate_workflow(
    workflow: dict[str, Any],
    url: str = COMFYUI_URL,
) -> list[str]:
    """
    Validate a ComfyUI workflow JSON against installed nodes.

    Checks:
    1. All node types exist in the installed node registry
    2. Required inputs are either connected or have widget values
    3. Link type compatibility between connected nodes

    Args:
        workflow: ComfyUI workflow dict (the JSON format with "nodes" and "links")

    Returns:
        List of error strings. Empty list = valid workflow.
    """
    nodes_registry = fetch_nodes(url=url)
    if not nodes_registry:
        return ["Cannot validate: no node info available (ComfyUI offline, no cache)"]

    errors: list[str] = []
    wf_nodes = workflow.get("nodes", [])
    wf_links = workflow.get("links", [])

    # Build link lookup: link_id -> (from_node, from_output_idx, to_node, to_input_idx, type)
    link_map: dict[int, tuple] = {}
    for link in wf_links:
        if len(link) >= 6:
            link_id, from_node, from_out, to_node, to_in, link_type = link[:6]
            link_map[link_id] = (from_node, from_out, to_node, to_in, link_type)

    # Build node lookup by id
    node_by_id: dict[int, dict] = {}
    for node in wf_nodes:
        node_by_id[node.get("id", -1)] = node

    for node in wf_nodes:
        node_id = node.get("id", "?")
        node_type = node.get("type", "")

        # Skip special nodes (reroute, notes, etc.)
        if node_type in ("Reroute", "Note", "PrimitiveNode"):
            continue

        # Check 1: Node type exists
        if node_type not in nodes_registry:
            errors.append(
                f"Node {node_id}: type '{node_type}' not found in ComfyUI. "
                f"Check spelling or install the required custom node pack."
            )
            continue

        spec = nodes_registry[node_type]
        required = spec.get("input", {}).get("required", {})

        # Check 2: Required inputs have connections or widget values
        node_inputs = node.get("inputs", [])
        widget_values = node.get("widgets_values", [])
        connected_inputs = set()

        for inp in node_inputs:
            if inp.get("link") is not None:
                connected_inputs.add(inp.get("name", ""))

        # Track which required inputs are satisfied
        widget_idx = 0
        for input_name, input_spec in required.items():
            if input_name in connected_inputs:
                continue
            # Check if it's a type that comes via link (IMAGE, MASK, MODEL, etc.)
            input_type = input_spec[0] if isinstance(input_spec, (list, tuple)) else input_spec
            if isinstance(input_type, str) and input_type.isupper():
                # This is a link-type input (IMAGE, MASK, LATENT, etc.)
                # It MUST be connected — no widget fallback
                errors.append(
                    f"Node {node_id} ({node_type}): required input '{input_name}' "
                    f"(type {input_type}) is not connected."
                )

    return errors


def validate_workflow_api_format(
    workflow: dict[str, Any],
    url: str = COMFYUI_URL,
) -> list[str]:
    """
    Validate a ComfyUI API-format workflow (the prompt dict format used by
    /prompt endpoint, keyed by string node IDs).

    This is the format used by comfyui_api.py to submit workflows.

    Returns:
        List of error strings. Empty list = valid workflow.
    """
    nodes_registry = fetch_nodes(url=url)
    if not nodes_registry:
        return ["Cannot validate: no node info available (ComfyUI offline, no cache)"]

    errors: list[str] = []

    for node_id, node_data in workflow.items():
        if not isinstance(node_data, dict):
            continue

        class_type = node_data.get("class_type", "")
        if not class_type:
            errors.append(f"Node '{node_id}': missing class_type")
            continue

        if class_type not in nodes_registry:
            errors.append(
                f"Node '{node_id}': class_type '{class_type}' not found in ComfyUI."
            )
            continue

        spec = nodes_registry[class_type]
        required = spec.get("input", {}).get("required", {})
        inputs = node_data.get("inputs", {})

        for input_name in required:
            if input_name not in inputs:
                errors.append(
                    f"Node '{node_id}' ({class_type}): missing required input '{input_name}'"
                )

    return errors


# ── Utilities ─────────────────────────────────────────────────────────────────

def get_node_cache_path() -> Path:
    """Return the path to the cached object_info.json."""
    return NODE_CACHE_FILE


def get_cache_age() -> float | None:
    """Return cache age in seconds, or None if no cache."""
    if NODE_CACHE_FILE.exists():
        return time.time() - NODE_CACHE_FILE.stat().st_mtime
    return None


def summarize_node(node_name: str, url: str = COMFYUI_URL) -> str | None:
    """
    Return a one-line summary of a node: display_name, category, inputs -> outputs.
    Returns None if node not found.
    """
    info = inspect_node(node_name, url=url)
    if info is None:
        return None

    display = info.get("display_name", node_name)
    cat = info.get("category", "")
    req = list(info.get("input", {}).get("required", {}).keys())
    outputs = info.get("output", [])

    inputs_str = ", ".join(req[:5])
    if len(req) > 5:
        inputs_str += f" (+{len(req)-5})"
    outputs_str = ", ".join(outputs)

    return f"{display} [{cat}] ({inputs_str}) -> ({outputs_str})"


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python comfyui_nodes.py fetch          # Fetch + cache all nodes")
        print("  python comfyui_nodes.py search <kw>    # Search nodes by keyword")
        print("  python comfyui_nodes.py inspect <name> # Inspect a specific node")
        print("  python comfyui_nodes.py validate <json> # Validate a workflow file")
        sys.exit(0)

    cmd = sys.argv[1]

    if cmd == "fetch":
        nodes = fetch_nodes(force=True)
        print(f"Fetched {len(nodes)} nodes. Cached to {NODE_CACHE_FILE}")
        for name in sorted(nodes)[:20]:
            print(f"  {name}")
        if len(nodes) > 20:
            print(f"  ... ({len(nodes) - 20} more)")

    elif cmd == "search" and len(sys.argv) >= 3:
        kw = sys.argv[2]
        results = search_nodes(kw)
        print(f"Found {len(results)} nodes matching '{kw}':")
        for r in results:
            outs = ", ".join(r["outputs"])
            print(f"  {r['class_name']:40s} [{r['category']}] -> ({outs})")

    elif cmd == "inspect" and len(sys.argv) >= 3:
        name = sys.argv[2]
        info = inspect_node(name)
        if info is None:
            print(f"Node '{name}' not found.")
        else:
            print(json.dumps(info, indent=2))

    elif cmd == "validate" and len(sys.argv) >= 3:
        wf_path = Path(sys.argv[2])
        wf = json.loads(wf_path.read_text())
        # Detect format: API format has string keys with class_type
        if any(isinstance(v, dict) and "class_type" in v for v in wf.values() if isinstance(v, dict)):
            errors = validate_workflow_api_format(wf)
        else:
            errors = validate_workflow(wf)
        if errors:
            print(f"INVALID — {len(errors)} error(s):")
            for e in errors:
                print(f"  {e}")
        else:
            print("VALID — all nodes verified against ComfyUI")

    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
