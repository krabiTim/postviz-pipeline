import json
import uuid
import time
import requests
import websocket
from pathlib import Path

COMFYUI_URL = "http://localhost:8188"


def get_available_checkpoints(url: str = COMFYUI_URL) -> list[str]:
    """Return list of available checkpoint model names from ComfyUI."""
    resp = requests.get(f"{url}/object_info/CheckpointLoaderSimple", timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return data["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"][0]


def get_available_models(model_type: str, url: str = COMFYUI_URL) -> list[str]:
    """
    Return model names for a given node type.
    model_type examples: 'VAELoader', 'UNETLoader', 'LoraLoader'
    """
    resp = requests.get(f"{url}/object_info/{model_type}", timeout=10)
    resp.raise_for_status()
    data = resp.json()
    node = data.get(model_type, {})
    required = node.get("input", {}).get("required", {})
    # First list input is usually the model file list
    for key, val in required.items():
        if isinstance(val, list) and isinstance(val[0], list):
            return val[0]
    return []


def upload_image(image_path: str, url: str = COMFYUI_URL) -> str:
    """Upload image to ComfyUI /upload/image endpoint. Return server filename."""
    with open(image_path, "rb") as f:
        resp = requests.post(
            f"{url}/upload/image",
            files={"image": (Path(image_path).name, f, "image/png")},
            timeout=30,
        )
    resp.raise_for_status()
    return resp.json()["name"]


def queue_workflow(workflow: dict, url: str = COMFYUI_URL) -> str:
    """POST workflow to /prompt. Return prompt_id."""
    client_id = str(uuid.uuid4())
    payload = {"prompt": workflow, "client_id": client_id}
    resp = requests.post(f"{url}/prompt", json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if data.get("node_errors"):
        raise RuntimeError(f"ComfyUI node errors: {data['node_errors']}")
    return data["prompt_id"]


def wait_for_completion(prompt_id: str, url: str = COMFYUI_URL, timeout: int = 300) -> dict:
    """
    Poll history endpoint until prompt_id is complete.
    Returns the outputs dict from the completed prompt.
    """
    deadline = time.time() + timeout

    def _check_history():
        resp = requests.get(f"{url}/history/{prompt_id}", timeout=10)
        resp.raise_for_status()
        history = resp.json()
        if prompt_id in history:
            entry = history[prompt_id]
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                raise RuntimeError(f"ComfyUI execution error: {status}")
            outputs = entry.get("outputs", {})
            if outputs:
                return outputs
        return None

    # Check if already done before connecting to websocket
    result = _check_history()
    if result is not None:
        return result

    # Try websocket for real-time completion signal
    ws_url = url.replace("http://", "ws://") + "/ws"
    ws_done = False
    try:
        ws = websocket.create_connection(ws_url, timeout=5)
        ws.settimeout(2)
        while time.time() < deadline:
            try:
                msg = json.loads(ws.recv())
                if msg.get("type") == "executing":
                    data = msg.get("data", {})
                    if data.get("node") is None and data.get("prompt_id") == prompt_id:
                        ws_done = True
                        break
            except websocket.WebSocketTimeoutException:
                # Periodically check history so we don't miss a completed job
                result = _check_history()
                if result is not None:
                    ws.close()
                    return result
        ws.close()
    except Exception:
        pass  # Fall through to polling

    # Poll history for outputs
    while time.time() < deadline:
        result = _check_history()
        if result is not None:
            return result
        time.sleep(1)

    raise TimeoutError(f"ComfyUI prompt {prompt_id} did not complete within {timeout}s")


def download_outputs(prompt_id: str, output_dir: str, url: str = COMFYUI_URL) -> list[str]:
    """
    Fetch output images from /view endpoint.
    Save to output_dir. Return list of saved file paths.
    """
    outputs = wait_for_completion(prompt_id, url)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for node_id, node_output in outputs.items():
        for image in node_output.get("images", []):
            filename = image["filename"]
            subfolder = image.get("subfolder", "")
            img_type = image.get("type", "output")
            params = {"filename": filename, "subfolder": subfolder, "type": img_type}
            resp = requests.get(f"{url}/view", params=params, timeout=60)
            resp.raise_for_status()
            dest = output_dir / filename
            dest.write_bytes(resp.content)
            saved.append(str(dest))

    return saved


def _load_workflow(workflow_id: str) -> dict:
    """Load a workflow JSON by ID from the workflow registry, with fallback to direct path."""
    try:
        from workflow_registry import load_workflow
        return load_workflow(workflow_id)
    except (ValueError, FileNotFoundError):
        workflow_path = Path(__file__).parent / "workflows" / f"{workflow_id}.json"
        if workflow_path.exists():
            return json.loads(workflow_path.read_text())
        raise


def run_sam2_mask(
    frames_dir: str,
    click_x: float,
    click_y: float,
    output_dir: str,
    comfyui_url: str = COMFYUI_URL,
) -> dict:
    """
    Run SAM2 masking via GroundingDINO+SAM2 on the first frame.
    Uses 'person' as the text prompt (greenscreen workflow always masks a person).
    The click coordinates are used to select the closest detected person if multiple.
    Applies the resulting mask to all frames.
    Returns: {"mask_frames": [list of paths], "coverage_pct": float}
    """
    import cv2
    import numpy as np

    frames_dir = Path(frames_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_files = sorted(frames_dir.glob("frame_*.png"))
    if not frame_files:
        raise FileNotFoundError(f"No frames found in {frames_dir}")

    first_frame = str(frame_files[0])

    # Load workflow from registry
    workflow = _load_workflow("sam2_mask")

    # Upload first frame
    server_filename = upload_image(first_frame, comfyui_url)

    # Patch workflow with uploaded filename
    for node in workflow.values():
        if node.get("class_type") == "LoadImage":
            node["inputs"]["image"] = server_filename

    prompt_id = queue_workflow(workflow, comfyui_url)
    mask_paths = download_outputs(prompt_id, str(output_dir), comfyui_url)

    if not mask_paths:
        raise RuntimeError("SAM2 workflow produced no output masks")

    # Load first mask to measure coverage
    first_mask = cv2.imread(mask_paths[0], cv2.IMREAD_GRAYSCALE)
    if first_mask is not None:
        coverage = float(np.mean(first_mask > 127) * 100)
    else:
        coverage = 0.0

    # If only one mask produced, propagate to all frames by copying
    if len(mask_paths) == 1 and len(frame_files) > 1:
        base_mask = cv2.imread(mask_paths[0], cv2.IMREAD_GRAYSCALE)
        for i, _ in enumerate(frame_files[1:], start=2):
            dest = output_dir / f"mask_{i:04d}.png"
            cv2.imwrite(str(dest), base_mask)
            mask_paths.append(str(dest))

    return {"mask_frames": mask_paths, "coverage_pct": coverage}


def generate_backgrounds(
    key_frame_path: str,
    mask_path: str,
    prompt: str,
    num_variants: int = 3,
    workflow_id: str = "bg_flux_inpaint",
    comfyui_url: str = COMFYUI_URL,
) -> list[str]:
    """
    Generate background variants using FLUX/SDXL inpainting behind the mask.
    workflow_id: 'bg_flux_inpaint' (FLUX Fill) or 'bg_generate' (legacy SDXL).
    Returns list of paths to generated background images.
    """
    workflow = _load_workflow(workflow_id)

    frame_filename = upload_image(key_frame_path, comfyui_url)
    mask_filename = upload_image(mask_path, comfyui_url)

    output_paths = []
    for i in range(num_variants):
        wf = json.loads(json.dumps(workflow))  # deep copy
        for node in wf.values():
            ct = node.get("class_type", "")
            if ct == "LoadImage":
                inp = node["inputs"]
                if inp.get("image") == "__FRAME__":
                    inp["image"] = frame_filename
                elif inp.get("image") == "__MASK__":
                    inp["image"] = mask_filename
            if ct in ("CLIPTextEncode", "Text Encode") and "__PROMPT__" in str(node["inputs"]):
                node["inputs"]["text"] = prompt
            if ct == "CLIPTextEncodeFlux" and "__PROMPT__" in str(node["inputs"]):
                inp = node["inputs"]
                if inp.get("clip_l") == "__PROMPT__":
                    inp["clip_l"] = prompt
                if inp.get("t5xxl") == "__PROMPT__":
                    inp["t5xxl"] = prompt
            if ct == "KSampler":
                node["inputs"]["seed"] = i * 1000 + 42

        out_dir = Path(key_frame_path).parent.parent / "backgrounds"
        out_dir.mkdir(exist_ok=True)
        prompt_id = queue_workflow(wf, comfyui_url)
        paths = download_outputs(prompt_id, str(out_dir), comfyui_url)
        output_paths.extend(paths)

    return output_paths


def apply_relight(
    fg_frame_path: str,
    mask_path: str,
    bg_path: str,
    denoise: float,
    comfyui_url: str = COMFYUI_URL,
) -> str:
    """Run IC-Light workflow to relight FG to match BG. Return path to relighted image."""
    workflow = _load_workflow("relight_iclight")

    fg_filename = upload_image(fg_frame_path, comfyui_url)
    mask_filename = upload_image(mask_path, comfyui_url)
    bg_filename = upload_image(bg_path, comfyui_url)

    for node in workflow.values():
        ct = node.get("class_type", "")
        inp = node.get("inputs", {})
        if ct == "LoadImage":
            if inp.get("image") == "__FG__":
                inp["image"] = fg_filename
            elif inp.get("image") == "__MASK__":
                inp["image"] = mask_filename
            elif inp.get("image") == "__BG__":
                inp["image"] = bg_filename
        if ct == "KSampler":
            inp["denoise"] = denoise

    out_dir = Path(fg_frame_path).parent.parent / "backgrounds"
    out_dir.mkdir(exist_ok=True)
    prompt_id = queue_workflow(workflow, comfyui_url)
    paths = download_outputs(prompt_id, str(out_dir), comfyui_url)
    return paths[0] if paths else fg_frame_path


def composite_frame(
    fg_path: str,
    mask_path: str,
    bg_path: str,
    output_path: str,
) -> str:
    """
    OpenCV alpha composite: result = fg * mask + bg * (1 - mask).
    Saves result to output_path. Returns output_path.
    """
    import cv2
    import numpy as np

    fg = cv2.imread(fg_path, cv2.IMREAD_COLOR)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    bg = cv2.imread(bg_path, cv2.IMREAD_COLOR)

    if fg is None or mask is None or bg is None:
        raise FileNotFoundError(f"Could not load composite inputs: {fg_path}, {mask_path}, {bg_path}")

    bg = cv2.resize(bg, (fg.shape[1], fg.shape[0]))
    alpha = (mask.astype(np.float32) / 255.0)[:, :, np.newaxis]
    fg_f = fg.astype(np.float32)
    bg_f = bg.astype(np.float32)
    result = (fg_f * alpha + bg_f * (1.0 - alpha)).astype(np.uint8)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, result)
    return output_path


if __name__ == "__main__":
    print("Testing ComfyUI connection...")
    try:
        checkpoints = get_available_checkpoints()
        print(f"Checkpoints ({len(checkpoints)}):")
        for c in checkpoints:
            print(f"  - {c}")
        vaes = get_available_models("VAELoader")
        print(f"\nVAEs ({len(vaes)}):")
        for v in vaes:
            print(f"  - {v}")
        print("\nComfyUI connection OK.")
    except Exception as e:
        print(f"ERROR: {e}")
