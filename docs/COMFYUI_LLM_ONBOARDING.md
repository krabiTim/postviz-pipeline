# ComfyUI + LLM Workflow Generation — Onboarding Prompt

> **Source:** https://github.com/huikku/comfyui-llm-onboarding-prompt
>
> **What this is:** Reference prompt for any LLM conversation where you want the
> AI to generate or debug ComfyUI workflows. It teaches the LLM to query
> ComfyUI's live API instead of guessing node names.
>
> **Integration:** This methodology is baked into the PostViz pipeline via
> `comfyui_nodes.py` which queries `/object_info`, caches results, and
> validates workflow JSON against installed nodes.

---

## Prompt (copy everything below this line)

You are helping me build ComfyUI workflows. ComfyUI has a REST API you can use
to discover exactly what nodes are installed and how they work. **Always use this
API before generating workflow JSON.** Do not guess or hallucinate node names.

### How to discover nodes

ComfyUI runs at `http://localhost:8188` (or whatever port is configured). It
exposes these endpoints:

```bash
# Get ALL registered nodes (large JSON — every installed node with full input/output specs)
curl -s http://localhost:8188/object_info

# Get ONE specific node (fast — use this to verify a node exists and check its exact interface)
curl -s http://localhost:8188/object_info/NodeName
```

### What the API returns

Each node entry includes:
- `input.required` / `input.optional` — every input with its type, default value, min/max, and tooltip
- `output` — list of output types (e.g. `["IMAGE", "MASK"]`)
- `output_name` — human-readable output names
- `display_name` — the name shown in the ComfyUI UI
- `category` — where it appears in the node menu

### Workflow before generating any workflow JSON

1. **Check if ComfyUI is running:** `curl -s http://localhost:8188/object_info | head -c 100`
2. **Search for relevant nodes:** `curl -s http://localhost:8188/object_info | python3 -c "import json,sys; [print(k) for k in sorted(json.load(sys.stdin)) if 'keyword' in k.lower()]"` (replace `keyword` with what you're looking for, e.g. `sam`, `video`, `mask`, `depth`)
3. **Inspect each node you plan to use:** `curl -s http://localhost:8188/object_info/ExactNodeName` — verify the exact input names, types, and output types
4. **Generate the workflow JSON** using ONLY verified nodes with correct types
5. **Save to** `workflows/<name>.json` with a matching `workflows/<name>.meta.json`

### Using the PostViz comfyui_nodes.py module

Instead of raw curl commands, use the built-in module:

```bash
# Fetch + cache all nodes from live ComfyUI
python comfyui_nodes.py fetch

# Search for nodes by keyword
python comfyui_nodes.py search sam
python comfyui_nodes.py search video
python comfyui_nodes.py search mask

# Inspect a specific node
python comfyui_nodes.py inspect SAM2VideoSegmentorLoader

# Validate a workflow JSON against installed nodes
python comfyui_nodes.py validate workflows/sam2_mask.json
```

Or from Python:

```python
from comfyui_nodes import fetch_nodes, search_nodes, inspect_node, validate_workflow

# Refresh cache
nodes = fetch_nodes(force=True)

# Search
matches = search_nodes("sam")
for m in matches:
    print(f"  {m['class_name']:40s} -> {m['outputs']}")

# Validate
import json
wf = json.loads(open("workflows/sam2_mask.json").read())
errors = validate_workflow(wf)
if errors:
    for e in errors:
        print(f"  ERROR: {e}")
else:
    print("All nodes verified.")
```

### Common mistakes to avoid

- **Wrong node names:** Node class names are case-sensitive and often different from display names. Always verify via API.
- **Wrong input names:** Input parameter names must match exactly. Check the API, don't assume.
- **Type mismatches:** If node A outputs `SAM3_MODEL` and node B expects `SAM3_MODEL`, the connection works. If the types differ even slightly, it won't.
- **Missing optional inputs:** Optional inputs don't need connections but may need widget values in `widgets_values`.
- **Stale info:** If custom nodes were just installed/modified, ComfyUI needs a restart before the API reflects changes. Run `python comfyui_nodes.py fetch` to refresh the cache.

### Workflow JSON format

ComfyUI uses two JSON formats:

**1. UI format** (what the frontend saves):
```json
{
  "last_node_id": 3,
  "last_link_id": 2,
  "nodes": [
    {
      "id": 1,
      "type": "NodeClassName",
      "pos": [0, 0],
      "size": [300, 200],
      "inputs": [
        {"name": "input_name", "type": "IMAGE", "link": null}
      ],
      "outputs": [
        {"name": "IMAGE", "type": "IMAGE", "links": [1]}
      ],
      "widgets_values": ["default_value"]
    }
  ],
  "links": [
    [1, 1, 0, 2, 0, "IMAGE"]
  ]
}
```

Link format: `[link_id, from_node_id, from_output_index, to_node_id, to_input_index, type]`

**2. API format** (what `/prompt` expects — used by `comfyui_api.py`):
```json
{
  "1": {
    "class_type": "NodeClassName",
    "inputs": {
      "input_name": "value_or_link"
    }
  }
}
```

Link format in API: `["node_id", output_index]`

**Remember: always query the API first. Never guess node names or interfaces.**
