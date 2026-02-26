import requests, sys

BASE = "http://localhost:7861"
SHOT = "MOK1100_test_v001_20260226_100037"
PASS = []
FAIL = []

def check(name, condition, detail=""):
    if condition:
        PASS.append(name)
        print(f"  PASS  {name}" + (f" — {detail}" if detail else ""))
    else:
        FAIL.append(name)
        print(f"  FAIL  {name}" + (f" — {detail}" if detail else ""))

# ── /api/shots ────────────────────────────────────────────────────────────────
r = requests.get(f"{BASE}/api/shots")
check("/api/shots HTTP 200", r.status_code == 200)
shots = r.json().get("shots", [])
check("/api/shots returns shots", len(shots) > 0, f"{len(shots)} shots")
check(f"test shot in list", SHOT in shots, SHOT)

# ── /api/scene ────────────────────────────────────────────────────────────────
r = requests.get(f"{BASE}/api/scene/{SHOT}")
check("/api/scene HTTP 200", r.status_code == 200)
d = r.json()
check("ply_url non-null", d.get("ply_url") is not None, str(d.get("ply_url")))
check("total_frames > 0", d.get("total_frames", 0) > 0, str(d.get("total_frames")))
check("all frames registered",
      d.get("total_registered") == d.get("total_frames"),
      f"{d.get('total_registered')}/{d.get('total_frames')}")
check("fps set correctly", d.get("fps", 0) > 0, f"{d.get('fps')} fps")
check("solve_error present", d.get("solve_error") is not None, str(d.get("solve_error")))

intr = d.get("intrinsics", {})
check("intrinsics.model present", bool(intr.get("model")), intr.get("model"))
check("intrinsics.focal_length_x present",
      intr.get("focal_length_x") is not None,
      str(intr.get("focal_length_x")))
check("intrinsics.fov_y_deg present",
      intr.get("fov_y_deg") is not None,
      f"{intr.get('fov_y_deg'):.2f} deg" if intr.get("fov_y_deg") else "None")

# Simulate what the viewer does: fl = intr.focal_length || intr.focal_length_x
fl = intr.get("focal_length") or intr.get("focal_length_x")
check("viewer focal_length resolves", fl is not None, f"{fl:.1f} px" if fl else "None")

# ── /api/ply ──────────────────────────────────────────────────────────────────
r = requests.get(f"{BASE}/api/ply/{SHOT}", stream=True)
check("/api/ply HTTP 200", r.status_code == 200)
check("/api/ply content-type octet-stream",
      "octet-stream" in r.headers.get("content-type", ""),
      r.headers.get("content-type"))
first_bytes = next(r.iter_content(chunk_size=4))
check("/api/ply PLY magic bytes", first_bytes[:3] == b"ply", repr(first_bytes[:3]))
content_length = int(r.headers.get("content-length", 0))
check("/api/ply has content-length", content_length > 0, f"{content_length:,} bytes")

# ── /api/ply — 404 for nonexistent shot ──────────────────────────────────────
r404 = requests.get(f"{BASE}/api/ply/nonexistent_shot_xyz")
check("/api/ply 404 for missing shot", r404.status_code == 404)

# ── viewer.html — code pattern checks ────────────────────────────────────────
r = requests.get(f"{BASE}/viewer")
check("/viewer HTTP 200", r.status_code == 200)
html = r.text
check("viewer: try/catch around buildScene",
      "try {" in html and "await buildScene" in html and "buildScene error" in html)
check("viewer: null rotation_matrix guard",
      "!frame.rotation_matrix" in html)
check("viewer: Array.isArray rotT guard",
      "Array.isArray(rotT)" in html)
check("viewer: PLY error uses console.error",
      'console.error("PLY load error' in html)
check("viewer: focal_length_x fallback",
      "intr.focal_length_x" in html)
check("viewer: hideLoading after try/catch",
      "hideLoading();  // always runs" in html)

# ── app.py — code pattern checks ─────────────────────────────────────────────
with open("/home/krabiboy/postviz-pipeline/app.py") as f:
    src = f.read()
check("app.py: gr.Markdown for solve_quality",
      'gr.Markdown(value="", visible=True)' in src)
check("app.py: gr.Accordion for solve_log",
      'gr.Accordion("COLMAP log"' in src)
check("app.py: no visible=True in final yield for solve_quality",
      'gr.update(value=quality, visible=True)' not in src)
check("app.py: failure yield uses Markdown bold",
      '**SOLVE FAILED**' in src)

# ── viewer_server.py — find_ply checks ───────────────────────────────────────
with open("/home/krabiboy/postviz-pipeline/viewer_server.py") as f:
    src = f.read()
check("viewer_server: find_ply() defined", "def find_ply(" in src)
after_def = src.split("def find_ply")[1]
check("viewer_server: no hardcoded fused.ply after find_ply",
      src.count('"fused.ply"') == 1)  # only inside find_ply itself
check("viewer_server: ply_path.name used", "filename=ply_path.name" in src)

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print(f"Results: {len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("FAILED:")
    for f in FAIL:
        print(f"  - {f}")
    sys.exit(1)
else:
    print("All checks passed.")
