#!/usr/bin/env bash
set -euo pipefail

# Batch runner for workzone-setup-yolo-orin repo.
# Assumes the Python environment already has dependencies installed.

ROOT="/home/cvrr/projects/workzone_metrics"
REPO="$ROOT/workzone-setup-yolo-orin/workzone-setup-yolo-orin"
VIDEOS="$ROOT/ROADWORK_data/videos/videos_compressed"
GT_JSON="$ROOT/workzone_annotations.json"
OUT="$REPO/outputs/batch"
STRIDE="${STRIDE:-2}"

CONFIG_SRC="$REPO/configs/jetson_config.yaml"
CONFIG_TMP="$REPO/configs/jetson_config_batch.yaml"

# Create a batch config that writes outputs where we expect.
python - <<PY
from pathlib import Path
import yaml

src = Path("$CONFIG_SRC")
dst = Path("$CONFIG_TMP")
data = yaml.safe_load(src.read_text())
data.setdefault("video", {})
data["video"]["output_dir"] = "$OUT"
data.setdefault("model", {})
data["model"]["path"] = str(Path("$REPO") / "weights" / "yolo12s_hardneg_1280.pt")
data["model"]["disable_trt_export"] = True
data.setdefault("hardware", {})
# Use GPU device 0 if available.
data["hardware"]["device"] = 0
data["hardware"]["half"] = True
data.setdefault("video", {})
data["video"]["stride"] = int("$STRIDE")
dst.write_text(yaml.safe_dump(data))
print("Wrote", dst)
PY

python - <<PY
import json
import subprocess
import sys
from pathlib import Path

root = Path("$VIDEOS")
out_root = Path("$OUT")
out_root.mkdir(parents=True, exist_ok=True)

with open("$GT_JSON", "r", encoding="utf-8") as f:
    videos = sorted(json.load(f).keys())

script = str(Path("$REPO") / "scripts/jetson_app_sota.py")

ok = 0
fail = 0
for name in videos:
    video_path = root / name
    if not video_path.exists():
        print("MISSING", video_path)
        fail += 1
        continue
    # jetson_app_sota writes outputs to config video.output_dir with name "sota_<video_name>.csv"
    csv_path = out_root / f"sota_{video_path.name}".replace(".mp4", ".csv")
    if csv_path.exists():
        print("SKIP (exists)", name)
        ok += 1
        continue
    cmd = [
        sys.executable,
        script,
        "--config",
        str(Path("$CONFIG_TMP")),
        "--input",
        str(video_path),
    ]
    print("RUN", name)
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        if csv_path.exists():
            print("WARN (crashed after output)", name)
            ok += 1
        else:
            print("FAILED", name)
            fail += 1
    else:
        ok += 1

print("DONE ok", ok, "fail", fail)
PY
