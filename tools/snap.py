"""Render a chapter headlessly to PNG. Example:
    uv run tools/snap.py swarms.fish-school out.png --frames 300 --zoom 4 --center focus
"""
import argparse
import json

from ailab.app.snapshot import snapshot
from ailab.core import compute
from ailab.core.catalog import discover

ap = argparse.ArgumentParser()
ap.add_argument("chapter")
ap.add_argument("out")
ap.add_argument("--frames", type=int, default=240)
ap.add_argument("--zoom", type=float, default=1.0)
ap.add_argument("--center", default=None)
ap.add_argument("--overlays", default="default")
ap.add_argument("--mouse", default="auto")
ap.add_argument("--device", default="auto")
ap.add_argument("--size", default="1600x900")
ap.add_argument("--param", action="append", default=[], help="key=value")
a = ap.parse_args()
params = {}
for kv in a.param:
    k, v = kv.split("=", 1)
    params[k] = json.loads(v) if v[:1] in "0123456789-.tf[" else v
w, h = map(int, a.size.split("x"))
dev = compute.init(a.device)
info = discover().chapters[a.chapter]
print(snapshot(info, a.out, dev, a.frames, (w, h), mouse=a.mouse, overlays=a.overlays,
               zoom=a.zoom, params=params, center=a.center))
