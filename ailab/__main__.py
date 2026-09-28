"""Command line entry point: `ailab` (or `python -m ailab`)."""

from __future__ import annotations

import argparse
import os
import sys


def _prime_offload() -> None:
    """Draw with the NVIDIA GPU on a hybrid laptop (must run before Qt/GL starts).

    NVIDIA's PRIME render offload works through GLX, so on Wayland the window goes through
    XWayland (like `prime-run`). Forcing NVIDIA's EGL vendor under a Wayland compositor that
    runs on the iGPU fails with EGL_BAD_MATCH, so we don't do that."""
    os.environ.setdefault("__NV_PRIME_RENDER_OFFLOAD", "1")
    os.environ.setdefault("__GLX_VENDOR_LIBRARY_NAME", "nvidia")
    if os.environ.get("WAYLAND_DISPLAY") and os.environ.get("DISPLAY"):
        os.environ["QT_QPA_PLATFORM"] = "xcb"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="ailab", description="Algorithmic Intelligence Lab")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], help="compute backend")
    ap.add_argument("--render-gpu", choices=["system", "nvidia"],
                    help="which GPU draws the window on hybrid laptops")
    ap.add_argument("--chapter", help="open this chapter id directly, e.g. swarms.fish-school")
    ap.add_argument("--sysinfo", action="store_true", help="print the system check and exit")
    ap.add_argument("--list", action="store_true", help="list chapters and exit")
    ap.add_argument("--snapshot", metavar="PNG", help="render --chapter headlessly and exit")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--size", default="1600x900")
    ap.add_argument("--no-boot", action="store_true", help="skip the system check screen")
    ap.add_argument("--seed", type=int, metavar="N",
                    help="start with this seed (default: a fresh random seed each launch)")
    args = ap.parse_args(argv)
    # Windows consoles/pipes may not be UTF-8: never crash on "·" or "→" in the report.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    from .core import compute, settings, system

    prefs = settings.load()
    device = args.device or prefs.get("device") or "auto"
    info = system.probe()

    if args.sysinfo:
        compute.init(device, info)
        print(system.format_report(info))
        return 0
    if args.list:
        from .core.catalog import discover

        cat = discover()
        for t in cat.tracks:
            print(f"{t.title}")
            for c in t.chapters:
                print(f"  {c.id:<28} {c.title}  [{c.compute}]")
        for t in cat.planned:
            print(f"{t.title}  (planned)")
        return 0
    if args.snapshot:
        from .app.snapshot import snapshot
        from .core.catalog import discover

        if not args.chapter:
            ap.error("--snapshot needs --chapter")
        dev = compute.init(device, info)
        w, h = (int(v) for v in args.size.split("x"))
        print(snapshot(discover().chapters[args.chapter], args.snapshot, dev, args.frames,
                       (w, h), seed=1 if args.seed is None else args.seed))
        return 0

    compute.init(device, info)
    print(system.format_report(info), flush=True)
    render_gpu = args.render_gpu or prefs.get("render_gpu", "system")
    if render_gpu == "nvidia" and info.is_hybrid:
        if sys.platform.startswith("linux"):
            _prime_offload()
        else:
            print("  --render-gpu nvidia is Linux-only; on Windows choose the GPU for "
                  "python.exe in Settings → Display → Graphics.", flush=True)
    from .app.main import run_app

    return run_app(info, device, args.chapter or None, boot=not args.no_boot,
                   seed=args.seed)


if __name__ == "__main__":
    sys.exit(main())
