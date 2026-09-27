"""System awareness: discover the hardware this launch runs on.

Every launch probes the machine (CPU, RAM, every GPU, NVIDIA driver/CUDA, display
session, optional tools) so the app can
  * pick a compute backend (CUDA via Warp, else CPU),
  * scale simulation defaults to the hardware,
  * and *show* the learner what their computer is doing.

Linux reads /proc, /sys and lspci; Windows asks the OS (one CIM query + the memory API);
macOS uses sysctl. nvidia-smi is used everywhere it exists. Every step is optional: a
missing tool degrades to "unknown", never to a crash.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field

IS_WINDOWS = sys.platform == "win32"
IS_MAC = sys.platform == "darwin"

_PCI_VENDORS = {"0x10de": "NVIDIA", "0x8086": "Intel", "0x1002": "AMD"}


@dataclass
class GPU:
    vendor: str
    name: str
    driver: str = ""
    vram_mb: int = 0
    pci: str = ""
    cuda_capable: bool = False
    compute_capability: str = ""

    @property
    def short(self) -> str:
        mem = f" · {self.vram_mb / 1024:.0f} GB" if self.vram_mb else ""
        return f"{self.name}{mem}"


@dataclass
class SystemInfo:
    os_name: str = ""
    kernel: str = ""
    cpu_model: str = ""
    cpu_cores: int = 0
    cpu_threads: int = 0
    ram_total_mb: int = 0
    ram_available_mb: int = 0
    gpus: list[GPU] = field(default_factory=list)
    nvidia_driver: str = ""
    cuda_driver_version: str = ""
    session: str = ""
    desktop: str = ""
    python: str = ""
    tools: dict[str, bool] = field(default_factory=dict)
    # Filled in later by the compute + render layers.
    compute_device: str = "cpu"
    compute_name: str = ""
    warp_version: str = ""
    gl_renderer: str = ""
    gl_version: str = ""
    gl_vendor: str = ""

    # ------------------------------------------------------------------ derived
    @property
    def cuda_gpu(self) -> GPU | None:
        return next((g for g in self.gpus if g.cuda_capable), None)

    @property
    def is_hybrid(self) -> bool:
        return len({g.vendor for g in self.gpus}) > 1

    @property
    def on_gpu(self) -> bool:
        return self.compute_device.startswith("cuda")

    @property
    def render_gpu_label(self) -> str:
        r = self.gl_renderer.lower()
        if not r:
            return "not initialised"
        if "llvmpipe" in r or "softpipe" in r:
            return "software (CPU) rasteriser"
        for g in self.gpus:
            if g.vendor.lower() in r or g.vendor.lower() in self.gl_vendor.lower():
                kind = "dGPU" if g.vendor == "NVIDIA" or g.vram_mb else "iGPU"
                return f"{g.vendor} {kind}"
        return self.gl_vendor or "GPU"

    def summary_lines(self) -> list[tuple[str, str]]:
        """Human readable (label, value) rows for the UI and the CLI banner."""
        rows = [
            ("OS", f"{self.os_name} · {'build' if IS_WINDOWS else 'kernel'} {self.kernel}"),
            ("CPU", f"{self.cpu_model} · {self.cpu_cores} cores / {self.cpu_threads} threads"),
            ("Memory", f"{self.ram_total_mb / 1024:.1f} GB total · "
                       f"{self.ram_available_mb / 1024:.1f} GB free"),
        ]
        for i, g in enumerate(self.gpus):
            extra = []
            if g.driver:
                extra.append(f"driver {g.driver}")
            if g.compute_capability:
                extra.append(f"sm_{g.compute_capability.replace('.', '')}")
            if g.cuda_capable:
                extra.append("CUDA")
            rows.append((f"GPU {i}", f"{g.vendor} {g.short}" + (f" · {' · '.join(extra)}"
                                                                 if extra else "")))
        if not self.gpus:
            rows.append(("GPU", "none detected"))
        if self.cuda_driver_version:
            rows.append(("CUDA driver", self.cuda_driver_version))
        rows.append(("Compute", f"{self.compute_label}"))
        if self.gl_renderer:
            rows.append(("Rendering", f"{self.gl_renderer} (OpenGL {self.gl_version.split()[0]})"))
        rows.append(("Session", f"{self.session or 'unknown'} · {self.desktop or 'unknown'}"))
        rows.append(("Python", self.python + (f" · Warp {self.warp_version}"
                                              if self.warp_version else "")))
        have = [k for k, v in self.tools.items() if v]
        missing = [k for k, v in self.tools.items() if not v]
        rows.append(("Tools", ", ".join(have) + (f"  (missing: {', '.join(missing)})"
                                                 if missing else "")))
        return rows

    @property
    def compute_label(self) -> str:
        if self.on_gpu:
            return f"CUDA · {self.compute_name or self.compute_device}"
        return f"CPU · {self.cpu_model or 'host'} (Warp CPU backend)"

    def advice(self) -> list[str]:
        """Plain-language notes about this machine, shown at launch."""
        notes = []
        if self.on_gpu:
            notes.append("Heavy simulations run on the NVIDIA GPU; small ones stay on the CPU.")
        elif self.cuda_gpu:
            notes.append("A CUDA GPU exists but compute is on the CPU (forced or failed init).")
        else:
            notes.append("No CUDA GPU found: every chapter runs on the CPU fallback "
                         "with smaller default agent counts.")
        if self.is_hybrid:
            if IS_WINDOWS:
                notes.append("Hybrid graphics: Windows picks the drawing GPU. To draw on the "
                             "NVIDIA GPU, set python.exe to 'High performance' in Settings → "
                             "Display → Graphics.")
            else:
                notes.append("Hybrid graphics: the desktop GPU draws the picture, the NVIDIA GPU "
                             "does the maths. Use --render-gpu nvidia to draw on the dGPU too.")
        if "llvmpipe" in self.gl_renderer.lower():
            notes.append("OpenGL is software-rendered: visuals will be slow. Check GPU drivers.")
        if not self.tools.get("latex"):
            notes.append("LaTeX not found: equations use the built-in math renderer "
                         "(install TeX Live or MiKTeX for full typesetting).")
        return notes


# ---------------------------------------------------------------------- probes
def _run(cmd: list[str], timeout: float = 3.0) -> str:
    # No console window flashing up on Windows when launched from a shortcut.
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False,
                             creationflags=flags)
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _read(path: str) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return ""


def _os_name() -> str:
    if IS_WINDOWS:
        return f"Windows {platform.release()}"
    if IS_MAC:
        return f"macOS {platform.mac_ver()[0]}"
    for line in _read("/etc/os-release").splitlines():
        if line.startswith("PRETTY_NAME="):
            return line.split("=", 1)[1].strip('"')
    return platform.system()


def _cpu() -> tuple[str, int, int]:
    model, cores_per_socket, sockets = "", 0, 1
    info = _read("/proc/cpuinfo")
    physical = set()
    for line in info.splitlines():
        key, _, val = line.partition(":")
        key, val = key.strip(), val.strip()
        if key == "model name" and not model:
            model = val
        elif key == "cpu cores" and not cores_per_socket:
            cores_per_socket = int(val or 0)
        elif key == "physical id":
            physical.add(val)
    sockets = max(1, len(physical))
    threads = os.cpu_count() or 0
    model = " ".join(model.replace("(R)", "").replace("(TM)", "").split())
    return model or platform.processor(), (cores_per_socket * sockets) or threads, threads


def _memory() -> tuple[int, int]:
    if IS_WINDOWS:
        return _memory_windows()
    if IS_MAC:
        total = int(_run(["sysctl", "-n", "hw.memsize"]) or 0) // 2**20
        return total, 0
    total = avail = 0
    for line in _read("/proc/meminfo").splitlines():
        if line.startswith("MemTotal:"):
            total = int(line.split()[1]) // 1024
        elif line.startswith("MemAvailable:"):
            avail = int(line.split()[1]) // 1024
    return total, avail


# ------------------------------------------------------------------- windows
_WIN_QUERY = (
    "$c = Get-CimInstance Win32_Processor | Select-Object -First 1 Name,NumberOfCores;"
    "$g = @(Get-CimInstance Win32_VideoController | Select-Object Name,AdapterCompatibility,"
    "DriverVersion,AdapterRAM);"
    "@{cpu=$c; gpus=$g} | ConvertTo-Json -Depth 3 -Compress"
)


def _memory_windows() -> tuple[int, int]:
    import ctypes

    class MemoryStatus(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

    st = MemoryStatus()
    st.dwLength = ctypes.sizeof(MemoryStatus)
    try:
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
            return st.ullTotalPhys // 2**20, st.ullAvailPhys // 2**20
    except (AttributeError, OSError):
        pass
    return 0, 0


def parse_windows_query(text: str) -> tuple[str, int, list[GPU]]:
    """Parse the JSON printed by ``_WIN_QUERY``: (cpu model, physical cores, gpus)."""
    try:
        data = json.loads(text) if text else {}
    except ValueError:
        return "", 0, []
    cpu = data.get("cpu") or {}
    model = " ".join(str(cpu.get("Name", "")).replace("(R)", "").replace("(TM)", "").split())
    cores = int(cpu.get("NumberOfCores") or 0)
    raw = data.get("gpus") or []
    if isinstance(raw, dict):          # PowerShell unwraps one-element arrays
        raw = [raw]
    gpus = []
    for g in raw:
        name = str(g.get("Name") or "").strip()
        if not name or "Basic Display" in name or "Remote" in name:
            continue
        vendor = next((v for v in ("NVIDIA", "AMD", "Intel") if v.lower() in
                       (name + str(g.get("AdapterCompatibility") or "")).lower()), "GPU")
        if vendor == "AMD" or "Advanced Micro" in str(g.get("AdapterCompatibility") or ""):
            vendor = "AMD"
        for prefix in ("NVIDIA ", "AMD ", "Intel(R) ", "Intel "):
            name = name.removeprefix(prefix)
        # AdapterRAM is a uint32: it saturates at 4 GB, so nvidia-smi overrides it for NVIDIA.
        ram = int(g.get("AdapterRAM") or 0) // 2**20
        gpus.append(GPU(vendor=vendor, name=name, driver=str(g.get("DriverVersion") or ""),
                        vram_mb=ram if vendor != "Intel" else 0))
    return model, cores, gpus


def _windows_hw() -> tuple[str, int, list[GPU]]:
    out = _run(["powershell", "-NoProfile", "-NonInteractive", "-Command", _WIN_QUERY],
               timeout=8.0)
    return parse_windows_query(out)


# --------------------------------------------------------------------- linux
def _pci_gpus() -> list[GPU]:
    """All display controllers, from lspci names if available, else /sys vendor ids."""
    gpus: list[GPU] = []
    out = _run(["lspci", "-mm"]) if shutil.which("lspci") else ""
    for line in out.splitlines():
        if not any(k in line for k in ('"VGA', '"3D', '"Display')):
            continue
        parts = [p.strip('"') for p in _split_quoted(line)]
        if len(parts) >= 4:
            vendor = parts[2].split()[0].replace("Corporation", "").strip()
            name = parts[3]
            if "[" in name and "]" in name:
                name = name[name.index("[") + 1:name.rindex("]")]
            gpus.append(GPU(vendor=vendor, name=name, pci=parts[0]))
    if gpus:
        return gpus
    import glob

    for card in sorted(glob.glob("/sys/class/drm/card[0-9]")):
        vid = _read(f"{card}/device/vendor").strip()
        if vid in _PCI_VENDORS:
            gpus.append(GPU(vendor=_PCI_VENDORS[vid], name=f"{_PCI_VENDORS[vid]} GPU"))
    return gpus


def _split_quoted(line: str) -> list[str]:
    parts, cur, quoted = [], "", False
    for ch in line:
        if ch == '"':
            quoted = not quoted
            cur += ch
        elif ch == " " and not quoted:
            if cur:
                parts.append(cur)
            cur = ""
        else:
            cur += ch
    if cur:
        parts.append(cur)
    return parts


def _nvidia(gpus: list[GPU]) -> str:
    """Enrich NVIDIA entries using nvidia-smi. Returns the driver version."""
    if not shutil.which("nvidia-smi"):
        return ""
    out = _run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version,compute_cap",
                "--format=csv,noheader,nounits"])
    driver = ""
    nv = [g for g in gpus if g.vendor == "NVIDIA"]
    for i, line in enumerate(out.splitlines()):
        fields = [f.strip() for f in line.split(",")]
        if len(fields) < 3:
            continue
        name, mem, driver = fields[0], fields[1], fields[2]
        cc = fields[3] if len(fields) > 3 and fields[3] not in ("[N/A]", "") else ""
        name = name.replace("NVIDIA ", "")
        if i < len(nv):
            g = nv[i]
        else:
            g = GPU(vendor="NVIDIA", name=name)
            gpus.append(g)
        g.name, g.driver, g.compute_capability = name, driver, cc
        g.vram_mb = int(float(mem)) if mem.replace(".", "").isdigit() else 0
        g.cuda_capable = True
    return driver


def probe() -> SystemInfo:
    """Fast hardware probe (~50 ms). Compute/GL fields are filled in later."""
    info = SystemInfo()
    info.os_name = _os_name()
    info.kernel = platform.version() if IS_WINDOWS else platform.release()
    info.cpu_threads = os.cpu_count() or 1
    if IS_WINDOWS:
        info.cpu_model, info.cpu_cores, info.gpus = _windows_hw()
        info.cpu_model = info.cpu_model or platform.processor()
        info.cpu_cores = info.cpu_cores or info.cpu_threads
    elif IS_MAC:
        info.cpu_model = _run(["sysctl", "-n", "machdep.cpu.brand_string"]) or platform.processor()
        info.cpu_cores = int(_run(["sysctl", "-n", "hw.physicalcpu"]) or 0) or info.cpu_threads
        info.gpus = [GPU(vendor="Apple", name=info.cpu_model)] if "Apple" in info.cpu_model else []
    else:
        info.cpu_model, info.cpu_cores, info.cpu_threads = _cpu()
        info.gpus = _pci_gpus()
    info.ram_total_mb, info.ram_available_mb = _memory()
    info.nvidia_driver = _nvidia(info.gpus)
    # Integrated GPUs first is confusing; list the CUDA device first.
    info.gpus.sort(key=lambda g: not g.cuda_capable)
    if IS_WINDOWS or IS_MAC:
        info.session, info.desktop = "native", platform.system()
    else:
        info.session = os.environ.get("XDG_SESSION_TYPE", "")
        info.desktop = os.environ.get("XDG_CURRENT_DESKTOP", "")
    info.python = f"Python {sys.version.split()[0]}"
    info.tools = {t: shutil.which(t) is not None
                  for t in ("latex", "dvisvgm", "ffmpeg", "nvidia-smi")}
    try:
        import manim  # noqa: F401

        info.tools["manim"] = True
    except Exception:
        info.tools["manim"] = False
    return info


def format_report(info: SystemInfo) -> str:
    width = max(len(k) for k, _ in info.summary_lines())
    lines = ["Algorithmic Intelligence Lab: system check", "-" * 44]
    lines += [f"  {k:<{width}}  {v}" for k, v in info.summary_lines()]
    notes = info.advice()
    if notes:
        lines.append("")
        lines += [f"  * {n}" for n in notes]
    return "\n".join(lines)
