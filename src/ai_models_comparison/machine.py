# Copyright 2026 Yauhen Bichel
# SPDX-License-Identifier: Apache-2.0
"""What this machine has: system memory, GPU memory, and whether they are one pool. From that, two budgets:
how many bytes of weights fit the GPU, and how many fit GPU plus system memory together.

Detection reads what every machine already exposes: /proc/meminfo, `sysctl hw.memsize`, `nvidia-smi`, the
amdgpu sysfs counters, and on Windows GlobalMemoryStatusEx. Nothing is installed and nothing is written. Every
figure can be overridden. What cannot be detected is said to be unknown: it is never taken for zero.
"""
from __future__ import annotations

import ctypes
import glob
import math
import os
import platform
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any

GIB = 2**30


@dataclass(slots=True)
class Machine:
    ram: int            # bytes of system memory the OS sees
    gpu: int            # bytes of GPU memory (0 when there is no GPU worth using)
    kind: str           # nvidia | amd | apple | none | given | unknown (system memory could not be detected)
    unified: bool       # the GPU's memory is the system's memory (Apple silicon; an AMD APU without a carve-out)
    note: str = ""
    warning: str = ""   # what detection could not find out, and what to give instead


@dataclass(slots=True)
class Budgets:
    gpu: int            # bytes of weights that fit the GPU, after its reserve
    memory: int         # bytes of weights that fit GPU and system memory together, after both reserves
    gpu_reserve: int
    ram_reserve: int


Run = Callable[[list[str]], str]
ReadText = Callable[[str], str]


Probe = Callable[[], int]
GIVE = "give gpu_gb and ram_gb (--gpu-gb and --ram-gb, or gpu_gb and ram_gb in the file --config names)"


class Unknown(ValueError):
    """The machine's memory could not be detected and was not given: there is nothing to judge against."""


def _run(cmd: list[str]) -> str:
    """What the command printed; "" when it is not installed; what it complained of when it failed."""
    try:
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    except subprocess.SubprocessError:
        return "failed: no answer in 10 s"
    except OSError:
        return ""
    return done.stdout if done.returncode == 0 else f"failed: {(done.stderr or done.stdout).strip()[:200] or done.returncode}"


def _read(path: str) -> str:
    try:
        with open(path) as fh:
            return fh.read()
    except OSError:
        return ""


def _sysconf() -> int:
    """System memory as POSIX says it: Linux without /proc, the BSDs."""
    try:
        return int(os.sysconf("SC_PHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, ValueError, OSError):
        return 0


def _windows_ram() -> int:
    """System memory as Windows says it (GlobalMemoryStatusEx); 0 when it will not say."""
    class Status(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                    ("free", ctypes.c_ulonglong), ("page_total", ctypes.c_ulonglong), ("page_free", ctypes.c_ulonglong),
                    ("virtual_total", ctypes.c_ulonglong), ("virtual_free", ctypes.c_ulonglong), ("extended", ctypes.c_ulonglong)]

    try:
        status = Status()
        status.length = ctypes.sizeof(Status)
        ok = getattr(ctypes, "windll").kernel32.GlobalMemoryStatusEx(ctypes.byref(status))  # noqa: B009  absent off Windows
        return int(status.total) if ok else 0
    except (AttributeError, OSError):
        return 0


def _unknown(system: str, gpu: int = 0, note: str = "") -> Machine:
    return Machine(0, gpu, "unknown", False, note, f"system memory could not be detected on {system or 'this system'}: {GIVE}")


def _nvidia(run: Run) -> tuple[list[int], str]:
    """(MiB of each GPU with memory of its own, what else nvidia-smi said): "" nothing, "shared" a GPU that
    reports no memory of its own, otherwise its complaint."""
    out = run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"])
    lines = [x.strip() for x in out.splitlines() if x.strip()]
    sizes = [int(x) for x in lines if x.isdigit()]
    rest = [x for x in lines if not x.isdigit()]
    if rest and all("N/A" in x or "Not Supported" in x for x in rest):
        return sizes, "shared"
    return sizes, " ".join(rest)[:200]


def detect(run: Run = _run, read: ReadText = _read, system: str | None = None,
           amd_cards: list[str] | None = None, sysconf: Probe = _sysconf, windows_ram: Probe = _windows_ram) -> Machine:
    system = system or platform.system()
    if system == "Darwin":
        size = run(["sysctl", "-n", "hw.memsize"]).strip()
        ram = int(size) if size.isdigit() else 0
        if ram <= 0:
            return _unknown("macOS (sysctl hw.memsize gave no answer)")
        uname = run(["uname", "-m"]).strip()
        arm = "arm" in (platform.machine() if not uname or uname.startswith("failed:") else uname).lower()
        if arm:
            return Machine(ram, ram, "apple", True, "Apple silicon: one pool; the GPU may use most of it")
        return Machine(ram, 0, "none", False, "Intel Mac: no GPU used")
    ram = 0
    if system == "Windows":
        ram = windows_ram()
    else:
        for line in read("/proc/meminfo").splitlines():
            if line.startswith("MemTotal:"):
                ram = int(line.split()[1]) * 1024
                break
        ram = ram or sysconf()
    nvidia, said = _nvidia(run)
    if ram <= 0:
        return _unknown(system, sum(nvidia) * 2**20, f"{len(nvidia)} NVIDIA GPU(s)" if nvidia else "")
    if nvidia:
        return Machine(ram, sum(nvidia) * 2**20, "nvidia", False, f"{len(nvidia)} NVIDIA GPU(s)")
    if said == "shared":
        # NVIDIA's DGX Spark notes: nvidia-smi shows no memory for an integrated GPU, which has none of its own and
        # shares the system's (https://docs.nvidia.com/dgx/dgx-spark/known-issues.html); the Jetson boards are the same
        return Machine(ram, ram, "nvidia", True, "NVIDIA GPU without memory of its own (nvidia-smi reports none): it shares system memory",
                       "the GPU's memory was taken to be the system's; give gpu_gb if this GPU has its own")
    if system == "Windows":
        return Machine(ram, 0, "none", False, "no NVIDIA GPU found",
                       "on Windows only NVIDIA GPUs are detected: give gpu_gb if this machine has another")
    cards = amd_cards if amd_cards is not None else sorted(glob.glob("/sys/class/drm/card[0-9]*/device"))
    best_vram = best_gtt = 0
    for card in cards:
        vram = read(f"{card}/mem_info_vram_total").strip()
        gtt = read(f"{card}/mem_info_gtt_total").strip()
        if vram.isdigit() and int(vram) > best_vram:
            best_vram, best_gtt = int(vram), int(gtt) if gtt.isdigit() else 0
    if best_vram >= 4 * GIB:
        return Machine(ram, best_vram, "amd", False, "AMD GPU memory (a discrete card, or an APU with a BIOS carve-out)")
    if best_gtt >= 4 * GIB:
        return Machine(ram, best_gtt, "amd", True, "AMD APU without a carve-out: the GPU maps system memory (GTT)")
    if said:
        return Machine(ram, 0, "none", False, "no GPU memory found: models are judged for the CPU, from system memory",
                       f"nvidia-smi is installed but did not answer ({said}): give gpu_gb if this machine has a GPU")
    return Machine(ram, 0, "none", False, "no GPU found: models run on the CPU, from system memory")


def budgets(m: Machine, gpu_reserve_gib: float | None = None, ram_reserve_gib: float | None = None) -> Budgets:
    """The defaults leave room for the context cache and the rest of the machine; both can be set."""
    if m.unified:
        # one pool: what the GPU takes, the system loses. Leave the larger of 8 GiB and a quarter of it.
        reserve = int((ram_reserve_gib if ram_reserve_gib is not None else max(8.0, m.ram / GIB * 0.25)) * GIB)
        fit = max(0, min(m.gpu, m.ram) - reserve)
        return Budgets(fit, fit, 0, reserve)
    gpu_reserve = int((gpu_reserve_gib if gpu_reserve_gib is not None else max(1.0, m.gpu / GIB * 0.08)) * GIB) if m.gpu else 0
    ram_reserve = int((ram_reserve_gib if ram_reserve_gib is not None else max(8.0, m.ram / GIB * 0.25)) * GIB)
    gpu = max(0, m.gpu - gpu_reserve)
    return Budgets(gpu, gpu + max(0, m.ram - ram_reserve), gpu_reserve, ram_reserve)


SPEC = ("gpu_gb", "ram_gb", "unified", "gpu_reserve_gb", "ram_reserve_gb")   # how any interface describes a machine


def _number(spec: Mapping[str, Any], key: str) -> float | None:
    if spec.get(key) is None or spec[key] == "":
        return None
    try:
        v = float(spec[key])
    except (TypeError, ValueError):
        raise ValueError(f"{key} must be a number of GiB") from None
    if not math.isfinite(v) or v < 0 or v > 1_000_000:
        raise ValueError(f"{key} must be between 0 and 1000000 GiB")
    return v


def resolve(spec: Mapping[str, Any], detected: Machine | None = None) -> tuple[Machine, Budgets]:
    """The machine to judge for: the one detected, changed by what the spec gives. A machine given by hand is
    two pools unless it says `unified`; with `unified` and only `ram_gb`, the GPU may use all of it."""
    m = replace(detected) if detected is not None else detect()
    gpu_gb, ram_gb = _number(spec, "gpu_gb"), _number(spec, "ram_gb")
    undetected = m.ram <= 0
    if gpu_gb is not None or ram_gb is not None:
        m.unified, m.kind, m.note = False, "given", "as given, not detected"
        if not undetected or ram_gb is not None:
            m.warning = ""
    if gpu_gb is not None:
        m.gpu = int(gpu_gb * GIB)
    if ram_gb is not None:
        m.ram = int(ram_gb * GIB)
    elif undetected:   # nothing to judge against: say so, do not take it for a machine with no memory
        m.kind = "unknown"
    unified = spec.get("unified")
    if unified is not None and unified != "":
        m.unified = unified if isinstance(unified, bool) else str(unified).lower() in ("1", "true", "yes", "on")
        if m.unified and gpu_gb is None and ram_gb is not None:
            m.gpu = m.ram
    return m, budgets(m, _number(spec, "gpu_reserve_gb"), _number(spec, "ram_reserve_gb"))


def as_dict(m: Machine, b: Budgets) -> dict[str, Any]:
    def g(v: int) -> float:
        return round(v / GIB, 1)

    return {"ram_gib": g(m.ram), "gpu_gib": g(m.gpu), "kind": m.kind, "unified": m.unified, "note": m.note, "warning": m.warning,
            "fits_gpu_gib": g(b.gpu), "fits_memory_gib": g(b.memory), "gpu_reserve_gib": g(b.gpu_reserve),
            "ram_reserve_gib": g(b.ram_reserve)}


def describe(m: Machine, b: Budgets) -> str:
    def g(v: int) -> str:
        return f"{v / GIB:.1f} GiB"

    if m.kind == "unknown":
        return f"this machine     unknown: {m.warning}"
    lines = [f"system memory   {g(m.ram)}", f"GPU memory      {g(m.gpu)}  ({m.kind}{', unified' if m.unified else ''})"]
    if m.note:
        lines.append(f"                {m.note}")
    if m.warning:
        lines.append(f"warning         {m.warning}")
    lines += [f"fits the GPU    up to {g(b.gpu)} of weights" if b.gpu else "fits the GPU    nothing: no GPU",
              f"fits in memory  up to {g(b.memory)} of weights (GPU and system memory together)",
              f"reserves        GPU {g(b.gpu_reserve)}, system {g(b.ram_reserve)}"]
    return "\n".join(lines)


def known(m: Machine) -> Machine:
    """The machine, when there is one to judge against; Unknown, saying what to give, when its memory is not known."""
    if m.kind == "unknown":
        raise Unknown(m.warning or f"this machine's memory is not known: {GIVE}")
    return m
