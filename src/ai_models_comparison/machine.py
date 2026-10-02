# Copyright 2026 Yauhen Bichel
# SPDX-License-Identifier: Apache-2.0
"""What this machine has: system memory, GPU memory, and whether they are one pool. From that, two budgets:
how many bytes of weights fit the GPU, and how many fit GPU plus system memory together.

Detection reads what every machine already exposes: /proc/meminfo, `sysctl hw.memsize`, `nvidia-smi`, and
the amdgpu sysfs counters. Nothing is installed and nothing is written. Every figure can be overridden.
"""
from __future__ import annotations

import glob
import platform
import subprocess
from collections.abc import Callable
from dataclasses import dataclass

GIB = 2**30


@dataclass(slots=True)
class Machine:
    ram: int            # bytes of system memory the OS sees
    gpu: int            # bytes of GPU memory (0 when there is no GPU worth using)
    kind: str           # nvidia | amd | apple | none
    unified: bool       # the GPU's memory is the system's memory (Apple silicon; an AMD APU without a carve-out)
    note: str = ""


@dataclass(slots=True)
class Budgets:
    gpu: int            # bytes of weights that fit the GPU, after its reserve
    memory: int         # bytes of weights that fit GPU and system memory together, after both reserves
    gpu_reserve: int
    ram_reserve: int


Run = Callable[[list[str]], str]
ReadText = Callable[[str], str]


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def _read(path: str) -> str:
    try:
        with open(path) as fh:
            return fh.read()
    except OSError:
        return ""


def detect(run: Run = _run, read: ReadText = _read, system: str | None = None,
           amd_cards: list[str] | None = None) -> Machine:
    system = system or platform.system()
    if system == "Darwin":
        ram = int(run(["sysctl", "-n", "hw.memsize"]).strip() or 0)
        arm = "arm" in (run(["uname", "-m"]).strip() or platform.machine()).lower()
        if arm:
            return Machine(ram, ram, "apple", True, "Apple silicon: one pool; the GPU may use most of it")
        return Machine(ram, 0, "none", False, "Intel Mac: no GPU used")
    ram = 0
    for line in read("/proc/meminfo").splitlines():
        if line.startswith("MemTotal:"):
            ram = int(line.split()[1]) * 1024
            break
    nvidia = [int(x) for x in run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"]).split() if x.isdigit()]
    if nvidia:
        return Machine(ram, sum(nvidia) * 2**20, "nvidia", False, f"{len(nvidia)} NVIDIA GPU(s)")
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


def describe(m: Machine, b: Budgets) -> str:
    def g(v: int) -> str:
        return f"{v / GIB:.1f} GiB"

    lines = [f"system memory   {g(m.ram)}", f"GPU memory      {g(m.gpu)}  ({m.kind}{', unified' if m.unified else ''})"]
    if m.note:
        lines.append(f"                {m.note}")
    lines += [f"fits the GPU    up to {g(b.gpu)} of weights" if b.gpu else "fits the GPU    nothing: no GPU",
              f"fits in memory  up to {g(b.memory)} of weights (GPU and system memory together)",
              f"reserves        GPU {g(b.gpu_reserve)}, system {g(b.ram_reserve)}"]
    return "\n".join(lines)
