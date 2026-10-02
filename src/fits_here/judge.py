# Copyright 2026 Yauhen Bichel
# SPDX-License-Identifier: Apache-2.0
"""The verdict: which build of a model fits this machine's budgets, and what kind of fit that is.

  fits the GPU      a build of at least `gpu_min_bits` (4) within the GPU budget: the fast path
  fits in memory    a build of at least `memory_min_bits` (3) within GPU and system memory together: it runs
                    with part of the weights off the GPU (llama.cpp's --n-gpu-layers / --n-cpu-moe), slower
  low-bit only      only builds under `memory_min_bits` fit: such builds lose quality; measure before trusting
  too big           no build fits
  no GGUF yet       nothing to run with llama.cpp or Ollama today

A verdict is about memory. It says nothing about whether the model is good; that is what your own tests are for.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .hub import Build, Fetch, find_builds, parameters
from .machine import Budgets

ORDER = {"fits the GPU": 0, "fits in memory": 1, "low-bit only": 2, "too big": 3, "no GGUF yet": 4}


@dataclass(slots=True)
class Judged:
    model: str
    created: str
    kind: str
    params_b: float | None
    builds: list[Build] = field(default_factory=list)
    verdict: str = "no GGUF yet"
    pick: Build | None = None
    role: str = ""
    replaces: str = ""
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        pick = None if self.pick is None else {"repo": self.pick.repo, "quant": self.pick.quant, "gb": round(self.pick.size / 1e9, 1)}
        return {"model": self.model, "created": self.created, "kind": self.kind, "params_b": self.params_b,
                "verdict": self.verdict, "pick": pick, "role": self.role, "replaces": self.replaces, "note": self.note,
                "builds": [{"repo": b.repo, "quant": b.quant, "gb": round(b.size / 1e9, 1)} for b in self.builds]}


def judge(builds: list[Build], b: Budgets, gpu_min_bits: float = 4, memory_min_bits: float = 3) -> tuple[str, Build | None]:
    if not builds:
        return "no GGUF yet", None
    on_gpu = [x for x in builds if x.size <= b.gpu and x.bits >= gpu_min_bits]
    if on_gpu:
        # between a huge F16 of a tiny model and a Q6 the difference is nothing: cap what "more bits" is worth
        return "fits the GPU", max(on_gpu, key=lambda x: (min(x.bits, 6), x.size))
    in_memory = [x for x in builds if x.size <= b.memory and x.bits >= memory_min_bits]
    if in_memory:
        return "fits in memory", max(in_memory, key=lambda x: x.size)
    low = [x for x in builds if x.size <= b.memory]
    if low:
        return "low-bit only", max(low, key=lambda x: x.size)
    return "too big", min(builds, key=lambda x: x.size)


def role_of(model: str, kind: str) -> str:
    name = model.lower()
    if kind == "embed":
        return "embed"
    if kind == "speech":
        return "audio"
    if "ocr" in name:
        return "ocr"
    if kind == "vision":
        return "vision"
    if re.search(r"cod(e|er)|devstral|codestral", name):
        return "coder"
    return "general"


def judge_model(model: str, created: str, kind: str | None, fetch: Fetch, b: Budgets, roster: dict[str, str] | None = None,
                quantizers: list[str] | None = None, gpu_min_bits: float = 4, memory_min_bits: float = 3) -> Judged:
    params, page_kind = parameters(model, fetch)
    kind = kind or page_kind or "text"
    builds = find_builds(model, fetch, quantizers) if quantizers else find_builds(model, fetch)
    if params:  # under 0.75 bits a weight is no build of this model: a helper file that slipped through
        builds = [x for x in builds if x.size * 8 / (params * 1e9) >= 0.75]
    j = Judged(model, created[:10], kind, params, builds)
    j.verdict, j.pick = judge(builds, b, gpu_min_bits, memory_min_bits)
    j.role = role_of(model, kind)
    j.replaces = (roster or {}).get(j.role, "")
    if j.verdict == "low-bit only":
        j.note = "under 3 bits a weight: measure it on your own tasks before trusting it"
    elif j.verdict == "fits in memory":
        j.note = "part of the weights stay off the GPU: slower than a model that fits it"
    return j
