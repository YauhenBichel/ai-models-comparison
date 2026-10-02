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

Judging is arithmetic on a catalog entry and two budgets: no network, no machine. The page in `web/judge.js`
does the same arithmetic in the browser, and `tests/golden.json` holds the cases both must agree on.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .catalog import Entry
from .hub import Build
from .machine import Budgets

ORDER = {"fits the GPU": 0, "fits in memory": 1, "low-bit only": 2, "too big": 3, "no GGUF yet": 4}
NOTES = {"low-bit only": "under 3 bits a weight: measure it on your own tasks before trusting it",
         "fits in memory": "part of the weights stay off the GPU: slower than a model that fits it"}


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
    licence: str = ""
    context: int | None = None
    downloads: int | None = None
    likes: int | None = None

    def _build(self, x: Build, b: Budgets | None) -> dict[str, Any]:
        d: dict[str, Any] = {"repo": x.repo, "quant": x.quant, "gb": round(x.size / 1e9, 1), "bits": x.bits,
                             "bits_per_weight": round(x.size * 8 / (self.params_b * 1e9), 2) if self.params_b else None}
        if b is not None:
            d["fits"] = "gpu" if x.size <= b.gpu else "memory" if x.size <= b.memory else "no"
        return d

    def to_dict(self, b: Budgets | None = None) -> dict[str, Any]:
        """One model as every interface gives it. With the budgets, each build says where it fits."""
        pick = None
        if self.pick is not None:
            pick = self._build(self.pick, b)
            if self.verdict != "too big":
                pick["get"] = f'hf download {self.pick.repo} --include "*{self.pick.quant}*"'
                pick["run"] = f"llama-server -hf {self.pick.repo}:{self.pick.quant}"
                if b is not None:
                    room = (b.gpu if self.verdict == "fits the GPU" else b.memory) - self.pick.size
                    pick["headroom_gb"] = round(room / 1e9, 1)
        return {"model": self.model, "url": f"https://huggingface.co/{self.model}", "created": self.created, "kind": self.kind,
                "params_b": self.params_b, "licence": self.licence, "context": self.context, "downloads": self.downloads,
                "likes": self.likes, "verdict": self.verdict, "pick": pick, "role": self.role, "replaces": self.replaces,
                "note": self.note, "builds": [self._build(x, b) for x in self.builds]}


def _best(fitting: list[Build]) -> Build:
    """The most bits that are worth having, then the larger file. Above 8 bits a weight nothing is gained: an
    F16 of a small model is twice the memory of its Q8 for the same answers, so it is picked only when it is all there is."""
    usual = [x for x in fitting if x.bits <= 8]
    return max(usual, key=lambda x: (x.bits, x.size)) if usual else min(fitting, key=lambda x: x.size)


def judge(builds: list[Build], b: Budgets, gpu_min_bits: float = 4, memory_min_bits: float = 3) -> tuple[str, Build | None]:
    if not builds:
        return "no GGUF yet", None
    on_gpu = [x for x in builds if x.size <= b.gpu and x.bits >= gpu_min_bits]
    if on_gpu:
        return "fits the GPU", _best(on_gpu)
    in_memory = [x for x in builds if x.size <= b.memory and x.bits >= memory_min_bits]
    if in_memory:
        return "fits in memory", _best(in_memory)
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


def judge_entry(e: Entry, b: Budgets, roster: dict[str, str] | None = None, gpu_min_bits: float = 4,
                memory_min_bits: float = 3) -> Judged:
    j = Judged(e.model, e.created, e.kind, e.params_b, e.builds, licence=e.licence, context=e.context, downloads=e.downloads,
               likes=e.likes)
    j.verdict, j.pick = judge(e.builds, b, gpu_min_bits, memory_min_bits)
    j.role = role_of(e.model, e.kind)
    j.replaces = (roster or {}).get(j.role, "")
    j.note = NOTES.get(j.verdict, "")
    return j
