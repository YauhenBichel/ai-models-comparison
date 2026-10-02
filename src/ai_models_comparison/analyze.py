# Copyright 2026 Yauhen Bichel
# SPDX-License-Identifier: Apache-2.0
"""The analysis: of the models that fit, which are worth trying on this machine, and why.

A verdict says whether a model fits. The analysis compares the ones that do, role by role, on what can be
known without running them:

  size after quantization   the weights times what the build keeps of full precision. One published test
                            (DeepSeek V3.1 on Aider Polyglot, unsloth's dynamic builds) kept 97 % of the
                            score at 4 bits, 95.5 % at 3, 92 % at 2 and 78 % at 1. It orders candidates of
                            a role; it is not a benchmark, and a newer small model can beat an older large one.
  speed class               the whole build on the GPU, or part of it in system memory
  what you run today        the model your roster names for the role, when it is a Hugging Face id
  cautions                  a licence to read, a low-bit build, files a few days old, little room for context

Kept for a role: the best that fits the GPU, then only what is larger than it and still runs. A model that is
both slower and smaller than another is not listed. When the roster names what you run, a candidate smaller
than it is left out too, except the best one that is at least six months newer: new small models often beat
older large ones, and only a test says. A model that reads images is a candidate for `general` as well as
for `vision`. Everything here is arithmetic on the catalog: the same question gives the same answer, and
each line says its reason.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any

from .judge import Judged
from .machine import Budgets

KEEP = ((8, 1.0), (6, 0.99), (5, 0.98), (4, 0.97), (3, 0.955), (2, 0.92), (1, 0.78))   # nominal bits -> share of the score kept
OPEN = {"apache-2.0", "mit", "bsd-2-clause", "bsd-3-clause", "isc", "cc0-1.0", "cc-by-4.0", "unlicense"}
ROLES = ("coder", "general", "vision", "ocr", "embed", "audio")
FITS = ("fits the GPU", "fits in memory", "low-bit only")


def keep(bits: float) -> float:
    """What a build keeps of the full-precision score, by its nominal bits. Unknown bits: assume 3."""
    return next((k for b, k in KEEP if bits >= b), 0.955 if bits == 0 else 0.78)


def effective(j: Judged) -> float | None:
    """Billions of weights after the cost of quantization; from the file size when the weight count is unknown."""
    if j.pick is None:
        return None
    k = keep(j.pick.bits)
    if j.params_b:
        return round(j.params_b * k, 1)
    return round(j.pick.size * 8 / max(j.pick.bits, 1.0) / 1e9 * k, 1)


@dataclass(slots=True)
class Candidate:
    judged: Judged
    effective_b: float
    speed: str                        # "on the GPU" | "partly in system memory" | "in system memory"
    gpu_share: float                  # of the build's bytes that fit the GPU budget
    why: list[str] = field(default_factory=list)
    cautions: list[str] = field(default_factory=list)
    ratio: float | None = None        # to the model in use, by rank
    months_newer: int = 0             # than the model in use

    def to_dict(self, b: Budgets) -> dict[str, Any]:
        return {**self.judged.to_dict(b), "effective_b": self.effective_b, "speed": self.speed, "gpu_share": round(self.gpu_share, 2),
                "why": self.why, "cautions": self.cautions}


def _age(created: str, today: dt.date) -> int | None:
    try:
        return (today - dt.date.fromisoformat(created[:10])).days
    except ValueError:
        return None


def candidate(j: Judged, b: Budgets, today: dt.date, in_use: Judged | None = None) -> Candidate | None:
    """One model that fits, with the reasons to try it and the reasons to be careful. None when nothing of it fits."""
    eff = effective(j)
    if j.verdict not in FITS or j.pick is None or eff is None:
        return None
    pick = j.pick
    share = 1.0 if pick.size <= b.gpu else (b.gpu / pick.size if pick.size else 0.0)
    speed = "on the GPU" if share >= 1 else "partly in system memory" if b.gpu > 0 else "in system memory"
    c = Candidate(j, eff, speed, share)
    weights = f"{j.params_b:g}B weights" if j.params_b else "weight count unknown"
    c.why.append(f"{weights} at {pick.quant} ({pick.size / 1e9:.1f} GB): about {keep(pick.bits):.0%} of full precision, "
                 f"so it ranks as {eff:g}B")
    if speed == "on the GPU":
        c.why.append("the whole build fits the GPU: the fast way to run it")
    elif b.gpu > 0:
        c.why.append(f"{share:.0%} of the build fits the GPU, the rest runs from system memory: slower")
    else:
        c.why.append("no GPU: it runs on the CPU from system memory")
    if j.context:
        c.why.append(f"context {j.context // 1024}k" if j.context >= 1024 else f"context {j.context}")
    if in_use is not None:
        mine = effective(in_use)
        name = in_use.model.split("/")[-1]
        if mine:
            ratio = c.ratio = eff / mine
            c.why.append(f"{ratio:.1f} times the size of {name}, which you run today" if ratio >= 1.15
                         else f"about the size of {name}, which you run today" if ratio > 0.87
                         else f"smaller than {name}, which you run today ({eff:g}B against {mine:g}B)")
        newer, older = _age(in_use.created, today), _age(j.created, today)
        if newer is not None and older is not None and newer - older >= 30:
            c.months_newer = (newer - older) // 30
            c.why.append(f"{c.months_newer} months newer than {name}")
    if j.verdict == "low-bit only":
        c.cautions.append("only a build under 3 bits fits: it loses quality, measure it on your own tasks before trusting it")
    if not j.params_b:
        c.cautions.append("the weight count is unknown: its rank comes from the file size")
    if j.licence.lower() not in OPEN:
        c.cautions.append(f"licence {j.licence}: read it before commercial use" if j.licence else "no licence is stated: read the model page")
    age = _age(j.created, today)
    if age is not None and age < 14:
        c.cautions.append(f"released {age} days ago: builds this new are often replaced; check the build repository again before a long download")
    budget = b.gpu if j.verdict == "fits the GPU" else b.memory
    if budget and (budget - pick.size) / budget < 0.08:
        c.cautions.append(f"only {(budget - pick.size) / 1e9:.1f} GB of room left: the context cache needs some, so use a short context "
                          "or the next smaller build")
    if j.context and j.context < 16384:
        c.cautions.append(f"a short context ({j.context} tokens)")
    return c


def shortlist(cands: list[Candidate], limit: int = 3) -> list[Candidate]:
    """The best on the GPU first, then only what is larger than everything listed before it."""
    rank = {"fits the GPU": 0, "fits in memory": 1, "low-bit only": 2}
    out: list[Candidate] = []
    best = 0.0
    for tier in range(3):
        level = sorted((c for c in cands if rank[c.judged.verdict] == tier), key=lambda c: (-c.effective_b, c.judged.model))
        kept = [c for c in level if c.effective_b > best][: limit if tier == 0 else 1]
        out += kept
        best = max([best] + [c.effective_b for c in kept])
    return out


def analyse(judged: list[Judged], b: Budgets, in_use: dict[str, Judged] | None = None, names: dict[str, str] | None = None,
            today: dt.date | None = None, limit: int = 3) -> list[dict[str, Any]]:
    """Role by role: what you run today, the short list with reasons, and how many were left out and why."""
    today = today or dt.datetime.now(dt.UTC).date()
    out: list[dict[str, Any]] = []
    for role in ROLES:
        # a model that reads images also answers in text: it stands for `general` too
        mine = [j for j in judged if j.role == role or (role == "general" and j.role == "vision")]
        if not mine:
            continue
        current = (in_use or {}).get(role)
        cands = [c for j in mine if (current is None or j.model != current.model) and (c := candidate(j, b, today, current))]
        for c in cands:
            if c.judged.role != role:
                c.why.append("it also reads images")
        smaller = [c for c in cands if c.ratio is not None and c.ratio <= 0.87]
        worth = [c for c in cands if c not in smaller]
        newcomer = max((c for c in smaller if c.months_newer >= 6), key=lambda c: c.effective_b, default=None)
        listed = shortlist(worth, limit)
        if newcomer is not None:
            newcomer.cautions.insert(0, "smaller than what you run, but much newer: new small models often beat older large ones; "
                                        "only a test on your tasks says")
            listed.append(newcomer)
        left = {"too big": sum(1 for j in mine if j.verdict == "too big"), "no GGUF yet": sum(1 for j in mine if j.verdict == "no GGUF yet"),
                "smaller than what you run today": len(smaller) - (1 if newcomer else 0),
                "smaller or slower than a listed one": len(worth) - len(shortlist(worth, limit))}
        rest = [c for c in listed if c is not newcomer]
        fast = next((c for c in rest if c.speed == "on the GPU"), None)
        sound = [c for c in rest if c.judged.verdict != "low-bit only"] or rest       # a low-bit build leads only when it is all there is
        top = max(sound, key=lambda c: c.effective_b, default=None)
        if fast is not None and top is not None and top is not fast:
            summary = (f"fast: {fast.judged.model} ({fast.effective_b:g}B on the GPU); "
                       f"largest that runs: {top.judged.model} ({top.effective_b:g}B, {top.speed})")
        elif fast is not None:
            summary = f"{fast.judged.model} ({fast.effective_b:g}B, on the GPU)"
        elif top is not None:
            first = "nothing new fits the GPU" if current is None else "nothing new on the GPU is larger than what you run"
            summary = f"{first}; largest that runs: {top.judged.model} ({top.effective_b:g}B, {top.speed})"
        elif newcomer is not None:
            summary = f"nothing larger than what you run fits; newer and smaller: {newcomer.judged.model} ({newcomer.effective_b:g}B)"
        else:
            summary = "nothing new fits this machine" if current is None else "nothing new that fits is larger or much newer than what you run"
        out.append({"role": role, "in_use": (names or {}).get(role, ""),
                    "in_use_effective_b": effective(current) if current else None, "summary": summary,
                    "candidates": [c.to_dict(b) for c in listed], "left_out": {k: v for k, v in left.items() if v}})
    return out
