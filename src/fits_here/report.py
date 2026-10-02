# Copyright 2026 Yauhen Bichel
# SPDX-License-Identifier: Apache-2.0
"""The review as Markdown, as one HTML page, and as JSON."""
from __future__ import annotations

import html

from .judge import ORDER, Judged
from .machine import GIB, Budgets, Machine


def _sorted(judged: list[Judged]) -> list[Judged]:
    return sorted(judged, key=lambda j: (ORDER[j.verdict], j.role, -(j.params_b or 0), j.model))


def _pick(j: Judged) -> str:
    if j.pick is None:
        return ""
    if j.verdict == "too big":
        return f"smallest: {j.pick.quant}, {j.pick.size / 1e9:.0f} GB"
    return f"{j.pick.repo} {j.pick.quant}, {j.pick.size / 1e9:.1f} GB"


def counts(judged: list[Judged]) -> str:
    return ", ".join(f"{v} {n}" for v in ORDER if (n := sum(1 for j in judged if j.verdict == v)))


def markdown(judged: list[Judged], since: str, m: Machine, b: Budgets, today: str, roster: dict[str, str] | None = None) -> str:
    lines = [f"# Open-weights models since {since}, judged for this machine", "",
             f"Generated {today} by [fits-here](https://github.com/YauhenBichel/fits-here). This machine: {m.ram / GIB:.0f} GiB of "
             f"system memory, {m.gpu / GIB:.0f} GiB of GPU memory ({m.kind}{', unified' if m.unified else ''}). Budgets: "
             f"{b.gpu / GIB:.0f} GiB of weights on the GPU, {b.memory / GIB:.0f} GiB in memory. Nothing was downloaded.", ""]
    if roster:
        lines += ["In use today: " + ", ".join(f"{r} = `{name}`" for r, name in sorted(roster.items())) + ".", ""]
    lines += ["| Model | Released | Kind | Weights | Verdict | Best build that fits | Role | Would replace | Note |",
              "|---|---|---|---|---|---|---|---|---|"]
    for j in _sorted(judged):
        params = f"{j.params_b:g}B" if j.params_b else ""
        pick = f"`{_pick(j)}`" if j.pick and j.verdict != "too big" else _pick(j)
        lines.append(f"| [{j.model}](https://huggingface.co/{j.model}) | {j.created} | {j.kind} | {params} | **{j.verdict}** | {pick} | "
                     f"{j.role} | {f'`{j.replaces}`' if j.replaces else ''} | {j.note} |")
    lines += ["", f"Counts: {counts(judged)}." if judged else "No new models in the window.", "",
              "A verdict is about memory only. Whether a model is better than the one in use is for your own tests to say."]
    return "\n".join(lines) + "\n"


CSS = ("body{font:14px system-ui,sans-serif;margin:24px;background:#f6f5f1;color:#1c1b18}table{border-collapse:collapse;background:#fff}"
       "td,th{padding:6px 10px;border-bottom:1px solid #e2dfd7;text-align:left;vertical-align:top}.num{text-align:right}"
       ".v0{color:#2f8f5b;font-weight:600}.v1{color:#2a78d6;font-weight:600}.v2{color:#c9821b}.v3,.v4{color:#6f6a62}a{color:#2a78d6}"
       "p{max-width:70ch}@media(prefers-color-scheme:dark){body{background:#12141a;color:#eceaf2}table{background:#1a1d25}"
       "td,th{border-color:#2a2e3a}a{color:#5b9ae8}}")


def page(judged: list[Judged], since: str, m: Machine, b: Budgets, today: str) -> str:
    esc = html.escape
    rows = "".join(
        f"<tr><td><a href=\"https://huggingface.co/{esc(j.model)}\">{esc(j.model)}</a></td><td>{esc(j.created)}</td><td>{esc(j.kind)}</td>"
        f"<td class=num>{'' if j.params_b is None else f'{j.params_b:g}B'}</td><td class=\"v{ORDER[j.verdict]}\">{esc(j.verdict)}</td>"
        f"<td>{esc(_pick(j))}</td><td>{esc(j.role)}</td><td>{esc(j.replaces)}</td></tr>" for j in _sorted(judged))
    return ("<!doctype html><html lang=en><meta charset=utf-8><meta name=viewport content=\"width=device-width,initial-scale=1\">"
            f"<title>New open models</title><style>{CSS}</style>"
            f"<h1>Open-weights models since {esc(since)}</h1><p>Judged for this machine: {b.gpu / GIB:.0f} GiB of weights on the GPU, "
            f"{b.memory / GIB:.0f} GiB in memory. Generated {esc(today)}; nothing was downloaded. {esc(counts(judged))}.</p>"
            "<table><tr><th>model</th><th>released</th><th>kind</th><th>weights</th><th>verdict</th><th>best build that fits</th>"
            f"<th>role</th><th>would replace</th></tr>{rows}</table>"
            "<p>A verdict is about memory only. Whether a model is better than the one in use is for your own tests to say.</p>")
