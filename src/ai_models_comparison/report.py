# Copyright 2026 Yauhen Bichel
# SPDX-License-Identifier: Apache-2.0
"""The answers as text for a terminal or an agent, as Markdown, and as one HTML page."""
from __future__ import annotations

import html
from typing import Any

from .judge import ORDER, Judged
from .machine import GIB, Budgets, Machine


def _sorted(judged: list[Judged]) -> list[Judged]:
    return sorted(judged, key=lambda j: (ORDER[j.verdict], j.role, -(j.params_b or 0), j.model))


def _ctx(n: int | None) -> str:
    return "" if not n else f"{n // 1024}k" if n >= 1024 else str(n)


def text_machine(answer: dict[str, Any]) -> str:
    m = answer["machine"]
    return (f"machine: {m['gpu_gib']:g} GiB GPU, {m['ram_gib']:g} GiB system memory ({m['kind']}{', unified' if m['unified'] else ''}); "
            f"fits the GPU up to {m['fits_gpu_gib']:g} GiB of weights, fits in memory up to {m['fits_memory_gib']:g} GiB")


def text_models(answer: dict[str, Any], builds: bool = True) -> str:
    """One model after another, each with its builds: what `judge` prints and what an agent reads."""
    out = [text_machine(answer)]
    for j in answer["models"]:
        params = f", {j['params_b']:g}B weights" if j["params_b"] else ""
        out.append(f"{j['model']} ({j['kind']}{params}): {j['verdict']}")
        pick = j["pick"]
        for x in j["builds"] if builds else []:
            mark = "  <- " + j["verdict"] if pick and x["quant"] == pick["quant"] and j["verdict"] != "too big" else ""
            where = {"gpu": "GPU", "memory": "memory", "no": "no"}[x["fits"]]
            out.append(f"  {x['gb']:8.1f} GB  {x['quant']:<12} {where:<6}{mark}".rstrip())
        if not j["builds"]:
            out.append("  no GGUF builds found")
        if pick and "run" in pick:
            out.append(f"  get: {pick['get']}")
            out.append(f"  run: {pick['run']}")
        if j["note"]:
            out.append(f"  note: {j['note']}")
    out += [f"{e['model']}: {e['error']}" for e in answer.get("errors") or []]
    return "\n".join(out)


def text_new(answer: dict[str, Any]) -> str:
    """One line a model: what an agent needs to choose which to look at."""
    out = [text_machine(answer), f"models since {answer['since']}: " + (", ".join(f"{v} {n}" for v, n in answer["counts"].items()) or "none")]
    for j in answer["models"]:
        pick = j["pick"]
        best = f"{pick['quant']} {pick['gb']:g} GB" if pick else "-"
        params = f"{j['params_b']:g}B" if j["params_b"] else "?"
        out.append(f"{j['verdict']:<15} {j['model']}  {j['kind']}/{j['role']}  {params}  {best}  {j['created']}")
    return "\n".join(out)


def text_compare(answer: dict[str, Any]) -> str:
    """The models side by side: one column a model, one row a fact."""
    models = answer["models"]
    if not models:
        return "\n".join([text_machine(answer)] + [f"{e['model']}: {e['error']}" for e in answer.get("errors") or []])

    def pick(j: dict[str, Any], key: str, unit: str = "") -> str:
        return f"{j['pick'][key]:g}{unit}" if j["pick"] and j["pick"].get(key) is not None else "-"

    rows: list[tuple[str, list[str]]] = [
        ("verdict", [j["verdict"] for j in models]),
        ("best build", [j["pick"]["quant"] if j["pick"] else "-" for j in models]),
        ("size", [pick(j, "gb", " GB") for j in models]),
        ("bits a weight", [pick(j, "bits_per_weight") for j in models]),
        ("room left", [pick(j, "headroom_gb", " GB") for j in models]),
        ("weights", [f"{j['params_b']:g}B" if j["params_b"] else "-" for j in models]),
        ("context", [_ctx(j["context"]) or "-" for j in models]),
        ("kind", [j["kind"] for j in models]),
        ("licence", [j["licence"] or "-" for j in models]),
        ("released", [j["created"] or "-" for j in models]),
        ("downloads", [f"{j['downloads']:,}" if j["downloads"] is not None else "-" for j in models]),
    ]
    heads = [j["model"].split("/")[-1] for j in models]
    widths = [max(len(heads[i]), *(len(r[1][i]) for r in rows)) for i in range(len(models))]
    left = max(len(r[0]) for r in rows)
    out = [text_machine(answer), "", " " * left + "  " + "  ".join(h.ljust(w) for h, w in zip(heads, widths, strict=True))]
    out += [(name.ljust(left) + "  " + "  ".join(v.ljust(w) for v, w in zip(vals, widths, strict=True))).rstrip() for name, vals in rows]
    out += [f"{e['model']}: {e['error']}" for e in answer.get("errors") or []]
    return "\n".join(out)


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
             f"Generated {today} by [ai-models-comparison](https://github.com/YauhenBichel/ai-models-comparison). "
             f"This machine: {m.ram / GIB:.0f} GiB of system memory, {m.gpu / GIB:.0f} GiB of GPU memory "
             f"({m.kind}{', unified' if m.unified else ''}). Budgets: "
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


def text_analysis(answer: dict[str, Any]) -> str:
    """Role by role: the short list, each model with its reasons (+) and its cautions (!)."""
    out = [text_machine(answer), "models since " + answer["since"] if answer.get("since") else "the models named"]
    for r in answer["roles"]:
        out += ["", f"{r['role']}" + (f"  (in use: {r['in_use']})" if r["in_use"] else ""), f"  {r['summary']}"]
        for n, c in enumerate(r["candidates"], 1):
            pick = c["pick"]
            out.append(f"  {n}. {c['model']}  {c['verdict']}  {pick['quant']} {pick['gb']:g} GB  ranks as {c['effective_b']:g}B, {c['speed']}")
            out += [f"     + {w}" for w in c["why"]] + [f"     ! {w}" for w in c["cautions"]]
            if "run" in pick:
                out.append(f"     run: {pick['run']}")
        if r["left_out"]:
            out.append("  left out: " + ", ".join(f"{n} {why}" for why, n in r["left_out"].items()))
    if not answer["roles"]:
        out.append("no models to analyse")
    out += [f"{e['model']}: {e['error']}" for e in answer.get("errors") or []]
    out += ["", "Ranks are weights after the cost of quantization: an ordering, not a benchmark. Test the short list on your own tasks."]
    return "\n".join(out)
