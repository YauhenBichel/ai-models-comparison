# Copyright 2026 Yauhen Bichel
# SPDX-License-Identifier: Apache-2.0
"""fits-here: which new open-weights models fit this machine?

  fits-here machine                      what this machine has, and the budgets that follow
  fits-here judge MODEL [MODEL ...]      one model: its builds, and the best that fits here
  fits-here new [--days 45]              the watched publishers' new models, each judged; a report
  fits-here config                       an example configuration file

Nothing is downloaded but JSON from Hugging Face's public API, and nothing touches the GPU.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import os
import sys
import tomllib
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from . import __version__
from .hub import KINDS, PUBLISHERS, QUANTIZERS, Fetch, http_json, new_models
from .judge import Judged, judge_model
from .machine import GIB, Budgets, Machine, budgets, describe, detect
from .report import markdown, page

CONFIG = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "fits-here" / "config.toml"
EXAMPLE = '''# fits-here configuration. Every key is optional.
# publishers = ["deepseek-ai", "Qwen", "google", "mistralai"]   # whose new models to judge (default: about twenty)
# quantizers = ["unsloth", "bartowski"]                          # whose GGUF builds to prefer
# kinds = ["text", "vision"]                                     # of: text, vision, embed, speech
# gpu_gb = 24            # override what was detected (GiB)
# ram_gb = 64
# unified = false
# gpu_reserve_gb = 2     # kept free on the GPU for the context cache and other users
# ram_reserve_gb = 16    # kept free for the system
# gpu_min_bits = 4       # a build under this is not called "fits the GPU"
# memory_min_bits = 3    # a build under this is "low-bit only"
# notify = "https://ntfy.sh/your-topic"   # one line there when a new model fits; or the FITS_HERE_NOTIFY variable

# [roster]               # what you run today, by role: the report says what a new model would replace
# coder = "qwen3-coder-next"
# general = "gpt-oss:120b"
# vision = "qwen3.6:35b"
# embed = "bge-m3"
'''


def load_config(path: str | None) -> dict[str, Any]:
    p = Path(path) if path else CONFIG
    if not p.is_file():
        if path:
            raise SystemExit(f"fits-here: no such configuration file: {path}")
        return {}
    with open(p, "rb") as fh:
        return tomllib.load(fh)


def machine_and_budgets(cfg: dict[str, Any], a: argparse.Namespace) -> tuple[Machine, Budgets]:
    m = detect()
    gpu_gb = a.gpu_gb if a.gpu_gb is not None else cfg.get("gpu_gb")
    ram_gb = a.ram_gb if a.ram_gb is not None else cfg.get("ram_gb")
    if gpu_gb is not None or ram_gb is not None:
        # a machine described by hand is two pools unless it says otherwise: do not inherit this one's shape
        m.unified, m.kind, m.note = False, "given", "as given, not detected"
    if gpu_gb is not None:
        m.gpu = int(float(gpu_gb) * GIB)
    if ram_gb is not None:
        m.ram = int(float(ram_gb) * GIB)
    if "unified" in cfg:
        m.unified = bool(cfg["unified"])
    if a.unified:
        m.unified = True
    gr = a.gpu_reserve_gb if a.gpu_reserve_gb is not None else cfg.get("gpu_reserve_gb")
    rr = a.ram_reserve_gb if a.ram_reserve_gb is not None else cfg.get("ram_reserve_gb")
    return m, budgets(m, gr, rr)


def notify(url: str, text: str) -> None:
    try:
        urllib.request.urlopen(urllib.request.Request(url, data=text.encode(), headers={"Title": "fits-here: new models"}), timeout=15).read()  # noqa: S310
    except (OSError, urllib.error.URLError):
        print("fits-here: the notification could not be sent", file=sys.stderr)


def write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def cmd_machine(a: argparse.Namespace, cfg: dict[str, Any], fetch: Fetch) -> int:
    m, b = machine_and_budgets(cfg, a)
    if a.json:
        print(json.dumps({"ram_gib": round(m.ram / GIB, 1), "gpu_gib": round(m.gpu / GIB, 1), "kind": m.kind, "unified": m.unified,
                          "fits_gpu_gib": round(b.gpu / GIB, 1), "fits_memory_gib": round(b.memory / GIB, 1)}))
    else:
        print(describe(m, b))
    return 0


def _judge_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    return {"roster": cfg.get("roster") or {}, "quantizers": cfg.get("quantizers") or QUANTIZERS,
            "gpu_min_bits": float(cfg.get("gpu_min_bits", 4)), "memory_min_bits": float(cfg.get("memory_min_bits", 3))}


def cmd_judge(a: argparse.Namespace, cfg: dict[str, Any], fetch: Fetch) -> int:
    m, b = machine_and_budgets(cfg, a)
    today = dt.datetime.now(dt.UTC).strftime("%Y-%m-%d")
    out = [judge_model(model, today, None, fetch, b, **_judge_kwargs(cfg)) for model in a.models]
    if a.json:
        print(json.dumps([j.to_dict() for j in out], indent=1))
        return 0
    for j in out:
        params = f", {j.params_b:g}B weights" if j.params_b else ""
        print(f"{j.model} ({j.kind}{params}): {j.verdict}")
        for x in j.builds:
            mark = "  <- " + j.verdict if j.pick is not None and x == j.pick and j.verdict != "too big" else ""
            where = "GPU" if x.size <= b.gpu else "memory" if x.size <= b.memory else "no"
            print(f"  {x.size / 1e9:8.1f} GB  {x.quant:<12} {where:<6}{mark}")
        if not j.builds:
            print("  no GGUF builds found")
        if j.note:
            print(f"  note: {j.note}")
    return 0


def cmd_new(a: argparse.Namespace, cfg: dict[str, Any], fetch: Fetch) -> int:
    m, b = machine_and_budgets(cfg, a)
    now = dt.datetime.now(dt.UTC)
    today = now.strftime("%Y-%m-%d")
    since = a.since or (now - dt.timedelta(days=a.days)).strftime("%Y-%m-%d")
    publishers = a.publisher or cfg.get("publishers") or PUBLISHERS
    kinds = set(a.kind or cfg.get("kinds") or []) or None
    if kinds and not kinds <= set(KINDS.values()):
        raise SystemExit(f"fits-here: kinds are {sorted(set(KINDS.values()))}")
    found = new_models(since, fetch, publishers, kinds, on_error=lambda org, e: print(f"fits-here: {org}: {e}", file=sys.stderr))
    judged: list[Judged] = [judge_model(mid, created, kind, fetch, b, **_judge_kwargs(cfg)) for mid, created, kind in found]
    state: dict[str, Any] = {"seen": {}}
    if a.state:
        with contextlib.suppress(OSError, ValueError):
            state = json.loads(Path(a.state).read_text())
    first_run = not state.get("seen")
    fresh = [j for j in judged if j.verdict in ("fits the GPU", "fits in memory") and state["seen"].get(j.model) != j.verdict]
    text = markdown(judged, since, m, b, today, cfg.get("roster"))
    if a.out:
        write_atomic(Path(a.out) / f"{today}.md", text)
        write_atomic(Path(a.out) / "latest.md", text)
    if a.html:
        write_atomic(Path(a.html), page(judged, since, m, b, today))
    if a.state:
        write_atomic(Path(a.state), json.dumps({"time": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "since": since,
                                                 "seen": {**state.get("seen", {}), **{j.model: j.verdict for j in judged}},
                                                 "fresh": [j.to_dict() for j in fresh]}, indent=1))
    url = cfg.get("notify") or os.environ.get("FITS_HERE_NOTIFY")
    if url and fresh and a.state and not first_run:
        notify(url, "; ".join(f"{j.model} ({j.verdict}, {j.pick.quant} {j.pick.size / 1e9:.0f} GB, {j.role})" for j in fresh[:4] if j.pick))
    if a.json:
        print(json.dumps([j.to_dict() for j in judged], indent=1))
    elif not a.out and not a.html:
        sys.stdout.write(text)
    elif not a.quiet:
        print(f"{len(judged)} models since {since}; {len(fresh)} new that fit")
    return 0


def main(argv: list[str] | None = None, fetch: Fetch = http_json) -> int:
    ap = argparse.ArgumentParser(prog="fits-here", description=__doc__.splitlines()[0], epilog="\n".join(__doc__.splitlines()[2:]),
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"fits-here {__version__}")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", help=f"a TOML file (default {CONFIG}, if it exists)")
    common.add_argument("--gpu-gb", type=float, help="GPU memory in GiB, instead of what was detected")
    common.add_argument("--ram-gb", type=float, help="system memory in GiB, instead of what was detected")
    common.add_argument("--unified", action="store_true", help="GPU and system memory are one pool")
    common.add_argument("--gpu-reserve-gb", type=float)
    common.add_argument("--ram-reserve-gb", type=float)
    common.add_argument("--json", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("machine", parents=[common], help="what this machine has")
    s.set_defaults(fn=cmd_machine)
    s = sub.add_parser("judge", parents=[common], help="judge one or more models")
    s.add_argument("models", nargs="+", metavar="MODEL", help="a Hugging Face id, e.g. Qwen/Qwen3-Coder-Next")
    s.set_defaults(fn=cmd_judge)
    s = sub.add_parser("new", parents=[common], help="the publishers' new models, judged")
    s.add_argument("--days", type=int, default=45)
    s.add_argument("--since", help="YYYY-MM-DD, instead of --days")
    s.add_argument("--publisher", action="append", help="repeatable; instead of the configured list")
    s.add_argument("--kind", action="append", help="text, vision, embed or speech; repeatable")
    s.add_argument("--out", help="a directory for the Markdown report (latest.md and a dated copy)")
    s.add_argument("--html", help="a file for the report as one HTML page")
    s.add_argument("--state", help="a file remembering what was seen, so a daily run reports only what is new")
    s.add_argument("--quiet", action="store_true")
    s.set_defaults(fn=cmd_new)
    s = sub.add_parser("config", help="print an example configuration file")
    s.set_defaults(fn=lambda a, cfg, fetch: print(EXAMPLE, end="") or 0)
    a = ap.parse_args(argv)
    cfg = load_config(getattr(a, "config", None)) if a.cmd != "config" else {}
    return int(a.fn(a, cfg, fetch))


if __name__ == "__main__":
    raise SystemExit(main())
