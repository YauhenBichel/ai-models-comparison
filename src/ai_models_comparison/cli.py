# Copyright 2026 Yauhen Bichel
# SPDX-License-Identifier: Apache-2.0
"""ai-models-comparison: which new open-weights models fit this machine?

  machine                    what this machine has, and the budgets that follow
  judge MODEL [MODEL ...]    each model's builds, and the best that fits here
  compare MODEL MODEL ...    the models side by side
  new [--days 45]            the watched publishers' new models, each judged; a report
  analyze [MODEL ...]        which of them are worth trying on this machine, role by role, with reasons
  serve                      the page and the HTTP API, on this machine
  mcp                        the same answers for an agent (Model Context Protocol, on stdin and stdout)
  catalog | site             build the catalog as one JSON file, or the static site around it
  config                     an example configuration file

Every command takes --json, and --gpu-gb / --ram-gb to ask about another machine.
Exit status: 0 done, 1 the question could not be answered, 2 the command line was wrong, 3 --require was not met.
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
import webbrowser
from pathlib import Path
from typing import Any

from . import __version__
from . import catalog as cat
from .cache import cached
from .hub import KINDS, Fetch, http_json
from .judge import ORDER, Judged
from .machine import SPEC, describe
from .mcp import serve_stdio
from .report import markdown, page, text_analysis, text_compare, text_models
from .server import Options, serve, static
from .service import Service

CONFIG = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "ai-models-comparison" / "config.toml"
EXAMPLE = '''# ai-models-comparison configuration. Every key is optional.
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
# catalog = "https://example.org/catalog.json"   # read a published catalog instead of asking Hugging Face
# notify = "https://ntfy.sh/your-topic"   # one line there when a new model fits; or the AI_MODELS_COMPARISON_NOTIFY variable

# [roster]               # what you run today, by role: the report says what a new model would replace
# coder = "qwen3-coder-next"
# general = "gpt-oss:120b"
# vision = "qwen3.6:35b"
# embed = "bge-m3"
'''
RANK = {"gpu": 0, "memory": 1}   # --require: the verdict must be at least this good


def load_config(path: str | None) -> dict[str, Any]:
    p = Path(path) if path else CONFIG
    if not p.is_file():
        if path:
            raise ValueError(f"no such configuration file: {path}")
        return {}
    with open(p, "rb") as fh:
        return tomllib.load(fh)


def notify(url: str, text: str) -> None:
    try:
        req = urllib.request.Request(url, data=text.encode(), headers={"Title": "ai-models-comparison: new models"})
        urllib.request.urlopen(req, timeout=15).read()  # noqa: S310
    except (OSError, urllib.error.URLError):
        print("ai-models-comparison: the notification could not be sent", file=sys.stderr)


def write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def spec_of(a: argparse.Namespace) -> dict[str, Any]:
    spec = {k: getattr(a, k, None) for k in SPEC}
    spec["unified"] = True if getattr(a, "unified", False) else None
    return spec


def dump(answer: Any) -> None:
    print(json.dumps(answer, indent=1))


def cmd_machine(a: argparse.Namespace, s: Service) -> int:
    if a.json:
        dump(s.machine(spec_of(a)))
    else:
        print(describe(*s.resolve(spec_of(a))))
    return 0


def cmd_judge(a: argparse.Namespace, s: Service) -> int:
    answer = s.models(a.models, spec_of(a))
    if a.json:
        dump(answer)
    else:
        print(text_compare(answer) if a.cmd == "compare" else text_models(answer))
    if answer["errors"]:
        return 1
    need = getattr(a, "require", None)
    if need and any(ORDER[j["verdict"]] > RANK[need] for j in answer["models"]):
        return 3
    return 0


def cmd_new(a: argparse.Namespace, s: Service) -> int:
    m, b, c, judged = s.judged_new(spec_of(a), a.days, a.since, a.publisher, a.kind)
    now = dt.datetime.now(dt.UTC)
    today = now.strftime("%Y-%m-%d")
    state: dict[str, Any] = {"seen": {}}
    if a.state:
        with contextlib.suppress(OSError, ValueError):
            state = json.loads(Path(a.state).read_text())
    first_run = not state.get("seen")
    fresh: list[Judged] = [j for j in judged if j.verdict in ("fits the GPU", "fits in memory") and state["seen"].get(j.model) != j.verdict]
    text = markdown(judged, c.since, m, b, today, s.cfg.get("roster"))
    if a.out:
        write_atomic(Path(a.out) / f"{today}.md", text)
        write_atomic(Path(a.out) / "latest.md", text)
    if a.html:
        write_atomic(Path(a.html), page(judged, c.since, m, b, today))
    if a.state:
        write_atomic(Path(a.state), json.dumps({"time": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "since": c.since,
                                                 "seen": {**state.get("seen", {}), **{j.model: j.verdict for j in judged}},
                                                 "fresh": [j.to_dict(b) for j in fresh]}, indent=1))
    url = s.cfg.get("notify") or os.environ.get("AI_MODELS_COMPARISON_NOTIFY")
    if url and fresh and a.state and not first_run:
        notify(url, "; ".join(f"{j.model} ({j.verdict}, {j.pick.quant} {j.pick.size / 1e9:.0f} GB, {j.role})" for j in fresh[:4] if j.pick))
    if a.json:
        dump(s.new(spec_of(a), a.days, a.since, a.publisher, a.kind))
    elif not a.out and not a.html:
        sys.stdout.write(text)
    elif not a.quiet:
        print(f"{len(judged)} models since {c.since}; {len(fresh)} new that fit")
    return 0


def cmd_analyze(a: argparse.Namespace, s: Service) -> int:
    answer = s.analyze(spec_of(a), a.models, a.days, a.since, a.publisher, a.kind, a.top)
    if a.json:
        dump(answer)
    else:
        print(text_analysis(answer))
    return 1 if answer["errors"] else 0


def cmd_catalog(a: argparse.Namespace, s: Service) -> int:
    text = json.dumps(s.catalog(a.days, a.since, a.publisher, a.kind).to_dict(), separators=(",", ":"))
    if a.out:
        write_atomic(Path(a.out), text)
    else:
        print(text)
    return 0


def cmd_site(a: argparse.Namespace, s: Service) -> int:
    """The page and the catalog as plain files: any static host serves them, and nobody's visit asks Hugging Face anything."""
    out = Path(a.out)
    c = s.catalog(a.days, a.since, a.publisher, a.kind)
    for name in ("index.html", "app.js", "judge.js", "app.css"):
        write_atomic(out / name, static(name).decode())
    write_atomic(out / "catalog.json", json.dumps(c.to_dict(), separators=(",", ":")))
    write_atomic(out / "catalog.schema.json", json.dumps(cat.schema(), indent=1))
    if not a.quiet:
        print(f"{len(c.models)} models since {c.since} in {out}/")
    return 0


def cmd_serve(a: argparse.Namespace, s: Service) -> int:
    opts = Options(a.host, a.port, a.token or os.environ.get("AI_MODELS_COMPARISON_TOKEN", ""), a.cors or "", a.verbose)
    server = serve(s, opts)
    url = f"http://{'localhost' if a.host == '127.0.0.1' else a.host}:{server.server_address[1]}/"
    print(f"ai-models-comparison: the page is at {url}, the API at {url}api/v1/ (described by {url}openapi.json). Ctrl-C stops it.",
          file=sys.stderr)
    if a.open:
        webbrowser.open(url)
    with contextlib.suppress(KeyboardInterrupt):
        server.serve_forever()
    server.server_close()
    return 0


def cmd_mcp(a: argparse.Namespace, s: Service) -> int:
    return serve_stdio(s)


def main(argv: list[str] | None = None, fetch: Fetch | None = None) -> int:
    doc = (__doc__ or "").splitlines()
    ap = argparse.ArgumentParser(prog="ai-models-comparison", description=doc[0], epilog="\n".join(doc[2:]),
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"ai-models-comparison {__version__}")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", help=f"a TOML file (default {CONFIG}, if it exists)")
    common.add_argument("--gpu-gb", type=float, help="GPU memory in GiB, instead of what was detected")
    common.add_argument("--ram-gb", type=float, help="system memory in GiB, instead of what was detected")
    common.add_argument("--unified", action="store_true", help="GPU and system memory are one pool")
    common.add_argument("--gpu-reserve-gb", type=float, help="kept free on the GPU")
    common.add_argument("--ram-reserve-gb", type=float, help="kept free for the system")
    common.add_argument("--catalog", help="a published catalog (an address or a file) to read instead of asking Hugging Face")
    common.add_argument("--no-cache", action="store_true", help="ask Hugging Face again instead of reusing answers of the last hours")
    common.add_argument("--json", action="store_true", help="the answer as JSON, the same shape the HTTP API gives")
    window = argparse.ArgumentParser(add_help=False)
    window.add_argument("--days", type=int, default=45)
    window.add_argument("--since", help="YYYY-MM-DD, instead of --days")
    window.add_argument("--publisher", action="append", help="repeatable; instead of the configured list")
    window.add_argument("--kind", action="append", choices=sorted(set(KINDS.values())), help="repeatable")
    window.add_argument("--quiet", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("machine", parents=[common], help="what this machine has")
    p.set_defaults(fn=cmd_machine)
    for name, text in (("judge", "judge one or more models"), ("compare", "the models side by side")):
        p = sub.add_parser(name, parents=[common], help=text)
        p.add_argument("models", nargs="+", metavar="MODEL", help="a Hugging Face id or page address, e.g. Qwen/Qwen3-Coder-Next")
        p.add_argument("--require", choices=sorted(RANK), help="exit 3 unless every model fits the GPU (gpu) or at least memory (memory)")
        p.set_defaults(fn=cmd_judge)
    p = sub.add_parser("new", parents=[common, window], help="the publishers' new models, judged")
    p.add_argument("--out", help="a directory for the Markdown report (latest.md and a dated copy)")
    p.add_argument("--html", help="a file for the report as one HTML page")
    p.add_argument("--state", help="a file remembering what was seen, so a daily run reports only what is new")
    p.set_defaults(fn=cmd_new)
    p = sub.add_parser("analyze", parents=[common, window], help="what is worth trying here, role by role, with reasons")
    p.add_argument("models", nargs="*", metavar="MODEL", help="analyse these models instead of the new ones")
    p.add_argument("--top", type=int, default=3, help="how many to list for a role that fit the GPU (default 3)")
    p.set_defaults(fn=cmd_analyze)
    p = sub.add_parser("catalog", parents=[common, window], help="the new models without a verdict, as one JSON file")
    p.add_argument("--out", help="a file (default: standard output)")
    p.set_defaults(fn=cmd_catalog)
    p = sub.add_parser("site", parents=[common, window], help="the page and the catalog as static files")
    p.add_argument("--out", required=True, help="a directory")
    p.set_defaults(fn=cmd_site)
    p = sub.add_parser("serve", parents=[common], help="the page and the HTTP API")
    p.add_argument("--host", default="127.0.0.1", help="default: this machine only")
    p.add_argument("--port", type=int, default=8377)
    p.add_argument("--token", help="wanted on the API; required to listen beyond this machine (or AI_MODELS_COMPARISON_TOKEN)")
    p.add_argument("--cors", help="an origin allowed to read the API from a browser, or *")
    p.add_argument("--open", action="store_true", help="open the page in the browser")
    p.add_argument("--verbose", action="store_true", help="log requests")
    p.set_defaults(fn=cmd_serve)
    p = sub.add_parser("mcp", parents=[common], help="a Model Context Protocol server on stdin and stdout")
    p.set_defaults(fn=cmd_mcp)
    p = sub.add_parser("config", help="print an example configuration file")
    p.set_defaults(fn=None)
    a = ap.parse_args(argv)
    if a.cmd == "config":
        print(EXAMPLE, end="")
        return 0
    try:
        cfg = load_config(a.config)
        hub = fetch or (http_json if a.no_cache else cached(http_json))
        return int(a.fn(a, Service(cfg, hub, a.catalog)))
    except (ValueError, OSError) as e:   # a wrong value, a missing file, an unreachable hub: say it once, in the form asked for
        if getattr(a, "json", False):
            print(json.dumps({"error": {"code": "failed", "message": str(e)}}), file=sys.stderr)
        else:
            print(f"ai-models-comparison: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
