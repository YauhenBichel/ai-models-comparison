# Copyright 2026 Yauhen Bichel
# SPDX-License-Identifier: Apache-2.0
"""`ai-models-comparison serve`: the page and the HTTP API, from one small server.

  GET  /                      the page
  GET  /api/v1/machine        this machine and its budgets (or the machine the query describes)
  GET  /api/v1/models?id=..   the models named, judged: one, or several to compare
  GET  /api/v1/new?days=45    the publishers' new models, judged, best fit first
  GET  /api/v1/analyze        which of them are worth trying, role by role, with reasons
  GET  /api/v1/catalog        the same models without a verdict: for a client that judges by itself
  GET  /openapi.json          the API's description
  GET  /healthz

Read-only and stateless: a request names its machine with gpu_gb, ram_gb, unified, gpu_reserve_gb and
ram_reserve_gb, so one server answers for any machine. It listens on this machine only unless told otherwise,
and then it wants a token.
"""
from __future__ import annotations

import hmac
import json
import sys
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from typing import Any
from urllib.parse import parse_qs, urlsplit

from . import __version__
from .judge import ORDER
from .machine import SPEC
from .service import MAX_MODELS, SCHEMA, Service

LOOPBACK = {"127.0.0.1", "localhost", "::1"}
STATIC = {"/": ("index.html", "text/html; charset=utf-8"), "/index.html": ("index.html", "text/html; charset=utf-8"),
          "/app.js": ("app.js", "text/javascript; charset=utf-8"), "/judge.js": ("judge.js", "text/javascript; charset=utf-8"),
          "/app.css": ("app.css", "text/css; charset=utf-8")}
CSP = "default-src 'self'; img-src 'self' data:; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"


@dataclass(slots=True)
class Options:
    host: str = "127.0.0.1"
    port: int = 8377
    token: str = ""        # wanted on /api when set
    cors: str = ""         # an origin allowed to read /api from a browser, or "*"
    verbose: bool = False


def static(name: str) -> bytes:
    return resources.files(__package__ or "ai_models_comparison").joinpath("web", name).read_bytes()


def _host(header: str) -> str:
    header = header.strip().lower()
    if header.startswith("["):
        return header[1:header.find("]")]
    return header.rsplit(":", 1)[0] if ":" in header else header


def openapi() -> dict[str, Any]:
    def q(name: str, schema: dict[str, Any], description: str) -> dict[str, Any]:
        return {"name": name, "in": "query", "required": False, "schema": schema, "description": description}

    gib = {"type": "number", "minimum": 0}
    machine = [q("gpu_gb", gib, "GPU memory in GiB; omit for the server's own machine"),
               q("ram_gb", gib, "system memory in GiB; omit for the server's own machine"),
               q("unified", {"type": "boolean"}, "GPU and system memory are one pool"),
               q("gpu_reserve_gb", gib, "kept free on the GPU"), q("ram_reserve_gb", gib, "kept free for the system")]
    strings = {"type": "array", "items": {"type": "string"}}
    window = [q("days", {"type": "integer", "minimum": 1, "maximum": 365, "default": 45}, "how far back"),
              q("since", {"type": "string", "format": "date"}, "instead of days"),
              q("kind", {"type": "array", "items": {"type": "string", "enum": ["text", "vision", "embed", "speech"]}}, "repeatable"),
              q("publisher", strings, "repeatable; instead of the configured publishers")]
    build = {"type": "object", "properties": {
        "repo": {"type": "string"}, "quant": {"type": "string"}, "gb": {"type": "number"}, "bits": {"type": "number"},
        "bits_per_weight": {"type": ["number", "null"]}, "fits": {"type": "string", "enum": ["gpu", "memory", "no"]},
        "get": {"type": "string"}, "run": {"type": "string"}, "headroom_gb": {"type": "number"}}}
    model = {"type": "object", "properties": {
        "model": {"type": "string"}, "url": {"type": "string"}, "created": {"type": "string"}, "kind": {"type": "string"},
        "params_b": {"type": ["number", "null"]}, "licence": {"type": "string"}, "context": {"type": ["integer", "null"]},
        "downloads": {"type": ["integer", "null"]}, "likes": {"type": ["integer", "null"]},
        "verdict": {"type": "string", "enum": list(ORDER)}, "pick": {"oneOf": [build, {"type": "null"}]},
        "role": {"type": "string"}, "replaces": {"type": "string"}, "note": {"type": "string"},
        "builds": {"type": "array", "items": build}}}
    mach = {"type": "object", "properties": {k: {"type": "number"} for k in (
        "ram_gib", "gpu_gib", "fits_gpu_gib", "fits_memory_gib", "gpu_reserve_gib", "ram_reserve_gib")} | {
            "kind": {"type": "string"}, "unified": {"type": "boolean"}, "note": {"type": "string"}}}
    answer = {"type": "object", "required": ["schema", "machine"], "properties": {
        "schema": {"type": "integer", "const": SCHEMA}, "machine": mach, "models": {"type": "array", "items": model},
        "errors": {"type": "array", "items": {"type": "object"}}, "counts": {"type": "object"}, "since": {"type": "string"},
        "generated": {"type": "string"}}}

    def get(summary: str, params: list[dict[str, Any]], schema: dict[str, Any], op: str) -> dict[str, Any]:
        return {"get": {"operationId": op, "summary": summary, "parameters": params, "responses": {
            "200": {"description": "the answer", "content": {"application/json": {"schema": schema}}},
            "400": {"description": "the request was wrong; the message says how"},
            "502": {"description": "Hugging Face could not be reached"}}}}

    return {"openapi": "3.1.0",
            "info": {"title": "ai-models-comparison", "version": __version__, "license": {"name": "Apache-2.0"},
                     "description": "Which open-weights models fit a machine's memory. Read-only. Answers have `schema`: it grows by "
                                    "adding keys and changes only when a reader would break."},
            "paths": {
                "/api/v1/machine": get("A machine's memory and budgets", machine, answer, "machine"),
                "/api/v1/models": get("Judge the models named: one, or several to compare",
                                      [{"name": "id", "in": "query", "required": True, "schema": {**strings, "maxItems": MAX_MODELS},
                                        "description": "a Hugging Face id (org/Name); repeatable"}] + machine, answer, "models"),
                "/api/v1/new": get("New models, judged, best fit first", window + [
                    q("verdict", {"type": "array", "items": {"type": "string", "enum": list(ORDER)}}, "repeatable"),
                    q("limit", {"type": "integer", "minimum": 1}, "at most this many")] + machine, answer, "new"),
                "/api/v1/analyze": get("Which models are worth trying, role by role, with reasons", window + [
                    q("id", strings, "analyse these models instead of the new ones"),
                    q("top", {"type": "integer", "minimum": 1, "default": 3}, "how many to list for a role")] + machine,
                    {"type": "object"}, "analyze"),
                "/api/v1/catalog": get("The models without a verdict", window + [q("id", strings, "these models instead of the new ones")],
                                       {"type": "object"}, "catalog")}}


def make_handler(service: Service, opts: Options) -> type[BaseHTTPRequestHandler]:
    loopback = opts.host in LOOPBACK

    class Handler(BaseHTTPRequestHandler):
        server_version = f"ai-models-comparison/{__version__}"
        protocol_version = "HTTP/1.1"
        timeout = 60

        def log_message(self, fmt: str, *args: Any) -> None:
            if opts.verbose:
                sys.stderr.write(f"{self.address_string()} {fmt % args}\n")

        def send(self, status: int, body: bytes, ctype: str, extra: dict[str, str] | None = None) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def json(self, status: int, payload: Any, extra: dict[str, str] | None = None) -> None:
            headers = dict(extra or {})
            if opts.cors:
                headers["Access-Control-Allow-Origin"] = opts.cors
            self.send(status, json.dumps(payload, separators=(",", ":")).encode(), "application/json", headers)

        def error(self, status: int, code: str, message: str) -> None:
            self.json(status, {"error": {"code": code, "message": message}})

        def allowed(self, api: bool) -> bool:
            # a page on another site must not reach a server on this machine through a name that points here
            if loopback and _host(self.headers.get("Host", "")) not in LOOPBACK:
                self.error(403, "bad_host", "this server answers on this machine only")
                return False
            if api and opts.token:
                given = self.headers.get("Authorization", "").removeprefix("Bearer ").strip()
                if not hmac.compare_digest(given.encode(), opts.token.encode()):
                    self.error(401, "unauthorized", "send the token as: Authorization: Bearer <token>")
                    return False
            return True

        def do_OPTIONS(self) -> None:  # noqa: N802
            self.send(204, b"", "text/plain", {"Access-Control-Allow-Origin": opts.cors, "Access-Control-Allow-Methods": "GET",
                                               "Access-Control-Allow-Headers": "authorization"} if opts.cors else None)

        def do_HEAD(self) -> None:  # noqa: N802
            self.do_GET()

        def do_GET(self) -> None:  # noqa: N802
            url = urlsplit(self.path)
            path, query = url.path, parse_qs(url.query)
            api = path.startswith("/api/")
            if not self.allowed(api):
                return
            if path in STATIC:
                name, ctype = STATIC[path]
                return self.send(200, static(name), ctype, {"Cache-Control": "no-cache", "Content-Security-Policy": CSP})
            if path == "/healthz":
                return self.json(200, {"ok": True, "version": __version__})
            if path == "/openapi.json":
                return self.json(200, openapi())
            if not api:
                return self.error(404, "not_found", "see /openapi.json for what this server answers")

            def many(*names: str) -> list[str]:
                return [x for n in names for v in query.get(n, []) for x in v.split(",") if x]

            def one(name: str) -> str | None:
                return query[name][0] if query.get(name) else None

            def integer(name: str, default: int | None = None) -> int | None:
                return int(query[name][0]) if query.get(name) else default

            spec = {k: one(k) for k in SPEC}
            try:
                if path == "/api/v1/machine":
                    return self.json(200, service.machine(spec))
                if path == "/api/v1/models":
                    return self.json(200, service.models(many("id", "ids"), spec), {"Cache-Control": "max-age=300"})
                if path == "/api/v1/new":
                    return self.json(200, service.new(spec, integer("days", 45) or 45, one("since"), many("publisher") or None,
                                                      many("kind") or None, many("verdict") or None, integer("limit")),
                                     {"Cache-Control": "max-age=300"})
                if path == "/api/v1/analyze":
                    return self.json(200, service.analyze(spec, many("id", "ids") or None, integer("days", 45) or 45, one("since"),
                                                          many("publisher") or None, many("kind") or None, integer("top", 3) or 3),
                                     {"Cache-Control": "max-age=300"})
                if path == "/api/v1/catalog":
                    if many("id", "ids"):
                        entries, errors = service.entries(many("id", "ids"))
                        return self.json(200, {"schema": SCHEMA, "models": [e.to_dict() for e in entries], "errors": errors})
                    c = service.catalog(integer("days", 45) or 45, one("since"), many("publisher") or None, many("kind") or None)
                    return self.json(200, c.to_dict(), {"Cache-Control": "max-age=300"})
            except ValueError as e:
                return self.error(400, "bad_request", str(e))
            except OSError as e:
                return self.error(502, "hub_unreachable", f"Hugging Face could not be reached: {e}")
            return self.error(404, "not_found", "see /openapi.json for what this server answers")

    return Handler


def serve(service: Service, opts: Options) -> ThreadingHTTPServer:
    """The server, bound and ready: call `serve_forever()` on it."""
    if opts.host not in LOOPBACK and not opts.token:
        raise ValueError("listening beyond this machine needs a token: --token, or AI_MODELS_COMPARISON_TOKEN")
    server = ThreadingHTTPServer((opts.host, opts.port), make_handler(service, opts))
    server.daemon_threads = True
    return server
