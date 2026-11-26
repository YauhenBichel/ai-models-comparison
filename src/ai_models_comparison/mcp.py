# Copyright 2026 Yauhen Bichel
# SPDX-License-Identifier: Apache-2.0
"""The same answers for an agent, over the Model Context Protocol: `ai-models-comparison mcp` speaks it on
standard input and output.

Four tools, all read-only. Each gives a short text an agent can read in a few hundred tokens, and the full
answer as structured content in the same shape the HTTP API and `--json` give.

Two eras of the protocol are served from the same process. Since revision 2026-07-28 there is no handshake:
every request names its version in `_meta`, and `server/discover` says what the server speaks. Older clients
open with `initialize`; they are answered as before.
"""
from __future__ import annotations

import json
import math
import queue
import sys
import threading
import traceback
from typing import IO, Any

from . import __version__
from .hub import MAX_NAME, short
from .judge import ORDER
from .report import text_analysis, text_compare, text_machine, text_models, text_new
from .service import MAX_DAYS, MAX_MODELS, Service

MODERN = ("2026-07-28",)                                            # a version in every request's _meta; no handshake
LEGACY = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")   # opened with `initialize`; newest first
VERSION_KEY = "io.modelcontextprotocol/protocolVersion"
SERVER = {"name": "ai-models-comparison", "title": "AI models comparison", "version": __version__}
INSTRUCTIONS = ("Which open-weights models fit a machine's memory. `analyze` answers 'what should I try on this machine?' with "
                "reasons; `judge_models` judges named models (several at once to compare); `new_models` lists what was released "
                "lately; `machine` says what the machine has.")

_GIB = {"type": "number", "minimum": 0, "maximum": 1_000_000}
_MACHINE = {
    "gpu_gb": {**_GIB, "description": "GPU memory in GiB. Omit to use the machine this server runs on."},
    "ram_gb": {**_GIB, "description": "System memory in GiB. Omit to use the machine this server runs on."},
    "unified": {"type": "boolean", "description": "GPU and system memory are one pool (Apple silicon; an AMD APU without a carve-out)."},
}
_INCOMPLETE = {"type": "object", "description": "Present only when part of the answer is missing: how many lookups of Hugging Face "
                "failed of how many, and why. What is listed is true; what is not listed is not known.",
                "properties": {"failed": {"type": "integer"}, "of": {"type": "integer"}, "cause": {"type": "string"},
                               "out_of_time": {"type": "boolean"}}}
_ANSWER = {"type": "object", "properties": {"schema": {"type": "integer"}, "machine": {"type": "object"},
                                            "models": {"type": "array", "items": {"type": "object"}}, "incomplete": _INCOMPLETE},
           "required": ["schema", "machine"]}
_ID = {"type": "string", "minLength": 3, "maxLength": 2 * MAX_NAME + 40}   # org/Name, or its page's address
_READ_ONLY = {"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True}

TOOLS: list[dict[str, Any]] = [
    {"name": "machine", "title": "This machine's memory",
     "description": "GPU and system memory of the machine, and the two budgets that follow: how many GiB of model weights fit the "
                    "GPU, and how many fit GPU and system memory together. Give gpu_gb and ram_gb to ask about another machine.",
     "inputSchema": {"type": "object", "properties": _MACHINE, "additionalProperties": False},
     "outputSchema": _ANSWER, "annotations": {**_READ_ONLY, "openWorldHint": False}},
    {"name": "judge_models", "title": "Does this model fit? Compare models",
     "description": "For one or several Hugging Face models: every GGUF build with its size, the best build that fits the machine, "
                    "a verdict (fits the GPU / fits in memory / low-bit only / too big / no GGUF yet), and the commands to get and "
                    "run it. With several models the text is a side-by-side comparison. A verdict is about memory only: it does "
                    "not say the model is good.",
     "inputSchema": {"type": "object", "properties": {
         "models": {"type": "array", "minItems": 1, "maxItems": MAX_MODELS, "items": _ID,
                    "description": "Hugging Face ids (org/Name) or model page addresses. A GGUF repository is judged by its own files."},
         **_MACHINE}, "required": ["models"], "additionalProperties": False},
     "outputSchema": _ANSWER, "annotations": _READ_ONLY},
    {"name": "new_models", "title": "New open-weights models, judged",
     "description": "The models the main publishers released in the last days, each judged for the machine, best fit first: one "
                    "line a model. Use it to answer 'is there anything new worth trying on this machine?'.",
     "inputSchema": {"type": "object", "properties": {
         "days": {"type": "integer", "minimum": 1, "maximum": MAX_DAYS, "default": 45},
         "kinds": {"type": "array", "items": {"type": "string", "enum": ["text", "vision", "embed", "speech"]}},
         "verdicts": {"type": "array", "items": {"type": "string", "enum": list(ORDER)},
                      "description": "Only these verdicts, e.g. ['fits the GPU', 'fits in memory']."},
         "limit": {"type": "integer", "minimum": 1, "maximum": 200, "default": 25},
         **_MACHINE}, "additionalProperties": False},
     "outputSchema": _ANSWER, "annotations": _READ_ONLY},
    {"name": "analyze", "title": "What is worth trying on this machine",
     "description": "Role by role (coder, general, vision, ocr, embed, audio): the short list of models worth trying on the machine, "
                    "each with its reasons and cautions, compared with the model in use when the configuration names one. Of the "
                    "new models, or of the models named. Ranks are weights after the cost of quantization: an ordering, not a "
                    "benchmark.",
     "inputSchema": {"type": "object", "properties": {
         "models": {"type": "array", "maxItems": MAX_MODELS, "items": _ID,
                    "description": "Analyse these Hugging Face ids instead of the new models."},
         "days": {"type": "integer", "minimum": 1, "maximum": MAX_DAYS, "default": 45},
         "kinds": {"type": "array", "items": {"type": "string", "enum": ["text", "vision", "embed", "speech"]}},
         **_MACHINE}, "additionalProperties": False},
     "outputSchema": {"type": "object", "properties": {"schema": {"type": "integer"}, "machine": {"type": "object"},
                                                       "roles": {"type": "array", "items": {"type": "object"}},
                                                       "incomplete": _INCOMPLETE},
                      "required": ["schema", "machine", "roles"]}, "annotations": _READ_ONLY},
]


SCHEMAS = {t["name"]: t["inputSchema"] for t in TOOLS}
_TYPES = {"string": "a string", "boolean": "true or false", "number": "a number", "integer": "a whole number", "array": "a list"}


def _checked(name: str, value: Any, schema: dict[str, Any]) -> Any:
    """The value, when it is what the schema says; ValueError naming the argument when it is not."""
    kind = schema["type"]
    said = short(json.dumps(value, ensure_ascii=False), 60)

    def wrong(must: str) -> ValueError:
        return ValueError(f"invalid argument `{name}`: must be {must} (got {said})")

    if kind in ("number", "integer"):
        if isinstance(value, bool) or not isinstance(value, int | float) or (isinstance(value, float) and not math.isfinite(value)):
            raise wrong(_TYPES[kind])
        if kind == "integer":
            if value != int(value):
                raise wrong(_TYPES[kind])
            value = int(value)
        low, high = schema.get("minimum"), schema.get("maximum")
        if (low is not None and value < low) or (high is not None and value > high):
            raise wrong(f"{_TYPES[kind]} from {low} to {high}" if None not in (low, high) else f"{_TYPES[kind]} of at least {low}")
    elif kind == "boolean":
        if not isinstance(value, bool):
            raise wrong(_TYPES[kind])
    elif kind == "string":
        if not isinstance(value, str):
            raise wrong(_TYPES[kind])
        if not schema.get("minLength", 0) <= len(value) <= schema.get("maxLength", len(value)):
            raise wrong(f"a string of {schema.get('minLength', 0)} to {schema.get('maxLength')} characters")
        if "enum" in schema and value not in schema["enum"]:
            raise wrong(f"one of {schema['enum']}")
    elif kind == "array":
        if not isinstance(value, list):
            raise wrong(_TYPES[kind])
        if not schema.get("minItems", 0) <= len(value) <= schema.get("maxItems", len(value)):
            raise wrong(f"a list of {schema.get('minItems', 0)} to {schema.get('maxItems', 'any number of')} items")
        value = [_checked(f"{name}[{i}]", x, schema["items"]) for i, x in enumerate(value)]
    return value


def validate(tool: str, args: Any) -> dict[str, Any]:
    """A tool's arguments, checked against the schema `tools/list` declares: every type and range, in this one
    place. ValueError naming the argument; nothing a caller sent reaches a tool unchecked."""
    schema = SCHEMAS[tool]
    if not isinstance(args, dict):
        raise ValueError("invalid arguments: must be an object")
    unknown = sorted(str(k) for k in args if k not in schema["properties"])
    if unknown:
        raise ValueError(f"unknown argument `{short(unknown[0], 60)}`: {tool} takes {', '.join(schema['properties'])}")
    missing = [k for k in schema.get("required", []) if args.get(k) is None]
    if missing:
        raise ValueError(f"missing argument `{missing[0]}`")
    return {k: _checked(k, v, schema["properties"][k]) for k, v in args.items() if v is not None}


def call(name: str, args: dict[str, Any], service: Service) -> tuple[str, dict[str, Any]]:
    """(the text, the full answer) of one tool. LookupError for a tool there is not; ValueError for arguments
    that are not what its schema says."""
    if name not in SCHEMAS:
        raise LookupError(name)
    args = validate(name, args)
    if name == "machine":
        answer = service.machine(args)
        return text_machine(answer), answer
    if name == "judge_models":
        answer = service.models(args["models"], args)
        return (text_compare(answer) if len(answer["models"]) > 1 else text_models(answer)), answer
    if name == "new_models":
        answer = service.new(args, days=args.get("days", 45), kinds=args.get("kinds"), verdicts=args.get("verdicts"),
                             limit=args.get("limit", 25))
        return text_new(answer), answer
    answer = service.analyze(args, args.get("models") or None, args.get("days", 45), kinds=args.get("kinds"))
    return text_analysis(answer), answer


def handle(msg: Any, service: Service) -> dict[str, Any] | None:
    """One JSON-RPC message in, its response out; None for a notification."""
    if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0" or not isinstance(msg.get("method"), str):
        return {"jsonrpc": "2.0", "id": msg.get("id") if isinstance(msg, dict) else None,
                "error": {"code": -32600, "message": "not a JSON-RPC 2.0 request"}}
    if "id" not in msg:
        return None
    method = msg["method"]
    params: dict[str, Any] = msg["params"] if isinstance(msg.get("params"), dict) else {}

    def reply(result: dict[str, Any]) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": msg["id"], "result": result}

    if method == "initialize":   # a client of the handshake era
        asked = str(params.get("protocolVersion") or "")
        return reply({"protocolVersion": asked if asked in LEGACY else LEGACY[0], "capabilities": {"tools": {"listChanged": False}},
                      "serverInfo": SERVER, "instructions": INSTRUCTIONS})
    if method == "ping":
        return reply({})
    meta = params.get("_meta")
    named = meta.get(VERSION_KEY) if isinstance(meta, dict) else None
    if named is not None and named not in MODERN + LEGACY:
        return {"jsonrpc": "2.0", "id": msg["id"], "error": {"code": -32022, "message": "Unsupported protocol version",
                                                             "data": {"supported": list(MODERN + LEGACY), "requested": named}}}
    done = {"resultType": "complete", "_meta": {"io.modelcontextprotocol/serverInfo": SERVER}}
    if method == "server/discover":
        return reply({**done, "supportedVersions": list(MODERN + LEGACY), "capabilities": {"tools": {}}, "instructions": INSTRUCTIONS,
                      "ttlMs": 3_600_000, "cacheScope": "public"})
    if method == "tools/list":
        return reply({**done, "tools": TOOLS, "ttlMs": 3_600_000, "cacheScope": "public"})
    if method == "tools/call":
        name, args = params.get("name"), params.get("arguments")
        if not isinstance(name, str) or name not in SCHEMAS:   # the only place this is said: a tool's own errors are its own
            return {"jsonrpc": "2.0", "id": msg["id"], "error": {"code": -32602, "message": f"unknown tool: {short(str(name), 60)}"}}

        def failed(text: str) -> dict[str, Any]:
            return reply({**done, "content": [{"type": "text", "text": f"error: {text}"}], "isError": True})

        try:
            text, answer = call(name, {} if args is None else args, service)
        except ValueError as e:   # the caller can correct these: say what was wrong, as a tool result
            return failed(str(e))
        except OSError as e:      # the hub could not be reached, or a file could not be read: the message says which
            return failed(str(e))
        except Exception as e:  # noqa: BLE001  a fault of this program: it answers and goes on; the trace is for whoever runs it
            traceback.print_exc(file=sys.stderr)
            return failed(f"internal error ({type(e).__name__}); the server is still running, and its log has the details")
        return reply({**done, "content": [{"type": "text", "text": text}], "structuredContent": answer, "isError": False})
    return {"jsonrpc": "2.0", "id": msg["id"], "error": {"code": -32601, "message": f"method not found: {method}"}}


def answer(line: str, service: Service) -> dict[str, Any] | None:
    """One line in, its response out. Whatever happens inside, this returns: nothing a client sends ends the server."""
    msg: Any = None
    try:
        try:
            msg = json.loads(line)
        except (ValueError, RecursionError):
            return {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}}
        return handle(msg, service)
    except Exception as e:  # noqa: BLE001
        traceback.print_exc(file=sys.stderr)
        if not isinstance(msg, dict) or "id" not in msg:
            return None
        return {"jsonrpc": "2.0", "id": msg["id"], "error": {"code": -32603, "message": f"internal error ({type(e).__name__})"}}


def _is_call(line: str) -> bool:
    return '"tools/call"' in line


def serve_stdio(service: Service, stdin: IO[str] | None = None, stdout: IO[str] | None = None) -> int:
    """One JSON message a line, as the protocol's stdio transport says. Nothing else is ever written to stdout.

    Tool calls are answered one at a time, in the order they came, by one worker; everything else (`ping`,
    `tools/list`, the handshake) is answered at once, so a client can tell a slow lookup from a dead server."""
    stdin, stdout = stdin or sys.stdin, stdout or sys.stdout
    writing = threading.Lock()
    calls: queue.SimpleQueue[str | None] = queue.SimpleQueue()

    def write(out: dict[str, Any] | None) -> None:
        if out is None:
            return
        try:
            line = json.dumps(out, separators=(",", ":"))
        except (TypeError, ValueError):   # an id or an answer JSON cannot carry
            line = json.dumps({"jsonrpc": "2.0", "id": None, "error": {"code": -32603, "message": "internal error (the answer is not JSON)"}})
        with writing:
            stdout.write(line + "\n")
            stdout.flush()

    def work() -> None:
        while (line := calls.get()) is not None:
            write(answer(line, service))

    worker = threading.Thread(target=work, name="tool-calls", daemon=True)
    worker.start()
    try:
        for line in stdin:
            if not line.strip():
                continue
            if _is_call(line):
                calls.put(line)
            else:
                write(answer(line, service))
    finally:
        calls.put(None)
        worker.join()   # what was asked before the input ended is still answered
    return 0
