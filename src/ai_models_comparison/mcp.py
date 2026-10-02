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
import sys
from typing import IO, Any

from . import __version__
from .judge import ORDER
from .report import text_analysis, text_compare, text_machine, text_models, text_new
from .service import MAX_MODELS, Service

MODERN = ("2026-07-28",)                                            # a version in every request's _meta; no handshake
LEGACY = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")   # opened with `initialize`; newest first
VERSION_KEY = "io.modelcontextprotocol/protocolVersion"
SERVER = {"name": "ai-models-comparison", "title": "AI models comparison", "version": __version__}
INSTRUCTIONS = ("Which open-weights models fit a machine's memory. `analyze` answers 'what should I try on this machine?' with "
                "reasons; `judge_models` judges named models (several at once to compare); `new_models` lists what was released "
                "lately; `machine` says what the machine has.")

_MACHINE = {
    "gpu_gb": {"type": "number", "minimum": 0, "description": "GPU memory in GiB. Omit to use the machine this server runs on."},
    "ram_gb": {"type": "number", "minimum": 0, "description": "System memory in GiB. Omit to use the machine this server runs on."},
    "unified": {"type": "boolean", "description": "GPU and system memory are one pool (Apple silicon; an AMD APU without a carve-out)."},
}
_ANSWER = {"type": "object", "properties": {"schema": {"type": "integer"}, "machine": {"type": "object"},
                                            "models": {"type": "array", "items": {"type": "object"}}},
           "required": ["schema", "machine"]}
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
         "models": {"type": "array", "minItems": 1, "maxItems": MAX_MODELS, "items": {"type": "string"},
                    "description": "Hugging Face ids (org/Name) or model page addresses. A GGUF repository is judged by its own files."},
         **_MACHINE}, "required": ["models"], "additionalProperties": False},
     "outputSchema": _ANSWER, "annotations": _READ_ONLY},
    {"name": "new_models", "title": "New open-weights models, judged",
     "description": "The models the main publishers released in the last days, each judged for the machine, best fit first: one "
                    "line a model. Use it to answer 'is there anything new worth trying on this machine?'.",
     "inputSchema": {"type": "object", "properties": {
         "days": {"type": "integer", "minimum": 1, "maximum": 365, "default": 45},
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
         "models": {"type": "array", "maxItems": MAX_MODELS, "items": {"type": "string"},
                    "description": "Analyse these Hugging Face ids instead of the new models."},
         "days": {"type": "integer", "minimum": 1, "maximum": 365, "default": 45},
         "kinds": {"type": "array", "items": {"type": "string", "enum": ["text", "vision", "embed", "speech"]}},
         **_MACHINE}, "additionalProperties": False},
     "outputSchema": {"type": "object", "properties": {"schema": {"type": "integer"}, "machine": {"type": "object"},
                                                       "roles": {"type": "array", "items": {"type": "object"}}},
                      "required": ["schema", "machine", "roles"]}, "annotations": _READ_ONLY},
]


def call(name: str, args: dict[str, Any], service: Service) -> tuple[str, dict[str, Any]]:
    """(the text, the full answer) of one tool."""
    if name == "machine":
        answer = service.machine(args)
        return text_machine(answer), answer
    if name == "judge_models":
        models = args.get("models")
        if not isinstance(models, list) or not all(isinstance(x, str) for x in models):
            raise ValueError("models must be a list of Hugging Face ids")
        answer = service.models(models, args)
        return (text_compare(answer) if len(answer["models"]) > 1 else text_models(answer)), answer
    if name == "new_models":
        answer = service.new(args, days=int(args.get("days") or 45), kinds=args.get("kinds"), verdicts=args.get("verdicts"),
                             limit=int(args.get("limit") or 25))
        return text_new(answer), answer
    if name == "analyze":
        answer = service.analyze(args, args.get("models") or None, int(args.get("days") or 45), kinds=args.get("kinds"))
        return text_analysis(answer), answer
    raise KeyError(name)


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
        name, args = str(params.get("name") or ""), params.get("arguments") or {}
        try:
            text, answer = call(name, args if isinstance(args, dict) else {}, service)
        except KeyError:
            return {"jsonrpc": "2.0", "id": msg["id"], "error": {"code": -32602, "message": f"unknown tool: {name}"}}
        except (ValueError, TypeError) as e:   # the caller can correct these: say what was wrong, as a tool result
            return reply({**done, "content": [{"type": "text", "text": f"error: {e}"}], "isError": True})
        except OSError as e:
            return reply({**done, "content": [{"type": "text", "text": f"error: Hugging Face could not be reached ({e})"}],
                          "isError": True})
        return reply({**done, "content": [{"type": "text", "text": text}], "structuredContent": answer, "isError": False})
    return {"jsonrpc": "2.0", "id": msg["id"], "error": {"code": -32601, "message": f"method not found: {method}"}}


def serve_stdio(service: Service, stdin: IO[str] | None = None, stdout: IO[str] | None = None) -> int:
    """One JSON message a line, as the protocol's stdio transport says. Nothing else is ever written to stdout."""
    stdin, stdout = stdin or sys.stdin, stdout or sys.stdout
    for line in stdin:
        if not line.strip():
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            out: dict[str, Any] | None = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}}
        else:
            out = handle(msg, service)
        if out is not None:
            stdout.write(json.dumps(out, separators=(",", ":")) + "\n")
            stdout.flush()
    return 0
