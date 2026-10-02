# Copyright 2026 Yauhen Bichel
# SPDX-License-Identifier: Apache-2.0
"""The catalog: what is known about each model that does not depend on any machine. Its builds and their
sizes, its weights, kind, licence, context and popularity.

Building it is the slow part (several questions to Hugging Face a model). Judging it against a machine is
arithmetic. So the catalog is built once, cached, and can be published as one JSON file that any number of
people read without asking Hugging Face anything. `SCHEMA` changes only when a reader would break.
"""
from __future__ import annotations

import json
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .hub import API, KINDS, PUBLISHERS, QUANTIZERS, Build, Fetch, FetchError, builds_from_tree, facts, find_builds, model_page, new_models

SCHEMA = 1


@dataclass(slots=True)
class Entry:
    model: str
    created: str = ""                 # YYYY-MM-DD; "" when unknown
    kind: str = "text"                # text | vision | embed | speech
    params_b: float | None = None     # billions of weights
    licence: str = ""
    context: int | None = None        # the context length the build declares
    architecture: str = ""
    downloads: int | None = None
    likes: int | None = None
    builds: list[Build] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"model": self.model, "created": self.created, "kind": self.kind, "params_b": self.params_b, "licence": self.licence,
                "context": self.context, "architecture": self.architecture, "downloads": self.downloads, "likes": self.likes,
                "builds": [{"repo": b.repo, "quant": b.quant, "bits": b.bits, "bytes": b.size} for b in self.builds]}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Entry:
        return cls(str(d["model"]), str(d.get("created") or ""), str(d.get("kind") or "text"), d.get("params_b"),
                   str(d.get("licence") or ""), d.get("context"), str(d.get("architecture") or ""), d.get("downloads"), d.get("likes"),
                   [Build(str(b["repo"]), str(b["quant"]), float(b.get("bits") or 0), int(b["bytes"])) for b in d.get("builds") or []])


@dataclass(slots=True)
class Catalog:
    generated: str                    # UTC, ISO 8601
    since: str                        # models created on or after this day
    models: list[Entry] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"schema": SCHEMA, "generated": self.generated, "since": self.since, "models": [e.to_dict() for e in self.models]}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Catalog:
        if int(d.get("schema") or 0) > SCHEMA:
            raise ValueError(f"this catalog has schema {d.get('schema')}, newer than this version reads ({SCHEMA}): upgrade")
        return cls(str(d.get("generated") or ""), str(d.get("since") or ""), [Entry.from_dict(m) for m in d.get("models") or []])

    def find(self, model: str) -> Entry | None:
        want = model.lower()
        return next((e for e in self.models if e.model.lower() == want), None)


def entry_for(model: str, fetch: Fetch, created: str = "", kind: str | None = None, quantizers: list[str] | None = None) -> Entry:
    """Everything about one model. `model` may also be a GGUF repository: then its own files are the builds.
    Raises hub.NotFound when Hugging Face has no such model."""
    page = model_page(model, fetch)
    f = facts(page)
    e = Entry(model, created[:10] or f["created"], kind or f["kind"] or "text", f["params_b"], f["licence"], f["context"],
              f["architecture"], f["downloads"], f["likes"])
    if "gguf" in page or model.lower().endswith("gguf"):
        try:
            e.builds = builds_from_tree(model, fetch(f"{API}/models/{model}/tree/main?recursive=true"))
        except FetchError:
            e.builds = []
    else:
        e.builds = find_builds(model, fetch, quantizers or QUANTIZERS)
        if e.builds:  # the build repository knows what the model page does not: context, and weights when the page is silent
            try:
                b = facts(fetch(f"{API}/models/{e.builds[0].repo}"))
            except FetchError:
                b = {}
            e.context, e.architecture = b.get("context"), b.get("architecture") or ""
            e.params_b = e.params_b or b.get("params_b")
            e.licence = e.licence or b.get("licence") or ""
    if e.params_b:
        # no build of this model: a file under 0.75 bits a weight, or under half of what its own name promises
        # (an "F32" of half a gigabyte beside a model of four billion weights is a projector, not the model)
        def real(x: Build) -> bool:
            per_weight = x.size * 8 / (e.params_b * 1e9) if e.params_b else 0.0
            return per_weight >= max(0.75, x.bits / 2)
        e.builds = [x for x in e.builds if real(x)]
    return e


def build(since: str, fetch: Fetch, publishers: list[str] | None = None, kinds: set[str] | None = None,
          quantizers: list[str] | None = None, workers: int = 6,
          on_error: Callable[[str, Exception], None] | None = None) -> Catalog:
    """The watched publishers' models since a day, each with its builds. A few at a time: the answers are
    small, and Hugging Face allows a few hundred questions in five minutes."""
    if kinds and not kinds <= set(KINDS.values()):
        raise ValueError(f"kinds are {sorted(set(KINDS.values()))}")
    found = new_models(since, fetch, publishers or PUBLISHERS, kinds, on_error)

    def one(row: tuple[str, str, str]) -> Entry:
        try:
            return entry_for(row[0], fetch, row[1], row[2], quantizers)
        except LookupError:
            return Entry(row[0], row[1][:10], row[2])

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        entries = list(pool.map(one, found))
    return Catalog(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), since, entries)


def load(source: str, fetch: Fetch) -> Catalog:
    """A published catalog: an address or a file."""
    if source.startswith(("http://", "https://")):
        return Catalog.from_dict(fetch(source))
    return Catalog.from_dict(json.loads(Path(source).read_text()))


def schema() -> dict[str, Any]:
    """The published file's shape, for a reader in any language."""
    nullable_int = {"type": ["integer", "null"]}
    build_ = {"type": "object", "required": ["repo", "quant", "bits", "bytes"], "properties": {
        "repo": {"type": "string", "description": "the Hugging Face repository holding the files"},
        "quant": {"type": "string", "description": "the quantization, as its file names say it"},
        "bits": {"type": "number", "description": "nominal bits a weight (Q4_K_M is 4); 0 when the name does not say"},
        "bytes": {"type": "integer", "description": "all shards of this build together"}}}
    model = {"type": "object", "required": ["model", "builds"], "properties": {
        "model": {"type": "string", "description": "org/Name on Hugging Face"}, "created": {"type": "string"},
        "kind": {"type": "string", "enum": sorted(set(KINDS.values()))}, "params_b": {"type": ["number", "null"]},
        "licence": {"type": "string"}, "context": nullable_int, "architecture": {"type": "string"}, "downloads": nullable_int,
        "likes": nullable_int, "builds": {"type": "array", "items": build_}}}
    return {"$schema": "https://json-schema.org/draft/2020-12/schema", "title": "ai-models-comparison catalog", "type": "object",
            "required": ["schema", "generated", "since", "models"], "properties": {
                "schema": {"type": "integer", "const": SCHEMA, "description": "changes only when a reader would break; keys may be added"},
                "generated": {"type": "string", "format": "date-time"}, "since": {"type": "string", "format": "date"},
                "models": {"type": "array", "items": model}}}
