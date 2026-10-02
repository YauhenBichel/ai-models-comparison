# Copyright 2026 Yauhen Bichel
# SPDX-License-Identifier: Apache-2.0
"""What Hugging Face's public API says: which models a publisher released, how many weights one has, and
which GGUF builds of it exist with their sizes. Read-only. Nothing is downloaded but JSON."""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

API = "https://huggingface.co/api"
Fetch = Callable[[str], Any]
FetchError = (urllib.error.URLError, OSError, ValueError)

PUBLISHERS = ["deepseek-ai", "Qwen", "openai", "google", "mistralai", "zai-org", "moonshotai", "MiniMaxAI", "meta-llama",
              "nvidia", "microsoft", "ibm-granite", "tencent", "baidu", "xai-org", "CohereLabs", "LiquidAI", "stepfun-ai",
              "ByteDance-Seed", "allenai", "BAAI", "jinaai", "nomic-ai"]
QUANTIZERS = ["unsloth", "bartowski", "ggml-org", "lmstudio-community", "mradermacher"]
KINDS = {"text-generation": "text", "image-text-to-text": "vision", "any-to-any": "vision", "feature-extraction": "embed",
         "sentence-similarity": "embed", "automatic-speech-recognition": "speech", "text-to-speech": "speech"}
SKIP_NAME = re.compile(r"(?i)(-lora\b|adapter|-awq\b|-exl2\b|-mlx\b|onnx|safety|guard|reranker)")
AUXILIARY = re.compile(r"(?i)mmproj|draft|vision-?proj|tokenizer|vocoder|dspark")  # files beside a model, not builds of it
QUANT = re.compile(r"(?i)(?:^|[-._/])((?:UD-)?(?:I?Q\d(?:_[A-Z0-9]+)*|MXFP4|BF16|F16|F32))(?=[-._/]|$)")
VARIANT = re.compile(r"(?i)abliterat|uncensor|heretic")


@dataclass(slots=True, frozen=True)
class Build:
    repo: str
    quant: str
    bits: float
    size: int


def http_json(url: str) -> Any:
    headers = {"user-agent": "ai-models-comparison/0"}
    token = os.environ.get("HF_TOKEN")  # optional, for rate limits; never printed
    if token:
        headers["authorization"] = f"Bearer {token}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=40) as r:  # noqa: S310
        return json.loads(r.read())


def bits_of(quant: str) -> float:
    q = quant.upper().removeprefix("UD-")
    if q in ("F16", "BF16"):
        return 16.0
    if q == "F32":
        return 32.0
    if q == "MXFP4":
        return 4.0
    m = re.match(r"I?Q(\d)", q)
    return float(m.group(1)) if m else 0.0


def builds_from_tree(repo: str, tree: list[dict[str, Any]]) -> list[Build]:
    """A repository's .gguf files grouped by quantization: shards and folders summed, smallest first."""
    sizes: dict[str, int] = {}
    for f in tree:
        path = str(f.get("path", ""))
        if not path.endswith(".gguf") or AUXILIARY.search(path):
            continue
        found = QUANT.findall(path)
        if not found:
            continue
        quant = found[-1].upper()
        sizes[quant] = sizes.get(quant, 0) + int((f.get("lfs") or {}).get("size") or f.get("size") or 0)
    return sorted((Build(repo, q, bits_of(q), s) for q, s in sizes.items() if s > 0), key=lambda b: b.size)


def _same_model(repo: str, base: str) -> bool:
    """`unsloth/Name-GGUF` is a build of `Name`; `unsloth/Name-Flash-GGUF` is a build of something else."""
    name = repo.split("/")[-1].lower()
    name = re.sub(r"([-._](i1|imatrix))?[-._]gguf$", "", name)
    return name == base.lower()


def find_builds(model: str, fetch: Fetch, quantizers: list[str] = QUANTIZERS) -> list[Build]:
    """The GGUF builds of exactly this model: the publisher's own repository first, then the trusted
    quantizers, then the most downloaded."""
    base = model.split("/")[-1]
    try:
        found = fetch(f"{API}/models?search={urllib.parse.quote(base)}&filter=gguf&sort=downloads&direction=-1&limit=30")
    except FetchError:
        return []
    repos = [m["id"] for m in found if _same_model(m["id"], base) and not VARIANT.search(m["id"])]

    def rank(repo: str) -> int:
        owner = repo.split("/")[0]
        if owner == model.split("/")[0]:
            return 0
        return 1 + quantizers.index(owner) if owner in quantizers else 50

    for repo in sorted(repos, key=rank)[:3]:
        try:
            builds = builds_from_tree(repo, fetch(f"{API}/models/{repo}/tree/main?recursive=true"))
        except FetchError:
            continue
        if builds:
            return builds
    return []


def parameters(model: str, fetch: Fetch) -> tuple[float | None, str | None]:
    """(billions of weights, kind) from the model's own page; either may be unknown."""
    try:
        info = fetch(f"{API}/models/{model}")
    except FetchError:
        return None, None
    total = (info.get("safetensors") or {}).get("total")
    return (round(total / 1e9, 1) if total else None), KINDS.get(str(info.get("pipeline_tag") or ""))


def new_models(since: str, fetch: Fetch, publishers: list[str] = PUBLISHERS, kinds: set[str] | None = None,
               on_error: Callable[[str, Exception], None] | None = None) -> list[tuple[str, str, str]]:
    """(id, created, kind) of the publishers' models created on or after `since` (YYYY-MM-DD), newest first."""
    out: list[tuple[str, str, str]] = []
    for org in publishers:
        try:
            rows = fetch(f"{API}/models?author={urllib.parse.quote(org)}&sort=createdAt&direction=-1&limit=50")
        except FetchError as e:
            if on_error:
                on_error(org, e)
            continue
        for m in rows:
            created = str(m.get("createdAt") or "")
            kind = KINDS.get(str(m.get("pipeline_tag") or ""))
            if created[:10] < since or kind is None or m.get("private") or SKIP_NAME.search(m["id"]):
                continue
            if kinds and kind not in kinds:
                continue
            out.append((m["id"], created, kind))
    return sorted(out, key=lambda r: r[1], reverse=True)
