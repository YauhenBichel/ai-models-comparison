# Copyright 2026 Yauhen Bichel
# SPDX-License-Identifier: Apache-2.0
"""What Hugging Face's public API says: which models a publisher released, how many weights one has, and
which GGUF builds of it exist with their sizes. Read-only. Nothing is downloaded but JSON."""
from __future__ import annotations

import http.client
import json
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

API = "https://huggingface.co/api"
ENDPOINT = "HF_ENDPOINT"   # the hub's address without /api, as Hugging Face's own tools read it: a mirror, or a test's fake
Fetch = Callable[[str], Any]
FetchError = (urllib.error.URLError, OSError, ValueError, http.client.HTTPException)   # an answer cut short is an HTTPException
WORKERS = 8                # questions asked of the hub at once: enough to be quick, few enough to be polite
RETRY_AFTER = 10.0         # a 429 that asks for a wait no longer than this is waited for, once

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


def api() -> str:
    """Where the hub's API is: Hugging Face, unless HF_ENDPOINT names a mirror (or a test names its fake)."""
    endpoint = os.environ.get(ENDPOINT, "").strip().rstrip("/")
    return f"{endpoint}/api" if endpoint else API


def http_json(url: str, timeout: float = 40.0, sleep: Callable[[float], None] = time.sleep) -> Any:
    headers = {"user-agent": "ai-models-comparison/0"}
    token = os.environ.get("HF_TOKEN")  # optional, for rate limits; never printed
    if token:
        headers["authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    for again in (True, False):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as r:  # noqa: S310
                return json.loads(r.read())
        except urllib.error.HTTPError as e:   # asked to slow down: wait as long as it says, once, when that is short
            wait = str(e.headers.get("Retry-After") or "").strip() if e.headers else ""
            if not (again and e.code == 429 and wait.isdigit() and int(wait) <= RETRY_AFTER):
                raise
            sleep(float(wait))
        except (ConnectionError, urllib.error.URLError) as e:   # a connection refused or dropped is tried once more; a timeout is not
            reason = getattr(e, "reason", e)
            if not again or not isinstance(reason, OSError) or isinstance(reason, TimeoutError):
                raise
            sleep(0.1)
    raise AssertionError("unreachable")


def why(e: BaseException) -> str:
    """A failed lookup's cause in a few words a person can act on."""
    if isinstance(e, urllib.error.HTTPError):
        return f"HTTP {e.code} {e.reason}".strip()
    if isinstance(e, urllib.error.URLError):
        return str(e.reason)
    if isinstance(e, http.client.IncompleteRead):
        return "the answer was cut short"
    if isinstance(e, json.JSONDecodeError):
        return "the answer was not JSON"
    return str(e) or type(e).__name__


class Unanswered(TimeoutError):
    """A question still waiting when the time allowed ran out."""


class Unreachable(OSError):
    """Every lookup failed: there is no answer, which is not the same as an answer of nothing."""


class Lookups:
    """One question's lookups: each counted, the failures kept with their cause, and none started after the
    deadline. It is itself a Fetch, so everything that takes one can be given one; `scope()` counts a part of
    the question (one model) on its own as well. A 401 or 404 is an answer, not a failure."""

    def __init__(self, fetch: Fetch, seconds: float | None = None, clock: Callable[[], float] = time.monotonic,
                 parent: Lookups | None = None) -> None:
        self._fetch, self._clock, self._parent = fetch, clock, parent
        self._lock: threading.Lock = parent._lock if parent else threading.Lock()
        self.seconds: float | None = parent.seconds if parent else seconds
        self._end: float | None = parent._end if parent else (None if seconds is None else clock() + seconds)
        self.asked = self.failed = 0
        self.cause = ""        # of the first failure

    @classmethod
    def of(cls, fetch: Fetch) -> Lookups:
        return fetch if isinstance(fetch, Lookups) else cls(fetch)

    def scope(self) -> Lookups:
        return Lookups(self._fetch, clock=self._clock, parent=self)

    def left(self) -> float | None:
        """Seconds until the deadline; None when there is none."""
        return None if self._end is None else max(0.0, self._end - self._clock())

    @property
    def late(self) -> bool:
        return self._end is not None and self._clock() >= self._end

    def _count(self, cause: str | None) -> None:
        node: Lookups | None = self
        with self._lock:
            while node is not None:
                node.asked += 1
                if cause is not None:
                    node.failed, node.cause = node.failed + 1, node.cause or cause
                node = node._parent

    def missed(self, cause: str | BaseException) -> None:
        """A lookup that failed, or was never answered."""
        self._count(cause if isinstance(cause, str) else why(cause))

    def out_of_time(self) -> TimeoutError:
        return TimeoutError(f"the time allowed for one question ({self.seconds:g} s) ran out")

    def _ask(self, url: str, shape: type | None) -> Any:
        try:
            if self.late:
                raise self.out_of_time()
            body = self._fetch(url)
            if shape is not None and not isinstance(body, shape):   # a 200 with something else in it: an error page, a proxy's note
                raise ValueError(f"the answer was not the {'list' if shape is list else 'object'} expected")
        except urllib.error.HTTPError as e:
            self._count(None if e.code in (401, 404) else why(e))
            raise
        except FetchError as e:
            self.missed(e)
            raise
        self._count(None)
        return body

    def __call__(self, url: str) -> Any:
        return self._ask(url, None)

    def rows(self, url: str) -> list[dict[str, Any]]:
        return [m for m in self._ask(url, list) if isinstance(m, dict)]

    def page(self, url: str) -> dict[str, Any]:
        page: dict[str, Any] = self._ask(url, dict)
        return page

    def summary(self, **more: Any) -> dict[str, Any] | None:
        """What an answer says of itself when part of it is missing; None when nothing is."""
        if not self.failed:
            return None
        return {"failed": self.failed, "of": self.asked, "cause": self.cause, "out_of_time": self.late, **more}

    def unreachable(self) -> Unreachable:
        return Unreachable(f"Hugging Face could not be reached: {self.failed} of {self.asked} lookups failed ({self.cause})")


def each(fn: Callable[[Any], Any], items: Iterable[Any], workers: int = WORKERS, seconds: float | None = None) -> list[Any]:
    """`fn` of every item, a few at a time, in the order given. What `fn` raised is returned in its place; an
    item not answered within `seconds` gets an Unanswered, and its thread is left behind, not waited for."""
    todo = list(items)
    out: list[Any] = [Unanswered("not answered in the time allowed")] * len(todo)
    queue = iter(enumerate(todo))
    lock, stop = threading.Lock(), threading.Event()

    def work() -> None:
        while not stop.is_set():
            with lock:
                item = next(queue, None)
            if item is None:
                return
            try:
                got = fn(item[1])
            except Exception as e:  # noqa: BLE001  the caller decides what a failure means
                got = e
            if not stop.is_set():
                out[item[0]] = got

    threads = [threading.Thread(target=work, daemon=True) for _ in range(max(1, min(workers, len(todo))))]
    end = None if seconds is None else time.monotonic() + seconds
    for t in threads:
        t.start()
    for t in threads:
        t.join(None if end is None else max(0.0, end - time.monotonic()))
    with lock:
        stop.set()
    return list(out)


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
    asks = Lookups.of(fetch)
    base = model.split("/")[-1]
    try:
        found = asks.rows(f"{api()}/models?search={urllib.parse.quote(base)}&filter=gguf&sort=downloads&direction=-1&limit=30")
    except FetchError:
        return []
    ids = [m["id"] for m in found if isinstance(m.get("id"), str)]
    repos = [r for r in ids if _same_model(r, base) and not VARIANT.search(r)]

    def rank(repo: str) -> int:
        owner = repo.split("/")[0]
        if owner == model.split("/")[0]:
            return 0
        return 1 + quantizers.index(owner) if owner in quantizers else 50

    for repo in sorted(repos, key=rank)[:3]:
        try:
            builds = builds_from_tree(repo, asks.rows(f"{api()}/models/{repo}/tree/main?recursive=true"))
        except FetchError:
            continue
        if builds:
            return builds
    return []


class NotFound(LookupError):
    """Hugging Face has no such model (or it is private)."""


def model_id(text: str) -> str:
    """`org/Name` from what a person pastes: an id, a model page's address, with or without a trailing part."""
    text = re.sub(r"^https?://(www\.)?(huggingface\.co|hf\.co)/", "", text.strip()).strip("/")
    parts = text.split("/")
    if len(parts) < 2 or not all(_NAME.fullmatch(x) and ".." not in x and "--" not in x for x in parts[:2]):
        raise ValueError(f"not a Hugging Face model id: {short(text)!r} (expected org/Name: letters, digits, '-', '_' and '.', "
                         f"at most {MAX_NAME} a part)")
    return "/".join(parts[:2])


MAX_NAME = 96   # of an owner or of a name, as Hugging Face allows
_NAME = re.compile(rf"(?![-.])[A-Za-z0-9._-]{{1,{MAX_NAME}}}(?<![-.])")


def short(text: str, most: int = 120) -> str:
    """What someone sent, cut to what is worth repeating back to them."""
    return text if len(text) <= most else f"{text[:most]}... ({len(text)} characters)"


def model_page(model: str, fetch: Fetch) -> dict[str, Any]:
    """The model's own page as the API gives it; {} when it cannot be read; NotFound when it does not exist."""
    try:
        return Lookups.of(fetch).page(f"{api()}/models/{model}")
    except urllib.error.HTTPError as e:
        if e.code in (401, 404):
            raise NotFound(model) from e
        return {}
    except FetchError:
        return {}


def facts(info: dict[str, Any]) -> dict[str, Any]:
    """What a model page says that is worth comparing: weights, kind, licence, context, popularity."""
    card = info.get("cardData") or {}
    gguf = info.get("gguf") or {}
    total = (info.get("safetensors") or {}).get("total") or gguf.get("total")
    licence = str(card.get("license") or "")
    if licence in ("", "other"):
        licence = str(card.get("license_name") or licence)
    return {"params_b": round(total / 1e9, 1) if total else None, "kind": KINDS.get(str(info.get("pipeline_tag") or "")),
            "licence": licence, "context": gguf.get("context_length"), "architecture": str(gguf.get("architecture") or ""),
            "downloads": info.get("downloads"), "likes": info.get("likes"), "created": str(info.get("createdAt") or "")[:10]}


def new_models(since: str, fetch: Fetch, publishers: list[str] = PUBLISHERS, kinds: set[str] | None = None,
               on_error: Callable[[str, Exception], None] | None = None, workers: int = WORKERS) -> list[tuple[str, str, str]]:
    """(id, created, kind) of the publishers' models created on or after `since` (YYYY-MM-DD), newest first.
    A publisher whose list could not be read is left out, and `on_error` is told: the caller must not take
    the rest for the whole."""
    asks = Lookups.of(fetch)
    out: list[tuple[str, str, str]] = []

    def listed(org: str) -> list[dict[str, Any]]:
        return asks.rows(f"{api()}/models?author={urllib.parse.quote(org)}&sort=createdAt&direction=-1&limit=50")

    for org, rows in zip(publishers, each(listed, publishers, workers, asks.left()), strict=True):
        if isinstance(rows, Unanswered):
            asks.missed(asks.out_of_time())   # never answered: the lookup itself did not get to count it
        if isinstance(rows, FetchError):
            if on_error:
                on_error(org, rows)
            continue
        if isinstance(rows, BaseException):
            raise rows
        for m in rows:
            created = str(m.get("createdAt") or "")
            kind = KINDS.get(str(m.get("pipeline_tag") or ""))
            if not isinstance(m.get("id"), str) or created[:10] < since or kind is None or m.get("private") or SKIP_NAME.search(m["id"]):
                continue
            if kinds and kind not in kinds:
                continue
            out.append((m["id"], created, kind))
    return sorted(out, key=lambda r: r[1], reverse=True)
