# Copyright 2026 Yauhen Bichel
# SPDX-License-Identifier: Apache-2.0
"""Answers from Hugging Face kept on disk for a few hours: a second question costs nothing, a busy or
unreachable hub still gets an answer (the last one), and many clients of one server share them."""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

from .hub import Fetch, FetchError

TTL = 6 * 3600.0


def directory() -> Path:
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "ai-models-comparison"


def cached(fetch: Fetch, where: Path | None = None, ttl: float = TTL, now: Callable[[], float] = time.time) -> Fetch:
    where = where or directory()

    def get(url: str) -> object:
        path = where / (hashlib.sha256(url.encode()).hexdigest()[:32] + ".json")
        saved = None
        with contextlib.suppress(OSError, ValueError):
            saved = json.loads(path.read_text())
        if saved is not None and now() - float(saved["t"]) < ttl:
            return saved["body"]
        try:
            body = fetch(url)
        except FetchError:
            if saved is not None:
                return saved["body"]  # an old answer beats none
            raise
        with contextlib.suppress(OSError):  # a cache that cannot be written is only a slower run
            where.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=where, suffix=".tmp")
            with os.fdopen(fd, "w") as fh:
                json.dump({"t": now(), "url": url, "body": body}, fh)
            os.replace(tmp, path)
        return body

    return get
