# Copyright 2026 Yauhen Bichel
# SPDX-License-Identifier: Apache-2.0
"""The one layer every interface calls: the command line, the HTTP API, the MCP server and the Python
library all get their answers here, in the same shape. An interface only parses a request and prints.

    from ai_models_comparison.service import Service
    answer = Service().models(["Qwen/Qwen3-Coder-Next"], {"gpu_gb": 24, "ram_gb": 64})

A request names its machine (or takes the one detected), so one running server answers for any machine.
"""
from __future__ import annotations

import datetime as dt
import threading
import time
from collections.abc import Mapping
from typing import Any

from . import catalog as cat
from .analyze import analyse
from .cache import cached
from .hub import QUANTIZERS, Fetch, NotFound, http_json, model_id
from .judge import ORDER, Judged, judge_entry
from .machine import SPEC, Budgets, Machine, as_dict, detect, resolve

SCHEMA = 1          # of the answers; grows by adding keys, changes only when a reader would break
MAX_MODELS = 12     # in one question


def order(judged: list[Judged]) -> list[Judged]:
    return sorted(judged, key=lambda j: (ORDER[j.verdict], j.role, -(j.params_b or 0), j.model))


class Service:
    def __init__(self, cfg: Mapping[str, Any] | None = None, fetch: Fetch | None = None, catalog: str | None = None,
                 detected: Machine | None = None, keep: float = 1800.0) -> None:
        self.cfg: Mapping[str, Any] = cfg or {}
        self.fetch: Fetch = fetch or cached(http_json)
        self.source = catalog or self.cfg.get("catalog")      # a published catalog (address or file), instead of asking the hub
        self.keep = keep                                      # seconds a built catalog is reused within this process
        self._detected = detected
        self._lock = threading.Lock()
        self._catalogs: dict[tuple[Any, ...], tuple[float, cat.Catalog]] = {}

    # ---- the machine -------------------------------------------------------------------------------------

    def resolve(self, given: Mapping[str, Any] | None = None) -> tuple[Machine, Budgets]:
        """The configuration file describes this machine; a request may describe another."""
        spec = {k: self.cfg[k] for k in SPEC if k in self.cfg}
        spec.update({k: v for k, v in (given or {}).items() if k in SPEC and v is not None and v != ""})
        with self._lock:
            if self._detected is None:
                self._detected = detect()
            detected = self._detected
        return resolve(spec, detected)

    def machine(self, given: Mapping[str, Any] | None = None) -> dict[str, Any]:
        m, b = self.resolve(given)
        return {"schema": SCHEMA, "machine": as_dict(m, b)}

    # ---- the catalog -------------------------------------------------------------------------------------

    def catalog(self, days: int = 45, since: str | None = None, publishers: list[str] | None = None,
                kinds: list[str] | None = None) -> cat.Catalog:
        since = since or (dt.datetime.now(dt.UTC) - dt.timedelta(days=days)).strftime("%Y-%m-%d")
        dt.date.fromisoformat(since)
        publishers = publishers or list(self.cfg.get("publishers") or []) or None
        want = set(kinds or self.cfg.get("kinds") or []) or None
        key = (self.source, since, tuple(publishers or ()), tuple(sorted(want or ())))
        with self._lock:
            hit = self._catalogs.get(key)
        if hit and time.monotonic() - hit[0] < self.keep:
            return hit[1]
        if self.source:
            whole = cat.load(str(self.source), self.fetch)
            orgs = {p.lower() for p in publishers} if publishers else None
            got = cat.Catalog(whole.generated, max(since, whole.since), [
                e for e in whole.models
                if e.created >= since and (not want or e.kind in want) and (not orgs or e.model.split("/")[0].lower() in orgs)])
        else:
            got = cat.build(since, self.fetch, publishers, want, list(self.cfg.get("quantizers") or QUANTIZERS))
        with self._lock:
            self._catalogs[key] = (time.monotonic(), got)
        return got

    def entries(self, ids: list[str]) -> tuple[list[cat.Entry], list[dict[str, str]]]:
        """The entries of the models named, in the order named, and what could not be found."""
        if not ids:
            raise ValueError("name at least one model, as org/Name")
        if len(ids) > MAX_MODELS:
            raise ValueError(f"at most {MAX_MODELS} models in one question")
        known = cat.load(str(self.source), self.fetch) if self.source else None
        found: list[cat.Entry] = []
        errors: list[dict[str, str]] = []
        for raw in ids:
            try:
                model = model_id(raw)
                found.append((known.find(model) if known else None)
                             or cat.entry_for(model, self.fetch, quantizers=list(self.cfg.get("quantizers") or QUANTIZERS)))
            except NotFound:
                errors.append({"model": raw, "error": "Hugging Face has no such model, or it is private or gated"})
            except ValueError as e:
                errors.append({"model": raw, "error": str(e)})
        return found, errors

    # ---- the answers -------------------------------------------------------------------------------------

    def _judge(self, entries: list[cat.Entry], b: Budgets) -> list[Judged]:
        return [judge_entry(e, b, self.cfg.get("roster") or {}, float(self.cfg.get("gpu_min_bits", 4)),
                            float(self.cfg.get("memory_min_bits", 3))) for e in entries]

    def judged(self, ids: list[str], given: Mapping[str, Any] | None = None) -> tuple[Machine, Budgets, list[Judged], list[dict[str, str]]]:
        m, b = self.resolve(given)
        entries, errors = self.entries(ids)
        return m, b, self._judge(entries, b), errors

    def judged_new(self, given: Mapping[str, Any] | None = None, days: int = 45, since: str | None = None,
                   publishers: list[str] | None = None, kinds: list[str] | None = None) -> tuple[Machine, Budgets, cat.Catalog, list[Judged]]:
        m, b = self.resolve(given)
        c = self.catalog(days, since, publishers, kinds)
        return m, b, c, order(self._judge(c.models, b))

    def analyze(self, given: Mapping[str, Any] | None = None, ids: list[str] | None = None, days: int = 45, since: str | None = None,
                publishers: list[str] | None = None, kinds: list[str] | None = None, limit: int = 3) -> dict[str, Any]:
        """Which of the models that fit are worth trying, role by role, with reasons: of the models named, or of the new ones."""
        errors: list[dict[str, str]] = []
        if ids:
            m, b, judged, errors = self.judged(ids, given)
            window = ""
        else:
            m, b, c, judged = self.judged_new(given, days, since, publishers, kinds)
            window = c.since
        roster = {str(k): str(v) for k, v in (self.cfg.get("roster") or {}).items()}
        in_use: dict[str, Judged] = {}
        for role, name in roster.items():   # a roster entry that is a Hugging Face id can be compared with
            if "/" in name:
                try:
                    found, _ = self.entries([name])
                except (ValueError, OSError):
                    continue
                if found:
                    in_use[role] = self._judge(found, b)[0]
        return {"schema": SCHEMA, "machine": as_dict(m, b), "since": window, "roles": analyse(judged, b, in_use, roster, limit=limit),
                "errors": errors}

    def models(self, ids: list[str], given: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Judge the models named, in the order named: one model, or several to compare."""
        m, b, judged, errors = self.judged(ids, given)
        return {"schema": SCHEMA, "machine": as_dict(m, b), "models": [j.to_dict(b) for j in judged], "errors": errors}

    def new(self, given: Mapping[str, Any] | None = None, days: int = 45, since: str | None = None,
            publishers: list[str] | None = None, kinds: list[str] | None = None, verdicts: list[str] | None = None,
            limit: int | None = None) -> dict[str, Any]:
        """The watched publishers' new models, each judged, best fit first."""
        if verdicts and not set(verdicts) <= set(ORDER):
            raise ValueError(f"verdicts are {list(ORDER)}")
        m, b, c, judged = self.judged_new(given, days, since, publishers, kinds)
        counts = {v: n for v in ORDER if (n := sum(1 for j in judged if j.verdict == v))}
        shown = [j for j in judged if not verdicts or j.verdict in verdicts][:limit]
        return {"schema": SCHEMA, "machine": as_dict(m, b), "since": c.since, "generated": c.generated, "counts": counts,
                "models": [j.to_dict(b) for j in shown]}
