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
from .hub import QUANTIZERS, WORKERS, Fetch, FetchError, Lookups, NotFound, Unanswered, each, http_json, model_id, short
from .judge import ORDER, Judged, judge_entry
from .machine import SPEC, Budgets, Machine, as_dict, detect, known, resolve

SCHEMA = 1          # of the answers; grows by adding keys, changes only when a reader would break
MAX_MODELS = 12     # in one question
MAX_DAYS = 365      # how far back a question may look
DEADLINE = 60.0     # seconds a server gives one question before it answers with what it has, marked incomplete


def whole(name: str, value: Any, least: int, most: int | None = None) -> int:
    """An argument that must be a whole number in a range, whichever interface it came through."""
    if isinstance(value, bool) or not isinstance(value, int) or value < least or (most is not None and value > most):
        raise ValueError(f"{name} must be a whole number " + (f"from {least} to {most}" if most is not None else f"of at least {least}"))
    return value


def order(judged: list[Judged]) -> list[Judged]:
    return sorted(judged, key=lambda j: (ORDER[j.verdict], j.role, -(j.params_b or 0), j.model))


class Service:
    def __init__(self, cfg: Mapping[str, Any] | None = None, fetch: Fetch | None = None, catalog: str | None = None,
                 detected: Machine | None = None, keep: float = 1800.0, deadline: float | None = None) -> None:
        self.cfg: Mapping[str, Any] = cfg or {}
        self.fetch: Fetch = fetch or cached(http_json)
        self.source = catalog or self.cfg.get("catalog")      # a published catalog (address or file), instead of asking the hub
        self.keep = keep                                      # seconds a built catalog is reused within this process
        given = deadline if deadline is not None else self.cfg.get("deadline")
        if given is not None and (isinstance(given, bool) or not isinstance(given, int | float) or given < 0):
            raise ValueError("deadline must be a number of seconds (0 for none)")
        self.deadline: float | None = float(given) if given else None   # seconds for one question's lookups; None waits for all
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

    def known(self, given: Mapping[str, Any] | None = None) -> tuple[Machine, Budgets]:
        """The machine to judge against. machine.Unknown when its memory was neither detected nor given:
        a verdict against no memory would be a false one."""
        m, b = self.resolve(given)
        return known(m), b

    def asks(self) -> Lookups:
        """One question's lookups: counted, and given this service's deadline."""
        return Lookups(self.fetch, self.deadline)

    def machine(self, given: Mapping[str, Any] | None = None) -> dict[str, Any]:
        m, b = self.resolve(given)
        return {"schema": SCHEMA, "machine": as_dict(m, b)}

    # ---- the catalog -------------------------------------------------------------------------------------

    def catalog(self, days: int = 45, since: str | None = None, publishers: list[str] | None = None,
                kinds: list[str] | None = None, asks: Lookups | None = None) -> cat.Catalog:
        """The new models. `incomplete` on it says what could not be read; hub.Unreachable when nothing could."""
        since = since or (dt.datetime.now(dt.UTC) - dt.timedelta(days=whole("days", days, 1, MAX_DAYS))).strftime("%Y-%m-%d")
        try:
            dt.date.fromisoformat(since)
        except (TypeError, ValueError):
            raise ValueError(f"since must be a day as YYYY-MM-DD, not {short(str(since), 40)!r}") from None
        publishers = publishers or list(self.cfg.get("publishers") or []) or None
        want = set(kinds or self.cfg.get("kinds") or []) or None
        key = (self.source, since, tuple(publishers or ()), tuple(sorted(want or ())))
        with self._lock:
            hit = self._catalogs.get(key)
        if hit and time.monotonic() - hit[0] < self.keep:
            return hit[1]
        if self.source:
            whole_ = cat.load(str(self.source), self.fetch)
            orgs = {p.lower() for p in publishers} if publishers else None
            got = cat.Catalog(whole_.generated, max(since, whole_.since), [
                e for e in whole_.models
                if e.created >= since and (not want or e.kind in want) and (not orgs or e.model.split("/")[0].lower() in orgs)],
                whole_.incomplete)
        else:
            got = cat.build(since, asks or self.asks(), publishers, want, list(self.cfg.get("quantizers") or QUANTIZERS))
        if not got.incomplete:   # what is missing a part is asked for again: the answers it did get are in the cache
            with self._lock:
                self._catalogs[key] = (time.monotonic(), got)
        return got

    def entries(self, ids: list[str], asks: Lookups | None = None) -> tuple[list[cat.Entry], list[dict[str, str]]]:
        """The entries of the models named, in the order named, and what could not be found or looked up.
        hub.Unreachable when every lookup failed."""
        if not ids:
            raise ValueError("name at least one model, as org/Name")
        if len(ids) > MAX_MODELS:
            raise ValueError(f"at most {MAX_MODELS} models in one question")
        if not all(isinstance(x, str) for x in ids):
            raise ValueError("models must be Hugging Face ids, as org/Name")
        known_ = cat.load(str(self.source), self.fetch) if self.source else None
        asks = asks or self.asks()
        quantizers = list(self.cfg.get("quantizers") or QUANTIZERS)

        def one(raw: str) -> cat.Entry | dict[str, str]:
            mine = asks.scope()
            try:
                model = model_id(raw)
                e = (known_.find(model) if known_ else None) or cat.entry_for(model, mine, quantizers=quantizers)
            except NotFound:
                return {"model": short(raw), "code": "not_found", "error": "Hugging Face has no such model, or it is private or gated"}
            except ValueError as bad:
                return {"model": short(raw), "code": "invalid", "error": str(bad)}
            if mine.failed and not e.builds:   # not "no GGUF yet": nobody could look
                return {"model": short(raw), "code": "lookup_failed",
                        "error": f"could not be looked up: {mine.failed} of {mine.asked} lookups failed ({mine.cause})"}
            return e

        found: list[cat.Entry] = []
        errors: list[dict[str, str]] = []
        for raw, got in zip(ids, each(one, ids, WORKERS, asks.left()), strict=True):
            if isinstance(got, Unanswered):
                asks.missed(asks.out_of_time())
                got = {"model": short(raw), "code": "lookup_failed", "error": f"could not be looked up: {asks.out_of_time()}"}
            elif isinstance(got, BaseException):
                if not isinstance(got, FetchError):
                    raise got
                got = {"model": short(raw), "code": "lookup_failed", "error": f"could not be looked up: {got}"}
            if isinstance(got, dict):
                errors.append(got)
            else:
                found.append(got)
        if asks.asked and asks.failed == asks.asked:
            raise asks.unreachable()
        return found, errors

    # ---- the answers -------------------------------------------------------------------------------------

    def _judge(self, entries: list[cat.Entry], b: Budgets) -> list[Judged]:
        return [judge_entry(e, b, self.cfg.get("roster") or {}, float(self.cfg.get("gpu_min_bits", 4)),
                            float(self.cfg.get("memory_min_bits", 3))) for e in entries]

    def judged(self, ids: list[str], given: Mapping[str, Any] | None = None,
               asks: Lookups | None = None) -> tuple[Machine, Budgets, list[Judged], list[dict[str, str]]]:
        m, b = self.known(given)
        entries, errors = self.entries(ids, asks)
        return m, b, self._judge(entries, b), errors

    def judged_new(self, given: Mapping[str, Any] | None = None, days: int = 45, since: str | None = None,
                   publishers: list[str] | None = None, kinds: list[str] | None = None,
                   asks: Lookups | None = None) -> tuple[Machine, Budgets, cat.Catalog, list[Judged]]:
        m, b = self.known(given)
        c = self.catalog(days, since, publishers, kinds, asks)
        return m, b, c, order(self._judge(c.models, b))

    def analyze(self, given: Mapping[str, Any] | None = None, ids: list[str] | None = None, days: int = 45, since: str | None = None,
                publishers: list[str] | None = None, kinds: list[str] | None = None, limit: int = 3) -> dict[str, Any]:
        """Which of the models that fit are worth trying, role by role, with reasons: of the models named, or of the new ones."""
        whole("top", limit, 1)
        errors: list[dict[str, str]] = []
        asks = self.asks()
        if ids:
            m, b, judged, errors = self.judged(ids, given, asks)
            window, incomplete = "", asks.summary()
        else:
            m, b, c, judged = self.judged_new(given, days, since, publishers, kinds, asks)
            window, incomplete = c.since, c.incomplete
        roster = {str(k): str(v) for k, v in (self.cfg.get("roster") or {}).items()}
        in_use: dict[str, Judged] = {}
        for role, name in roster.items():   # a roster entry that is a Hugging Face id can be compared with
            if "/" in name:
                try:
                    found, _ = self.entries([name], asks.scope())
                except (ValueError, OSError):
                    continue
                if found:
                    in_use[role] = self._judge(found, b)[0]
        answer = {"schema": SCHEMA, "machine": as_dict(m, b), "since": window, "roles": analyse(judged, b, in_use, roster, limit=limit),
                  "errors": errors}
        incomplete = incomplete or asks.summary()   # the models in use could not be looked up: the comparison lacks them
        return {**answer, "incomplete": incomplete} if incomplete else answer

    def models(self, ids: list[str], given: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Judge the models named, in the order named: one model, or several to compare."""
        asks = self.asks()
        m, b, judged, errors = self.judged(ids, given, asks)
        answer = {"schema": SCHEMA, "machine": as_dict(m, b), "models": [j.to_dict(b) for j in judged], "errors": errors}
        incomplete = asks.summary()
        return {**answer, "incomplete": incomplete} if incomplete else answer

    def new(self, given: Mapping[str, Any] | None = None, days: int = 45, since: str | None = None,
            publishers: list[str] | None = None, kinds: list[str] | None = None, verdicts: list[str] | None = None,
            limit: int | None = None) -> dict[str, Any]:
        """The watched publishers' new models, each judged, best fit first."""
        if verdicts and not set(verdicts) <= set(ORDER):
            raise ValueError(f"verdicts are {list(ORDER)}")
        if limit is not None:
            whole("limit", limit, 1)
        m, b, c, judged = self.judged_new(given, days, since, publishers, kinds)
        counts = {v: n for v in ORDER if (n := sum(1 for j in judged if j.verdict == v))}
        shown = [j for j in judged if not verdicts or j.verdict in verdicts][:limit]
        answer = {"schema": SCHEMA, "machine": as_dict(m, b), "since": c.since, "generated": c.generated, "counts": counts,
                  "models": [j.to_dict(b) for j in shown]}
        return {**answer, "incomplete": c.incomplete} if c.incomplete else answer
