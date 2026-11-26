# Copyright 2026 Yauhen Bichel
# SPDX-License-Identifier: Apache-2.0
"""What happens when something goes wrong: Hugging Face is down, slow or answers nonsense, an argument is
outside what a tool declares, a file is not what it should be, the machine cannot be detected. A failure is
said to be one, with its cause; it is never an answer of "none", and it never stops a server."""
from __future__ import annotations

import http.client
import io
import json
import threading
import time
import urllib.error
import urllib.request
from email.message import Message
from pathlib import Path
from typing import Any

import pytest
from test_ai_models_comparison import fake_fetch
from test_interfaces import BOX, CODER, get, rpc, service

from ai_models_comparison import cache, catalog, cli, hub, machine, mcp, report, server
from ai_models_comparison.service import Service

GIB = 2**30
V9 = "deepseek-ai/DeepSeek-V9"
MEMINFO = "MemTotal: 33554432 kB\n"


def down(url: str) -> Any:
    raise urllib.error.HTTPError(url, 500, "Internal Server Error", Message(), None)


def without_v9_builds(url: str) -> Any:
    """Hugging Face answering everything but the search for one model's builds."""
    if "search=DeepSeek-V9" in url:
        raise OSError("connection reset")
    return fake_fetch(url)


def tool(s: Service, name: str, **args: Any) -> dict[str, Any]:
    out = mcp.handle(rpc("tools/call", {"name": name, "arguments": args}), s)
    assert out is not None
    return dict(out.get("result") or out)


def said(result: dict[str, Any]) -> str:
    return str(result["content"][0]["text"])


# ---- Hugging Face fails (M1) ---------------------------------------------------------------------------------

def test_an_outage_is_an_error_with_its_cause_and_never_an_answer_of_none(capsys: pytest.CaptureFixture[str]) -> None:
    s = Service({"publishers": ["Qwen", "deepseek-ai"]}, down, detected=BOX)
    for name, args in (("judge_models", {"models": [CODER]}), ("new_models", {}), ("analyze", {}), ("analyze", {"models": [CODER]})):
        out = tool(s, name, **args)
        assert out["isError"] is True and "structuredContent" not in out, name
        assert "Hugging Face could not be reached" in said(out) and "HTTP 500" in said(out) and "no GGUF yet" not in said(out), name
    with pytest.raises(hub.Unreachable, match="2 of 2 lookups failed"):
        s.new()
    assert cli.main(["judge", CODER, "--gpu-gb", "64", "--ram-gb", "62"], fetch=down) == 1
    assert cli.main(["new", "--publisher", "Qwen", "--gpu-gb", "64", "--ram-gb", "62"], fetch=down) == 1
    err = capsys.readouterr()
    assert err.err.count("Hugging Face could not be reached") == 2 and "no GGUF" not in err.out and "none" not in err.out
    srv = server.serve(s, server.Options(port=0))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        status, body, _ = get(f"http://127.0.0.1:{srv.server_address[1]}/api/v1/new")
    finally:
        srv.shutdown()
        srv.server_close()
    assert status == 502 and body["error"]["code"] == "hub_unreachable" and "HTTP 500" in body["error"]["message"]


def test_a_partial_answer_says_what_is_missing_in_the_text_and_in_the_data(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    s = Service({"publishers": ["Qwen", "deepseek-ai", "broken"]}, without_v9_builds, detected=BOX)
    new = s.new(since="2026-09-01")
    assert [m["model"] for m in new["models"]] == [CODER] and "no GGUF yet" not in new["counts"]   # not judged, so not "no GGUF yet"
    gap = new["incomplete"]
    assert gap["publishers"] == ["broken"] and gap["models"] == [V9] and 0 < gap["failed"] < gap["of"] and gap["out_of_time"] is False
    assert "rate limited" in gap["cause"] or "connection reset" in gap["cause"]
    text = report.text_new(new)
    assert f"incomplete: {gap['failed']} of {gap['of']} lookups of Hugging Face failed" in text
    assert "publishers not read: broken" in text and f"models not judged: {V9}" in text
    out = tool(s, "new_models")
    assert out["isError"] is False and out["structuredContent"]["incomplete"]["publishers"] == ["broken"] and "\nincomplete: " in said(out)

    named = s.models([CODER, V9])
    assert [m["model"] for m in named["models"]] == [CODER] and named["incomplete"]["failed"] >= 1
    assert [(e["model"], e["code"]) for e in named["errors"]] == [(V9, "lookup_failed")] and "connection reset" in named["errors"][0]["error"]
    assert "incomplete: " in report.text_models(named) and "no GGUF" not in report.text_models(named)
    assert "incomplete" not in service().models([CODER, V9]) and "incomplete" not in service().new(since="2026-09-01")   # a whole answer

    c = s.catalog(since="2026-09-01")
    again = catalog.Catalog.from_dict(json.loads(json.dumps(c.to_dict())))
    assert again.incomplete == c.incomplete and again.incomplete is not None and again.incomplete["models"] == [V9]
    assert "incomplete" in catalog.schema()["properties"] and "incomplete" not in service().catalog(since="2026-09-01").to_dict()
    box = ["--gpu-gb", "64", "--ram-gb", "62", "--since", "2026-09-01", "--publisher", "Qwen", "--publisher", "deepseek-ai"]
    assert cli.main(["site", "--out", str(tmp_path), *box], fetch=without_v9_builds) == 0
    assert json.loads((tmp_path / "catalog.json").read_text())["incomplete"]["models"] == [V9]
    assert "incomplete: " in capsys.readouterr().err


# ---- nothing stops the server (M2, M4) -----------------------------------------------------------------------

def test_no_failed_lookup_and_no_fault_of_its_own_stops_the_mcp_server(capsys: pytest.CaptureFixture[str]) -> None:
    def cut_short(url: str) -> Any:
        raise http.client.IncompleteRead(b"{", 10)

    def faulty(url: str) -> Any:
        raise KeyError("createdAt")
    assert isinstance(http.client.IncompleteRead(b""), hub.FetchError)
    asked = "".join(json.dumps(rpc("tools/call", {"name": n, "arguments": a}, id_=i)) + "\n" for i, (n, a) in enumerate(
        (("judge_models", {"models": [CODER]}), ("new_models", {}), ("machine", {}))))
    for fetch, expected in ((cut_short, "the answer was cut short"), (faulty, "internal error (KeyError)")):
        out = io.StringIO()
        assert mcp.serve_stdio(Service({"publishers": ["Qwen"]}, fetch, detected=BOX), io.StringIO(asked + "[1, 2]\n"), out) == 0
        lines = sorted((json.loads(x) for x in out.getvalue().splitlines()), key=lambda x: (x["id"] is None, x["id"]))
        assert [x["id"] for x in lines] == [0, 1, 2, None] and lines[3]["error"]["code"] == -32600
        assert all(x["result"]["isError"] is True and expected in said(x["result"]) for x in lines[:2]), lines
        assert lines[2]["result"]["isError"] is False   # the server is still answering
    err = capsys.readouterr()
    assert "Traceback" in err.err and "KeyError: 'createdAt'" in err.err and err.out == ""   # the trace is for the log, not the client
    s = Service({"publishers": ["Qwen"]}, faulty, detected=BOX)
    assert "error" not in tool(s, "judge_models", models=[CODER])   # a KeyError inside a tool is not "unknown tool"
    unknown = tool(s, "no_such_tool" * 1000)
    assert unknown["error"]["code"] == -32602 and unknown["error"]["message"].startswith("unknown tool: no_such") and len(str(unknown)) < 400
    assert mcp.answer("[" * 100_000, s) == {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}}


def test_a_cache_file_this_cache_did_not_write_is_a_miss_and_is_overwritten(tmp_path: Path) -> None:
    asked: list[str] = []

    def fetch(url: str) -> Any:
        asked.append(url)
        return {"n": len(asked)}
    get_ = cache.cached(fetch, tmp_path)
    assert get_("u") == {"n": 1}
    (path,) = tmp_path.glob("*.json")
    for n, broken in enumerate(("[]", "{}", '{"body": {"n": 0}}', '{"t": "now", "body": 1}', "7", "null", "{"), 2):
        path.write_text(broken)
        assert get_("u") == {"n": n} and get_("u") == {"n": n}, broken   # asked again once, then answered from the file
        assert json.loads(path.read_text())["body"] == {"n": n}


# ---- arguments (M3, M7, M8) ----------------------------------------------------------------------------------

def test_an_argument_outside_what_a_tool_declares_is_refused_by_name_and_echoed_short() -> None:
    s = service()
    huge = "org/" + "a" * 1_000_000
    for name, args, wrong in (
            ("new_models", {"days": 10**12}, "days"), ("new_models", {"days": float("inf")}, "days"), ("new_models", {"days": -5}, "days"),
            ("new_models", {"days": 0}, "days"), ("new_models", {"days": True}, "days"), ("new_models", {"days": 1.5}, "days"),
            ("new_models", {"limit": -1}, "limit"), ("new_models", {"limit": 0}, "limit"), ("new_models", {"limit": 10**9}, "limit"),
            ("new_models", {"kinds": "text"}, "kinds"), ("new_models", {"kinds": ["poetry"]}, "kinds[0]"),
            ("new_models", {"verdicts": ["maybe"]}, "verdicts[0]"),
            ("analyze", {"models": [1, 2]}, "models[0]"), ("analyze", {"days": 10**400}, "days"),
            ("judge_models", {}, "models"), ("judge_models", {"models": "Qwen"}, "models"), ("judge_models", {"models": [huge]}, "models[0]"),
            ("judge_models", {"models": [CODER] * 13}, "models"), ("judge_models", {"models": [CODER], "gpu_gb": "nan"}, "gpu_gb"),
            ("machine", {"ram_gb": float("nan")}, "ram_gb"), ("machine", {"gpu_gb": -1}, "gpu_gb"), ("machine", {"unified": "yes"}, "unified"),
            ("machine", {"surprise": "x" * 1_000_000}, "surprise")):
        out = tool(s, name, **args)
        assert out.get("isError") is True and f"`{wrong}`" in said(out), (name, wrong, out)
        assert len(json.dumps(out)) < 700, (name, wrong)   # what was sent is not sent back whole
    odd = mcp.handle(rpc("tools/call", {"name": "machine", "arguments": "x"}), s)
    assert odd is not None and odd["result"]["isError"] is True and "must be an object" in said(odd["result"])
    assert tool(s, "new_models", days=365.0, limit=1)["isError"] is False   # a whole number, however JSON wrote it
    for call in (lambda: s.new(days=10**12), lambda: s.new(days=-5), lambda: s.new(limit=0), lambda: s.new(limit=-1),
                 lambda: s.analyze(limit=0), lambda: s.catalog(since="yesterday")):
        with pytest.raises(ValueError, match="must be"):
            call()


def test_days_and_limits_are_refused_alike_over_http() -> None:
    srv = server.serve(service(), server.Options(port=0))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{srv.server_address[1]}/api/v1/new"
        for query in ("days=1000000000000", "days=-5", "days=0", "limit=0", "limit=-1", "days=1e400"):
            status, body, _ = get(f"{base}?{query}")
            assert status == 400 and body["error"]["code"] == "bad_request", (query, status, body)
        assert get(f"{base}?days=365&limit=1")[0] == 200
    finally:
        srv.shutdown()
        srv.server_close()


def test_only_a_hugging_face_id_is_ever_asked_for() -> None:
    asked: list[str] = []

    def fetch(url: str) -> Any:
        asked.append(url)
        return fake_fetch(url)
    bad = ["../..", "a/..", "./x", "a/.", "../../etc/passwd", "a/../b", "org/-x", "org/a..b", "org/Name?x=1", "org/na me", "org/" + "a" * 97,
           "https://huggingface.co/../..", "org/%2e%2e", "oрg/name"]
    for name in bad:
        with pytest.raises(ValueError, match="org/Name"):
            hub.model_id(name)
    assert hub.model_id("Qwen/Qwen2.5-Coder-32B-Instruct_v1.1") == "Qwen/Qwen2.5-Coder-32B-Instruct_v1.1"
    s = Service({}, fetch, detected=BOX)
    answer = s.models([*bad[:10], "x" * 5000, CODER])
    assert [m["model"] for m in answer["models"]] == [CODER] and "incomplete" not in answer
    assert [e["code"] for e in answer["errors"]] == ["invalid"] * 11
    assert all(len(e["model"]) < 200 and len(e["error"]) < 400 for e in answer["errors"])   # what was sent is not sent back whole
    assert asked and all(".." not in url and "passwd" not in url and "/./" not in url for url in asked)


# ---- time (M5) -----------------------------------------------------------------------------------------------

class Counting:
    """A hub that takes a moment to answer, and knows how many questions it had at once."""

    def __init__(self, wait: float = 0.03) -> None:
        self.wait, self.now, self.most, self.asked, self.lock = wait, 0, 0, 0, threading.Lock()

    def __call__(self, url: str) -> Any:
        with self.lock:
            self.now, self.asked = self.now + 1, self.asked + 1
            self.most = max(self.most, self.now)
        time.sleep(self.wait)
        with self.lock:
            self.now -= 1
        return fake_fetch(url)


def test_the_publishers_are_asked_a_few_at_a_time_and_answered_in_order() -> None:
    fetch = Counting()
    orgs = ["Qwen", "deepseek-ai", *(f"other{n}" for n in range(22))]
    failed: list[str] = []
    got = hub.new_models("2026-09-01", fetch, orgs, on_error=lambda org, e: failed.append(org))
    assert [g[0] for g in got] == [V9, CODER] and failed == orgs[2:]
    assert 1 < fetch.most <= hub.WORKERS and fetch.asked == len(orgs)
    assert hub.each(lambda n: 10 // n, [1, 2, 5]) == [10, 5, 2] and isinstance(hub.each(lambda n: 10 // n, [0])[0], ZeroDivisionError)


def test_a_question_out_of_time_is_answered_with_what_is_known_and_says_so() -> None:
    def slow(url: str) -> Any:
        if url.endswith("/models/" + V9):
            time.sleep(5)
        return fake_fetch(url)
    s = Service({"publishers": ["Qwen", "deepseek-ai"], "deadline": 0.5}, slow, detected=BOX)
    began = time.monotonic()
    new = s.new(since="2026-09-01")
    named = s.models([V9, CODER])
    assert time.monotonic() - began < 4
    assert [m["model"] for m in new["models"]] == [CODER] and new["incomplete"]["out_of_time"] is True and new["incomplete"]["models"] == [V9]
    assert "0.5 s" in new["incomplete"]["cause"] and "incomplete: " in report.text_new(new)
    assert [m["model"] for m in named["models"]] == [CODER] and named["errors"][0]["code"] == "lookup_failed" and named["incomplete"]["out_of_time"]
    assert Service({}, fake_fetch, detected=BOX).deadline is None and Service({}, fake_fetch, detected=BOX, deadline=0).deadline is None
    with pytest.raises(ValueError, match="deadline"):
        Service({"deadline": "soon"}, fake_fetch, detected=BOX)
    with pytest.raises(ValueError, match="deadline"):
        Service({}, fake_fetch, detected=BOX, deadline=-1)


def test_a_ping_is_answered_while_a_tool_call_waits_and_calls_keep_their_order() -> None:
    def slow(url: str) -> Any:
        time.sleep(0.05)
        return fake_fetch(url)
    lines = [rpc("tools/call", {"name": "judge_models", "arguments": {"models": [CODER]}}, id_=1),
             rpc("tools/call", {"name": "machine", "arguments": {}}, id_=2), rpc("ping", id_=3), rpc("tools/list", id_=4)]
    out = io.StringIO()
    assert mcp.serve_stdio(Service({}, slow, detected=BOX), io.StringIO("".join(json.dumps(x) + "\n" for x in lines)), out) == 0
    order = [json.loads(x)["id"] for x in out.getvalue().splitlines()]
    assert sorted(order) == [1, 2, 3, 4] and order.index(3) < order.index(1) < order.index(2)   # every line still one whole message


class Answer(io.BytesIO):
    def __enter__(self) -> Answer:
        return self


def test_asked_to_slow_down_it_waits_as_told_once_and_a_dropped_connection_is_asked_for_again(monkeypatch: pytest.MonkeyPatch) -> None:
    def busy(wait: str) -> urllib.error.HTTPError:
        headers = Message()
        headers["Retry-After"] = wait
        return urllib.error.HTTPError("u", 429, "Too Many Requests", headers, None)
    script: list[Any] = []
    waited: list[float] = []

    def urlopen(request: Any, timeout: float = 0) -> Any:
        step = script.pop(0)
        if isinstance(step, BaseException):
            raise step
        return Answer(step)
    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    script[:] = [busy("2"), b'{"ok": 1}']
    assert hub.http_json("http://hub.test/u", sleep=waited.append) == {"ok": 1} and waited == [2.0]
    for script[:] in ([busy("2"), busy("2")], [busy("3600")], [busy("soon")]):   # still busy, or too long a wait to hold a question for
        with pytest.raises(urllib.error.HTTPError):
            hub.http_json("http://hub.test/u", sleep=waited.append)
    assert waited == [2.0, 2.0]
    script[:] = [ConnectionResetError(54, "Connection reset by peer"), b"[]"]
    assert hub.http_json("http://hub.test/u", sleep=waited.append) == []
    script[:] = [urllib.error.URLError(ConnectionRefusedError(61, "refused")), ConnectionResetError(54, "reset")]
    with pytest.raises(ConnectionResetError):
        hub.http_json("http://hub.test/u", sleep=waited.append)
    script[:] = [urllib.error.URLError(TimeoutError("timed out"))]   # a slow hub is not asked to be slow twice
    with pytest.raises(urllib.error.URLError):
        hub.http_json("http://hub.test/u", sleep=waited.append)
    assert not script and waited == [2.0, 2.0, 0.1, 0.1]


def test_the_hub_is_where_hf_endpoint_says_and_hugging_face_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    asked: list[str] = []

    def fetch(url: str) -> Any:
        asked.append(url)
        return fake_fetch(url)
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    assert hub.api() == hub.API == "https://huggingface.co/api"
    catalog.build("2026-09-01", fetch, ["Qwen"])
    assert asked and all(url.startswith("https://huggingface.co/api/") for url in asked)
    del asked[:]
    monkeypatch.setenv("HF_ENDPOINT", "http://127.0.0.1:9/")
    assert hub.api() == "http://127.0.0.1:9/api"
    catalog.build("2026-09-01", fetch, ["Qwen"])
    assert asked and all(url.startswith("http://127.0.0.1:9/api/") for url in asked)


# ---- the machine (M6) ----------------------------------------------------------------------------------------

def test_a_machine_that_cannot_be_detected_is_unknown_and_nothing_is_judged_against_no_memory(capsys: pytest.CaptureFixture[str]) -> None:
    def nothing(cmd: list[str]) -> str:
        return ""
    for m in (machine.detect(run=nothing, read=lambda p: "", system="FreeBSD", amd_cards=[], sysconf=lambda: 0),
              machine.detect(run=nothing, read=lambda p: "", system="Linux", amd_cards=[], sysconf=lambda: 0),
              machine.detect(run=lambda c: "failed: sysctl: unknown oid", read=lambda p: "", system="Darwin"),
              machine.detect(run=nothing, read=lambda p: "", system="Windows", windows_ram=lambda: 0)):
        assert (m.kind, m.ram, m.unified) == ("unknown", 0, False) and "gpu_gb and ram_gb" in m.warning and "--config" in m.warning
        s = Service({}, fake_fetch, detected=m)
        shown = s.machine()["machine"]
        assert shown["kind"] == "unknown" and shown["warning"] == m.warning and "unknown" in machine.describe(*machine.resolve({}, m))
        with pytest.raises(machine.Unknown, match="gpu_gb and ram_gb"):
            s.models([CODER])
        with pytest.raises(machine.Unknown):
            s.models([CODER], {"gpu_gb": 24})   # half a machine is not one
        with pytest.raises(machine.Unknown):
            s.new(since="2026-09-01")
        out = tool(s, "judge_models", models=[CODER])
        assert out["isError"] is True and "gpu_gb and ram_gb" in said(out) and "too big" not in said(out)
        assert tool(s, "machine")["isError"] is False and said(tool(s, "machine")).startswith("machine: unknown")
        given = s.models([CODER], {"gpu_gb": 64, "ram_gb": 62})
        assert given["models"][0]["verdict"] == "fits the GPU" and given["machine"]["kind"] == "given" and given["machine"]["warning"] == ""
    bsd = machine.detect(run=nothing, read=lambda p: "", system="FreeBSD", amd_cards=[], sysconf=lambda: 32 * GIB)
    assert (bsd.kind, bsd.ram, bsd.warning) == ("none", 32 * GIB, "")
    assert Service({}, fake_fetch, detected=BOX).machine()["machine"]["warning"] == ""


def test_an_nvidia_gpu_that_reports_no_memory_shares_the_systems_and_a_failed_probe_is_not_no_gpu() -> None:
    def smi(answer: str) -> machine.Run:
        return lambda cmd: answer if cmd[0] == "nvidia-smi" else ""

    def linux(answer: str) -> machine.Machine:
        return machine.detect(run=smi(answer), read=lambda p: MEMINFO if "meminfo" in p else "", system="Linux", amd_cards=[])
    for answer in ("[N/A]\n", "[Not Supported]\n", "N/A\n"):   # an integrated GPU: a DGX Spark, a Jetson
        shared = linux(answer)
        assert (shared.kind, shared.unified, shared.gpu, shared.ram) == ("nvidia", True, 32 * GIB, 32 * GIB), answer
        assert "gpu_gb" in shared.warning and machine.budgets(shared).gpu == 24 * GIB
    for answer in ("failed: NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver", "failed: no answer in 10 s", "what?\n"):
        broken = linux(answer)
        assert (broken.kind, broken.gpu) == ("none", 0) and "nvidia-smi" in broken.warning and "gpu_gb" in broken.warning
        assert "warning: " in report.text_machine(Service({}, fake_fetch, detected=broken).machine())
    assert linux("24564\n").warning == linux("").warning == "" and linux("24564\n[N/A]\n").gpu == 24564 * 2**20
    card = machine.detect(run=smi("8192\n"), read=lambda p: "", system="Windows", windows_ram=lambda: 16 * GIB)
    assert (card.kind, card.gpu, card.ram, card.warning) == ("nvidia", 8 * GIB, 16 * GIB, "")
    other = machine.detect(run=smi(""), read=lambda p: "", system="Windows", windows_ram=lambda: 16 * GIB)
    assert (other.kind, other.ram) == ("none", 16 * GIB) and "only NVIDIA" in other.warning


# ---- files (M9) ----------------------------------------------------------------------------------------------

def test_a_bad_file_is_named_with_what_is_wrong_with_it(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    box = ["--gpu-gb", "64", "--ram-gb", "62"]
    missing, not_json, not_catalog, toml = (tmp_path / n for n in ("missing.json", "page.json", "list.json", "bad.toml"))
    not_json.write_text("<html>not found</html>")
    not_catalog.write_text("[1, 2]")
    toml.write_text("gpu_gb = = 24\n")
    for args, wrong in ((["new", "--catalog", str(missing)], "No such file"), (["new", "--catalog", str(not_json)], "not valid JSON"),
                        (["judge", CODER, "--catalog", str(not_catalog)], "a catalog is a JSON object"),
                        (["machine", "--config", str(toml)], "not valid TOML"),
                        (["machine", "--config", str(tmp_path / "none.toml")], "no such configuration file")):
        assert cli.main([*args, *box], fetch=fake_fetch) == 1
        err = capsys.readouterr().err
        assert str(tmp_path) in err and wrong in err and "Hugging Face could not be reached" not in err, err
    with pytest.raises(catalog.Unreadable, match="could not be read: HTTP 500"):
        catalog.load("https://example.org/catalog.json", down)
    assert cli.main(["new", *box, "--deadline", "-1"], fetch=fake_fetch) == 1 and "deadline" in capsys.readouterr().err
    assert "deadline" in cli.EXAMPLE
