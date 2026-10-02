# Copyright 2026 Yauhen Bichel
# SPDX-License-Identifier: Apache-2.0
"""The catalog, the cache, the service and its three interfaces (HTTP, MCP, the command line), against a fake
Hugging Face: no network, no GPU. The server tests listen on a free port of this machine."""
from __future__ import annotations

import datetime as dt
import io
import json
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from test_ai_models_comparison import fake_fetch, tree

from ai_models_comparison import analyze, cache, catalog, cli, hub, judge, machine, mcp, server
from ai_models_comparison.service import Service

GIB = 2**30
BOX = machine.Machine(62 * GIB, 64 * GIB, "amd", False)   # the machine every test's service "detects"
CODER = "Qwen/Qwen4-Coder-90B"


def service(**kw: Any) -> Service:
    return Service({"publishers": ["Qwen", "deepseek-ai"], "roster": {"coder": "old-coder"}} | kw.pop("cfg", {}), fake_fetch, detected=BOX, **kw)


# ---- the catalog -------------------------------------------------------------------------------------------

def test_an_entry_has_what_is_worth_comparing_and_survives_a_round_trip() -> None:
    e = catalog.entry_for(CODER, fake_fetch)
    assert (e.params_b, e.kind, e.licence, e.context, e.created, e.downloads) == (90.0, "text", "apache-2.0", 262144, "2026-09-20", 12345)
    assert [b.quant for b in e.builds] == ["Q3_K_M", "Q4_K_M", "Q8_0"]
    c = catalog.build("2026-09-01", fake_fetch, ["Qwen", "deepseek-ai"])
    again = catalog.Catalog.from_dict(json.loads(json.dumps(c.to_dict())))
    assert [m.to_dict() for m in again.models] == [m.to_dict() for m in c.models] and again.find(CODER.lower()) is not None
    assert c.to_dict()["schema"] == catalog.SCHEMA == catalog.schema()["properties"]["schema"]["const"]
    with pytest.raises(ValueError, match="newer than this version"):
        catalog.Catalog.from_dict({"schema": catalog.SCHEMA + 1})


def test_a_gguf_repository_is_judged_by_its_own_files_and_a_missing_model_is_said_to_be_missing() -> None:
    def fetch(url: str) -> Any:
        if url.endswith("/models/u/Thing-GGUF"):
            return {"gguf": {"total": 8_000_000_000, "context_length": 8192, "architecture": "llama"}, "pipeline_tag": "text-generation"}
        if url.endswith("/models/u/Thing-GGUF/tree/main?recursive=true"):
            return tree(("Thing-Q4_K_M.gguf", 4.9), ("Thing-Q8_0.gguf", 8.5), ("mmproj-F16.gguf", 0.6), ("encoder/Thing-enc-F32.gguf", 0.5))
        raise AssertionError(url)
    e = catalog.entry_for("u/Thing-GGUF", fetch)
    assert [b.quant for b in e.builds] == ["Q4_K_M", "Q8_0"] and e.params_b == 8.0 and e.context == 8192
    with pytest.raises(hub.NotFound):
        catalog.entry_for("no/such", fake_fetch)
    assert hub.model_id(" https://huggingface.co/Qwen/Qwen4-Coder-90B/tree/main ") == CODER
    with pytest.raises(ValueError, match="org/Name"):
        hub.model_id("just-a-name")


def test_the_cache_asks_once_and_an_old_answer_beats_none(tmp_path: Path) -> None:
    asked: list[str] = []
    clock = [1000.0]
    down = [False]

    def fetch(url: str) -> Any:
        if down[0]:
            raise OSError("unreachable")
        asked.append(url)
        return {"n": len(asked)}
    get = cache.cached(fetch, tmp_path, ttl=60, now=lambda: clock[0])
    assert get("u") == {"n": 1} and get("u") == {"n": 1} and asked == ["u"]
    clock[0] += 61
    assert get("u") == {"n": 2}
    clock[0] += 61
    down[0] = True
    assert get("u") == {"n": 2}                 # the hub is down: the last answer
    with pytest.raises(OSError, match="unreachable"):
        get("never-seen")


# ---- the service -------------------------------------------------------------------------------------------

def test_the_service_answers_in_one_shape_for_this_machine_or_any_other() -> None:
    s = service()
    here = s.models([CODER], {})
    assert here["schema"] == 1 and here["machine"]["kind"] == "amd" and here["models"][0]["verdict"] == "fits the GPU"
    small = s.models([CODER, "no/such", "nonsense"], {"gpu_gb": 24, "ram_gb": 64})
    j = small["models"][0]
    assert small["machine"]["kind"] == "given" and j["verdict"] == "fits in memory" and j["replaces"] == "old-coder"
    assert j["pick"] == {"repo": "unsloth/Qwen4-Coder-90B-GGUF", "quant": "Q4_K_M", "gb": 54.0, "bits": 4.0, "bits_per_weight": 4.8,
                         "fits": "memory", "get": 'hf download unsloth/Qwen4-Coder-90B-GGUF --include "*Q4_K_M*"',
                         "run": "llama-server -hf unsloth/Qwen4-Coder-90B-GGUF:Q4_K_M", "headroom_gb": 21.2}
    assert [b["fits"] for b in j["builds"]] == ["memory", "memory", "no"]
    assert [e["model"] for e in small["errors"]] == ["no/such", "nonsense"] and "no such model" in small["errors"][0]["error"]
    mac = s.machine({"ram_gb": 64, "unified": "true"})["machine"]
    assert mac["unified"] and mac["gpu_gib"] == 64.0 and mac["fits_gpu_gib"] == mac["fits_memory_gib"] == 48.0
    for bad in ({"gpu_gb": "abc"}, {"ram_gb": -1}):
        with pytest.raises(ValueError, match="GiB"):
            s.machine(bad)
    with pytest.raises(ValueError, match="at least one"):
        s.models([], {})


def test_new_models_are_counted_filtered_and_limited_and_a_published_catalog_replaces_the_hub(tmp_path: Path) -> None:
    s = service()
    new = s.new({}, since="2026-09-01")
    assert new["counts"] == {"fits the GPU": 1, "no GGUF yet": 1} and [m["model"] for m in new["models"]] == [CODER, "deepseek-ai/DeepSeek-V9"]
    assert [m["model"] for m in s.new({}, since="2026-09-01", verdicts=["fits the GPU"], limit=5)["models"]] == [CODER]
    with pytest.raises(ValueError, match="verdicts are"):
        s.new({}, verdicts=["great"])
    published = tmp_path / "catalog.json"
    published.write_text(json.dumps(s.catalog(since="2026-09-01").to_dict()))

    def no_hub(url: str) -> Any:
        raise AssertionError(f"the hub was asked: {url}")
    offline = Service({}, no_hub, catalog=str(published), detected=BOX)
    assert offline.new({}, since="2026-09-01")["counts"] == new["counts"]
    assert offline.new({}, since="2026-09-22")["counts"] == {"no GGUF yet": 1}          # the window narrows what is read
    assert offline.new({}, since="2026-09-01", publishers=["qwen"])["counts"] == {"fits the GPU": 1}
    assert offline.models([CODER], {})["models"][0]["verdict"] == "fits the GPU"


# ---- the analysis ------------------------------------------------------------------------------------------

def model(name: str, params: float, *builds: tuple[str, float, float], created: str = "2026-09-01", licence: str = "apache-2.0",
          context: int | None = 131072) -> catalog.Entry:
    return catalog.Entry(name, created, "text", params, licence, context,
                         builds=[hub.Build(f"u/{name.split('/')[-1]}-GGUF", q, bits, int(gb * 1e9)) for q, bits, gb in builds])


def test_the_analysis_keeps_the_best_on_the_gpu_and_only_what_is_larger_and_says_why() -> None:
    b = machine.Budgets(52 * GIB, 92 * GIB, 0, 0)
    today = dt.date(2026, 10, 2)
    entries = [model("a/Small-20B", 20, ("Q8_0", 8, 21)), model("a/Tiny-8B", 8, ("Q8_0", 8, 9)),
               model("a/Mid-40B", 40, ("Q4_K_M", 4, 24), ("Q8_0", 8, 42)),
               model("a/Slow-30B", 30, ("Q8_0", 8, 60)),                                  # in memory, yet smaller than Mid: left out
               model("a/Big-180B", 180, ("IQ4_XS", 4, 93.7), licence="qwen-community-1.0", created="2026-09-25"),
               model("a/Huge-300B", 300, ("IQ1_S", 1, 82.5), ("Q4_K", 4, 155)),            # low-bit only, but the largest that runs
               model("a/Giant-700B", 700, ("Q2", 2, 366)), model("a/Unbuilt", 50)]
    judged = [judge.judge_entry(e, b) for e in entries]
    mine = judge.judge_entry(model("a/Mine-30B", 30, ("Q8_0", 8, 32), created="2026-03-01"), b)
    [role] = analyze.analyse(judged, b, {"general": mine}, {"general": "a/Mine-30B"}, today)
    got = [(c["model"], c["verdict"], c["effective_b"], c["speed"]) for c in role["candidates"]]
    assert got == [("a/Mid-40B", "fits the GPU", 40.0, "on the GPU"),
                   ("a/Big-180B", "fits in memory", 174.6, "partly in system memory"),
                   ("a/Huge-300B", "low-bit only", 234.0, "partly in system memory"),
                   ("a/Small-20B", "fits the GPU", 20.0, "on the GPU")]      # smaller than Mine, kept as the one much newer
    assert role["left_out"] == {"too big": 1, "no GGUF yet": 1, "smaller than what you run today": 1,
                                "smaller or slower than a listed one": 1}
    assert role["in_use"] == "a/Mine-30B" and role["in_use_effective_b"] == 30.0
    assert role["summary"] == "fast: a/Mid-40B (40B on the GPU); largest that runs: a/Big-180B (174.6B, partly in system memory)"
    mid, big, huge, small = role["candidates"]
    assert "1.3 times the size of Mine-30B, which you run today" in mid["why"] and "6 months newer than Mine-30B" in mid["why"]
    assert any(w.startswith("smaller than Mine-30B") for w in small["why"]) and mid["cautions"] == []
    assert small["cautions"][0].startswith("smaller than what you run, but much newer")
    eyes = judge.judge_entry(catalog.Entry("a/Sees-60B", "2026-09-01", "vision", 60, "mit", builds=[hub.Build("r", "Q4_K_M", 4, int(36e9))]), b)
    both = {r["role"]: [c["model"] for c in r["candidates"]] for r in analyze.analyse([eyes], b, today=today)}
    assert both == {"general": ["a/Sees-60B"], "vision": ["a/Sees-60B"]}                 # a model that reads images stands for both
    alone = analyze.analyse([judge.judge_entry(entries[0], b), judge.judge_entry(entries[1], b)], b, today=today)[0]
    assert alone["summary"] == "a/Small-20B (20B, on the GPU)"                          # no "largest that runs" when it is the same model
    assert any("60% of the build fits the GPU" in w for w in big["why"])
    assert [c.split(":")[0] for c in big["cautions"]] == ["licence qwen-community-1.0", "released 7 days ago", "only 5.1 GB of room left"]
    assert huge["cautions"][0].startswith("only a build under 3 bits fits")
    assert analyze.keep(4) == 0.97 and analyze.keep(1) == 0.78 and analyze.keep(16) == 1.0
    no_gpu = analyze.candidate(judge.judge_entry(entries[0], machine.Budgets(0, 24 * GIB, 0, 0)), machine.Budgets(0, 24 * GIB, 0, 0), today)
    assert no_gpu is not None and no_gpu.speed == "in system memory" and "no GPU: it runs on the CPU from system memory" in no_gpu.why


def test_analyze_through_the_service_and_the_command_line(capsys: pytest.CaptureFixture[str]) -> None:
    s = service(cfg={"roster": {"coder": CODER, "general": "gpt-oss:120b"}})
    new = s.analyze({}, since="2026-09-01")
    assert [r["role"] for r in new["roles"]] == ["coder", "general"]
    assert new["roles"][0]["candidates"] == []                                              # the only coder is the one in use
    assert new["roles"][0]["summary"] == "nothing new that fits is larger or much newer than what you run"
    named = s.analyze({"gpu_gb": 24, "ram_gb": 64}, [CODER, "no/such"])
    assert named["since"] == "" and named["errors"][0]["model"] == "no/such" and named["roles"][0]["in_use"] == CODER
    assert cli.main(["analyze", "--since", "2026-09-01", "--publisher", "Qwen", "--gpu-gb", "24", "--ram-gb", "64"], fetch=fake_fetch) == 0
    out = capsys.readouterr().out
    assert "1. Qwen/Qwen4-Coder-90B  fits in memory  Q4_K_M 54 GB  ranks as 87.3B, partly in system memory" in out
    assert "+ 44% of the build fits the GPU, the rest runs from system memory: slower" in out and "an ordering, not a benchmark" in out


# ---- the HTTP API -----------------------------------------------------------------------------------------

@pytest.fixture
def base() -> Iterator[str]:
    srv = server.serve(service(), server.Options(port=0))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def get(url: str, headers: dict[str, str] | None = None) -> tuple[int, Any, Any]:
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers or {}), timeout=10) as r:  # noqa: S310
            body = r.read()
            return r.status, (json.loads(body) if "json" in r.headers["Content-Type"] else body.decode()), r.headers
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read()), e.headers


def test_the_api_answers_for_any_machine_and_says_what_was_wrong(base: str) -> None:
    assert get(f"{base}/healthz")[1]["ok"] is True
    here = get(f"{base}/api/v1/machine")[1]["machine"]
    assert here["gpu_gib"] == 64.0 and here["gpu_reserve_gib"] > 0 and here["ram_reserve_gib"] > 0   # the page judges with these
    status, body, _ = get(f"{base}/api/v1/models?id={CODER}&gpu_gb=24&ram_gb=64")
    assert status == 200 and body["models"][0]["verdict"] == "fits in memory" and body["machine"]["fits_gpu_gib"] == 22.1
    status, body, _ = get(f"{base}/api/v1/new?since=2026-09-01&verdict=fits%20the%20GPU")
    assert status == 200 and [m["model"] for m in body["models"]] == [CODER] and body["counts"]["no GGUF yet"] == 1
    status, body, _ = get(f"{base}/api/v1/catalog?since=2026-09-01")
    assert status == 200 and body["schema"] == 1 and body["models"][1]["builds"][0].keys() == {"repo", "quant", "bits", "bytes"}
    assert get(f"{base}/api/v1/catalog?id={CODER},no/such")[1]["errors"][0]["model"] == "no/such"
    for bad, code in ((f"{base}/api/v1/machine?gpu_gb=abc", 400), (f"{base}/api/v1/models", 400), (f"{base}/api/v1/new?since=soon", 400),
                      (f"{base}/api/v1/nothing", 404), (f"{base}/nothing", 404)):
        status, body, _ = get(bad)
        assert status == code and body["error"]["message"]
    paths = get(f"{base}/openapi.json")[1]["paths"]
    assert set(paths) == {"/api/v1/machine", "/api/v1/models", "/api/v1/new", "/api/v1/analyze", "/api/v1/catalog"}
    status, body, _ = get(f"{base}/api/v1/analyze?since=2026-09-01")
    assert status == 200 and body["roles"][0]["role"] == "coder" and body["roles"][0]["candidates"][0]["model"] == CODER


def test_the_page_is_served_and_a_foreign_name_for_this_machine_is_refused(base: str) -> None:
    status, html, headers = get(f"{base}/")
    assert status == 200 and "<title>AI models comparison" in html and "default-src 'self'" in headers["Content-Security-Policy"]
    for name in ("app.js", "judge.js", "app.css"):
        assert get(f"{base}/{name}")[0] == 200
    status, body, _ = get(f"{base}/api/v1/machine", {"Host": "evil.example:8377"})
    assert status == 403 and body["error"]["code"] == "bad_host"


def test_beyond_this_machine_a_token_is_required() -> None:
    with pytest.raises(ValueError, match="needs a token"):
        server.serve(service(), server.Options(host="0.0.0.0", port=0))  # noqa: S104
    srv = server.serve(service(), server.Options(port=0, token="s3", cors="https://example.org"))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/api/v1/machine"
    try:
        assert get(url)[0] == 401 and get(url, {"Authorization": "Bearer wrong"})[0] == 401
        status, _, headers = get(url, {"Authorization": "Bearer s3"})
        assert status == 200 and headers["Access-Control-Allow-Origin"] == "https://example.org"
    finally:
        srv.shutdown()
        srv.server_close()


# ---- MCP --------------------------------------------------------------------------------------------------

def rpc(method: str, params: dict[str, Any] | None = None, id_: int | None = 1) -> dict[str, Any]:
    msg: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
    if id_ is not None:
        msg["id"] = id_
    if params is not None:
        msg["params"] = params
    return msg


def test_mcp_speaks_both_eras_of_the_protocol() -> None:
    s = service()
    modern = {"_meta": {mcp.VERSION_KEY: "2026-07-28"}}
    found = mcp.handle(rpc("server/discover", modern), s)
    assert found is not None and found["result"]["supportedVersions"][0] == "2026-07-28" and found["result"]["resultType"] == "complete"
    assert found["result"]["capabilities"] == {"tools": {}} and found["result"]["_meta"]["io.modelcontextprotocol/serverInfo"]["name"]
    old = mcp.handle(rpc("initialize", {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}}), s)
    assert old is not None and old["result"]["protocolVersion"] == "2025-06-18" and old["result"]["serverInfo"]["name"] == "ai-models-comparison"
    assert mcp.handle(rpc("notifications/initialized", id_=None), s) is None
    refused = mcp.handle(rpc("tools/list", {"_meta": {mcp.VERSION_KEY: "1900-01-01"}}), s)
    assert refused is not None and refused["error"]["code"] == -32022 and "2026-07-28" in refused["error"]["data"]["supported"]
    listed = mcp.handle(rpc("tools/list", modern), s)
    assert listed is not None and [t["name"] for t in listed["result"]["tools"]] == ["machine", "judge_models", "new_models", "analyze"]
    assert all(t["annotations"]["readOnlyHint"] and t["outputSchema"] for t in listed["result"]["tools"]) and listed["result"]["ttlMs"] > 0
    missing = mcp.handle(rpc("nothing/here"), s)
    assert missing is not None and missing["error"]["code"] == -32601


def test_mcp_tools_give_a_short_text_and_the_full_answer() -> None:
    s = service()

    def call(name: str, **args: Any) -> dict[str, Any]:
        out = mcp.handle(rpc("tools/call", {"name": name, "arguments": args}), s)
        assert out is not None
        return dict(out.get("result") or out)
    one = call("judge_models", models=[CODER], gpu_gb=24, ram_gb=64)
    assert one["isError"] is False and one["structuredContent"]["models"][0]["verdict"] == "fits in memory"
    assert "Qwen/Qwen4-Coder-90B (text, 90B weights): fits in memory" in one["content"][0]["text"]
    two = call("judge_models", models=[CODER, "deepseek-ai/DeepSeek-V9"])["content"][0]["text"]
    assert "verdict        fits the GPU     no GGUF yet" in two and "context        256k             -\n" in two
    new = call("new_models", days=400, verdicts=["fits the GPU"])
    assert len(new["structuredContent"]["models"]) <= 1 and new["content"][0]["text"].startswith("machine: 64 GiB GPU")
    assert "fits the GPU up to" in call("machine")["content"][0]["text"]
    wrong = call("judge_models", models="Qwen")
    assert wrong["isError"] is True and "must be a list" in wrong["content"][0]["text"]
    assert call("no_such_tool")["error"]["code"] == -32602
    out = io.StringIO()
    mcp.serve_stdio(s, io.StringIO(json.dumps(rpc("server/discover")) + "\nnot json\n\n" + json.dumps(rpc("ping", id_=None)) + "\n"), out)
    lines = [json.loads(x) for x in out.getvalue().splitlines()]
    assert len(lines) == 2 and lines[0]["result"]["supportedVersions"] and lines[1]["error"]["code"] == -32700


# ---- the command line -------------------------------------------------------------------------------------

def test_compare_require_site_and_errors_for_an_agent(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    box = ["--gpu-gb", "64", "--ram-gb", "62"]
    assert cli.main(["compare", CODER, "deepseek-ai/DeepSeek-V9", *box], fetch=fake_fetch) == 0
    out = capsys.readouterr().out
    assert "Qwen4-Coder-90B  DeepSeek-V9" in out and "licence        apache-2.0" in out and "best build     Q4_K_M" in out
    assert cli.main(["judge", CODER, *box, "--require", "gpu"], fetch=fake_fetch) == 0
    assert cli.main(["judge", CODER, "--gpu-gb", "24", "--ram-gb", "64", "--require", "gpu"], fetch=fake_fetch) == 3
    assert cli.main(["judge", CODER, "--gpu-gb", "24", "--ram-gb", "64", "--require", "memory"], fetch=fake_fetch) == 0
    capsys.readouterr()
    assert cli.main(["judge", "no/such", "--json"], fetch=fake_fetch) == 1
    assert json.loads(capsys.readouterr().out)["errors"][0]["model"] == "no/such"
    assert cli.main(["new", "--since", "tomorrow", "--json"], fetch=fake_fetch) == 1
    assert json.loads(capsys.readouterr().err)["error"]["message"]
    site = tmp_path / "site"
    assert cli.main(["site", "--out", str(site), "--since", "2026-09-01", "--publisher", "Qwen", "--publisher", "deepseek-ai"], fetch=fake_fetch) == 0
    assert {p.name for p in site.iterdir()} == {"index.html", "app.js", "judge.js", "app.css", "catalog.json", "catalog.schema.json"}
    published = json.loads((site / "catalog.json").read_text())
    assert [m["model"] for m in published["models"]] == ["deepseek-ai/DeepSeek-V9", CODER]
    capsys.readouterr()
    assert cli.main(["new", "--catalog", str(site / "catalog.json"), "--since", "2026-09-01", "--json", *box], fetch=fake_fetch) == 0
    assert json.loads(capsys.readouterr().out)["counts"] == {"fits the GPU": 1, "no GGUF yet": 1}


def test_the_golden_cases_hold_in_python_as_they_do_in_the_browser() -> None:
    cases = json.loads((Path(__file__).parent / "golden.json").read_text())
    assert len(cases) >= 50
    for c in cases:
        _, b = machine.resolve(c["machine"], machine.Machine(0, 0, "none", False))
        verdict, pick = judge.judge([hub.Build("r", x["quant"], x["bits"], x["bytes"]) for x in c["builds"]], b)
        assert (b.gpu, b.memory, verdict, pick.quant if pick else None) == (c["budget_gpu"], c["budget_memory"], c["verdict"], c["pick"]), c
