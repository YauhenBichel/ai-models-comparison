# Copyright 2026 Yauhen Bichel
# SPDX-License-Identifier: Apache-2.0
"""Everything against a fake machine and a fake Hugging Face: no network, no GPU."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from fits_here import cli, hub, judge, machine, report

GB = 10**9
GIB = 2**30


def tree(*files: tuple[str, float]) -> list[dict[str, Any]]:
    return [{"path": p, "lfs": {"size": int(s * GB)}} for p, s in files]


# ---- the machine ------------------------------------------------------------------------------------------

def test_linux_with_an_nvidia_card() -> None:
    m = machine.detect(run=lambda c: "24564\n" if c[0] == "nvidia-smi" else "",
                       read=lambda p: "MemTotal:       65806000 kB\n" if "meminfo" in p else "", system="Linux", amd_cards=[])
    assert (m.kind, m.unified, m.gpu) == ("nvidia", False, 24564 * 2**20) and m.ram == 65806000 * 1024
    b = machine.budgets(m)
    assert 21 * GIB < b.gpu < 23 * GIB and b.memory == b.gpu + m.ram - b.ram_reserve and b.ram_reserve > 15 * GIB


def test_an_amd_apu_with_a_carve_out_and_one_without() -> None:
    def read(vram: int, gtt: int):  # type: ignore[no-untyped-def]
        files = {"/proc/meminfo": "MemTotal: 65467120 kB\n", "c0/mem_info_vram_total": str(vram), "c0/mem_info_gtt_total": str(gtt)}
        return lambda p: files.get(p, "")
    carve = machine.detect(run=lambda c: "", read=read(64 * GIB, 16 * GIB), system="Linux", amd_cards=["c0"])
    assert (carve.kind, carve.unified, carve.gpu) == ("amd", False, 64 * GIB)
    b = machine.budgets(carve, gpu_reserve_gib=12, ram_reserve_gib=22)
    assert b.gpu == 52 * GIB and b.memory // GIB == 52 + 62 - 22
    pool = machine.detect(run=lambda c: "", read=read(512 * 2**20, 112 * GIB), system="Linux", amd_cards=["c0"])
    assert pool.unified and pool.gpu == 112 * GIB
    bp = machine.budgets(pool)
    assert bp.gpu == bp.memory and bp.gpu < pool.ram  # one pool: never more than the system has, less its reserve


def test_apple_silicon_and_no_gpu() -> None:
    mac = machine.detect(run=lambda c: "137438953472\n" if c[0] == "sysctl" else "arm64\n", read=lambda p: "", system="Darwin")
    assert mac.kind == "apple" and mac.unified and machine.budgets(mac).gpu == 96 * GIB
    cpu = machine.detect(run=lambda c: "", read=lambda p: "MemTotal: 33554432 kB\n" if "meminfo" in p else "", system="Linux", amd_cards=[])
    b = machine.budgets(cpu)
    assert cpu.kind == "none" and b.gpu == 0 and b.memory == 24 * GIB
    assert "no GPU" in machine.describe(cpu, b)


# ---- the builds ------------------------------------------------------------------------------------------

FLASH = tree(("UD-IQ1_S/M-UD-IQ1_S-00001-of-00003.gguf", 0.005), ("UD-IQ1_S/M-UD-IQ1_S-00002-of-00003.gguf", 49.1),
             ("UD-IQ1_S/M-UD-IQ1_S-00003-of-00003.gguf", 33.4), ("UD-IQ3_XXS/M-UD-IQ3_XXS-00001-of-00002.gguf", 60.0),
             ("UD-IQ3_XXS/M-UD-IQ3_XXS-00002-of-00002.gguf", 44.2), ("M-UD-Q4_K_XL.gguf", 155.1), ("mmproj-F16.gguf", 1.0),
             ("dspark/M-dspark-Q8_0.gguf", 10.9), ("README.md", 0.0))


def test_builds_group_shards_and_skip_helper_files() -> None:
    got = [(b.quant, round(b.size / GB, 1), b.bits) for b in hub.builds_from_tree("u/M-GGUF", FLASH)]
    assert got == [("UD-IQ1_S", 82.5, 1.0), ("UD-IQ3_XXS", 104.2, 3.0), ("UD-Q4_K_XL", 155.1, 4.0)]
    assert [hub.bits_of(q) for q in ("Q4_K_M", "UD-IQ2_M", "Q8_0", "MXFP4", "BF16", "odd")] == [4, 2, 8, 4, 16, 0]


def test_a_build_repo_must_be_of_exactly_this_model() -> None:
    assert hub._same_model("unsloth/GLM-5.3-GGUF", "GLM-5.3") and hub._same_model("mradermacher/GLM-5.3-i1-GGUF", "GLM-5.3")
    assert not hub._same_model("unsloth/GLM-5.3-Flash-GGUF", "GLM-5.3")  # a different model with the same beginning


def budgets(gpu: float, memory: float) -> machine.Budgets:
    return machine.Budgets(int(gpu * GIB), int(memory * GIB), 0, 0)


def test_verdicts() -> None:
    B = hub.Build
    b = budgets(52, 92)
    coder = [B("r", "Q4_K_M", 4, 20 * GB), B("r", "Q8_0", 8, 36 * GB), B("r", "F16", 16, 70 * GB)]
    assert judge.judge(coder, b) == ("fits the GPU", coder[1])
    verdict, pick = judge.judge(hub.builds_from_tree("r", FLASH), b)
    assert verdict == "low-bit only" and pick is not None and pick.quant == "UD-IQ1_S"
    assert judge.judge([B("r", "IQ4_XS", 4, 93 * GB)], b)[0] == "fits in memory"
    assert judge.judge([B("r", "Q2_K", 2, 264 * GB)], b)[0] == "too big"
    assert judge.judge([], b) == ("no GGUF yet", None)
    assert judge.judge(coder, budgets(0, 24))[0] == "fits in memory"  # no GPU: the CPU path


def test_roles() -> None:
    assert [judge.role_of(m, k) for m, k in (("Q/Qwen4-Coder", "text"), ("d/DeepSeek-OCR-2", "vision"), ("g/gemma-5", "vision"),
                                             ("B/bge-m4", "embed"), ("z/GLM-6", "text"), ("m/Voice", "speech"))] == \
        ["coder", "ocr", "vision", "embed", "general", "audio"]


# ---- a run ------------------------------------------------------------------------------------------------

def fake_fetch(url: str) -> Any:
    if "author=Qwen" in url:
        return [{"id": "Qwen/Qwen4-Coder-90B", "createdAt": "2026-09-20T00:00:00.000Z", "pipeline_tag": "text-generation"},
                {"id": "Qwen/Qwen-Image-9", "createdAt": "2026-09-21T00:00:00.000Z", "pipeline_tag": "text-to-image"},
                {"id": "Qwen/Old", "createdAt": "2026-01-01T00:00:00.000Z", "pipeline_tag": "text-generation"},
                {"id": "Qwen/Qwen4-Coder-90B-AWQ", "createdAt": "2026-09-20T00:00:00.000Z", "pipeline_tag": "text-generation"}]
    if "author=deepseek-ai" in url:
        return [{"id": "deepseek-ai/DeepSeek-V9", "createdAt": "2026-09-25T00:00:00.000Z", "pipeline_tag": "text-generation"}]
    if "author=" in url:
        raise OSError("rate limited")
    if "search=Qwen4-Coder-90B" in url:
        return [{"id": "x/Qwen4-Coder-90B-abliterated-GGUF"}, {"id": "x/Qwen4-Coder-90B-Mini-GGUF"}, {"id": "unsloth/Qwen4-Coder-90B-GGUF"}]
    if "search=" in url:
        return []
    if url.endswith("/models/unsloth/Qwen4-Coder-90B-GGUF/tree/main?recursive=true"):
        return tree(("Qwen4-Coder-90B-Q4_K_M.gguf", 54.0), ("Qwen4-Coder-90B-Q3_K_M.gguf", 42.0), ("Qwen4-Coder-90B-Q8_0.gguf", 95.0))
    if url.endswith("/models/Qwen/Qwen4-Coder-90B"):
        return {"safetensors": {"total": 90_000_000_000}, "pipeline_tag": "text-generation"}
    if url.endswith("/models/deepseek-ai/DeepSeek-V9"):
        return {}
    raise AssertionError(url)


def test_new_models_filters_and_survives_a_failing_publisher() -> None:
    errors: list[str] = []
    got = hub.new_models("2026-09-01", fake_fetch, ["Qwen", "deepseek-ai", "broken"], on_error=lambda org, e: errors.append(org))
    assert [g[0] for g in got] == ["deepseek-ai/DeepSeek-V9", "Qwen/Qwen4-Coder-90B"] and errors == ["broken"]


def test_the_cli_writes_a_report_a_page_and_state_and_notifies_only_what_is_new(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    cfg = tmp_path / "c.toml"
    cfg.write_text('publishers = ["Qwen", "deepseek-ai"]\ngpu_gb = 64\nram_gb = 62\ngpu_reserve_gb = 12\nram_reserve_gb = 22\n'
                   'notify = "https://example.test/topic"\n[roster]\ncoder = "qwen3-coder-next"\n')
    sent: list[str] = []
    monkeypatch.setattr(cli, "notify", lambda url, text: sent.append(text))
    args = ["new", "--config", str(cfg), "--since", "2026-09-01", "--out", str(tmp_path / "out"), "--html", str(tmp_path / "p.html"),
            "--state", str(tmp_path / "s.json")]
    assert cli.main(args, fetch=fake_fetch) == 0
    text = (tmp_path / "out" / "latest.md").read_text()
    assert "**fits the GPU**" in text and "`unsloth/Qwen4-Coder-90B-GGUF Q4_K_M, 54.0 GB`" in text and "`qwen3-coder-next`" in text
    assert "**no GGUF yet**" in text and "52 GiB of weights on the GPU, 92 GiB in memory" in text and "abliterated" not in text
    assert "Qwen4-Coder-90B" in (tmp_path / "p.html").read_text()
    state = json.loads((tmp_path / "s.json").read_text())
    assert state["seen"] == {"Qwen/Qwen4-Coder-90B": "fits the GPU", "deepseek-ai/DeepSeek-V9": "no GGUF yet"}
    assert sent == [] and "1 new that fit" in capsys.readouterr().out  # the first run fills the state quietly
    cli.main(args, fetch=fake_fetch)
    assert sent == []
    state["seen"]["Qwen/Qwen4-Coder-90B"] = "no GGUF yet"  # a build appears between two runs
    (tmp_path / "s.json").write_text(json.dumps(state))
    cli.main(args, fetch=fake_fetch)
    assert sent == ["Qwen/Qwen4-Coder-90B (fits the GPU, Q4_K_M 54 GB, coder)"]


def test_judge_and_machine_commands(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["judge", "Qwen/Qwen4-Coder-90B", "--gpu-gb", "24", "--ram-gb", "64"], fetch=fake_fetch) == 0
    out = capsys.readouterr().out
    assert "Qwen/Qwen4-Coder-90B (text, 90B weights): fits in memory" in out and "54.0 GB  Q4_K_M       memory  <- fits in memory" in out
    assert cli.main(["machine", "--gpu-gb", "24", "--ram-gb", "64", "--json"], fetch=fake_fetch) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["gpu_gib"] == 24.0 and d["fits_gpu_gib"] == 22.1 and d["fits_memory_gib"] == 70.1 and d["kind"] == "given"
    assert cli.main(["config"]) == 0 and "[roster]" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        cli.main(["judge", "x/y", "--config", str(tmp_path / "missing.toml")], fetch=fake_fetch)


def test_report_marks_too_big_with_its_smallest_build() -> None:
    m = machine.Machine(62 * GIB, 64 * GIB, "amd", False)
    b = budgets(52, 92)
    big = hub.Build("r", "Q2", 2, 366 * GB)
    j = judge.Judged("a/Huge", "2026-09-10", "text", 763.0, [big], "too big", big, "general")
    md = report.markdown([j], "2026-09-01", m, b, "2026-10-02")
    assert "smallest: Q2, 366 GB" in md and "Counts: too big 1." in md
    assert "No new models" in report.markdown([], "2026-09-01", m, b, "2026-10-02")
