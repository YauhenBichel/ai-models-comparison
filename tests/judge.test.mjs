// Copyright 2026 Yauhen Bichel
// SPDX-License-Identifier: Apache-2.0
// The browser's judge against the cases the Python judge produced: `node --test tests/`.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { budgets, judge, roleOf } from "../src/ai_models_comparison/web/judge.js";

const cases = JSON.parse(readFileSync(new URL("./golden.json", import.meta.url), "utf8"));

test("the browser and Python agree on every golden case", () => {
  assert.ok(cases.length >= 50);
  for (const c of cases) {
    const b = budgets(c.machine);
    const where = `${JSON.stringify(c.machine)} / ${c.set}`;
    assert.equal(b.gpu, c.budget_gpu, `GPU budget: ${where}`);
    assert.equal(b.memory, c.budget_memory, `memory budget: ${where}`);
    const got = judge(c.builds, b);
    assert.equal(got.verdict, c.verdict, `verdict: ${where}`);
    assert.equal(got.pick ? got.pick.quant : null, c.pick, `pick: ${where}`);
  }
});

test("roles", () => {
  const got = [["Q/Qwen4-Coder", "text"], ["d/DeepSeek-OCR-2", "vision"], ["g/gemma-5", "vision"], ["B/bge-m4", "embed"], ["z/GLM-6", "text"],
    ["m/Voice", "speech"]].map(([m, k]) => roleOf(m, k));
  assert.deepEqual(got, ["coder", "ocr", "vision", "embed", "general", "audio"]);
});
