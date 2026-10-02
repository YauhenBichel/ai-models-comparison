// Copyright 2026 Yauhen Bichel
// SPDX-License-Identifier: Apache-2.0
// The verdict, in the browser: the same arithmetic as machine.py and judge.py. tests/golden.json holds the
// cases both must agree on; change one side and the other's test fails.

export const GIB = 2 ** 30;
export const ORDER = ["fits the GPU", "fits in memory", "low-bit only", "too big", "no GGUF yet"];
export const NOTES = {
  "low-bit only": "under 3 bits a weight: measure it on your own tasks before trusting it",
  "fits in memory": "part of the weights stay off the GPU: slower than a model that fits it",
};

const given = (v) => v !== undefined && v !== null && v !== "" && Number.isFinite(Number(v));

// machine: {gpu_gb, ram_gb, unified, gpu_reserve_gb?, ram_reserve_gb?} in GiB -> budgets in bytes
export function budgets(m) {
  const ram = Math.trunc(Number(m.ram_gb || 0) * GIB);
  const gpu = m.unified && !given(m.gpu_gb) ? ram : Math.trunc(Number(m.gpu_gb || 0) * GIB);
  const ramReserve = Math.trunc((given(m.ram_reserve_gb) ? Number(m.ram_reserve_gb) : Math.max(8, (ram / GIB) * 0.25)) * GIB);
  if (m.unified) {
    const fit = Math.max(0, Math.min(gpu, ram) - ramReserve);
    return { gpu: fit, memory: fit, gpu_reserve: 0, ram_reserve: ramReserve };
  }
  const gpuReserve = gpu ? Math.trunc((given(m.gpu_reserve_gb) ? Number(m.gpu_reserve_gb) : Math.max(1, (gpu / GIB) * 0.08)) * GIB) : 0;
  const onGpu = Math.max(0, gpu - gpuReserve);
  return { gpu: onGpu, memory: onGpu + Math.max(0, ram - ramReserve), gpu_reserve: gpuReserve, ram_reserve: ramReserve };
}

// the first of the largest, as Python's max() gives it
function best(items, key) {
  let top = null, topKey = null;
  for (const x of items) {
    const k = key(x);
    if (top === null || k[0] > topKey[0] || (k[0] === topKey[0] && k[1] > topKey[1])) { top = x; topKey = k; }
  }
  return top;
}

// the most bits worth having, then the larger file; above 8 bits nothing is gained, so F16 only when it is all there is
function bestFit(fitting) {
  const usual = fitting.filter((x) => x.bits <= 8);
  return usual.length ? best(usual, (x) => [x.bits, x.bytes]) : best(fitting, (x) => [-x.bytes, 0]);
}

// builds: [{quant, bits, bytes}] -> {verdict, pick}
export function judge(builds, b, gpuMinBits = 4, memoryMinBits = 3) {
  if (!builds || !builds.length) return { verdict: "no GGUF yet", pick: null };
  const onGpu = builds.filter((x) => x.bytes <= b.gpu && x.bits >= gpuMinBits);
  if (onGpu.length) return { verdict: "fits the GPU", pick: bestFit(onGpu) };
  const inMemory = builds.filter((x) => x.bytes <= b.memory && x.bits >= memoryMinBits);
  if (inMemory.length) return { verdict: "fits in memory", pick: bestFit(inMemory) };
  const low = builds.filter((x) => x.bytes <= b.memory);
  if (low.length) return { verdict: "low-bit only", pick: best(low, (x) => [x.bytes, 0]) };
  return { verdict: "too big", pick: best(builds, (x) => [-x.bytes, 0]) };
}

export function roleOf(model, kind) {
  const name = model.toLowerCase();
  if (kind === "embed") return "embed";
  if (kind === "speech") return "audio";
  if (name.includes("ocr")) return "ocr";
  if (kind === "vision") return "vision";
  if (/cod(e|er)|devstral|codestral/.test(name)) return "coder";
  return "general";
}

export function fitsWhere(build, b) {
  return build.bytes <= b.gpu ? "gpu" : build.bytes <= b.memory ? "memory" : "no";
}
