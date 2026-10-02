// Copyright 2026 Yauhen Bichel
// SPDX-License-Identifier: Apache-2.0
// The page. It reads a catalog (from the local server's API, or catalog.json beside it on a static host),
// and judges every model in the browser as the machine form changes. No framework, no build step.
import { GIB, NOTES, ORDER, budgets, fitsWhere, judge, roleOf } from "./judge.js";

const PRESETS = [
  ["24 GB GPU, 64 GB memory", { gpu_gb: 24, ram_gb: 64 }],
  ["16 GB GPU, 32 GB memory", { gpu_gb: 16, ram_gb: 32 }],
  ["12 GB GPU, 32 GB memory", { gpu_gb: 12, ram_gb: 32 }],
  ["8 GB GPU, 16 GB memory", { gpu_gb: 8, ram_gb: 16 }],
  ["Two 24 GB GPUs, 128 GB memory", { gpu_gb: 48, ram_gb: 128 }],
  ["Apple silicon, 32 GB", { ram_gb: 32, unified: true }],
  ["Apple silicon, 64 GB", { ram_gb: 64, unified: true }],
  ["Apple silicon, 128 GB", { ram_gb: 128, unified: true }],
  ["AMD Strix Halo 128 GB (64 to the GPU)", { gpu_gb: 64, ram_gb: 62 }],
  ["No GPU, 32 GB memory", { gpu_gb: 0, ram_gb: 32 }],
];
const $ = (id) => document.getElementById(id);
const state = { machine: { gpu_gb: 24, ram_gb: 64, unified: false }, detected: null, catalog: null, served: false, added: [],
  hidden: new Set(["no GGUF yet"]), open: new Set(), compare: [], token: "" };

function h(tag, attrs = {}, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === false || v === null || v === undefined) continue;
    if (k === "class") el.className = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const kid of kids.flat()) if (kid !== null && kid !== undefined && kid !== false) el.append(kid);
  return el;
}
const gb = (bytes) => (bytes / 1e9).toFixed(bytes >= 1e11 ? 0 : 1) + " GB";
const gib = (bytes) => (bytes / GIB).toFixed(0);
const ctx = (n) => (!n ? "" : n >= 1024 ? Math.round(n / 1024) + "k" : String(n));

// ---- storage and the address: a machine survives a reload and travels in a link -------------------------
function readHash() {
  const p = new URLSearchParams(location.hash.slice(1));
  if (p.get("token")) {
    state.token = p.get("token");
    try { sessionStorage.setItem("token", state.token); } catch { /* a private window */ }
    p.delete("token");
  } else {
    try { state.token = sessionStorage.getItem("token") || ""; } catch { /* a private window */ }
  }
  state.compare = (p.get("c") || "").split(",").filter(Boolean).slice(0, 4);
  state.open = new Set((p.get("o") || "").split(",").filter(Boolean));
  state.query = p.get("q") || "";
  if (state.compare.length || state.open.size) state.hidden.clear();
  if (!p.has("ram")) return null;
  const num = (k) => (p.has(k) && p.get(k) !== "" ? Number(p.get(k)) : undefined);
  return { gpu_gb: num("gpu"), ram_gb: num("ram"), unified: p.get("u") === "1", gpu_reserve_gb: num("gr"), ram_reserve_gb: num("rr") };
}
function writeHash() {
  const m = state.machine, p = new URLSearchParams();
  if (!m.unified) p.set("gpu", m.gpu_gb ?? 0);
  p.set("ram", m.ram_gb ?? 0);
  if (m.unified) p.set("u", "1");
  if (m.gpu_reserve_gb !== undefined) p.set("gr", m.gpu_reserve_gb);
  if (m.ram_reserve_gb !== undefined) p.set("rr", m.ram_reserve_gb);
  if (state.compare.length) p.set("c", state.compare.join(","));
  if (state.open.size) p.set("o", [...state.open].join(","));
  if ($("search").value.trim()) p.set("q", $("search").value.trim());
  history.replaceState(null, "", "#" + p.toString().replaceAll("%2F", "/").replaceAll("%2C", ","));
  try { localStorage.setItem("machine", JSON.stringify(m)); } catch { /* a private window */ }
}
function stored() {
  try { return JSON.parse(localStorage.getItem("machine") || "null"); } catch { return null; }
}
async function api(path) {
  const r = await fetch(path, state.token ? { headers: { authorization: "Bearer " + state.token } } : {});
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(body.error?.message || r.statusText);
  return body;
}

// ---- the machine form ------------------------------------------------------------------------------------
function fillForm() {
  const m = state.machine;
  $("unified").checked = !!m.unified;
  $("gpu").disabled = !!m.unified;
  $("gpu").value = m.unified ? "" : m.gpu_gb ?? "";
  $("ram").value = m.ram_gb ?? "";
  $("gpu-reserve").value = m.gpu_reserve_gb ?? "";
  $("ram-reserve").value = m.ram_reserve_gb ?? "";
  $("gpu-reserve").disabled = !!m.unified;
}
function readForm() {
  const num = (id) => ($(id).value === "" ? undefined : Math.max(0, Number($(id).value)));
  const unified = $("unified").checked;
  state.machine = { gpu_gb: unified ? undefined : num("gpu") ?? 0, ram_gb: num("ram") ?? 0, unified,
    gpu_reserve_gb: unified ? undefined : num("gpu-reserve"), ram_reserve_gb: num("ram-reserve") };
  $("preset").value = "";
  fillForm();
  writeHash();
  render();
}
function setupForm() {
  const sel = $("preset");
  sel.append(h("option", { value: "" }, "choose a machine like yours"));
  if (state.detected) sel.append(h("option", { value: "detected" }, "this machine, as detected"));
  PRESETS.forEach(([name], i) => sel.append(h("option", { value: String(i) }, name)));
  sel.addEventListener("change", () => {
    if (sel.value === "") return;
    state.machine = sel.value === "detected" ? { ...state.detected } : { unified: false, ...PRESETS[Number(sel.value)][1] };
    fillForm(); writeHash(); render();
  });
  for (const id of ["gpu", "ram", "unified", "gpu-reserve", "ram-reserve"]) $(id).addEventListener("input", readForm);
  $("search").value = state.query || "";
  for (const id of ["kind", "search"]) $(id).addEventListener("input", () => { writeHash(); render(); });
  $("compare-clear").addEventListener("click", () => { state.compare = []; writeHash(); render(); });
  try { $("howto").open = localStorage.getItem("howto") !== "closed"; } catch { /* a private window */ }
  $("howto").addEventListener("toggle", () => { try { localStorage.setItem("howto", $("howto").open ? "open" : "closed"); } catch { /* same */ } });
  $("add").addEventListener("submit", addModel);
}

// ---- judging and drawing ---------------------------------------------------------------------------------
function judged() {
  const b = budgets(state.machine);
  const all = [...state.added, ...(state.catalog?.models || []).filter((e) => !state.added.some((a) => a.model === e.model))];
  return { b, models: all.map((e) => {
    const { verdict, pick } = judge(e.builds, b);
    return { ...e, verdict, pick, role: roleOf(e.model, e.kind), note: NOTES[verdict] || "", rank: ORDER.indexOf(verdict) };
  }) };
}
function bar(bytes, b, rank, scale) {
  const pct = (v) => Math.min(100, (v / scale) * 100) + "%";
  const track = h("div", { class: "track", role: "img",
    "aria-label": `${gb(bytes)} against ${gib(b.gpu)} GiB on the GPU and ${gib(b.memory)} GiB in memory` });
  const mem = h("div", { class: "zone memz" }), gpu = h("div", { class: "zone gpu" });
  mem.style.left = "0"; mem.style.width = pct(b.memory); gpu.style.left = "0"; gpu.style.width = pct(b.gpu);
  const fill = h("div", { class: `fill v${Math.min(rank, 3)}${bytes > scale ? " over" : ""}` });
  fill.style.width = pct(bytes);
  track.append(mem, gpu, fill);
  return track;
}
function verdictCell(v, rank) {
  return h("span", { class: `verdict v${rank}` }, h("span", { class: "dot" }), v);
}
function copyButton(text) {
  const btn = h("button", { type: "button", onclick: async (ev) => {
    ev.stopPropagation();
    try { await navigator.clipboard.writeText(text); btn.textContent = "Copied"; } catch { btn.textContent = "Select and copy"; }
    setTimeout(() => (btn.textContent = "Copy"), 1500);
  } }, "Copy");
  return btn;
}
function detail(j, b, scale) {
  const facts = [["Licence", j.licence], ["Context", ctx(j.context)], ["Kind", `${j.kind} / ${j.role}`],
    ["Downloads", j.downloads?.toLocaleString()], ["Likes", j.likes?.toLocaleString()]].filter(([, v]) => v);
  const rows = j.builds.flatMap((x) => {
    const where = { gpu: "GPU", memory: "memory", no: "does not fit" }[fitsWhere(x, b)];
    const cls = j.pick && x.quant === j.pick.quant && j.rank < 3 ? "picked" : "";
    return [h("span", { class: cls }, x.quant), h("span", { class: cls }, gb(x.bytes)), h("span", { class: cls }, where),
      bar(x.bytes, b, x.bytes <= b.gpu ? 0 : x.bytes <= b.memory ? 1 : 3, scale)];
  });
  const cmds = j.pick && j.rank < 3 ? [
    h("div", { class: "cmd" }, h("code", {}, `llama-server -hf ${j.pick.repo}:${j.pick.quant}`), copyButton(`llama-server -hf ${j.pick.repo}:${j.pick.quant}`)),
    h("div", { class: "cmd" }, h("code", {}, `hf download ${j.pick.repo} --include "*${j.pick.quant}*"`),
      copyButton(`hf download ${j.pick.repo} --include "*${j.pick.quant}*"`))] : [];
  return h("tr", { class: "detail" }, h("td", { colspan: "7" },
    h("dl", { class: "facts" }, facts.map(([k, v]) => h("div", {}, h("dt", {}, k), h("dd", {}, v)))),
    j.builds.length ? h("div", { class: "builds" }, rows) : h("p", { class: "note" }, "No GGUF builds were found for this model yet."),
    cmds, j.note ? h("p", { class: "note" }, "Note: " + j.note + ".") : null));
}
function render() {
  const { b, models } = judged();
  const m = state.machine;
  $("budgets").replaceChildren(
    b.gpu > 0 && !m.unified ? h("span", {}, "A build fits the GPU up to ", h("b", {}, gib(b.gpu) + " GiB"), ", and fits in memory up to ",
      h("b", {}, gib(b.memory) + " GiB"), " (GPU and system memory together).")
      : h("span", {}, "A build fits up to ", h("b", {}, gib(b.memory) + " GiB"), m.unified ? " of the shared pool." : " of system memory: no GPU."));
  $("legend").replaceChildren(...[
    b.gpu > 0 && !m.unified ? h("span", {}, h("span", { class: "key gpu" }), "GPU budget") : null,
    h("span", {}, h("span", { class: "key mem" }), "memory budget"), h("span", {}, "bar: the best build's size")].filter(Boolean));

  const counts = ORDER.map((v) => models.filter((j) => j.verdict === v).length);
  $("verdicts").replaceChildren(...ORDER.map((v, i) => h("button", { type: "button", class: `chip v${i}`,
    "aria-pressed": String(!state.hidden.has(v)), onclick: () => { state.hidden.has(v) ? state.hidden.delete(v) : state.hidden.add(v); render(); } },
    h("span", { class: "dot" }), v, h("span", { class: "n" }, String(counts[i])))));

  const kind = $("kind").value, q = $("search").value.trim().toLowerCase();
  const shown = models.filter((j) => !state.hidden.has(j.verdict) && (!kind || j.kind === kind) && (!q || j.model.toLowerCase().includes(q)))
    .sort((x, y) => x.rank - y.rank || (y.created || "").localeCompare(x.created || "") || x.model.localeCompare(y.model));
  const scale = Math.max(b.memory, 8 * GIB) * 1.25;
  const rows = [];
  for (const j of shown) {
    const [org, ...rest] = j.model.split("/");
    const checked = state.compare.includes(j.model);
    const box = h("input", { type: "checkbox", "aria-label": `Compare ${j.model}`, checked,
      onclick: (ev) => ev.stopPropagation(),
      onchange: () => { state.compare = checked ? state.compare.filter((x) => x !== j.model) : [...state.compare, j.model].slice(-4); writeHash(); render(); } });
    box.checked = checked;
    const toggle = () => { state.open.has(j.model) ? state.open.delete(j.model) : state.open.add(j.model); writeHash(); render(); };
    const name = h("a", { href: `https://huggingface.co/${j.model}`, target: "_blank", rel: "noopener", onclick: (ev) => ev.stopPropagation() },
      h("span", { class: "org" }, org + "/"), rest.join("/"));
    rows.push(h("tr", { class: "model", onclick: toggle },
      h("td", { class: "pick" }, box),
      h("td", {}, h("div", { class: "name" }, name), h("div", { class: "sub" }, j.kind === j.role ? j.kind : `${j.kind} / ${j.role}`)),
      h("td", { class: "opt date" }, j.created || ""),
      h("td", { class: "num opt" }, j.params_b ? j.params_b + "B" : ""),
      h("td", {}, verdictCell(j.verdict, j.rank)),
      h("td", { class: "build" }, j.pick ? (j.rank === 3 ? `smallest: ${j.pick.quant}, ${gb(j.pick.bytes)}` : `${j.pick.quant}, ${gb(j.pick.bytes)}`) : ""),
      h("td", { class: "mem" }, j.pick ? bar(j.pick.bytes, b, j.rank, scale) : null,
        h("button", { type: "button", class: "sr", "aria-expanded": String(state.open.has(j.model)), onclick: (ev) => { ev.stopPropagation(); toggle(); } },
          `All builds of ${j.model}`))));
    if (state.open.has(j.model)) rows.push(detail(j, b, scale));
  }
  $("rows").replaceChildren(...rows);
  $("empty").hidden = shown.length > 0;
  $("empty").textContent = models.length ? "No model matches these filters. Switch a verdict back on above." : "The catalog has no models yet.";
  renderCompare(models, b);
  clearTimeout(state.timer);
  if (state.served) state.timer = setTimeout(analysis, 350);
}
async function analysis() {
  const m = state.machine, p = new URLSearchParams();
  for (const [k, v] of Object.entries(m)) if (v !== undefined && v !== false) p.set(k, v);
  const key = p.toString();
  if (key === state.analysed) return;
  try {
    const got = await api("api/v1/analyze?" + key);
    state.analysed = key;
    const roles = got.roles.filter((r) => r.candidates.length);
    $("analysis").hidden = roles.length === 0;
    $("analysis-body").replaceChildren(...roles.map((r) => h("details", { class: "role" },
      h("summary", {}, h("span", { class: "role-name" }, r.role), h("span", {}, r.summary),
        r.in_use ? h("span", { class: "sub" }, `in use: ${r.in_use}`) : null),
      r.candidates.map((c) => h("div", { class: "cand" },
        h("div", { class: "cand-head" },
          h("a", { href: c.url, target: "_blank", rel: "noopener", class: "name" }, c.model),
          verdictCell(c.verdict, ORDER.indexOf(c.verdict)),
          h("span", { class: "build" }, `${c.pick.quant}, ${c.pick.gb} GB`),
          h("span", { class: "sub" }, `ranks as ${c.effective_b}B · ${c.speed}`)),
        h("ul", { class: "why" }, c.why.map((w) => h("li", {}, w)), c.cautions.map((w) => h("li", { class: "caution" }, w))))))));
  } catch { $("analysis").hidden = true; }
}
function renderCompare(models, b) {
  const picked = state.compare.map((id) => models.find((j) => j.model === id)).filter(Boolean);
  $("compare").hidden = picked.length < 2;
  if (picked.length < 2) return;
  const bpw = (j) => (j.pick && j.params_b ? ((j.pick.bytes * 8) / (j.params_b * 1e9)).toFixed(2) : "");
  const room = (j) => (j.pick && j.rank < 3 ? gb((j.rank === 0 ? b.gpu : b.memory) - j.pick.bytes) : "");
  const facts = [
    ["Verdict", (j) => verdictCell(j.verdict, j.rank)], ["Best build", (j) => (j.pick ? j.pick.quant : "")],
    ["Size", (j) => (j.pick ? gb(j.pick.bytes) : "")], ["Bits a weight", bpw], ["Room left", room],
    ["Weights", (j) => (j.params_b ? j.params_b + "B" : "")], ["Context", (j) => ctx(j.context)], ["Kind", (j) => (j.kind === j.role ? j.kind : `${j.kind} / ${j.role}`)],
    ["Licence", (j) => j.licence || ""], ["Released", (j) => j.created || ""], ["Downloads", (j) => j.downloads?.toLocaleString() || ""]];
  $("compare-table").replaceChildren(
    h("thead", {}, h("tr", {}, h("th", {}, ""), picked.map((j) => h("th", { scope: "col" }, j.model.split("/").pop())))),
    h("tbody", {}, facts.map(([name, f]) => h("tr", {}, h("th", { scope: "row" }, name), picked.map((j) => h("td", {}, f(j) || "–"))))));
}
async function addModel(ev) {
  ev.preventDefault();
  const id = $("add-id").value.trim();
  if (!id) return;
  $("add-status").textContent = "asking Hugging Face…";
  try {
    const got = await api("api/v1/catalog?id=" + encodeURIComponent(id));
    if (got.errors?.length) throw new Error(got.errors[0].error);
    for (const e of got.models) {
      state.added = [e, ...state.added.filter((a) => a.model !== e.model)];
      state.open.add(e.model);
      state.hidden.clear();
    }
    $("add-id").value = ""; $("add-status").textContent = "";
    render();
  } catch (e) { $("add-status").textContent = String(e.message || e); }
}

// ---- start -----------------------------------------------------------------------------------------------
async function start() {
  const fromHash = readHash();
  try {
    const d = (await api("api/v1/machine")).machine;        // a local server: it knows this machine
    state.served = true;
    state.detected = { gpu_gb: d.unified ? undefined : d.gpu_gib, ram_gb: d.ram_gib, unified: d.unified };
  } catch { /* a static host: the visitor describes the machine */ }
  state.machine = fromHash || (state.served ? { ...state.detected } : stored()) || state.machine;
  setupForm(); fillForm();
  if (state.served && !fromHash) $("preset").value = "detected";
  $("add").hidden = !state.served;
  $("meta").textContent = "Reading the catalog…";
  render();
  try {
    state.catalog = state.served ? await api("api/v1/catalog?days=45") : await (await fetch("catalog.json", { cache: "no-cache" })).json();
    const age = (Date.now() - Date.parse(state.catalog.generated)) / 86400000;
    $("meta").textContent = `${state.catalog.models.length} models since ${state.catalog.since} · catalog of ${state.catalog.generated.slice(0, 10)}`
      + (age > 3 ? ` · ${Math.floor(age)} days old` : "");
    $("meta").classList.toggle("stale", age > 3);
  } catch (e) {
    $("meta").textContent = "The catalog could not be read: " + String(e.message || e);
    $("meta").classList.add("stale");
  }
  render();
}
start();
