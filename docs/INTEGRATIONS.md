# Integrations

The same answers, in the same JSON shape, through five doors. Pick the one your program or agent already speaks.

| Door | For | Start |
|---|---|---|
| [MCP](#mcp-for-an-agent) | Claude Code, Cursor, any agent that speaks the Model Context Protocol | `ai-models-comparison mcp` |
| [Command line with `--json`](#command-line-for-scripts-and-agents) | shell scripts, CI, an agent with a terminal | any command + `--json` |
| [HTTP API](#http-api) | a program in any language, a dashboard | `ai-models-comparison serve` |
| [Python](#python) | a Python program | `from ai_models_comparison.service import Service` |
| [The catalog file](#the-catalog-file-no-server) | a web page, many readers, no server | `ai-models-comparison site --out DIR` |

## The answer's shape

Every door returns this envelope. `schema` is 1; keys may be added, and it changes only when a reader would break.

```json
{
  "schema": 1,
  "machine": {"gpu_gib": 24.0, "ram_gib": 64.0, "kind": "nvidia", "unified": false,
              "fits_gpu_gib": 22.1, "fits_memory_gib": 70.1, "gpu_reserve_gib": 1.9, "ram_reserve_gib": 16.0},
  "models": [{
    "model": "tencent/ContextPilot-14B", "url": "https://huggingface.co/tencent/ContextPilot-14B",
    "created": "2026-08-27", "kind": "text", "role": "general", "params_b": 14.8, "licence": "other", "context": 40960,
    "verdict": "fits the GPU",
    "pick": {"repo": "mradermacher/ContextPilot-14B-i1-GGUF", "quant": "Q6_K", "gb": 12.1, "bits": 6.0, "bits_per_weight": 6.55,
             "fits": "gpu", "headroom_gb": 11.6,
             "get": "hf download mradermacher/ContextPilot-14B-i1-GGUF --include \"*Q6_K*\"",
             "run": "llama-server -hf mradermacher/ContextPilot-14B-i1-GGUF:Q6_K"},
    "builds": [{"repo": "...", "quant": "Q4_K_M", "gb": 9.0, "bits": 4.0, "bits_per_weight": 4.86, "fits": "gpu"}],
    "note": "", "replaces": ""
  }],
  "errors": [{"model": "no/such", "error": "Hugging Face has no such model, or it is private or gated"}]
}
```

`verdict` is one of `fits the GPU`, `fits in memory`, `low-bit only`, `too big`, `no GGUF yet`. The analysis
returns `roles` instead of `models`: each role has `summary`, `in_use`, `candidates` (a model as above plus
`effective_b`, `speed`, `why`, `cautions`) and `left_out`.

**A machine is described the same way everywhere:** `gpu_gb`, `ram_gb`, `unified`, `gpu_reserve_gb`,
`ram_reserve_gb` (GiB). Leave them out and the machine the tool runs on is used. So one server can answer
for any machine.

## MCP, for an agent

Four read-only tools: `machine`, `judge_models` (one model, or several to compare), `new_models`, `analyze`.
Each returns a short text (a few hundred tokens) and the full answer as structured content.

**Claude Code:**

```bash
claude mcp add ai-models-comparison -- uvx --from git+https://github.com/YauhenBichel/ai-models-comparison ai-models-comparison mcp
```

**Any client with a JSON configuration** (Cursor, Claude Desktop, VS Code, ...):

```json
{
  "mcpServers": {
    "ai-models-comparison": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/YauhenBichel/ai-models-comparison", "ai-models-comparison", "mcp"]
    }
  }
}
```

**For a server you reach over SSH** (the answers are then about that machine, not your laptop):

```bash
claude mcp add my-server-models -- ssh my-server ai-models-comparison mcp
```

Then ask in plain words: "what new models are worth trying on this machine?", "does Qwen3-Coder-Next fit a 24
GB card?", "compare these three models for a Mac with 64 GB".

The server speaks protocol revision 2026-07-28 (no handshake; `server/discover`) and the older revisions that
open with `initialize`, from the same process. It writes nothing but protocol messages to standard output.

An agent **skill** that teaches the command line instead is in
[`skills/ai-models-comparison/SKILL.md`](../skills/ai-models-comparison/SKILL.md): copy the folder into your
agent's skills directory.

## Command line, for scripts and agents

```bash
ai-models-comparison judge Qwen/Qwen3-Coder-Next --json | jq -r '.models[0].pick.get'       # the download command
ai-models-comparison new --json --kind text | jq '.models[] | select(.verdict == "fits the GPU") | .model'
ai-models-comparison judge "$MODEL" --require memory || exit 0                              # stop a pipeline early
```

| Exit status | Means |
|---|---|
| 0 | answered |
| 1 | the question could not be answered: a model not found, a wrong value, Hugging Face unreachable. With `--json`, standard error carries `{"error": {"code", "message"}}` |
| 2 | the command line was wrong |
| 3 | `--require gpu` or `--require memory` was not met |

Nothing is ever asked on the terminal, and with `--json` nothing but JSON is written to standard output.

## HTTP API

```bash
ai-models-comparison serve                     # http://localhost:8377/ ; the description is /openapi.json
```

| Request | Returns |
|---|---|
| `GET /api/v1/machine` | the machine and its budgets |
| `GET /api/v1/models?id=A&id=B` | the models named, judged, in the order named |
| `GET /api/v1/new?days=45&kind=text&verdict=fits the GPU&limit=10` | new models, judged, best fit first, with counts |
| `GET /api/v1/analyze?days=45` (or `?id=A&id=B`) | what is worth trying, role by role, with reasons |
| `GET /api/v1/catalog?days=45` (or `?id=A`) | the same models without a verdict |
| `GET /openapi.json`, `GET /healthz` | the OpenAPI 3.1 description; liveness |

```bash
curl 'http://localhost:8377/api/v1/models?id=Qwen/Qwen3-Coder-Next&gpu_gb=24&ram_gb=64' | jq '.models[0].verdict'
```

Errors are `{"error": {"code", "message"}}` with status 400 (the request was wrong, the message says how),
401, 403, 404 or 502 (Hugging Face unreachable).

**Beyond your own machine.** The server listens on `127.0.0.1` and refuses requests that reach it under
another host name. To serve a network, a token is required: `serve --host 0.0.0.0 --token SECRET`, and clients
send `Authorization: Bearer SECRET`. `--cors https://your.site` lets a page on that origin read the API.

## Python

```python
from ai_models_comparison.service import Service

service = Service()                                              # your configuration file, cached answers
answer = service.models(["Qwen/Qwen3-Coder-Next"], {"gpu_gb": 24, "ram_gb": 64})
print(answer["models"][0]["verdict"], answer["models"][0]["pick"]["run"])

for role in service.analyze()["roles"]:                          # this machine, the last 45 days
    print(role["role"], "->", role["summary"])
```

`Service(cfg={...}, catalog="https://.../catalog.json")` takes a configuration as a dictionary and reads a
published catalog instead of asking Hugging Face.

## The catalog file: no server

Judging is arithmetic; only building the catalog needs Hugging Face. So the catalog can be one static file
that any number of readers share:

```bash
ai-models-comparison site --out public/       # index.html, app.js, judge.js, app.css, catalog.json, catalog.schema.json
```

Serve `public/` from any static host (GitHub Pages, S3, nginx). The page judges in the browser, for the
machine each visitor types. `catalog.json` is described by `catalog.schema.json`:

```json
{"schema": 1, "generated": "2026-10-02T19:42:54Z", "since": "2026-08-18",
 "models": [{"model": "org/Name", "created": "2026-09-01", "kind": "text", "params_b": 14.8, "licence": "apache-2.0",
             "context": 131072, "builds": [{"repo": "org/Name-GGUF", "quant": "Q4_K_M", "bits": 4, "bytes": 9000000000}]}]}
```

Any tool can then run against it, with no question to Hugging Face at all:

```bash
ai-models-comparison new --catalog https://your.site/catalog.json
```

`web/judge.js` in the package is the judge as an ES module (`budgets(machine)`, `judge(builds, budgets)`), if
you want the verdict in your own page.

## The local model stack

| With | Do |
|---|---|
| llama.cpp | the `run` line of a pick is a ready `llama-server -hf repo:quant` command |
| Ollama | `ollama run hf.co/REPO:QUANT` with the pick's `repo` and `quant` (single-file builds) |
| a mixture-of-experts model that only fits in memory | plan the placement with [moe-fit](https://github.com/YauhenBichel/moe-fit) |
| ntfy, or any webhook | `notify = "https://..."` in the configuration: one POST when a model newly fits |
