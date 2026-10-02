---
name: ai-models-comparison
description: Check which open-weights AI models (GGUF builds on Hugging Face) fit a machine's GPU and system memory, compare models side by side, list new releases that fit, and get a short list of models worth trying with reasons. Use when the user asks "can I run X", "which build of X fits", "what new models fit my machine", "compare X and Y for my GPU", or wants to pick a local model.
---

# ai-models-comparison

A command-line tool. It reads public JSON from Hugging Face, downloads no model and never touches the GPU.

Run it with `ai-models-comparison ...` if installed, otherwise with
`uvx --from git+https://github.com/YauhenBichel/ai-models-comparison ai-models-comparison ...`.

## Commands

| Question | Command |
|---|---|
| What does this machine have? | `ai-models-comparison machine --json` |
| Does this model fit, and which build? | `ai-models-comparison judge ORG/NAME --json` |
| Several models side by side | `ai-models-comparison compare ORG/A ORG/B` |
| What was released lately, judged | `ai-models-comparison new --days 45 --json` |
| What is worth trying, with reasons | `ai-models-comparison analyze` (or `analyze ORG/A ORG/B`) |

- Another machine: add `--gpu-gb N --ram-gb N` (GiB), and `--unified` for one shared pool (Apple silicon).
- `--json` gives `{"schema": 1, "machine": {...}, "models": [...]}`; each model has `verdict`, `pick` (the best
  build, with `get` and `run` commands and `headroom_gb`) and `builds` (each with `fits`: `gpu`, `memory`, `no`).
- Exit status: 0 answered, 1 not answered (see `errors` or standard error), 2 wrong command line, 3 `--require` not met.
- `judge ORG/NAME --require gpu` (or `memory`) exits 3 when the model does not fit that well: use it before a download.

## Reading the answer

Verdicts, best first: `fits the GPU`, `fits in memory` (runs, slower), `low-bit only` (under 3 bits a weight:
quality is lost), `too big`, `no GGUF yet`.

A verdict is about memory only. Do not tell the user a model is good because it fits. The analysis ranks by
size after quantization, which is an ordering, not a benchmark: present its list as "worth testing", and
repeat its cautions (licence, low-bit build, little room left, very new files).

Never start a download for the user without asking: builds are tens of gigabytes. Give the `get` command.
