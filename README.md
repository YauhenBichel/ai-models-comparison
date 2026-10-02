# ai-models-comparison

**Compare new open-weights AI models with the machine you have: which of them fit it?**

New models appear every week, and each comes in a dozen builds from 20 GB to 400 GB. The question for
anyone who runs models at home or on one server is always the same: *is there a build of this that fits my
GPU, or at least my memory, and is it a real build or a 1-bit one?* Answering it by hand means opening the
model page, finding a GGUF repository, adding up shards and comparing with what the machine has.

`ai-models-comparison` does that for you, for every new model of the publishers you watch:

```console
$ ai-models-comparison machine
system memory   62.4 GiB
GPU memory      64.0 GiB  (amd)
fits the GPU    up to 58.9 GiB of weights
fits in memory  up to 105.7 GiB of weights (GPU and system memory together)

$ ai-models-comparison judge deepseek-ai/DeepSeek-V4-Flash-0731 Qwen/Qwen3.8-Flash-Next
deepseek-ai/DeepSeek-V4-Flash-0731 (text, 304.2B weights): low-bit only
      82.5 GB  UD-IQ1_S     memory
      96.8 GB  UD-Q2_K_XL   memory  <- low-bit only
     104.2 GB  UD-IQ3_XXS   memory
     155.1 GB  UD-Q4_K_XL   no
  note: under 3 bits a weight: measure it on your own tasks before trusting it
Qwen/Qwen3.8-Flash-Next (vision, 180B weights): fits in memory
      90.0 GB  UD-Q3_K_XL   memory
      93.7 GB  UD-IQ4_XS    memory  <- fits in memory
     111.3 GB  UD-Q4_K_XL   no

$ ai-models-comparison new --days 45
# Open-weights models since 2026-08-18, judged for this machine
| Model | Released | Kind | Weights | Verdict | Best build that fits | Role | Would replace | Note |
...
```

It downloads nothing but JSON from Hugging Face's public API, never touches the GPU, and needs no token.
Standard library only.

## Install

```bash
pip install ai-models-comparison        # or: uv tool install ai-models-comparison, or run once with: uvx ai-models-comparison machine
```

Python 3.11 or newer. Linux and macOS.

## What it reads from your machine

| Machine | Where the numbers come from | The budgets |
|---|---|---|
| NVIDIA GPUs | `nvidia-smi` (all cards summed), `/proc/meminfo` | GPU: its memory less 8 %; memory: that plus system memory less a quarter |
| AMD GPU, or an AMD APU with a BIOS carve-out | `/sys/class/drm/card*/device/mem_info_vram_total` | the same |
| AMD APU without a carve-out (the GPU maps system memory) | `mem_info_gtt_total` | one pool: system memory less a quarter |
| Apple silicon | `sysctl hw.memsize` | one pool: memory less a quarter |
| no GPU | `/proc/meminfo` | memory only: models run on the CPU |

Every figure can be set: `--gpu-gb`, `--ram-gb`, `--unified`, `--gpu-reserve-gb`, `--ram-reserve-gb`, or the
same keys in the configuration file. The reserves are what you keep free for the context cache and the rest
of the machine; the defaults are cautious.

## The verdicts

| Verdict | Means |
|---|---|
| **fits the GPU** | a build of at least 4 bits a weight fits the GPU budget: the fast path |
| **fits in memory** | a build of at least 3 bits fits GPU and system memory together: it runs with part of the weights off the GPU (llama.cpp's `--n-gpu-layers` or `--n-cpu-moe`), slower |
| **low-bit only** | only builds under 3 bits fit. They lose quality: one published coding benchmark of a large model shows 3 bits nearly free and 1 bit costing 16 to 18 points. Measure before trusting |
| **too big** | no build fits; the report shows the smallest |
| **no GGUF yet** | nothing to run with llama.cpp or Ollama today |

**A verdict is about memory.** It does not say the model is good, or better than what you run. That is what
your own tests are for.

## A daily review

```bash
ai-models-comparison config > ~/.config/ai-models-comparison/config.toml     # then edit: publishers, reserves, your roster, a notify URL
ai-models-comparison new --state ~/.local/state/ai-models-comparison.json --out ~/reports/models --html ~/public/models.html --quiet
```

With `--state`, a run remembers what it has seen. With `notify` in the configuration (an [ntfy](https://ntfy.sh)
topic URL or any endpoint that takes a POST), a model that newly fits is one line on your phone; the first
run fills the state and stays quiet. A systemd timer:

```ini
# ~/.config/systemd/user/ai-models-comparison.service
[Service]
Type=oneshot
ExecStart=%h/.local/bin/ai-models-comparison new --state %h/.local/state/ai-models-comparison.json --out %h/reports/models --quiet

# ~/.config/systemd/user/ai-models-comparison.timer
[Timer]
OnCalendar=*-*-* 07:10
Persistent=true
[Install]
WantedBy=timers.target
```

The `[roster]` table in the configuration names what you run today by role (coder, general, vision, embed,
ocr, audio); the report then says what a new model would replace.

## How it finds builds

For a model `org/Name` it searches Hugging Face for GGUF repositories named exactly `Name` (so
`Name-Flash-GGUF` is not mistaken for `Name`), prefers the publisher's own, then `unsloth`, `bartowski`,
`ggml-org`, `lmstudio-community`, `mradermacher`, and leaves out abliterated and uncensored variants. In the
chosen repository it groups `.gguf` files by quantization, sums shards and folders, and skips helper files
(projectors, draft heads, tokenizers). When the model page gives a weight count, a "build" under 0.75 bits a
weight is dropped: it is something else that slipped through.

## Honest limits

- **Memory only.** No speed estimate. For a mixture-of-experts model, its companion
  [moe-fit](https://github.com/YauhenBichel/moe-fit) plans the placement and estimates the speed from the
  model's index.
- **File size is not loaded size.** The context cache and the runtime take more; that is what the reserves
  are for. A model at the edge of a budget deserves a look at its card.
- **GGUF only.** Models that ship only as safetensors, MLX or ONNX show as "no GGUF yet".
- **Names are matched, not understood.** A build repository with an unusual name is missed. A role is a
  guess from the name and the task tag.
- **Public repositories only**, and Hugging Face's API is rate-limited without a token; `HF_TOKEN` in the
  environment is used when present and never printed.

## Contributing

Issues and pull requests are welcome: a machine it detects wrongly (paste `ai-models-comparison machine --json`), a
model whose builds it misses, a publisher worth watching by default.

```bash
uv run ruff check . && uv run mypy && uv run pytest -q     # no network, no GPU
```

## Licence

Apache-2.0.
