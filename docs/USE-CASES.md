# Use cases, shown on a real home server

Every example here is real output from 2 October 2026, for one machine: **yserver**, a mini PC at home with an
AMD Ryzen AI MAX+ 395 and 128 GB of memory shared between CPU and GPU, half of it given to the GPU. It
[serves open models to coding tools and side projects](https://github.com/YauhenBichel/yserver-local-llm-system).
Its configuration file is the whole setup:

```toml
# ~/.config/ai-models-comparison/config.toml
gpu_gb = 64            # the half given to the GPU
ram_gb = 62
gpu_reserve_gb = 12    # context caches of the models that stay loaded
ram_reserve_gb = 22    # the system, an embedding model on the CPU, small speech servers

[roster]               # what it runs today
coder = "Qwen/Qwen3-Coder-Next"
general = "openai/gpt-oss-120b"
vision = "Qwen/Qwen3.6-35B-A3B"
embed = "BAAI/bge-m3"
ocr = "deepseek-ai/DeepSeek-OCR"
```

Budgets that follow: a build fits the GPU up to 52 GiB, and fits in memory up to 92 GiB.

## 1. "Is there a bigger general model than the one I run?"

The question that started this tool. One command, no model named:

```console
$ ai-models-comparison analyze
general  (in use: openai/gpt-oss-120b)
  nothing new on the GPU is larger than what you run; largest that runs: Qwen/Qwen3.8-Flash-Next (174.6B, partly in system memory)
  1. Qwen/Qwen3.8-Flash-Next  fits in memory  UD-IQ4_XS 93.7 GB  ranks as 174.6B, partly in system memory
     + 180B weights at UD-IQ4_XS (93.7 GB): about 97% of full precision, so it ranks as 174.6B
     + 60% of the build fits the GPU, the rest runs from system memory: slower
     + context 256k
     + 1.5 times the size of gpt-oss-120b, which you run today
     + 12 months newer than gpt-oss-120b
     + it also reads images
     ! licence qwen-community-1.0: read it before commercial use
     ! only 5.1 GB of room left: the context cache needs some, so use a short context or the next smaller build
     run: llama-server -hf unsloth/Qwen3.8-Flash-Next-GGUF:UD-IQ4_XS
  2. zai-org/GLM-5.3-Flash-BF16  low-bit only  IQ2_XXS 96.6 GB  ranks as 295.6B, partly in system memory
     ! only a build under 3 bits fits: it loses quality, measure it on your own tasks before trusting it
  3. BAAI/AREX-2  fits the GPU  Q6_K 22.1 GB  ranks as 27.1B, on the GPU
     ! smaller than what you run, but much newer: new small models often beat older large ones; only a test on your tasks says
  left out: 3 too big, 26 no GGUF yet, 7 smaller than what you run today, 2 smaller or slower than a listed one
```

**What it gave:** out of 49 models released in 45 days, three to look at, with the reason for each and the
honest cost of each. The first is a real 4-bit build, 1.5 times the size of the model in use; it needs 40 % of
its weights in system memory, and leaves 5 GB for context. The other 38 are accounted for in one line.

**What was done with it:** the first candidate was downloaded, to be tested against the model in use on the
server's own task suite. The analysis chose what to test; it did not decide.

## 2. "I want DeepSeek. Which build, and is it better than the alternative?"

```console
$ ai-models-comparison compare openai/gpt-oss-120b Qwen/Qwen3.8-Flash-Next deepseek-ai/DeepSeek-V4-Flash-0731

               gpt-oss-120b    Qwen3.8-Flash-Next  DeepSeek-V4-Flash-0731
verdict        fits in memory  fits in memory      low-bit only
best build     UD-Q8_K_XL      UD-IQ4_XS           UD-Q2_K_XL
size           64.5 GB         93.7 GB             96.8 GB
bits a weight  4.42            4.16                2.55
room left      34.3 GB         5.1 GB              2 GB
weights        116.8B          180B                304.2B
context        128k            256k                1024k
licence        apache-2.0      qwen-community-1.0  mit
released       2025-08-04      2026-08-24          2026-07-31
```

**What it gave:** the larger name is not the better choice here. DeepSeek V4 Flash has more weights, but on
this machine only at 2.55 bits a weight with 2 GB of room; the Qwen model runs at 4.16 bits. `judge` then
lists DeepSeek's thirteen builds and shows that the first 3-bit one (104 GB) is 5 GB over the budget: it
would fit only by cutting what is kept free for the system.

## 3. "What would more memory buy me?"

Before buying hardware, ask about the machine you do not have:

```console
$ ai-models-comparison analyze deepseek-ai/DeepSeek-V4-Flash-0731 Qwen/Qwen3.8-Flash-Next openai/gpt-oss-120b \
      --ram-gb 192 --unified --ram-reserve-gb 24
general
  1. deepseek-ai/DeepSeek-V4-Flash-0731  fits the GPU  UD-Q8_K_XL 161.9 GB  ranks as 304.2B, on the GPU
  2. Qwen/Qwen3.8-Flash-Next  fits the GPU  UD-Q6_K_XL 169.2 GB  ranks as 178.2B, on the GPU
     ! only 11.2 GB of room left: the context cache needs some, so use a short context or the next smaller build
  3. openai/gpt-oss-120b  fits the GPU  UD-Q8_K_XL 64.5 GB  ranks as 116.8B, on the GPU
```

**What it gave:** with 192 GB in one pool, DeepSeek V4 Flash moves from "low-bit only" to a full build on the
GPU. And `judge deepseek-ai/DeepSeek-V4.1-Flash` says "too big" on both machines (its smallest build is 366
GB): that upgrade would not buy the newest DeepSeek. A purchase decision, answered in two commands.

## 4. "Tell me when something that fits appears"

On the server, a timer runs once a day:

```bash
ai-models-comparison new --state ~/.local/state/ai-models-comparison.json --out ~/reports/models --html ~/www/models.html --quiet
```

It writes the day's report and a page, and sends one line to a phone (through [ntfy](https://ntfy.sh)) only
when a model **newly** fits:

```
Qwen/Qwen3.8-Flash-Next (fits in memory, UD-IQ4_XS 94 GB, vision)
```

**What it gave:** nobody has to read release news to keep a home server current. In 45 days, 49 models were
judged; 15 fit, and one of them was larger than anything the server runs.

## 5. "Do not start a 94 GB download that cannot work"

A download script on the server checks first, and stops with a reason:

```bash
ai-models-comparison judge "$MODEL" --require memory || { echo "$MODEL does not fit this machine at 3 bits or more"; exit 1; }
eval "$(ai-models-comparison judge "$MODEL" --json | jq -r '.models[0].pick.get')"
```

Exit status 3 means "judged, and it does not meet the requirement"; 1 means "could not be judged" (not found,
no network). The download command comes from the answer, so the build that was judged is the build fetched.

## 6. "Ask from the laptop, about the server"

The coding agent runs on a laptop; the models run on the server. Registered once:

```bash
claude mcp add yserver-models -- ssh yserver ai-models-comparison mcp
```

and then, in a session on the laptop: *"is there anything new worth trying on yserver?"* The agent calls
`analyze`, which runs on the server, with the server's memory and the server's roster, and gets the short
text of use case 1 instead of reading model pages.

## 7. "One page for the household"

`ai-models-comparison serve` runs as a service on the server. Anyone on a device that can reach it opens the
page, sees the server as the detected machine, and can change the numbers to their own laptop: the list is
judged again in the browser. A link such as
`.../#gpu=24&ram=64&c=nvidia/Qwen3.8-27B-NVFP4,tencent/ContextPilot-14B` carries a machine and a comparison
to someone else.

## What the examples have in common

- **The machine is described once**, then every question is short.
- **The roster turns a list into advice**: "larger than what you run", "newer than what you run".
- **Nothing here ran a model.** Each answer ends where a test begins, and says so.
