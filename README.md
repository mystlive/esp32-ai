# Running a 28.9M parameter LLM on a microcontroller

[日本語](README_ja.md)

<p align="center">
  <a href="https://x.com/slvDev">𝕏 slvDev</a> &nbsp;·&nbsp;
  <a href="https://www.linkedin.com/in/slvdev/">LinkedIn</a>
</p>

![28.9M-parameter LLM running on an ESP32-S3](media/esp32-ple-demo.gif)

This is a 28.9 million parameter language model that generates text on an ESP32-S3
microcontroller. It runs on the chip itself, with nothing sent to a server, and it
displays generated text at 9.88 tokens per second on a small screen wired to the
chip. It fits because most of the model lives in flash instead of RAM, using
Per-Layer Embeddings, an idea from Google's Gemma 3n.

This fork preserves the original ESP32-S3 project and adds host-side execution
and optimization experiments performed on a Raspberry Pi 3 Model B. The
Raspberry Pi work is an addition by the fork and does not reattribute the
original project's results.

## Raspberry Pi 3 Host Experiment

This experiment runs the ESP32-S3 TinyStories model with the portable C runtime
on a 64-bit ARM Linux host, then measures several changes to the matrix-vector
work. It is based on upstream commit
[`97018d5`](https://github.com/slvDev/esp32-ai/commit/97018d5). Later upstream
changes are intentionally not included so that the validated benchmark
conditions remain fixed.

### Test environment

| Component | Value |
| --------- | ----- |
| Board | Raspberry Pi 3 Model B |
| SoC | Broadcom BCM2837 |
| CPU | ARM Cortex-A53, four cores at 1.2 GHz |
| Memory | 1 GB RAM |
| OS | Debian GNU/Linux 12 (bookworm) |
| Architecture | aarch64 |
| Compiler | GCC 12.2.0 |

### Results

| Configuration | Throughput | Forward |
| ------------- | ---------- | ------- |
| Portable int4 | 31.22 tok/s | 31.89 ms/token |
| int8 staging | 171.81 tok/s | 5.68 ms/token |
| int8 + OpenMP 4-core | 146.32 tok/s | 6.54 ms/token |
| int8 + NEON | 177.98 tok/s | 5.47 ms/token |
| int8 + NEON 32B / 2 accumulators | 195.81 tok/s | 4.96 ms/token |

The portable int4 version established the 31.22 tok/s host baseline. Expanding
the 4-bit weights into an int8 staging representation before inference produced
the largest improvement, reaching 171.81 tok/s. Using four OpenMP threads
reduced throughput to 146.32 tok/s; synchronization cost, cache behavior, and
memory bandwidth are possible contributors, but this experiment does not
isolate a single cause.

ARM NEON SIMD raised throughput to 177.98 tok/s. Processing 32 bytes per loop
with two independent accumulators raised the final result to 195.81 tok/s,
about 6.27 times the initial host baseline. After a reboot and recompilation,
the final version measured 195.55 tok/s and 4.97 ms/token, confirming nearly
the same performance.

The profiled time distribution was approximately 59.1% in the output head,
27.9% in attention, 5.7% in PLE, 4.8% in the FFN, and 2.4% in input processing.
The output head and attention together accounted for about 87% of the measured
time.

These numbers compare token generation for this model and these inference
implementations. They are not CPU performance ratios between the Raspberry Pi
3 and the ESP32-S3. Under this TinyStories inference setup, the final Raspberry
Pi version has substantially higher throughput than the published ESP32-S3
result.

### Experiment sources

| File | Role |
| ---- | ---- |
| `runtime/host_generate/tinystories_generate.c` | int8 staging with a four-thread OpenMP output head |
| `runtime/host_generate/tinystories_generate_i8_single.c` | single-core int8 staging baseline |
| `runtime/host_generate/tinystories_generate_neon.c` | int8 staging with a 16-byte ARM NEON dot product |
| `runtime/host_generate/tinystories_generate_neon2.c` | final 32-byte ARM NEON dot product with two accumulators |
| `runtime/host_generate/tinystories_generate_profile.c` | 16-byte NEON version with `LLM_PROFILE` timing |

There is no separate Portable int4 source in this directory. The file named
`tinystories_generate.c` is the OpenMP int8 experiment, not the Portable int4
baseline.

### Build and run the final version

The model and generated vocabulary header are intentionally excluded from Git.
From the repository root, fetch the pinned TinyStories artifacts and generate
the vocabulary header before compiling:

```bash
scripts/fetch_model.sh tinystories

uv run python firmware/esp32_tinystories/tools/generate_vocab.py \
  --tokenizer artifacts/tinystories/tokenizer.json \
  --out firmware/esp32_tinystories/generated/vocab.h

gcc -std=c11 -O3 -Wall -Wextra -march=armv8-a+simd \
  -o tinystories_generate_neon2 \
  runtime/host_generate/tinystories_generate_neon2.c -lm

./tinystories_generate_neon2 artifacts/tinystories/model.bin
```

The benchmark initially placed the executable in `/tmp`. A reboot removed that
temporary executable while leaving the source, model, and virtual environment
intact. Compile into a normal working directory for persistent use.

### Model scope

TinyStories generates short English stories. It is not a general question
answering model, an instruction model, a ChatGPT replacement, a programming
model, or a source of general-knowledge answers. This experiment evaluates how
far a small LLM and lightweight C runtime can be optimized for an ARM CPU; it
does not evaluate general model intelligence.

Original project: [slvDev/esp32-ai](https://github.com/slvDev/esp32-ai)

Fork and Raspberry Pi 3 experiment:
[mystlive/esp32-ai](https://github.com/mystlive/esp32-ai)

## The numbers

|              |                                                    |
| ------------ | -------------------------------------------------- |
| Parameters   | 28.9M stored (25M of them in a flash lookup table) |
| Chip         | ESP32-S3, 512KB SRAM, 8MB PSRAM and 16MB flash     |
| Speed        | 9.88 tok/s end to end, 94.9 ms/token of compute    |
| Connectivity | none, everything runs on the device                |
| Model size   | 14.9MB at 4-bit                                    |

## Why it is hard, and how it fits anyway

A microcontroller has very little fast memory. The ESP32-S3 gives you 512KB of
SRAM, and only the values touched many times per token can live there:
activations and norm weights. The dense core and output head, scanned once per
position, sit in PSRAM. What is left is the embedding tables, and their size is
what normally decides how big a model can be.

In this model, most parameters sit in an embedding table, which the model reads
from rather than computes on. So that 25-million-parameter table stays in slow
flash, and only the few rows each token needs are pulled from it, about 450
bytes. Most of the model is therefore never loaded to run it: it sits in flash
and is sampled a little at a time.

That idea is Google's Per-Layer Embeddings, from
[Gemma 3n](https://ai.google.dev/gemma/docs/gemma-3n). Here it runs
on the memory layout of a microcontroller instead of a phone or a GPU.

Each tier holds whatever is read at its own frequency:

```
  SRAM  (fast, tiny)   activations and norm weights, touched many times a token
  PSRAM (medium)       the core and output head, read once per position
  FLASH (huge, slow)   the 25M-param table, about 6 rows read per token (~450 B)
```

## What it does, and what it does not

The model was trained on TinyStories, so it writes short, simple stories and mostly
keeps them coherent. It will not answer questions, follow instructions, write code,
or know facts. That limit comes from the small part of the model that does the
reasoning, and the memory trick does not change it. What is interesting here is the
architecture, fitting a large model onto a tiny chip, rather than what a 28.9 million
parameter model can say.

## Models

- [Barista](https://huggingface.co/slvDev/esp32-ai-barista) - espresso question answering
- [TinyStories](https://huggingface.co/slvDev/esp32-ai-tinystories) - story generation

## Running it yourself

Download and deployment are separate operations: one reaches the network, the
other touches the board.

```bash
scripts/fetch_model.sh barista   # download, verify, install into artifacts/
scripts/deploy.sh barista        # generate headers, run gates, compile, flash
```

`tinystories` is the other model, and takes the same two commands. Both require
the model to be named, because the board holds one at a time and deploying
replaces it.

`fetch_model.sh` checks the inference assets against a SHA-256 and byte size
pinned in the script, and cross-checks the release's own `metadata.json` against
those same pins. It installs nothing unless every check passes, so a failed
download leaves what you already have untouched. `deploy.sh` downloads no model:
it works from whatever is already in `artifacts/<model>/`. It does run two of its
header tools through `uv`, which fetches one pinned wheel the first time on a
machine that has never cached it.

The firmware details and the boot output to expect live in
[`firmware/esp32_barista/README.md`](firmware/esp32_barista/README.md) and
[`firmware/esp32_tinystories/README.md`](firmware/esp32_tinystories/README.md). The reusable
architecture is in `src/`; the training, ablation and quantization code that
reproduces the published numbers is in `research/tinystories/`. The full method,
the ablations, and the on-chip measurements are written up in
[`RESULTS.md`](RESULTS.md).

## Credit

TinyStories is the dataset this trains on: short synthetic stories simple enough
that a small model can still learn to write coherently (Ronen Eldan and Yuanzhi Li,
Microsoft Research, [arXiv:2305.07759](https://arxiv.org/abs/2305.07759)). The other
half is Per-Layer Embeddings, Google's design from Gemma 3n, which is what
lets a big model fit on a small chip.

Andrej Karpathy's [llama2.c](https://github.com/karpathy/llama2.c) is the
reference for training a small language model and running it in plain C.

## Measurements

Detailed measurements and ablations are documented in `RESULTS.md`.
