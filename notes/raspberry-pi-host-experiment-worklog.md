# Raspberry Pi 3 Host Experiment Work Record

## Confirmed

- The local branch is `main` at commit `97018d5`.
- The local `origin` fetch and push URLs are `https://github.com/mystlive/esp32-ai.git`.
- The validated benchmark snapshot must not be synchronized with later upstream changes.
- Five host experiment C files are present under `runtime/host_generate/`.
- `tinystories_generate.c` is the int8 staging and four-thread OpenMP experiment.
- `tinystories_generate_i8_single.c` is the single-core int8 staging experiment.
- `tinystories_generate_neon.c` uses a 16-byte NEON dot-product loop.
- `tinystories_generate_neon2.c` uses a 32-byte NEON loop and two accumulators.
- `tinystories_generate_profile.c` enables `LLM_PROFILE` around the 16-byte NEON implementation.
- `artifacts/` and `firmware/**/generated/` are excluded by the existing `.gitignore`.
- The original MIT license and author copyright remain unchanged.
- `README_ja.md` translates the complete English README and links back to `README.md`.
- `README.md` links to the Japanese version at its beginning.

## Rejected

- Synchronizing, fetching, rebasing, or otherwise updating from upstream.
- Treating `tinystories_generate.c` as the Portable int4 baseline.
- Expressing benchmark throughput as a hardware performance ratio.
- Adding ignored model artifacts, generated vocabulary headers, or `.venv` content.

## Unknown

- The Portable int4 baseline source used for the reported measurement is not present in `runtime/host_generate/`.
- Native compilation and benchmark execution cannot be verified on this Windows x64 host because the final source requires ARM NEON and the Raspberry Pi model environment.

## Current state

- README update drafted and checked with `git diff --check`.
- Japanese and English READMEs have matching heading, table-separator, and code-fence counts.
- The documented final build command matches the source's relative includes, one model-path argument, and `libm` dependency.
- No local ARM cross-compiler is available, so compilation was not executed on this Windows x64 host.
- The user approved the reviewed files for commit on 2026-08-10.
- Push has not been approved and must not be performed.
