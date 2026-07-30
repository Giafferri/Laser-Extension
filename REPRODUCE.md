# Running the FEVER Study

This guide covers the complete paper-aligned evaluation and fixed-intervention prompt-transfer study. It does not assume a particular GPU model.

## Requirements

- Python 3.11 or 3.12; use Python 3.11 on native Windows.
- A CUDA-enabled NVIDIA GPU with at least 20 GiB of memory for the full runner.
- At least 30 GiB free on the output volume, plus space for the GPT-J cache.

Create an environment from the repository root:

```bash
python -m venv .venv
```

Activate it with `source .venv/bin/activate` on macOS/Linux or `.\.venv\Scripts\Activate.ps1` in PowerShell. Install a CUDA-compatible PyTorch build from the [official selector](https://pytorch.org/get-started/locally/), then install the remaining dependencies:

```bash
python -m pip install --upgrade pip wheel "setuptools<82"
python -m pip install -r requirements.txt
```

The model and FEVER dataset are downloaded from Hugging Face on first use.

## Checks

Run the environment check before inference:

```text
python laser-main/src/check_fever_environment.py --model-path EleutherAI/gpt-j-6B --device cuda --output preflight.json
```

The check validates CUDA, FEVER splits, label tokenization, the prompt catalog, and seeded SVD behavior. A short smoke test is also run automatically by the full pipeline; failure stops the study before the validation grid.

## Full Run

Use a persistent terminal session and choose a stable output directory:

```text
python laser-main/src/run_fever_study.py --output-root results --device cuda --model-path EleutherAI/gpt-j-6B --revision float16 --seeds 0 1 2 3 4 --bootstrap-samples 10000
```

The workflow runs the canonical-prompt validation grid, evaluates the published intervention, selects one intervention on validation data, and freezes it across 40 prompt variants and five SVD seeds. It then generates reports, plots, and manifests.

To inspect the protocol without inference, append `--plan-only`. If execution is interrupted, rerun the exact same command with the same output directory. Valid completed runs are reused; missing or inconsistent outputs are recomputed. Do not use `--overwrite` when resuming.

A study is complete only when `<output-root>/RUN_COMPLETE.json` exists with `"status": "complete"`.
