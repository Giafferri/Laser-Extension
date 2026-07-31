# LASER on FEVER: Fixed-Intervention Prompt Transfer

This repository extends the public [LASER](https://github.com/pratyushasharma/laser) codebase with a GPT-J/FEVER prompt-transfer study. The original LASER source in `laser-main/` is unchanged; the extension is implemented in separate files.

## Experiment

The study first performs a paper-aligned, but not exact, FEVER replication. A LASER intervention configuration is then selected on the canonical validation prompt and evaluated without retuning across 40 prompt-template variants and five SVD seeds.

Predictions use only the GPT-J tokens `" true"` and `" false"`. The primary analysis reports binary accuracy, binary NLL, seed sensitivity, prompt-level gains, and a claim-level paired bootstrap interval over the fixed prompt suite.

Mean accuracy improves by **1.28 points**, but 14 of 40 prompt-level gains are negative. The result supports positive average transfer, not uniform robustness.

## Added files

| Path | Purpose |
| --- | --- |
| `laser-main/src/dataset_utils/fever_study.py` | FEVER filtering, provenance, and study splits. |
| `laser-main/src/fever_experiment_utils.py` | Shared seed, metric, token, split, and `rate`/`rho` utilities. |
| `laser-main/src/fever_prompting.py` | Canonical prompt and 40 categorized prompt-template variants. |
| `laser-main/src/intervention_gptj_fever_study.py` | Canonical GPT-J/FEVER evaluator and binary scoring. |
| `laser-main/src/intervention_gptj_fever_paraphrase.py` | Single-template transfer evaluator. |
| `laser-main/src/run_fever_reproduction_grid.py` | Canonical-prompt intervention search. |
| `laser-main/src/run_fever_paraphrase_suite.py` | Resumable multi-template and multi-seed runner. |
| `laser-main/src/run_fever_study.py` | End-to-end preflight, runs, analysis, and artifact validation. |
| `laser-main/src/check_fever_environment.py` | Environment, dataset, tokenization, prompt, and SVD checks. |
| `analysis/` | Reproduction and transfer reports, exports, and plots. |
| `tests/test_fever_pipeline.py` | Pipeline, metric, split, prompt, seed, and output tests. |

Setup, execution, and resumption instructions are in [`REPRODUCE.md`](REPRODUCE.md).
