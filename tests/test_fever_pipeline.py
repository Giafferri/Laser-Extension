"""Test the FEVER reproduction and prompt-transfer pipeline."""

import csv
import json
import os
import math
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "laser-main" / "src"
ANALYSIS = ROOT / "analysis"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ANALYSIS))

from dataset_utils.fever_study import FEVER
from fever_experiment_utils import (
    FEVER_EXPECTED_PAPER_DEV_FINGERPRINT,
    FEVER_EXPECTED_PAPER_TEST_FINGERPRINT,
    FEVER_LASER_DEV_SIZE,
    FEVER_LASER_TEST_SIZE,
    atomic_pickle_dump,
    compute_binary_label_scores,
    configure_utf8_stdio,
    expected_fever_split_size,
    rate_to_rho,
    resolve_rate_and_rho,
    rho_to_rate,
    torch_rng_seed_scope,
)
from fever_prompting import build_fever_prompt, list_fever_prompt_template_ids, validate_fever_prompt_specs
from fever_report import classify_row_split
from fever_reproduction_report import (
    EXPECTED_GRID_LNAMES,
    EXPECTED_GRID_LNUMS,
    EXPECTED_GRID_RHOS,
    validate_grid,
)
from fever_transfer_report import claim_bootstrap_ci, validate_suite_configuration
from laser.matrix_utils import do_low_rank
import run_fever_study as study_runner
from run_fever_study import validate_artifacts, validate_smoke_gate
from run_fever_paraphrase_suite import prediction_path, summary_path, validate_existing_output


class FeverPipelineTests(unittest.TestCase):
    def test_rate_rho_mapping(self):
        self.assertEqual(rate_to_rho(9.9), 0.01)
        self.assertEqual(rho_to_rate(0.01), 9.9)
        self.assertEqual(resolve_rate_and_rho(rho=0.10), (9.0, 0.10))
        with self.assertRaises(ValueError):
            resolve_rate_and_rho(rate=9.9, rho=0.01)

    def test_paper_aligned_sizes(self):
        self.assertEqual(FEVER_LASER_DEV_SIZE, 2617)
        self.assertEqual(FEVER_LASER_TEST_SIZE, 10469)
        self.assertEqual(expected_fever_split_size("laser_dev"), 2617)
        self.assertEqual(expected_fever_split_size("laser_test"), 10469)

    def test_split_tagging_is_non_destructive(self):
        source = [{"global_ix": 4, "paper_split": "paper_dev"}]
        tagged = FEVER._with_evaluation_split(source, "laser_test")
        self.assertNotIn("evaluation_split", source[0])
        self.assertEqual(tagged[0]["evaluation_split"], "laser_test")
        self.assertEqual(tagged[0]["evaluation_split_local_ix"], 0)

    def test_prompt_catalog_and_canonical_prompt(self):
        self.assertEqual(len(list_fever_prompt_template_ids()), 41)
        self.assertEqual(validate_fever_prompt_specs(), {})
        self.assertEqual(
            build_fever_prompt("A claim", "original"),
            "Consider the following claim: A claim. Is this claim true or false. The claim is",
        )
        self.assertEqual(
            build_fever_prompt("A claim?", "original"),
            "Consider the following claim: A claim? Is this claim true or false. The claim is",
        )
        self.assertEqual(
            build_fever_prompt("A claim!", "original"),
            "Consider the following claim: A claim!. Is this claim true or false. The claim is",
        )
        self.assertEqual(
            build_fever_prompt('A "quoted claim."', "original"),
            'Consider the following claim: A "quoted claim.". Is this claim true or false. The claim is',
        )

    def test_binary_scoring_uses_only_label_pair(self):
        logprobs = torch.tensor([-100.0, -2.0, -1.0, 50.0])
        scores = compute_binary_label_scores(logprobs, gold_answer_ix=1, true_token_id=2, false_token_id=1)
        self.assertEqual(scores["binary_pred_label"], "true")
        self.assertTrue(scores["binary_correct"])
        expected = -torch.log_softmax(torch.tensor([-2.0, -1.0]), dim=0)[1].item()
        self.assertAlmostEqual(scores["binary_nll_normalized"], expected, places=6)

        tied = compute_binary_label_scores(
            torch.tensor([-100.0, -1.0, -1.0]),
            gold_answer_ix=0,
            true_token_id=2,
            false_token_id=1,
        )
        self.assertEqual(tied["binary_pred_label"], "false")
        self.assertEqual(tied["binary_pred_label_ix"], 0)
        self.assertTrue(tied["binary_tie"])
        self.assertTrue(tied["binary_correct"])

    def test_utf8_stdio_configuration_is_inherited_and_safe(self):
        stdout = mock.Mock()
        stderr = mock.Mock()
        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch.object(sys, "stdout", stdout), mock.patch.object(sys, "stderr", stderr):
                configure_utf8_stdio()
            self.assertEqual(os.environ["PYTHONUTF8"], "1")
            self.assertEqual(os.environ["PYTHONIOENCODING"], "utf-8")
        stdout.reconfigure.assert_called_once_with(encoding="utf-8", errors="backslashreplace")
        stderr.reconfigure.assert_called_once_with(encoding="utf-8", errors="backslashreplace")

    def test_seeded_low_rank_is_reproducible(self):
        matrix = torch.randn(48, 32, generator=torch.Generator().manual_seed(7))
        with torch_rng_seed_scope(0):
            first = do_low_rank(matrix, 0.25).detach()
        with torch_rng_seed_scope(0):
            second = do_low_rank(matrix, 0.25).detach()
        with torch_rng_seed_scope(1):
            other = do_low_rank(matrix, 0.25).detach()
        self.assertTrue(torch.equal(first, second))
        self.assertFalse(torch.equal(first, other))

    def test_strict_reproduction_grid_requires_exact_candidates(self):
        rows = []
        for lname in EXPECTED_GRID_LNAMES:
            for lnum in EXPECTED_GRID_LNUMS:
                for rho in EXPECTED_GRID_RHOS:
                    rows.append(
                        {
                            "lname": lname,
                            "lnum": lnum,
                            "rho": rho,
                            "seed": 0,
                            "intervention_source": "validation_candidate",
                            "claim_count": FEVER_LASER_DEV_SIZE,
                            "binary_accuracy": 50.0,
                            "binary_nll": 0.7,
                        }
                    )
        validate_grid(rows, expected_runs=72, expected_claim_count=FEVER_LASER_DEV_SIZE)
        rows[-1] = dict(rows[-1], intervention_source="validation_selected")
        with self.assertRaisesRegex(ValueError, "validation_candidate"):
            validate_grid(rows, expected_runs=72, expected_claim_count=FEVER_LASER_DEV_SIZE)

    def test_strict_transfer_configuration_requires_fixed_suite(self):
        template_ids = list_fever_prompt_template_ids()
        shared = {
            "fever_split": "laser_test",
            "rate": 9.9,
            "rho": 0.01,
            "true_token_id": 2081,
            "false_token_id": 3991,
            "true_text": " true",
            "false_text": " false",
            "model_path": "EleutherAI/gpt-j-6B",
            "revision": "float16",
            "model_commit_hash": None,
            "tokenizer_commit_hash": None,
            "runtime_device": "cuda",
            "dataset_name": "EleutherAI/fever",
            "dataset_config": "v1.0",
            "dataset_paper_dev_fingerprint": FEVER_EXPECTED_PAPER_DEV_FINGERPRINT,
            "dataset_paper_test_fingerprint": FEVER_EXPECTED_PAPER_TEST_FINGERPRINT,
            "accuracy": 50.0,
            "binary_nll": 0.7,
            "raw_logprob": -1.0,
            "report": {"test": {"n": FEVER_LASER_TEST_SIZE}},
            "split_name": "test",
        }
        baseline = {
            template_id: dict(shared, lname="dont", lnum=26, intervention_source="unspecified")
            for template_id in template_ids
        }
        laser = {
            seed: {
                template_id: dict(
                    shared,
                    lname="fc_in",
                    lnum=24,
                    rho=0.10,
                    rate=9.0,
                    intervention_source="validation_selected",
                )
                for template_id in template_ids
            }
            for seed in [0, 1]
        }
        configuration = validate_suite_configuration(
            baseline,
            laser,
            expected_seeds=[0, 1],
            require_complete=True,
            expected_claim_count=FEVER_LASER_TEST_SIZE,
        )
        self.assertEqual(configuration["template_count"], 41)
        self.assertEqual(configuration["seeds"], [0, 1])
        del laser[1][template_ids[-1]]
        with self.assertRaisesRegex(ValueError, "Incomplete LASER prompt suite"):
            validate_suite_configuration(
                baseline,
                laser,
                expected_seeds=[0, 1],
                require_complete=True,
                expected_claim_count=FEVER_LASER_TEST_SIZE,
            )

    def test_claim_bootstrap_is_seeded_and_claim_sensitive(self):
        values = torch.tensor([0.5, -0.25, 0.0, 0.75], dtype=torch.float64).numpy()
        self.assertEqual(
            claim_bootstrap_ci(values, samples=250, seed=3),
            claim_bootstrap_ci(values, samples=250, seed=3),
        )
        shifted = values + 0.25
        self.assertNotEqual(
            claim_bootstrap_ci(values, samples=250, seed=3),
            claim_bootstrap_ci(shifted, samples=250, seed=3),
        )

    def test_artifact_gate_requires_complete_result_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reproduction = root / "analysis" / "reproduction"
            transfer = root / "analysis" / "transfer"
            plots = transfer / "plots"
            manifests = root / "manifests"
            for directory in [reproduction, transfer, plots, manifests]:
                directory.mkdir(parents=True, exist_ok=True)

            def write_csv(path, count):
                with path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=["value"])
                    writer.writeheader()
                    for index in range(count):
                        writer.writerow({"value": index})

            write_csv(reproduction / "reproduction_summary.csv", 3)
            write_csv(reproduction / "validation_grid.csv", 72)
            write_csv(transfer / "template_metrics.csv", 41)
            write_csv(transfer / "template_seed_metrics.csv", 82)
            write_csv(transfer / "category_metrics.csv", 5)
            write_csv(transfer / "seed_metrics.csv", 2)
            write_csv(transfer / "claim_metrics.csv", 3)
            write_csv(transfer / "claim_seed_metrics.csv", 6)

            with (reproduction / "reproduction_summary.json").open("w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "summary_rows": [{}, {}, {}],
                        "validation_grid": [{} for _ in range(72)],
                        "metadata": {"validation_winner": {"lname": "fc_in"}},
                    },
                    handle,
                )
            with (transfer / "summary.json").open("w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "metadata": {
                            "validated_configuration": {
                                "seeds": [0, 1],
                                "template_count": 41,
                                "claim_count": 3,
                            }
                        },
                        "global_summary": {"prompt_count": 40},
                        "category_summary": [{} for _ in range(5)],
                        "seed_summary": [{}, {}],
                    },
                    handle,
                )
            (manifests / "environment.json").write_text("{}", encoding="utf-8")

            plot_names = [
                "01_category_mean_accuracy",
                "02_category_mean_gain",
                "03_prompt_baseline_vs_laser_scatter",
                "04_prompt_gain_ranked",
                "05_category_gain_distribution",
                "06_divergence_vs_gain",
                "07_seed_sensitivity",
                "08_prompt_sign_stability",
                "09_binary_nll_scatter",
                "10_claim_gain_distribution",
            ]
            for basename in plot_names:
                for extension in ["png", "pdf"]:
                    (plots / f"{basename}.{extension}").write_bytes(b"plot")

            with mock.patch.object(study_runner, "FEVER_LASER_TEST_SIZE", 3):
                result = validate_artifacts(
                    output_root=root,
                    reproduction_analysis=reproduction,
                    transfer_analysis=transfer,
                    plots_root=plots,
                    manifests_root=manifests,
                    seeds=[0, 1],
                    smoke_required=False,
                )
                self.assertEqual(result["status"], "pass")
                self.assertEqual(result["plot_count"], 20)
                (plots / "10_claim_gain_distribution.pdf").unlink()
                with self.assertRaisesRegex(RuntimeError, "missing or empty plot"):
                    validate_artifacts(
                        output_root=root,
                        reproduction_analysis=reproduction,
                        transfer_analysis=transfer,
                        plots_root=plots,
                        manifests_root=manifests,
                        seeds=[0, 1],
                        smoke_required=False,
                    )

    def test_report_prefers_evaluation_split(self):
        row = {"paper_split": "paper_dev", "evaluation_split": "laser_test", "dataset_global_ix": 0}
        self.assertEqual(classify_row_split(row, 0, 0, 2617, 10469, {}), "test")

    def test_atomic_pickle_and_existing_output_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            prediction_path = os.path.join(tmp, "GPTJ-predictions-9-22000-24-splitlaser_test-s0-n2.p")
            predictions = [
                {
                    "dataset_global_ix": index,
                    "question": f"Claim {index}.",
                    "prompted-question": build_fever_prompt(f"Claim {index}.", "original"),
                    "gold-answer-ix": 1,
                    "binary_correct": True,
                    "binary_pred_label": "true",
                    "binary_pred_label_ix": 1,
                    "binary_tie": False,
                    "binary_true_logprob": math.log(0.7),
                    "binary_false_logprob": math.log(0.3),
                    "binary_nll_normalized": -math.log(0.7),
                    "binary_margin": math.log(0.7) - math.log(0.3),
                    "evaluation_split": "laser_test",
                }
                for index in range(2)
            ]
            summary = {
                "dataset_size": 2,
                "prompt/template_id": "original",
                "args/fever_split": "laser_test",
                "args/lname": "fc_in",
                "args/lnum": 24,
                "args/rate": 9.0,
                "args/rho": 0.1,
                "args/seed": 0,
                "metric/binary_accuracy": 100.0,
                "metric/binary_nll_normalized": -math.log(0.7),
            }
            atomic_pickle_dump(predictions, prediction_path)
            atomic_pickle_dump(summary, summary_path(prediction_path))
            valid, reason = validate_existing_output(
                prediction_path,
                {
                    "prediction_count": 2,
                    "template": "original",
                    "fever_split": "laser_test",
                    "lname": "fc_in",
                    "lnum": 24,
                    "rate": 9.0,
                    "seed": 0,
                },
            )
            self.assertTrue(valid, reason)

    def test_smoke_gate_accepts_valid_pair_and_rejects_non_finite_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            specs = [
                ("dont", 26, 9.9, None, False),
                ("fc_in", 24, 9.0, 0, True),
            ]
            output_paths = []
            for lname, lnum, rate, seed, seeded_path in specs:
                output_path = prediction_path(
                    home_dir=tmp,
                    template_id="original",
                    lnum=lnum,
                    lname=lname,
                    rate=rate,
                    dtpts=22000,
                    fever_split="laser_test",
                    start_index=0,
                    max_examples=5,
                    seed=seed,
                    seeded_path=seeded_path,
                )
                predictions = [
                    {
                        "dataset_global_ix": 2617 + index,
                        "question": f"Claim {index}.",
                        "prompted-question": build_fever_prompt(f"Claim {index}.", "original"),
                        "binary_correct": index % 2 == 0,
                        "binary_pred_label": "true",
                        "binary_pred_label_ix": 1,
                        "binary_tie": False,
                        "gold-answer-ix": 1 if index % 2 == 0 else 0,
                        "binary_true_logprob": math.log(0.7),
                        "binary_false_logprob": math.log(0.3),
                        "binary_margin": math.log(0.7) - math.log(0.3),
                        "binary_nll_normalized": (
                            -math.log(0.7) if index % 2 == 0 else -math.log(0.3)
                        ),
                        "evaluation_split": "laser_test",
                    }
                    for index in range(5)
                ]
                summary = {
                    "dataset_size": 5,
                    "prompt/template_id": "original",
                    "args/fever_split": "laser_test",
                    "args/lname": lname,
                    "args/lnum": lnum,
                    "args/rate": rate,
                    "args/rho": 1.0 - 0.1 * rate,
                    "args/seed": seed,
                    "args/model_path": "EleutherAI/gpt-j-6B",
                    "args/revision": "float16",
                    "args/intervention_source": "unspecified" if lname == "dont" else "validation_selected",
                    "args/dtpts": 22000,
                    "args/intervention": "rank-reduction",
                    "metric/binary_accuracy": 60.0,
                    "metric/binary_nll_normalized": (
                        3 * -math.log(0.7) + 2 * -math.log(0.3)
                    ) / 5,
                    "runtime/device": "cuda",
                    "runtime/cuda_device_name": "NVIDIA Test GPU",
                    "dataset/name": "EleutherAI/fever",
                    "dataset/config": "v1.0",
                    "dataset/paper_dev_fingerprint": FEVER_EXPECTED_PAPER_DEV_FINGERPRINT,
                    "dataset/paper_test_fingerprint": FEVER_EXPECTED_PAPER_TEST_FINGERPRINT,
                    "labels/true_text": " true",
                    "labels/false_text": " false",
                    "labels/true_token_id": 2081,
                    "labels/false_token_id": 3991,
                }
                atomic_pickle_dump(predictions, output_path)
                atomic_pickle_dump(summary, summary_path(output_path))
                output_paths.append(output_path)

            result = validate_smoke_gate(tmp)
            self.assertEqual(result["status"], "pass")
            self.assertEqual(result["examples"], 5)

            laser_summary_path = summary_path(output_paths[1])
            with open(laser_summary_path, "rb") as handle:
                valid_summary = pickle.load(handle)
            invalid_summary = dict(valid_summary)
            invalid_summary["dataset/paper_test_fingerprint"] = "wrong-fingerprint"
            atomic_pickle_dump(invalid_summary, laser_summary_path)
            with self.assertRaisesRegex(RuntimeError, "provenance mismatch"):
                validate_smoke_gate(tmp)
            atomic_pickle_dump(valid_summary, laser_summary_path)

            with open(output_paths[1], "rb") as handle:
                invalid_predictions = pickle.load(handle)
            invalid_predictions[-1]["binary_nll_normalized"] = float("nan")
            atomic_pickle_dump(invalid_predictions, output_paths[1])
            with self.assertRaisesRegex(RuntimeError, "non-finite"):
                validate_smoke_gate(tmp)


if __name__ == "__main__":
    unittest.main()
