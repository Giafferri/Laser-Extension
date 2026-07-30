"""Run resumable baseline and LASER evaluations across FEVER prompt templates."""

import argparse
import math
import os
import pickle
import shlex
import subprocess
import sys

from fever_experiment_utils import (
    FEVER_EXPECTED_PAPER_DEV_FINGERPRINT,
    FEVER_EXPECTED_PAPER_TEST_FINGERPRINT,
    GPTJ_FEVER_FALSE_TOKEN_ID,
    GPTJ_FEVER_TRUE_TOKEN_ID,
    FEVER_SPLIT_CHOICES,
    INTERVENTION_SOURCE_CHOICES,
    expected_fever_split_size,
    format_float_tag,
    rate_to_rho,
    resolve_rate_and_rho,
)
from fever_prompting import build_fever_prompt, list_fever_prompt_categories, list_fever_prompt_template_ids


DEFAULT_BASELINE = {"lname": "dont", "lnum": 26, "rate": 9.9}
DEFAULT_LASER = {"lname": "fc_in", "lnum": 24, "rate": 9.5}


def run_slice_suffix(start_index, max_examples):
    if start_index == 0 and max_examples is None:
        return ""
    max_examples_part = "all" if max_examples is None else str(max_examples)
    return f"-s{start_index}-n{max_examples_part}"


def split_and_slice_suffix(fever_split, start_index, max_examples):
    split_suffix = "" if fever_split == "combined" else f"-split{fever_split}"
    return f"{split_suffix}{run_slice_suffix(start_index, max_examples)}"


def category_ids():
    return [spec["category_id"] for spec in list_fever_prompt_categories()]


def ordered_template_ids(categories=None, templates=None):
    available_templates = set(list_fever_prompt_template_ids())

    if templates:
        deduped = []
        seen = set()
        for template_id in templates:
            if template_id in seen:
                continue
            seen.add(template_id)
            deduped.append(template_id)

        unknown = [template_id for template_id in deduped if template_id not in available_templates]
        if unknown:
            raise KeyError(f"Unknown template ids: {', '.join(unknown)}")

        if categories:
            allowed = set(list_fever_prompt_template_ids(include_original=False, categories=categories))
            for template_id in deduped:
                if template_id == "original":
                    continue
                if template_id not in allowed:
                    raise ValueError(
                        f"Template {template_id} is not part of the selected categories: {', '.join(categories)}"
                    )
        return deduped

    selected = ["original"]
    selected.extend(list_fever_prompt_template_ids(include_original=False, categories=categories))
    return selected


def prediction_path(home_dir, template_id, lnum, lname, rate, dtpts, fever_split, start_index, max_examples, seed=None, seeded_path=False):
    suffix = split_and_slice_suffix(fever_split=fever_split, start_index=start_index, max_examples=max_examples)
    parts = [home_dir, "GPTJ", "rank-reduction", lname]
    if seeded_path and seed is not None and lname != "dont":
        parts.extend(["seeds", f"seed_{seed}"])
    if template_id != "original":
        parts.extend(["prompts", template_id])
    save_dir = os.path.join(*parts)
    filename = f"GPTJ-predictions-{format_float_tag(rate)}-{dtpts}-{lnum}{suffix}.p"
    return os.path.join(save_dir, filename)


def summary_path(prediction_file):
    directory, filename = os.path.split(prediction_file)
    filename = filename.replace("-predictions-", "-result-summary-")
    return os.path.join(directory, filename[:-2] + ".pkl")


def expected_example_count(fever_split, start_index, max_examples):
    available = max(0, expected_fever_split_size(fever_split) - start_index)
    return available if max_examples is None else min(available, max_examples)


def validate_existing_output(output_path, expected):
    companion = summary_path(output_path)
    if not os.path.exists(companion):
        return False, f"missing summary: {companion}"

    try:
        with open(companion, "rb") as handle:
            summary = pickle.load(handle)
        with open(output_path, "rb") as handle:
            predictions = pickle.load(handle)
    except Exception as exc:
        return False, f"unreadable output: {exc}"

    checks = {
        "prediction_count": (len(predictions), expected["prediction_count"]),
        "summary_dataset_size": (summary.get("dataset_size"), expected["prediction_count"]),
        "template": (summary.get("prompt/template_id"), expected["template"]),
        "fever_split": (summary.get("args/fever_split"), expected["fever_split"]),
        "lname": (summary.get("args/lname"), expected["lname"]),
        "lnum": (summary.get("args/lnum"), expected["lnum"]),
        "seed": (summary.get("args/seed"), expected["seed"]),
    }
    for label, (actual, wanted) in checks.items():
        if actual != wanted:
            return False, f"{label} mismatch: found {actual!r}, expected {wanted!r}"

    actual_rate = summary.get("args/rate")
    if actual_rate is None or abs(float(actual_rate) - float(expected["rate"])) > 1e-9:
        return False, f"rate mismatch: found {actual_rate!r}, expected {expected['rate']!r}"

    actual_rho = summary.get("args/rho", summary.get("intervention/rho"))
    expected_rho = rate_to_rho(expected["rate"])
    if actual_rho is None or abs(float(actual_rho) - expected_rho) > 1e-9:
        return False, f"rho mismatch: found {actual_rho!r}, expected {expected_rho!r}"

    optional_summary_checks = {
        "model_path": "args/model_path",
        "revision": "args/revision",
        "intervention_source": "args/intervention_source",
        "dtpts": "args/dtpts",
        "intervention": "args/intervention",
    }
    for expected_key, summary_key in optional_summary_checks.items():
        if expected_key in expected and summary.get(summary_key) != expected[expected_key]:
            return False, (
                f"{expected_key} mismatch: found {summary.get(summary_key)!r}, "
                f"expected {expected[expected_key]!r}"
            )

    expected_device = expected.get("device")
    if expected_device not in {None, "auto"} and summary.get("runtime/device") != expected_device:
        return False, (
            f"runtime device mismatch: found {summary.get('runtime/device')!r}, "
            f"expected {expected_device!r}"
        )

    if expected.get("require_provenance"):
        provenance_fields = [
            "dataset/name",
            "dataset/config",
            "dataset/paper_dev_fingerprint",
            "dataset/paper_test_fingerprint",
            "labels/true_token_id",
            "labels/false_token_id",
        ]
        missing = [field for field in provenance_fields if summary.get(field) is None]
        if missing:
            return False, f"summary is missing provenance fields: {', '.join(missing)}"
        if summary.get("labels/true_text") != " true" or summary.get("labels/false_text") != " false":
            return False, "summary uses inconsistent binary label texts"
        expected_provenance = {
            "dataset/name": "EleutherAI/fever",
            "dataset/config": "v1.0",
            "dataset/paper_dev_fingerprint": FEVER_EXPECTED_PAPER_DEV_FINGERPRINT,
            "dataset/paper_test_fingerprint": FEVER_EXPECTED_PAPER_TEST_FINGERPRINT,
            "labels/true_token_id": GPTJ_FEVER_TRUE_TOKEN_ID,
            "labels/false_token_id": GPTJ_FEVER_FALSE_TOKEN_ID,
        }
        for field, wanted in expected_provenance.items():
            if summary.get(field) != wanted:
                return False, (
                    f"provenance mismatch for {field}: found {summary.get(field)!r}, "
                    f"expected {wanted!r}"
                )

    required_prediction_fields = {
        "dataset_global_ix",
        "question",
        "prompted-question",
        "gold-answer-ix",
        "binary_correct",
        "binary_pred_label",
        "binary_pred_label_ix",
        "binary_tie",
        "binary_true_logprob",
        "binary_false_logprob",
        "binary_nll_normalized",
        "binary_margin",
        "evaluation_split",
    }
    claim_ids = []
    correct_count = 0
    nll_total = 0.0
    for row_index, row in enumerate(predictions):
        missing = sorted(required_prediction_fields - set(row))
        if missing:
            return False, f"prediction row {row_index} is missing fields: {', '.join(missing)}"

        claim_id = row["dataset_global_ix"]
        if claim_id is None:
            return False, f"prediction row {row_index} has no dataset_global_ix"
        claim_ids.append(claim_id)
        if row["evaluation_split"] != expected["fever_split"]:
            return False, (
                f"prediction row {row_index} split mismatch: found {row['evaluation_split']!r}, "
                f"expected {expected['fever_split']!r}"
            )
        expected_prompt = build_fever_prompt(row["question"], expected["template"])
        if row["prompted-question"] != expected_prompt:
            return False, f"prediction row {row_index} does not match template {expected['template']}"

        if row["gold-answer-ix"] not in {0, 1}:
            return False, f"prediction row {row_index} has invalid gold-answer-ix"
        if row["binary_pred_label"] not in {"true", "false"}:
            return False, f"prediction row {row_index} has invalid binary_pred_label"
        expected_pred_ix = 1 if row["binary_pred_label"] == "true" else 0
        if row["binary_pred_label_ix"] != expected_pred_ix:
            return False, f"prediction row {row_index} has inconsistent binary_pred_label_ix"
        if not isinstance(row["binary_tie"], bool):
            return False, f"prediction row {row_index} has invalid binary_tie"
        expected_correct = expected_pred_ix == row["gold-answer-ix"]
        if not isinstance(row["binary_correct"], bool) or row["binary_correct"] != expected_correct:
            return False, f"prediction row {row_index} has inconsistent binary_correct"

        numeric_fields = [
            "binary_true_logprob",
            "binary_false_logprob",
            "binary_margin",
            "binary_nll_normalized",
        ]
        values = {}
        for field in numeric_fields:
            try:
                values[field] = float(row[field])
            except (TypeError, ValueError):
                return False, f"prediction row {row_index} has non-numeric {field}"
            if not math.isfinite(values[field]):
                return False, f"prediction row {row_index} has non-finite {field}"

        true_logprob = values["binary_true_logprob"]
        false_logprob = values["binary_false_logprob"]
        margin = values["binary_margin"]
        nll = values["binary_nll_normalized"]
        if nll < 0.0:
            return False, f"prediction row {row_index} has negative binary NLL"
        if not math.isclose(margin, true_logprob - false_logprob, rel_tol=0.0, abs_tol=1e-6):
            return False, f"prediction row {row_index} has inconsistent binary margin"
        margin_pred_ix = 1 if margin > 0.0 else 0
        if expected_pred_ix != margin_pred_ix:
            return False, f"prediction row {row_index} binary prediction disagrees with its margin"
        if row["binary_tie"] != (margin == 0.0):
            return False, f"prediction row {row_index} has inconsistent binary_tie"
        gold_logprob = true_logprob if row["gold-answer-ix"] == 1 else false_logprob
        if not math.isclose(nll, -gold_logprob, rel_tol=0.0, abs_tol=1e-6):
            return False, f"prediction row {row_index} has inconsistent binary NLL"
        log_denom = max(true_logprob, false_logprob) + math.log(
            math.exp(true_logprob - max(true_logprob, false_logprob))
            + math.exp(false_logprob - max(true_logprob, false_logprob))
        )
        if not math.isclose(log_denom, 0.0, rel_tol=0.0, abs_tol=1e-5):
            return False, f"prediction row {row_index} binary label probabilities are not normalized"

        correct_count += int(row["binary_correct"])
        nll_total += nll

    if len(set(claim_ids)) != len(claim_ids):
        return False, "prediction rows contain duplicate dataset_global_ix values"

    if predictions:
        computed_accuracy = 100.0 * correct_count / len(predictions)
        computed_nll = nll_total / len(predictions)
        summary_accuracy = summary.get("metric/binary_accuracy")
        summary_nll = summary.get("metric/binary_nll_normalized")
        if summary_accuracy is None or not math.isfinite(float(summary_accuracy)):
            return False, "summary binary accuracy is missing or non-finite"
        if summary_nll is None or not math.isfinite(float(summary_nll)):
            return False, "summary binary NLL is missing or non-finite"
        if not math.isclose(float(summary_accuracy), computed_accuracy, rel_tol=0.0, abs_tol=1e-9):
            return False, "summary binary accuracy does not match prediction rows"
        if not math.isclose(float(summary_nll), computed_nll, rel_tol=0.0, abs_tol=1e-9):
            return False, "summary binary NLL does not match prediction rows"
    return True, "complete"


def build_command(script_path, args, template_id, mode_name, seed):
    if mode_name == "baseline":
        lname = args.baseline_lname
        lnum = args.baseline_lnum
        rate = args.baseline_rate
        rho = args.baseline_rho
        intervention_source = "unspecified"
        save_seed_in_path = False
    else:
        lname = args.laser_lname
        lnum = args.laser_lnum
        rate = args.laser_rate
        rho = args.laser_rho
        intervention_source = args.intervention_source
        save_seed_in_path = True

    command = [
        sys.executable,
        script_path,
        "--model_path",
        args.model_path,
        "--revision",
        args.revision,
        "--home_dir",
        args.home_dir,
        "--device",
        args.device,
        "--lname",
        lname,
        "--lnum",
        str(lnum),
        "--dtpts",
        str(args.dtpts),
        "--batch_size",
        str(args.batch_size),
        "--max_len",
        str(args.max_len),
        "--k",
        str(args.k),
        "--intervention",
        args.intervention,
        "--prompt-template",
        template_id,
        "--fever-split",
        args.fever_split,
        "--start_index",
        str(args.start_index),
        "--intervention-source",
        intervention_source,
    ]

    if rho is not None:
        command.extend(["--rho", str(rho)])
    else:
        command.extend(["--rate", str(rate)])

    if mode_name == "laser":
        command.extend(["--seed", str(seed), "--save-seed-in-path"])
    if args.max_examples is not None:
        command.extend(["--max_examples", str(args.max_examples)])
    return command


def should_run(output_path, overwrite, skip_existing, expected):
    if not os.path.exists(output_path):
        return True, None
    if overwrite:
        return True, None
    valid, reason = validate_existing_output(output_path, expected)
    if not valid:
        return True, f"rerun invalid existing output: {reason}"
    if skip_existing:
        return False, f"skip existing: {output_path}"
    raise FileExistsError(f"Output already exists and overwrite is disabled: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run the FEVER fixed-intervention prompt transfer suite for GPT-J")
    parser.add_argument("--mode", choices=["baseline", "laser", "all"], default="all")
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=category_ids(),
        default=None,
        help="Optional prompt categories to include. Original is prepended unless explicit templates are passed.",
    )
    parser.add_argument(
        "--templates",
        nargs="+",
        default=None,
        help="Optional explicit template ids. If set, these exact templates are used in the provided order.",
    )
    parser.add_argument("--fever-split", choices=FEVER_SPLIT_CHOICES, default="paper_test")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--home-dir", required=True)
    parser.add_argument("--overwrite", action="store_true", help="Re-run and overwrite outputs if they already exist.")
    parser.add_argument(
        "--skip-existing",
        dest="skip_existing",
        action="store_true",
        default=True,
        help="Skip runs whose prediction file already exists.",
    )
    parser.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        help="Fail on existing outputs unless --overwrite is also set.",
    )
    parser.add_argument("--keep-going", action="store_true", help="Continue the suite after an individual run fails.")
    parser.add_argument("--dry-run", action="store_true", help="Print the complete execution plan without launching runs.")
    parser.add_argument("--model_path", default="EleutherAI/gpt-j-6B")
    parser.add_argument("--revision", default="float16")
    parser.add_argument("--dtpts", type=int, default=22000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_len", type=int, default=1)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--intervention", choices=["dropout", "rank-reduction"], default="rank-reduction")
    parser.add_argument("--baseline-lname", default=DEFAULT_BASELINE["lname"])
    parser.add_argument("--baseline-lnum", type=int, default=DEFAULT_BASELINE["lnum"])
    parser.add_argument("--baseline-rate", type=float, default=None)
    parser.add_argument("--baseline-rho", type=float, default=None)
    parser.add_argument("--laser-lname", default=DEFAULT_LASER["lname"])
    parser.add_argument("--laser-lnum", type=int, default=DEFAULT_LASER["lnum"])
    parser.add_argument("--laser-rate", type=float, default=None)
    parser.add_argument("--laser-rho", type=float, default=None)
    parser.add_argument(
        "--intervention-source",
        default="validation_selected",
        choices=INTERVENTION_SOURCE_CHOICES,
    )
    args = parser.parse_args()

    if args.start_index < 0:
        parser.error("--start-index must be non-negative")
    if args.max_examples is not None and args.max_examples <= 0:
        parser.error("--max-examples must be positive")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must not contain duplicates")

    args.baseline_rate, args.baseline_rho = resolve_rate_and_rho(
        rate=args.baseline_rate,
        rho=args.baseline_rho,
        default_rate=DEFAULT_BASELINE["rate"],
    )
    args.laser_rate, args.laser_rho = resolve_rate_and_rho(
        rate=args.laser_rate,
        rho=args.laser_rho,
        default_rate=DEFAULT_LASER["rate"],
    )
    return args


def main():
    args = parse_args()
    template_ids = ordered_template_ids(categories=args.categories, templates=args.templates)
    mode_names = ["baseline", "laser"] if args.mode == "all" else [args.mode]
    script_path = os.path.join(os.path.dirname(__file__), "intervention_gptj_fever_paraphrase.py")

    total_planned = len(template_ids) * (
        (1 if "baseline" in mode_names else 0) + (len(args.seeds) if "laser" in mode_names else 0)
    )
    print(f"SUITE_TEMPLATES: {', '.join(template_ids)}")
    print(f"SUITE_MODES: {', '.join(mode_names)}")
    print(f"SUITE_FEVER_SPLIT: {args.fever_split}")
    print(f"TOTAL_RUNS_PLANNED: {total_planned}")

    completed = 0
    skipped = 0
    failed = 0

    for mode_name in mode_names:
        if mode_name == "baseline":
            seeds = [None]
            lname = args.baseline_lname
            lnum = args.baseline_lnum
            rate = args.baseline_rate
            seeded_path = False
        else:
            seeds = args.seeds
            lname = args.laser_lname
            lnum = args.laser_lnum
            rate = args.laser_rate
            seeded_path = True
        intervention_source = "unspecified" if mode_name == "baseline" else args.intervention_source

        for seed in seeds:
            for template_id in template_ids:
                seed_label = "none" if seed is None else str(seed)
                output_path = prediction_path(
                    home_dir=args.home_dir,
                    template_id=template_id,
                    lnum=lnum,
                    lname=lname,
                    rate=rate,
                    dtpts=args.dtpts,
                    fever_split=args.fever_split,
                    start_index=args.start_index,
                    max_examples=args.max_examples,
                    seed=seed,
                    seeded_path=seeded_path,
                )
                expected = {
                    "prediction_count": expected_example_count(args.fever_split, args.start_index, args.max_examples),
                    "template": template_id,
                    "fever_split": args.fever_split,
                    "lname": lname,
                    "lnum": lnum,
                    "rate": rate,
                    "seed": seed,
                    "model_path": args.model_path,
                    "revision": args.revision,
                    "device": args.device,
                    "intervention_source": intervention_source,
                    "dtpts": args.dtpts,
                    "intervention": args.intervention,
                    "require_provenance": True,
                }
                run_it, reason = should_run(output_path, args.overwrite, args.skip_existing, expected)
                if not run_it:
                    skipped += 1
                    print(f"[skip] mode={mode_name} seed={seed_label} template={template_id} reason={reason}")
                    continue

                if reason is not None:
                    print(f"[repair] mode={mode_name} seed={seed_label} template={template_id} reason={reason}")

                command = build_command(script_path=script_path, args=args, template_id=template_id, mode_name=mode_name, seed=seed)
                print(f"[run] mode={mode_name} seed={seed_label} template={template_id}")
                print(shlex.join(command))

                if args.dry_run:
                    continue

                try:
                    subprocess.run(command, check=True, cwd=os.path.dirname(__file__))
                    valid, validation_reason = validate_existing_output(output_path, expected)
                    if not valid:
                        raise RuntimeError(
                            f"Run completed but output validation failed for mode={mode_name} "
                            f"seed={seed_label} template={template_id}: {validation_reason}"
                        )
                    completed += 1
                except subprocess.CalledProcessError as exc:
                    failed += 1
                    print(f"[fail] mode={mode_name} seed={seed_label} template={template_id} exit_code={exc.returncode}")
                    if not args.keep_going:
                        raise

    print(f"SUITE_DONE completed={completed} skipped={skipped} failed={failed}")
    if failed > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
