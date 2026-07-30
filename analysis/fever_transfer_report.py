"""Analyze the multi-seed fixed-intervention FEVER transfer study."""

import argparse
import csv
import json
import math
import os
import pickle
import sys

import numpy as np

from fever_report import fmt, parse_slice_from_path, report_file, short_name


SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "laser-main", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from fever_prompting import get_fever_prompt_spec, list_fever_prompt_categories, list_fever_prompt_template_ids


def recursive_prediction_files(root_dir):
    paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".p") and "-predictions-" in filename:
                paths.append(os.path.join(dirpath, filename))
    return sorted(paths)


def prediction_to_summary_path(prediction_path):
    dirname, filename = os.path.split(prediction_path)
    summary_name = filename.replace("-predictions-", "-result-summary-")
    if summary_name.endswith(".p"):
        summary_name = summary_name[:-2] + ".pkl"
    return os.path.join(dirname, summary_name)


def load_summary(prediction_path):
    summary_path = prediction_to_summary_path(prediction_path)
    if not os.path.exists(summary_path):
        return {}
    with open(summary_path, "rb") as f:
        return pickle.load(f)


def matches_requested_slice(path, start_index, max_examples):
    path_start, path_max = parse_slice_from_path(path)
    if start_index is not None and path_start != start_index:
        return False
    if max_examples is not None and path_max != max_examples:
        return False
    return True


def metric_from_report_or_summary(summary, report, split_name, metric_name):
    if metric_name == "accuracy":
        return summary.get("metric/binary_accuracy", report[split_name]["accuracy"])
    if metric_name == "binary_nll":
        return summary.get("metric/binary_nll_normalized", report[split_name]["mean_binary_nll"])
    if metric_name == "raw_logprob":
        return summary.get("metric/mean_raw_gold_token_logprob", report[split_name]["mean_log_prob"])
    raise KeyError(metric_name)


def infer_split_name(summary, report):
    declared = summary.get("args/fever_split", summary.get("fever/returned_split"))
    if declared in {"paper_dev", "laser_dev"}:
        return "dev"
    if declared in {"paper_test", "laser_test"}:
        return "test"
    if report["test"]["n"] == report["total"] and report["total"] > 0:
        return "test"
    if report["dev"]["n"] == report["total"] and report["total"] > 0:
        return "dev"
    return "all"


def load_descriptor(prediction_path, dev_size, test_size):
    summary = load_summary(prediction_path)
    report = report_file(prediction_path, dev_size=dev_size, test_size=test_size)
    split_name = infer_split_name(summary, report)
    template_id = summary.get("prompt/template_id")
    if template_id is None:
        template_id = "original" if os.path.basename(os.path.dirname(prediction_path)) != "original" else "original"
    prompt_spec = get_fever_prompt_spec(template_id)
    return {
        "path": prediction_path,
        "summary": summary,
        "report": report,
        "split_name": split_name,
        "template_id": prompt_spec["template_id"],
        "category_id": prompt_spec["category_id"],
        "category_label": prompt_spec["category_label"],
        "variant_index": prompt_spec["variant_index"],
        "is_original": prompt_spec["is_original"],
        "wrapper_char_len": prompt_spec["wrapper_char_len"],
        "wrapper_word_len": prompt_spec["wrapper_word_len"],
        "divergence_char_norm": prompt_spec["divergence_char_norm"],
        "divergence_token_jaccard": prompt_spec["divergence_token_jaccard"],
        "fever_split": summary.get("args/fever_split", summary.get("fever/returned_split")),
        "intervention_source": summary.get("args/intervention_source", "unspecified"),
        "seed": summary.get("args/seed"),
        "lname": summary.get("args/lname"),
        "lnum": summary.get("args/lnum"),
        "rate": summary.get("args/rate"),
        "rho": summary.get("args/rho", summary.get("intervention/rho")),
        "true_token_id": summary.get("labels/true_token_id"),
        "false_token_id": summary.get("labels/false_token_id"),
        "true_text": summary.get("labels/true_text"),
        "false_text": summary.get("labels/false_text"),
        "model_path": summary.get("args/model_path"),
        "revision": summary.get("args/revision"),
        "model_commit_hash": summary.get("runtime/model_commit_hash"),
        "tokenizer_commit_hash": summary.get("runtime/tokenizer_commit_hash"),
        "runtime_device": summary.get("runtime/device"),
        "dataset_name": summary.get("dataset/name"),
        "dataset_config": summary.get("dataset/config"),
        "dataset_paper_dev_fingerprint": summary.get("dataset/paper_dev_fingerprint"),
        "dataset_paper_test_fingerprint": summary.get("dataset/paper_test_fingerprint"),
        "accuracy": metric_from_report_or_summary(summary, report, split_name, "accuracy"),
        "binary_nll": metric_from_report_or_summary(summary, report, split_name, "binary_nll"),
        "raw_logprob": metric_from_report_or_summary(summary, report, split_name, "raw_logprob"),
    }


def scan_descriptors(root_dir, dev_size, test_size, fever_split=None, start_index=None, max_examples=None):
    descriptors = []
    for path in recursive_prediction_files(root_dir):
        if start_index is not None or max_examples is not None:
            if not matches_requested_slice(path, start_index, max_examples):
                continue
        descriptor = load_descriptor(path, dev_size=dev_size, test_size=test_size)
        if fever_split is not None and descriptor["fever_split"] not in {None, fever_split}:
            continue
        descriptors.append(descriptor)
    return descriptors


def mean(values):
    values = [value for value in values if value is not None]
    if not values:
        return None
    return sum(values) / len(values)


def stddev(values):
    values = [value for value in values if value is not None]
    if not values:
        return None
    mu = mean(values)
    return math.sqrt(sum((value - mu) ** 2 for value in values) / len(values))


def percentile(values, q):
    values = [value for value in values if value is not None]
    if not values:
        return None
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def build_baseline_map(descriptors):
    baseline_map = {}
    for descriptor in descriptors:
        template_id = descriptor["template_id"]
        if template_id in baseline_map:
            raise ValueError(f"Duplicate baseline run for template {template_id}: {short_name(descriptor['path'])}")
        baseline_map[template_id] = descriptor
    return baseline_map


def build_laser_map(descriptors):
    laser_map = {}
    for descriptor in descriptors:
        seed = descriptor["seed"]
        if seed is None:
            raise ValueError(f"Laser run is missing seed metadata: {short_name(descriptor['path'])}")
        laser_map.setdefault(seed, {})
        template_id = descriptor["template_id"]
        if template_id in laser_map[seed]:
            raise ValueError(
                f"Duplicate LASER run for seed={seed} template={template_id}: {short_name(descriptor['path'])}"
            )
        laser_map[seed][template_id] = descriptor
    return laser_map


def validate_suite_configuration(
    baseline_map,
    laser_map,
    expected_seeds=None,
    require_complete=False,
    expected_claim_count=None,
):
    expected_templates = set(sorted_template_ids())
    baseline_templates = set(baseline_map)
    actual_seeds = set(laser_map)

    if expected_seeds is not None and actual_seeds != set(expected_seeds):
        raise ValueError(
            f"LASER seed mismatch: found {sorted(actual_seeds)}, expected {sorted(expected_seeds)}"
        )

    if require_complete and baseline_templates != expected_templates:
        missing = sorted(expected_templates - baseline_templates)
        extra = sorted(baseline_templates - expected_templates)
        raise ValueError(f"Incomplete baseline prompt suite: missing={missing}, extra={extra}")

    for seed, template_map in sorted(laser_map.items()):
        template_ids = set(template_map)
        if require_complete and template_ids != expected_templates:
            missing = sorted(expected_templates - template_ids)
            extra = sorted(template_ids - expected_templates)
            raise ValueError(f"Incomplete LASER prompt suite for seed={seed}: missing={missing}, extra={extra}")

    all_descriptors = list(baseline_map.values()) + [
        row for template_map in laser_map.values() for row in template_map.values()
    ]
    if require_complete:
        required_descriptor_fields = [
            "fever_split",
            "lname",
            "lnum",
            "rate",
            "rho",
            "true_token_id",
            "false_token_id",
            "true_text",
            "false_text",
            "model_path",
            "revision",
            "runtime_device",
            "dataset_name",
            "dataset_config",
            "dataset_paper_dev_fingerprint",
            "dataset_paper_test_fingerprint",
        ]
        for descriptor in all_descriptors:
            missing = [field for field in required_descriptor_fields if descriptor.get(field) is None]
            if missing:
                raise ValueError(
                    f"Run is missing required metadata {missing}: {short_name(descriptor['path'])}"
                )

    counts = {
        descriptor["report"][descriptor["split_name"]]["n"]
        for descriptor in all_descriptors
    }
    if len(counts) != 1 or not counts or next(iter(counts)) <= 0:
        raise ValueError(f"Runs have inconsistent or empty claim counts: {sorted(counts)}")
    if require_complete and expected_claim_count is not None and next(iter(counts)) != expected_claim_count:
        raise ValueError(
            f"Runs contain {next(iter(counts))} claims; expected {expected_claim_count} for the requested split"
        )

    for descriptor in all_descriptors:
        accuracy = descriptor["accuracy"]
        binary_nll = descriptor["binary_nll"]
        raw_logprob = descriptor["raw_logprob"]
        if accuracy is None or not math.isfinite(float(accuracy)) or not 0.0 <= float(accuracy) <= 100.0:
            raise ValueError(f"Run has invalid binary accuracy: {short_name(descriptor['path'])}")
        if binary_nll is None or not math.isfinite(float(binary_nll)) or float(binary_nll) < 0.0:
            raise ValueError(f"Run has invalid normalized binary NLL: {short_name(descriptor['path'])}")
        if raw_logprob is None or not math.isfinite(float(raw_logprob)):
            raise ValueError(f"Run has invalid raw gold-token log probability: {short_name(descriptor['path'])}")

    shared_fields = [
        "model_path",
        "revision",
        "model_commit_hash",
        "tokenizer_commit_hash",
        "runtime_device",
        "dataset_name",
        "dataset_config",
        "dataset_paper_dev_fingerprint",
        "dataset_paper_test_fingerprint",
        "fever_split",
        "true_text",
        "false_text",
        "true_token_id",
        "false_token_id",
    ]
    shared_configs = {tuple(row.get(field) for field in shared_fields) for row in all_descriptors}
    if len(shared_configs) != 1:
        raise ValueError("Baseline/LASER runs mix model, tokenizer, dataset, device, split, or label provenance")

    baseline_configs = {
        (row["lname"], row["lnum"], row["rate"], row["intervention_source"])
        for row in baseline_map.values()
    }
    if len(baseline_configs) != 1:
        raise ValueError(f"Baseline runs mix configurations: {sorted(baseline_configs, key=str)}")

    laser_configs = {
        (
            row["lname"],
            row["lnum"],
            row["rho"],
            row["intervention_source"],
        )
        for template_map in laser_map.values()
        for row in template_map.values()
    }
    if len(laser_configs) != 1:
        raise ValueError(f"LASER runs mix frozen-intervention configurations: {sorted(laser_configs, key=str)}")

    if require_complete:
        if next(iter(baseline_configs))[3] != "unspecified":
            raise ValueError("Baseline runs must use intervention_source=unspecified")
        if next(iter(laser_configs))[3] != "validation_selected":
            raise ValueError("Transfer LASER runs must use intervention_source=validation_selected")

    return {
        "shared_configuration": dict(zip(shared_fields, next(iter(shared_configs)))),
        "baseline_configuration": list(next(iter(baseline_configs))),
        "laser_configuration": list(next(iter(laser_configs))),
        "seeds": sorted(actual_seeds),
        "template_count": len(baseline_templates),
        "claim_count": next(iter(counts)),
    }


def sorted_template_ids():
    return list_fever_prompt_template_ids()


def sort_rows(rows, sort_by):
    category_order = {spec["category_id"]: index for index, spec in enumerate(list_fever_prompt_categories())}
    template_order = {template_id: index for index, template_id in enumerate(sorted_template_ids())}

    def safe_neg(value):
        return -10 ** 9 if value is None else -value

    def safe_pos(value):
        return 10 ** 9 if value is None else value

    if sort_by == "template":
        return sorted(rows, key=lambda row: (template_order.get(row["template"], 10 ** 9), row["template"]))
    if sort_by == "category":
        return sorted(
            rows,
            key=lambda row: (
                category_order.get(row["category_id"], 10 ** 9),
                template_order.get(row["template"], 10 ** 9),
                row["template"],
            ),
        )
    if sort_by == "base_acc":
        return sorted(rows, key=lambda row: (safe_neg(row["baseline_acc"]), row["template"]))
    if sort_by == "laser_acc":
        return sorted(rows, key=lambda row: (safe_neg(row["laser_acc"]), row["template"]))
    if sort_by == "gain":
        return sorted(rows, key=lambda row: (safe_neg(row["laser_vs_same_template_acc"]), row["template"]))
    if sort_by == "divergence":
        return sorted(rows, key=lambda row: (safe_neg(row["divergence_char_norm"]), row["template"]))
    raise ValueError(f"Unknown sort_by={sort_by}")


def build_template_seed_rows(baseline_map, laser_map):
    rows = []
    common_templates = set(baseline_map)
    for seed, template_map in laser_map.items():
        common_templates &= set(template_map)

    if not common_templates:
        raise ValueError("No common templates found between baseline and LASER runs.")

    for seed in sorted(laser_map):
        for template_id in sorted(common_templates, key=lambda template: sorted_template_ids().index(template)):
            baseline = baseline_map[template_id]
            laser = laser_map[seed][template_id]
            if baseline["split_name"] != laser["split_name"]:
                raise ValueError(
                    f"Split mismatch for template={template_id}, seed={seed}: "
                    f"baseline={baseline['split_name']} laser={laser['split_name']}"
                )
            rows.append(
                {
                    "seed": seed,
                    "template": template_id,
                    "category_id": baseline["category_id"],
                    "category_label": baseline["category_label"],
                    "variant_index": baseline["variant_index"],
                    "is_original": baseline["is_original"],
                    "split": baseline["split_name"],
                    "wrapper_char_len": baseline["wrapper_char_len"],
                    "wrapper_word_len": baseline["wrapper_word_len"],
                    "divergence_char_norm": baseline["divergence_char_norm"],
                    "divergence_token_jaccard": baseline["divergence_token_jaccard"],
                    "intervention_source": laser["intervention_source"],
                    "baseline_acc": baseline["accuracy"],
                    "baseline_n": baseline["report"][baseline["split_name"]]["n"],
                    "baseline_binary_nll": baseline["binary_nll"],
                    "baseline_logp": baseline["raw_logprob"],
                    "baseline_raw_logprob": baseline["raw_logprob"],
                    "laser_acc": laser["accuracy"],
                    "laser_n": laser["report"][laser["split_name"]]["n"],
                    "laser_binary_nll": laser["binary_nll"],
                    "laser_logp": laser["raw_logprob"],
                    "laser_raw_logprob": laser["raw_logprob"],
                    "laser_vs_same_template_acc": laser["accuracy"] - baseline["accuracy"],
                }
            )
    return rows


def classify_sign(value, eps=1e-12):
    if value > eps:
        return "positive"
    if value < -eps:
        return "negative"
    return "zero"


def build_template_rows(template_seed_rows):
    grouped = {}
    for row in template_seed_rows:
        grouped.setdefault(row["template"], []).append(row)

    rows = []
    for template_id, template_rows in grouped.items():
        gains = [row["laser_vs_same_template_acc"] for row in template_rows]
        laser_accs = [row["laser_acc"] for row in template_rows]
        laser_nlls = [row["laser_binary_nll"] for row in template_rows]
        sign_labels = [classify_sign(gain) for gain in gains]
        unique_signs = set(sign_labels)
        if len(unique_signs) == 1:
            sign_stability = next(iter(unique_signs))
        else:
            sign_stability = "mixed"

        first = template_rows[0]
        rows.append(
            {
                "template": template_id,
                "category_id": first["category_id"],
                "category_label": first["category_label"],
                "variant_index": first["variant_index"],
                "is_original": first["is_original"],
                "split": first["split"],
                "wrapper_char_len": first["wrapper_char_len"],
                "wrapper_word_len": first["wrapper_word_len"],
                "divergence_char_norm": first["divergence_char_norm"],
                "divergence_token_jaccard": first["divergence_token_jaccard"],
                "intervention_source": first["intervention_source"],
                "baseline_n": first["baseline_n"],
                "baseline_acc": first["baseline_acc"],
                "baseline_binary_nll": first["baseline_binary_nll"],
                "baseline_logp": first["baseline_logp"],
                "baseline_raw_logprob": first["baseline_raw_logprob"],
                "laser_n": first["laser_n"],
                "laser_acc": mean(laser_accs),
                "laser_acc_std": stddev(laser_accs),
                "laser_binary_nll": mean(laser_nlls),
                "laser_binary_nll_std": stddev(laser_nlls),
                "laser_logp": mean([row["laser_logp"] for row in template_rows]),
                "laser_raw_logprob": mean([row["laser_raw_logprob"] for row in template_rows]),
                "laser_vs_same_template_acc": mean(gains),
                "gain_mean_acc": mean(gains),
                "gain_median_acc": percentile(gains, 0.5),
                "gain_min_acc": min(gains),
                "gain_max_acc": max(gains),
                "gain_std_acc": stddev(gains),
                "positive_seed_fraction": sum(label == "positive" for label in sign_labels) / len(sign_labels),
                "negative_seed_fraction": sum(label == "negative" for label in sign_labels) / len(sign_labels),
                "zero_seed_fraction": sum(label == "zero" for label in sign_labels) / len(sign_labels),
                "sign_stability": sign_stability,
            }
        )
    return rows


def add_original_relative_deltas(template_rows):
    original_row = next((row for row in template_rows if row["is_original"]), None)
    original_baseline_acc = None if original_row is None else original_row["baseline_acc"]

    for row in template_rows:
        if original_baseline_acc is None:
            row["baseline_vs_original_acc"] = None
            row["laser_vs_original_baseline_acc"] = None
        else:
            row["baseline_vs_original_acc"] = row["baseline_acc"] - original_baseline_acc
            row["laser_vs_original_baseline_acc"] = row["laser_acc"] - original_baseline_acc


def build_category_rows(template_rows):
    grouped = {}
    for row in template_rows:
        if row["is_original"]:
            continue
        grouped.setdefault(row["category_id"], []).append(row)

    category_order = {spec["category_id"]: index for index, spec in enumerate(list_fever_prompt_categories())}
    rows = []
    for category_id, category_rows in sorted(grouped.items(), key=lambda item: (category_order.get(item[0], 10 ** 9), item[0])):
        gains = [row["gain_mean_acc"] for row in category_rows]
        rows.append(
            {
                "category_id": category_id,
                "category_label": category_rows[0]["category_label"],
                "n": len(category_rows),
                "baseline_mean_acc": mean([row["baseline_acc"] for row in category_rows]),
                "baseline_std_acc": stddev([row["baseline_acc"] for row in category_rows]),
                "laser_mean_acc": mean([row["laser_acc"] for row in category_rows]),
                "laser_std_acc": stddev([row["laser_acc"] for row in category_rows]),
                "gain_mean_acc": mean(gains),
                "gain_std_acc": stddev(gains),
                "gain_median_acc": percentile(gains, 0.5),
                "gain_min_acc": min(gains),
                "gain_max_acc": max(gains),
            }
        )
    return rows


def build_seed_rows(template_seed_rows):
    grouped = {}
    for row in template_seed_rows:
        if row["is_original"]:
            continue
        grouped.setdefault(row["seed"], []).append(row)

    rows = []
    for seed, seed_rows in sorted(grouped.items()):
        gains = [row["laser_vs_same_template_acc"] for row in seed_rows]
        rows.append(
            {
                "seed": seed,
                "prompt_count": len(seed_rows),
                "baseline_mean_acc": mean([row["baseline_acc"] for row in seed_rows]),
                "laser_mean_acc": mean([row["laser_acc"] for row in seed_rows]),
                "gain_mean_acc": mean(gains),
                "gain_std_acc": stddev(gains),
                "gain_min_acc": min(gains),
                "gain_max_acc": max(gains),
                "positive_fraction": sum(classify_sign(gain) == "positive" for gain in gains) / len(gains),
                "negative_fraction": sum(classify_sign(gain) == "negative" for gain in gains) / len(gains),
                "zero_fraction": sum(classify_sign(gain) == "zero" for gain in gains) / len(gains),
            }
        )
    return rows


def load_binary_correct_arrays(path):
    with open(path, "rb") as f:
        predictions = pickle.load(f)

    claim_ids = []
    correctness = []
    for local_ix, row in enumerate(predictions):
        claim_id = row.get("dataset_global_ix")
        if claim_id is None:
            claim_id = local_ix
        claim_ids.append(int(claim_id))
        correctness.append(1.0 if row.get("binary_correct", row.get("correct")) else 0.0)
    return np.asarray(claim_ids, dtype=np.int64), np.asarray(correctness, dtype=np.float64)


def build_claim_delta_arrays(baseline_map, laser_map):
    prompt_ids = [template_id for template_id in sorted_template_ids() if template_id in baseline_map and template_id != "original"]
    seeds = sorted(laser_map)
    if not prompt_ids:
        raise ValueError("No paraphrase templates are available for claim-level transfer analysis.")

    baseline_arrays = {}
    reference_claim_ids = None
    for template_id in prompt_ids:
        claim_ids, correctness = load_binary_correct_arrays(baseline_map[template_id]["path"])
        if reference_claim_ids is None:
            reference_claim_ids = claim_ids
        elif not np.array_equal(reference_claim_ids, claim_ids):
            raise ValueError(f"Baseline claim ordering mismatch for template {template_id}")
        baseline_arrays[template_id] = correctness

    baseline_claim_mean = np.mean(
        np.stack([baseline_arrays[template_id] for template_id in prompt_ids], axis=0),
        axis=0,
    )

    per_seed_claim_mean_delta = {}
    per_seed_laser_claim_mean = {}
    for seed in seeds:
        delta_rows = []
        laser_rows = []
        for template_id in prompt_ids:
            claim_ids, laser_correct = load_binary_correct_arrays(laser_map[seed][template_id]["path"])
            if not np.array_equal(reference_claim_ids, claim_ids):
                raise ValueError(f"LASER claim ordering mismatch for template {template_id}, seed {seed}")
            delta_rows.append(laser_correct - baseline_arrays[template_id])
            laser_rows.append(laser_correct)
        per_seed_claim_mean_delta[seed] = np.mean(np.stack(delta_rows, axis=0), axis=0)
        per_seed_laser_claim_mean[seed] = np.mean(np.stack(laser_rows, axis=0), axis=0)

    overall_claim_mean_delta = np.mean(np.stack([per_seed_claim_mean_delta[seed] for seed in seeds], axis=0), axis=0)
    overall_laser_claim_mean = np.mean(
        np.stack([per_seed_laser_claim_mean[seed] for seed in seeds], axis=0),
        axis=0,
    )
    return {
        "claim_ids": reference_claim_ids,
        "baseline_claim_mean": baseline_claim_mean,
        "per_seed_laser_claim_mean": per_seed_laser_claim_mean,
        "per_seed_claim_mean_delta": per_seed_claim_mean_delta,
        "overall_laser_claim_mean": overall_laser_claim_mean,
        "overall_claim_mean_delta": overall_claim_mean_delta,
        "prompt_count": len(prompt_ids),
    }


def build_claim_export_rows(claim_analysis):
    claim_rows = []
    claim_seed_rows = []
    seeds = sorted(claim_analysis["per_seed_claim_mean_delta"])
    for index, claim_id in enumerate(claim_analysis["claim_ids"]):
        per_seed_gains = [claim_analysis["per_seed_claim_mean_delta"][seed][index] * 100.0 for seed in seeds]
        claim_rows.append(
            {
                "claim_id": int(claim_id),
                "prompt_count": claim_analysis["prompt_count"],
                "seed_count": len(seeds),
                "baseline_prompt_mean_accuracy": float(claim_analysis["baseline_claim_mean"][index] * 100.0),
                "laser_prompt_seed_mean_accuracy": float(claim_analysis["overall_laser_claim_mean"][index] * 100.0),
                "gain_prompt_seed_mean_accuracy": float(claim_analysis["overall_claim_mean_delta"][index] * 100.0),
                "gain_across_seeds_std": stddev(per_seed_gains),
            }
        )
        for seed in seeds:
            claim_seed_rows.append(
                {
                    "claim_id": int(claim_id),
                    "seed": seed,
                    "prompt_count": claim_analysis["prompt_count"],
                    "baseline_prompt_mean_accuracy": float(claim_analysis["baseline_claim_mean"][index] * 100.0),
                    "laser_prompt_mean_accuracy": float(
                        claim_analysis["per_seed_laser_claim_mean"][seed][index] * 100.0
                    ),
                    "gain_prompt_mean_accuracy": float(
                        claim_analysis["per_seed_claim_mean_delta"][seed][index] * 100.0
                    ),
                }
            )
    return claim_rows, claim_seed_rows


def claim_bootstrap_ci(claim_values, samples=10000, seed=0):
    if claim_values.size == 0:
        return None, None
    if claim_values.size == 1:
        value = float(claim_values[0] * 100.0)
        return value, value

    rng = np.random.default_rng(seed)
    n = claim_values.shape[0]
    bootstrap_values = []
    for _ in range(samples):
        sample_indices = rng.integers(0, n, size=n)
        bootstrap_values.append(float(claim_values[sample_indices].mean() * 100.0))
    bootstrap_values.sort()
    return percentile(bootstrap_values, 0.025), percentile(bootstrap_values, 0.975)


def compute_global_summary(template_rows, seed_rows, per_seed_claim_mean_delta, overall_claim_mean_delta, bootstrap_samples, bootstrap_seed):
    paraphrase_rows = [row for row in template_rows if not row["is_original"]]
    gains = [row["gain_mean_acc"] for row in paraphrase_rows]
    gain_ci_low, gain_ci_high = claim_bootstrap_ci(
        overall_claim_mean_delta,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )

    per_seed_ci = {
        seed: claim_bootstrap_ci(claim_values, samples=bootstrap_samples, seed=bootstrap_seed + int(seed) + 1)
        for seed, claim_values in sorted(per_seed_claim_mean_delta.items())
    }

    suite_mean_gains = [row["gain_mean_acc"] for row in seed_rows]
    min_gain = min(gains)
    max_gain = max(gains)

    gain_mean = mean(gains)
    if gain_mean > 0 and min_gain < 0:
        conclusion = "positive average transfer, but not uniform robustness"
    elif gain_mean > 0 and min_gain >= 0:
        conclusion = "positive average transfer with non-negative gain across the fixed prompt suite"
    elif gain_mean <= 0 and max_gain > 0:
        conclusion = "non-positive average transfer with heterogeneous prompt-level effects"
    else:
        conclusion = "non-positive transfer across the fixed prompt suite"

    return {
        "prompt_count": len(paraphrase_rows),
        "paraphrase_count": len(paraphrase_rows),
        "gain_mean_acc": gain_mean,
        "gain_median_acc": percentile(gains, 0.5),
        "gain_min_acc": min_gain,
        "gain_max_acc": max_gain,
        "gain_iqr_acc": percentile(gains, 0.75) - percentile(gains, 0.25),
        "gain_std_acc": stddev(gains),
        "positive_fraction": sum(classify_sign(gain) == "positive" for gain in gains) / len(gains),
        "negative_fraction": sum(classify_sign(gain) == "negative" for gain in gains) / len(gains),
        "zero_fraction": sum(classify_sign(gain) == "zero" for gain in gains) / len(gains),
        "stability_metric_min_prompt_gain": min_gain,
        "suite_mean_gain_across_seeds_mean": mean(suite_mean_gains),
        "suite_mean_gain_across_seeds_std": stddev(suite_mean_gains),
        "claim_bootstrap_ci_low": gain_ci_low,
        "claim_bootstrap_ci_high": gain_ci_high,
        "claim_bootstrap_per_seed": {str(seed): {"low": low, "high": high} for seed, (low, high) in per_seed_ci.items()},
        "stable_positive_prompts": sum(row["sign_stability"] == "positive" for row in paraphrase_rows),
        "stable_negative_prompts": sum(row["sign_stability"] == "negative" for row in paraphrase_rows),
        "mixed_sign_prompts": sum(row["sign_stability"] == "mixed" for row in paraphrase_rows),
        "conclusion": conclusion,
    }


def print_original_control_row(template_rows):
    original_rows = [row for row in template_rows if row["is_original"]]
    if not original_rows:
        return
    row = original_rows[0]
    print("ORIGINAL_CONTROL")
    print(
        f"baseline_acc={fmt(row['baseline_acc'])} "
        f"laser_mean_acc={fmt(row['laser_acc'])} "
        f"gain_mean_acc={fmt(row['gain_mean_acc'])} "
        f"gain_std_acc={fmt(row['gain_std_acc'])} "
        f"sign_stability={row['sign_stability']}"
    )


def print_template_table(rows):
    header = (
        "template".ljust(12)
        + " category".ljust(9)
        + " split".ljust(7)
        + " base_acc".rjust(12)
        + " laser_acc".rjust(12)
        + " gain_mean".rjust(12)
        + " gain_std".rjust(11)
        + " gain_min".rjust(11)
        + " gain_max".rjust(11)
        + " sign".rjust(10)
        + " base_bnll".rjust(12)
        + " laser_bnll".rjust(12)
    )
    print("\nTEMPLATE_SUMMARY")
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            row["template"].ljust(12)
            + row["category_id"].ljust(9)
            + row["split"].ljust(7)
            + fmt(row["baseline_acc"]).rjust(12)
            + fmt(row["laser_acc"]).rjust(12)
            + fmt(row["gain_mean_acc"]).rjust(12)
            + fmt(row["gain_std_acc"]).rjust(11)
            + fmt(row["gain_min_acc"]).rjust(11)
            + fmt(row["gain_max_acc"]).rjust(11)
            + row["sign_stability"].rjust(10)
            + fmt(row["baseline_binary_nll"]).rjust(12)
            + fmt(row["laser_binary_nll"]).rjust(12)
        )


def print_category_table(category_rows):
    print("\nCATEGORY_SUMMARY")
    header = (
        "category".ljust(9)
        + " n".rjust(5)
        + " base_mean".rjust(12)
        + " base_std".rjust(11)
        + " laser_mean".rjust(12)
        + " laser_std".rjust(11)
        + " gain_mean".rjust(11)
        + " gain_std".rjust(10)
    )
    print(header)
    print("-" * len(header))
    for row in category_rows:
        print(
            row["category_id"].ljust(9)
            + str(row["n"]).rjust(5)
            + fmt(row["baseline_mean_acc"]).rjust(12)
            + fmt(row["baseline_std_acc"]).rjust(11)
            + fmt(row["laser_mean_acc"]).rjust(12)
            + fmt(row["laser_std_acc"]).rjust(11)
            + fmt(row["gain_mean_acc"]).rjust(11)
            + fmt(row["gain_std_acc"]).rjust(10)
        )


def print_seed_table(seed_rows):
    print("\nSEED_SUMMARY")
    header = (
        "seed".ljust(8)
        + " prompts".rjust(9)
        + " base_mean".rjust(12)
        + " laser_mean".rjust(12)
        + " gain_mean".rjust(12)
        + " gain_std".rjust(11)
        + " gain_min".rjust(11)
        + " gain_max".rjust(11)
    )
    print(header)
    print("-" * len(header))
    for row in seed_rows:
        print(
            str(row["seed"]).ljust(8)
            + str(row["prompt_count"]).rjust(9)
            + fmt(row["baseline_mean_acc"]).rjust(12)
            + fmt(row["laser_mean_acc"]).rjust(12)
            + fmt(row["gain_mean_acc"]).rjust(12)
            + fmt(row["gain_std_acc"]).rjust(11)
            + fmt(row["gain_min_acc"]).rjust(11)
            + fmt(row["gain_max_acc"]).rjust(11)
        )


def print_global_summary(summary, bootstrap_samples):
    print("\nGLOBAL_FIXED_INTERVENTION_PROMPT_TRANSFER")
    print(f"prompt_count={summary['prompt_count']}")
    print(f"gain_mean_acc={fmt(summary['gain_mean_acc'])}")
    print(f"gain_median_acc={fmt(summary['gain_median_acc'])}")
    print(f"gain_min_acc={fmt(summary['gain_min_acc'])}")
    print(f"gain_max_acc={fmt(summary['gain_max_acc'])}")
    print(f"gain_iqr_acc={fmt(summary['gain_iqr_acc'])}")
    print(f"gain_std_acc={fmt(summary['gain_std_acc'])}")
    print(f"positive_fraction={fmt(100.0 * summary['positive_fraction'])}")
    print(f"negative_fraction={fmt(100.0 * summary['negative_fraction'])}")
    print(f"zero_fraction={fmt(100.0 * summary['zero_fraction'])}")
    print(f"stability_metric_min_p={fmt(summary['stability_metric_min_prompt_gain'])}")
    print(f"suite_mean_gain_across_seeds_mean={fmt(summary['suite_mean_gain_across_seeds_mean'])}")
    print(f"suite_mean_gain_across_seeds_std={fmt(summary['suite_mean_gain_across_seeds_std'])}")
    print(
        f"claim_level_paired_bootstrap_ci95(samples={bootstrap_samples})="
        f"[{fmt(summary['claim_bootstrap_ci_low'])}, {fmt(summary['claim_bootstrap_ci_high'])}]"
    )
    print(f"stable_positive_prompts={summary['stable_positive_prompts']}")
    print(f"stable_negative_prompts={summary['stable_negative_prompts']}")
    print(f"mixed_sign_prompts={summary['mixed_sign_prompts']}")
    print(f"CONCLUSION={summary['conclusion']}")


def write_exports(
    export_dir,
    template_rows,
    template_seed_rows,
    category_rows,
    seed_rows,
    claim_rows,
    claim_seed_rows,
    global_summary,
    metadata,
):
    os.makedirs(export_dir, exist_ok=True)

    template_fields = [
        "template",
        "category_id",
        "category_label",
        "variant_index",
        "is_original",
        "split",
        "wrapper_char_len",
        "wrapper_word_len",
        "divergence_char_norm",
        "divergence_token_jaccard",
        "intervention_source",
        "baseline_n",
        "baseline_acc",
        "baseline_logp",
        "baseline_binary_nll",
        "baseline_raw_logprob",
        "laser_n",
        "laser_acc",
        "laser_acc_std",
        "laser_logp",
        "laser_binary_nll",
        "laser_binary_nll_std",
        "laser_raw_logprob",
        "laser_vs_same_template_acc",
        "baseline_vs_original_acc",
        "laser_vs_original_baseline_acc",
        "gain_mean_acc",
        "gain_median_acc",
        "gain_min_acc",
        "gain_max_acc",
        "gain_std_acc",
        "positive_seed_fraction",
        "negative_seed_fraction",
        "zero_seed_fraction",
        "sign_stability",
    ]
    with open(os.path.join(export_dir, "template_metrics.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=template_fields)
        writer.writeheader()
        for row in template_rows:
            writer.writerow({field: row.get(field) for field in template_fields})

    template_seed_fields = [
        "seed",
        "template",
        "category_id",
        "category_label",
        "variant_index",
        "is_original",
        "split",
        "wrapper_char_len",
        "wrapper_word_len",
        "divergence_char_norm",
        "divergence_token_jaccard",
        "intervention_source",
        "baseline_n",
        "baseline_acc",
        "baseline_logp",
        "baseline_binary_nll",
        "baseline_raw_logprob",
        "laser_n",
        "laser_acc",
        "laser_logp",
        "laser_binary_nll",
        "laser_raw_logprob",
        "laser_vs_same_template_acc",
    ]
    with open(os.path.join(export_dir, "template_seed_metrics.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=template_seed_fields)
        writer.writeheader()
        for row in template_seed_rows:
            writer.writerow({field: row.get(field) for field in template_seed_fields})

    category_fields = [
        "category_id",
        "category_label",
        "n",
        "baseline_mean_acc",
        "baseline_std_acc",
        "laser_mean_acc",
        "laser_std_acc",
        "gain_mean_acc",
        "gain_std_acc",
        "gain_median_acc",
        "gain_min_acc",
        "gain_max_acc",
    ]
    with open(os.path.join(export_dir, "category_metrics.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=category_fields)
        writer.writeheader()
        for row in category_rows:
            writer.writerow({field: row.get(field) for field in category_fields})

    seed_fields = [
        "seed",
        "prompt_count",
        "baseline_mean_acc",
        "laser_mean_acc",
        "gain_mean_acc",
        "gain_std_acc",
        "gain_min_acc",
        "gain_max_acc",
        "positive_fraction",
        "negative_fraction",
        "zero_fraction",
    ]
    with open(os.path.join(export_dir, "seed_metrics.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=seed_fields)
        writer.writeheader()
        for row in seed_rows:
            writer.writerow({field: row.get(field) for field in seed_fields})

    claim_fields = [
        "claim_id",
        "prompt_count",
        "seed_count",
        "baseline_prompt_mean_accuracy",
        "laser_prompt_seed_mean_accuracy",
        "gain_prompt_seed_mean_accuracy",
        "gain_across_seeds_std",
    ]
    with open(os.path.join(export_dir, "claim_metrics.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=claim_fields)
        writer.writeheader()
        for row in claim_rows:
            writer.writerow({field: row.get(field) for field in claim_fields})

    claim_seed_fields = [
        "claim_id",
        "seed",
        "prompt_count",
        "baseline_prompt_mean_accuracy",
        "laser_prompt_mean_accuracy",
        "gain_prompt_mean_accuracy",
    ]
    with open(os.path.join(export_dir, "claim_seed_metrics.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=claim_seed_fields)
        writer.writeheader()
        for row in claim_seed_rows:
            writer.writerow({field: row.get(field) for field in claim_seed_fields})

    with open(os.path.join(export_dir, "summary.json"), "w") as f:
        json.dump(
            {
                "metadata": metadata,
                "global_summary": global_summary,
                "category_summary": category_rows,
                "seed_summary": seed_rows,
            },
            f,
            indent=2,
            sort_keys=True,
        )


def main():
    parser = argparse.ArgumentParser(description="Summarize FEVER fixed-intervention prompt transfer runs")
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--laser-root", required=True)
    parser.add_argument("--dev-size", type=int, default=6510)
    parser.add_argument("--test-size", type=int, default=6576)
    parser.add_argument("--fever-split", default="paper_test")
    parser.add_argument("--start-index", type=int, default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--sort-by", choices=["template", "category", "base_acc", "laser_acc", "gain", "divergence"], default="template")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--export-dir", default=None)
    parser.add_argument("--expected-seeds", nargs="+", type=int, default=None)
    parser.add_argument("--require-complete-suite", action="store_true")
    args = parser.parse_args()

    print(f"BASELINE_ROOT: {short_name(args.baseline_root)}")
    print(f"LASER_ROOT: {short_name(args.laser_root)}")
    print(f"FEVER_SPLIT: {args.fever_split}")
    print(f"SLICE_FILTER: start_index={args.start_index} max_examples={args.max_examples}")

    baseline_descriptors = scan_descriptors(
        args.baseline_root,
        dev_size=args.dev_size,
        test_size=args.test_size,
        fever_split=args.fever_split,
        start_index=args.start_index,
        max_examples=args.max_examples,
    )
    laser_descriptors = scan_descriptors(
        args.laser_root,
        dev_size=args.dev_size,
        test_size=args.test_size,
        fever_split=args.fever_split,
        start_index=args.start_index,
        max_examples=args.max_examples,
    )

    baseline_map = build_baseline_map(baseline_descriptors)
    laser_map = build_laser_map(laser_descriptors)
    configuration = validate_suite_configuration(
        baseline_map,
        laser_map,
        expected_seeds=args.expected_seeds,
        require_complete=args.require_complete_suite,
        expected_claim_count=(
            args.test_size
            if args.fever_split in {"paper_test", "laser_test"}
            else args.dev_size if args.fever_split in {"paper_dev", "laser_dev"} else None
        ),
    )

    template_seed_rows = build_template_seed_rows(baseline_map, laser_map)
    template_rows = build_template_rows(template_seed_rows)
    add_original_relative_deltas(template_rows)
    template_rows = sort_rows(template_rows, args.sort_by)
    category_rows = build_category_rows(template_rows)
    seed_rows = build_seed_rows(template_seed_rows)
    claim_analysis = build_claim_delta_arrays(baseline_map, laser_map)
    claim_rows, claim_seed_rows = build_claim_export_rows(claim_analysis)
    global_summary = compute_global_summary(
        template_rows,
        seed_rows,
        claim_analysis["per_seed_claim_mean_delta"],
        claim_analysis["overall_claim_mean_delta"],
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )

    print_original_control_row(template_rows)
    print_template_table(template_rows)
    print_category_table(category_rows)
    print_seed_table(seed_rows)
    print_global_summary(global_summary, bootstrap_samples=args.bootstrap_samples)

    if args.export_dir is not None:
        write_exports(
            export_dir=args.export_dir,
            template_rows=template_rows,
            template_seed_rows=template_seed_rows,
            category_rows=category_rows,
            seed_rows=seed_rows,
            claim_rows=claim_rows,
            claim_seed_rows=claim_seed_rows,
            global_summary=global_summary,
            metadata={
                "baseline_root": args.baseline_root,
                "laser_root": args.laser_root,
                "fever_split": args.fever_split,
                "start_index": args.start_index,
                "max_examples": args.max_examples,
                "sort_by": args.sort_by,
                "require_complete_suite": args.require_complete_suite,
                "expected_seeds": args.expected_seeds,
                "validated_configuration": configuration,
            },
        )
        print(f"EXPORT_DIR: {short_name(args.export_dir)}")


if __name__ == "__main__":
    main()
