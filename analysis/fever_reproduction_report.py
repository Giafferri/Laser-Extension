"""Summarize the paper-aligned FEVER reproduction runs."""

import argparse
import csv
import itertools
import json
import math
import os
import pickle

from fever_report import fmt, report_file, short_name


EXPECTED_GRID_LNAMES = ["fc_in", "fc_out", "q_proj", "k_proj", "v_proj", "out_proj"]
EXPECTED_GRID_LNUMS = [20, 22, 24, 26]
EXPECTED_GRID_RHOS = [0.10, 0.05, 0.01]


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


def metric(summary, report, key, split_name):
    if key == "accuracy":
        value = summary.get("metric/binary_accuracy")
        if value is not None:
            return value
        return report[split_name]["accuracy"]
    if key == "binary_nll":
        value = summary.get("metric/binary_nll_normalized")
        if value is not None:
            return value
        return report[split_name]["mean_binary_nll"]
    raise KeyError(key)


def build_descriptors(paths, dev_size, test_size):
    descriptors = []
    for path in paths:
        summary = load_summary(path)
        report = report_file(path, dev_size=dev_size, test_size=test_size)
        template_id = summary.get("prompt/template_id", "original")
        descriptors.append(
            {
                "path": path,
                "summary": summary,
                "report": report,
                "template_id": template_id,
                "fever_split": summary.get("args/fever_split", summary.get("fever/returned_split")),
                "intervention_source": summary.get("args/intervention_source", "unspecified"),
                "lname": summary.get("args/lname"),
                "lnum": summary.get("args/lnum"),
                "rate": summary.get("args/rate"),
                "rho": summary.get("args/rho", summary.get("intervention/rho")),
                "seed": summary.get("args/seed"),
                "model_path": summary.get("args/model_path"),
                "revision": summary.get("args/revision"),
                "model_commit_hash": summary.get("runtime/model_commit_hash"),
                "tokenizer_commit_hash": summary.get("runtime/tokenizer_commit_hash"),
                "runtime_device": summary.get("runtime/device"),
                "dataset_name": summary.get("dataset/name"),
                "dataset_config": summary.get("dataset/config"),
                "dataset_paper_dev_fingerprint": summary.get("dataset/paper_dev_fingerprint"),
                "dataset_paper_test_fingerprint": summary.get("dataset/paper_test_fingerprint"),
                "true_text": summary.get("labels/true_text"),
                "false_text": summary.get("labels/false_text"),
                "true_token_id": summary.get("labels/true_token_id"),
                "false_token_id": summary.get("labels/false_token_id"),
            }
        )
    return descriptors


def is_canonical_original(descriptor):
    return descriptor["template_id"] == "original"


def select_published_run(descriptors, published_lname, published_lnum, published_rho, test_split="paper_test"):
    candidates = [
        descriptor
        for descriptor in descriptors
        if is_canonical_original(descriptor)
        and descriptor["fever_split"] == test_split
        and descriptor["lname"] == published_lname
        and descriptor["lnum"] == published_lnum
        and descriptor["rho"] is not None
        and abs(float(descriptor["rho"]) - float(published_rho)) < 1e-9
    ]
    if not candidates:
        return None

    source_matched = [descriptor for descriptor in candidates if descriptor["intervention_source"] == "published"]
    chosen = source_matched if source_matched else candidates
    chosen.sort(key=lambda descriptor: (descriptor["seed"] is None, descriptor["seed"], descriptor["path"]))
    return chosen[0]


def select_validation_winner(descriptors, validation_split="paper_dev"):
    dev_candidates = [
        descriptor
        for descriptor in descriptors
        if is_canonical_original(descriptor)
        and descriptor["fever_split"] == validation_split
        and descriptor["lname"] != "dont"
    ]
    if not dev_candidates:
        return None
    source_matched = [
        descriptor for descriptor in dev_candidates
        if descriptor["intervention_source"] == "validation_candidate"
    ]
    if source_matched:
        dev_candidates = source_matched

    def sort_key(descriptor):
        accuracy = metric(descriptor["summary"], descriptor["report"], "accuracy", "dev")
        binary_nll = metric(descriptor["summary"], descriptor["report"], "binary_nll", "dev")
        return (
            -(float("-inf") if accuracy is None else accuracy),
            float("inf") if binary_nll is None else binary_nll,
            descriptor["lname"],
            descriptor["lnum"],
            descriptor["rho"],
            descriptor["path"],
        )

    dev_candidates.sort(key=sort_key)
    return dev_candidates[0]


def find_matching_test_run(descriptors, dev_winner, test_split="paper_test"):
    if dev_winner is None:
        return None

    candidates = [
        descriptor
        for descriptor in descriptors
        if is_canonical_original(descriptor)
        and descriptor["fever_split"] == test_split
        and descriptor["lname"] == dev_winner["lname"]
        and descriptor["lnum"] == dev_winner["lnum"]
        and descriptor["rho"] is not None
        and dev_winner["rho"] is not None
        and abs(float(descriptor["rho"]) - float(dev_winner["rho"])) < 1e-9
    ]
    if not candidates:
        return None

    source_matched = [
        descriptor for descriptor in candidates if descriptor["intervention_source"] == "validation_selected"
    ]
    chosen = source_matched if source_matched else candidates
    chosen.sort(key=lambda descriptor: (descriptor["seed"] is None, descriptor["seed"], descriptor["path"]))
    return chosen[0]


def render_row(label, descriptor, split_name):
    if descriptor is None:
        return {
            "label": label,
            "split": split_name,
            "lname": "NA",
            "lnum": "NA",
            "rho": "NA",
            "rate": "NA",
            "seed": "NA",
            "acc": "NA",
            "binary_nll": "NA",
            "note": "missing",
        }

    report_split = "test" if split_name in {"paper_test", "laser_test"} else "dev"
    return {
        "label": label,
        "split": split_name,
        "lname": descriptor["lname"],
        "lnum": descriptor["lnum"],
        "rho": fmt(descriptor["rho"]),
        "rate": fmt(descriptor["rate"]),
        "seed": descriptor["seed"],
        "acc": fmt(metric(descriptor["summary"], descriptor["report"], "accuracy", report_split)),
        "binary_nll": fmt(
            metric(descriptor["summary"], descriptor["report"], "binary_nll", report_split)
        ),
        "note": short_name(descriptor["path"]),
    }


def print_rows(rows, paper_reported_accuracy, paper_reported_note):
    header = (
        "result".ljust(38)
        + " split".ljust(13)
        + " lname".ljust(10)
        + " lnum".rjust(6)
        + " rho".rjust(10)
        + " rate".rjust(10)
        + " seed".rjust(8)
        + " acc".rjust(12)
        + " binary_nll".rjust(14)
        + " note".rjust(18)
    )
    print(header)
    print("-" * len(header))
    paper_acc = "NA" if paper_reported_accuracy is None else fmt(paper_reported_accuracy)
    print(
        "paper-reported result".ljust(38)
        + " paper".ljust(13)
        + "NA".ljust(10)
        + "NA".rjust(6)
        + "NA".rjust(10)
        + "NA".rjust(10)
        + "NA".rjust(8)
        + paper_acc.rjust(12)
        + "NA".rjust(14)
        + paper_reported_note.rjust(18)
    )
    for row in rows:
        print(
            row["label"].ljust(38)
            + row["split"].ljust(13)
            + str(row["lname"]).ljust(10)
            + str(row["lnum"]).rjust(6)
            + str(row["rho"]).rjust(10)
            + str(row["rate"]).rjust(10)
            + str(row["seed"]).rjust(8)
            + str(row["acc"]).rjust(12)
            + str(row["binary_nll"]).rjust(14)
            + short_name(str(row["note"])).rjust(18)
        )


def validation_grid_rows(descriptors, validation_split):
    rows = []
    for descriptor in descriptors:
        if not is_canonical_original(descriptor) or descriptor["fever_split"] != validation_split:
            continue
        report_split = "dev"
        rows.append(
            {
                "lname": descriptor["lname"],
                "lnum": descriptor["lnum"],
                "rho": descriptor["rho"],
                "rate": descriptor["rate"],
                "seed": descriptor["seed"],
                "binary_accuracy": metric(descriptor["summary"], descriptor["report"], "accuracy", report_split),
                "binary_nll": metric(descriptor["summary"], descriptor["report"], "binary_nll", report_split),
                "intervention_source": descriptor["intervention_source"],
                "claim_count": descriptor["report"][report_split]["n"],
                "path": descriptor["path"],
            }
        )
    return sorted(rows, key=lambda row: (row["lname"], row["lnum"], row["rho"], row["path"]))


def validate_grid(rows, expected_runs, expected_claim_count):
    if len(rows) != expected_runs:
        raise ValueError(f"Validation grid has {len(rows)} runs; expected {expected_runs}")
    keys = [(row["lname"], row["lnum"], row["rho"], row["seed"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Validation grid contains duplicate (lname, lnum, rho, seed) configurations")
    observed = {
        (row["lname"], int(row["lnum"]), round(float(row["rho"]), 10), row["seed"])
        for row in rows
    }
    expected = {
        (lname, lnum, round(rho, 10), 0)
        for lname, lnum, rho in itertools.product(
            EXPECTED_GRID_LNAMES,
            EXPECTED_GRID_LNUMS,
            EXPECTED_GRID_RHOS,
        )
    }
    if observed != expected:
        raise ValueError(
            f"Validation grid configuration mismatch: missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )
    for row in rows:
        if row["intervention_source"] != "validation_candidate":
            raise ValueError("Validation grid runs must use intervention_source=validation_candidate")
        if row["claim_count"] != expected_claim_count:
            raise ValueError(
                f"Validation grid run has {row['claim_count']} claims; expected {expected_claim_count}"
            )
        if row["binary_accuracy"] is None or not math.isfinite(float(row["binary_accuracy"])):
            raise ValueError("Validation grid contains an invalid binary accuracy")
        if row["binary_nll"] is None or not math.isfinite(float(row["binary_nll"])):
            raise ValueError("Validation grid contains an invalid binary NLL")


def validate_reproduction_provenance(descriptors, validation_split, test_split, dev_size, test_size):
    relevant = [
        row for row in descriptors
        if row["fever_split"] in {validation_split, test_split} and is_canonical_original(row)
    ]
    if not relevant:
        raise ValueError("No canonical reproduction runs were found")
    required = [
        "model_path",
        "revision",
        "runtime_device",
        "dataset_name",
        "dataset_config",
        "dataset_paper_dev_fingerprint",
        "dataset_paper_test_fingerprint",
        "true_text",
        "false_text",
        "true_token_id",
        "false_token_id",
    ]
    for row in relevant:
        missing = [field for field in required if row.get(field) is None]
        if missing:
            raise ValueError(f"Reproduction run is missing provenance {missing}: {short_name(row['path'])}")
        split_name = "dev" if row["fever_split"] == validation_split else "test"
        expected_count = dev_size if split_name == "dev" else test_size
        if row["report"][split_name]["n"] != expected_count:
            raise ValueError(
                f"Reproduction run has {row['report'][split_name]['n']} claims; expected {expected_count}: "
                f"{short_name(row['path'])}"
            )
        accuracy = metric(row["summary"], row["report"], "accuracy", split_name)
        binary_nll = metric(row["summary"], row["report"], "binary_nll", split_name)
        if accuracy is None or not math.isfinite(float(accuracy)) or not 0 <= float(accuracy) <= 100:
            raise ValueError(f"Reproduction run has invalid binary accuracy: {short_name(row['path'])}")
        if binary_nll is None or not math.isfinite(float(binary_nll)) or float(binary_nll) < 0:
            raise ValueError(f"Reproduction run has invalid binary NLL: {short_name(row['path'])}")

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
        "true_text",
        "false_text",
        "true_token_id",
        "false_token_id",
    ]
    signatures = {tuple(row.get(field) for field in shared_fields) for row in relevant}
    if len(signatures) != 1:
        raise ValueError("Reproduction runs mix model, tokenizer, dataset, device, or label provenance")
    return dict(zip(shared_fields, next(iter(signatures))))


def export_results(export_dir, paper_row, rows, grid_rows, metadata):
    os.makedirs(export_dir, exist_ok=True)
    summary_rows = [paper_row] + rows
    summary_fields = ["label", "split", "lname", "lnum", "rho", "rate", "seed", "acc", "binary_nll", "note"]
    with open(os.path.join(export_dir, "reproduction_summary.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    grid_fields = [
        "lname", "lnum", "rho", "rate", "seed", "binary_accuracy", "binary_nll",
        "intervention_source", "claim_count", "path",
    ]
    with open(os.path.join(export_dir, "validation_grid.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=grid_fields)
        writer.writeheader()
        writer.writerows(grid_rows)

    with open(os.path.join(export_dir, "reproduction_summary.json"), "w") as handle:
        json.dump(
            {"metadata": metadata, "summary_rows": summary_rows, "validation_grid": grid_rows},
            handle,
            indent=2,
            sort_keys=True,
        )


def main():
    parser = argparse.ArgumentParser(description="Summarize paper-aligned FEVER reproduction runs")
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--dev-size", type=int, default=6510)
    parser.add_argument("--test-size", type=int, default=6576)
    parser.add_argument("--published-lname", default="fc_in")
    parser.add_argument("--published-lnum", type=int, default=24)
    parser.add_argument("--published-rho", type=float, default=0.01)
    parser.add_argument("--validation-split", default="paper_dev")
    parser.add_argument("--test-split", default="paper_test")
    parser.add_argument("--paper-reported-accuracy", type=float, default=56.2)
    parser.add_argument("--paper-reported-note", default="paper baseline 50.2; LASER 56.2")
    parser.add_argument("--export-dir", default=None)
    parser.add_argument("--require-complete-grid", action="store_true")
    parser.add_argument("--expected-grid-runs", type=int, default=72)
    args = parser.parse_args()

    paths = recursive_prediction_files(args.results_root)
    descriptors = build_descriptors(paths, dev_size=args.dev_size, test_size=args.test_size)
    published = select_published_run(
        descriptors,
        args.published_lname,
        args.published_lnum,
        args.published_rho,
        test_split=args.test_split,
    )
    validation_winner = select_validation_winner(descriptors, validation_split=args.validation_split)
    validation_test = find_matching_test_run(descriptors, validation_winner, test_split=args.test_split)
    grid_rows = validation_grid_rows(descriptors, args.validation_split)
    if args.require_complete_grid:
        validate_grid(grid_rows, args.expected_grid_runs, args.dev_size)
        shared_provenance = validate_reproduction_provenance(
            descriptors,
            args.validation_split,
            args.test_split,
            args.dev_size,
            args.test_size,
        )
        if published is None:
            raise ValueError("Published intervention test run is missing")
        if validation_winner is None:
            raise ValueError("Validation winner is missing")
        if validation_test is None:
            raise ValueError("Validation-selected test run is missing")
        if published["intervention_source"] != "published":
            raise ValueError("Published intervention test run has incorrect provenance")
        if (
            validation_test["intervention_source"] != "validation_selected"
            and validation_test["path"] != published["path"]
        ):
            raise ValueError("Validation-selected test run has incorrect provenance")
    else:
        shared_provenance = None

    print(f"RESULTS_ROOT: {short_name(args.results_root)}")
    print(f"FOUND_RUNS: {len(descriptors)}")
    if validation_winner is not None:
        print(
            "VALIDATION_WINNER: "
            f"lname={validation_winner['lname']} "
            f"lnum={validation_winner['lnum']} "
            f"rho={fmt(validation_winner['rho'])} "
            f"acc={fmt(metric(validation_winner['summary'], validation_winner['report'], 'accuracy', 'dev'))} "
            f"binary_nll={fmt(metric(validation_winner['summary'], validation_winner['report'], 'binary_nll', 'dev'))}"
        )
    else:
        print("VALIDATION_WINNER: missing")

    rows = [
        render_row("published intervention on our split", published, args.test_split),
        render_row("validation-selected intervention", validation_test, args.test_split),
    ]
    print_rows(rows, args.paper_reported_accuracy, args.paper_reported_note)
    print(
        "\nNOTE: treat this as a paper-aligned evaluation on the repo FEVER split, "
        "not an exact reproduction unless split, preprocessing, and intervention all match the paper."
    )
    if args.export_dir is not None:
        paper_row = {
            "label": "paper-reported result",
            "split": "paper",
            "lname": "NA",
            "lnum": "NA",
            "rho": "NA",
            "rate": "NA",
            "seed": "NA",
            "acc": args.paper_reported_accuracy,
            "binary_nll": "NA",
            "note": args.paper_reported_note,
        }
        export_results(
            args.export_dir,
            paper_row,
            rows,
            grid_rows,
            metadata={
                "results_root": args.results_root,
                "validation_split": args.validation_split,
                "test_split": args.test_split,
                "dev_size": args.dev_size,
                "test_size": args.test_size,
                "paper_reported_accuracy": args.paper_reported_accuracy,
                "paper_reported_note": args.paper_reported_note,
                "validation_winner": None if validation_winner is None else {
                    "lname": validation_winner["lname"],
                    "lnum": validation_winner["lnum"],
                    "rho": validation_winner["rho"],
                    "rate": validation_winner["rate"],
                    "seed": validation_winner["seed"],
                },
                "shared_provenance": shared_provenance,
            },
        )
        print(f"EXPORT_DIR: {short_name(args.export_dir)}")


if __name__ == "__main__":
    main()
