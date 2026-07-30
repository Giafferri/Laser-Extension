"""Report split-aware metrics for individual FEVER result files."""

import argparse
import os
import pickle
import re


def prediction_to_summary_path(path):
    dirname, filename = os.path.split(path)
    summary_name = filename.replace("-predictions-", "-result-summary-")
    if summary_name.endswith(".p"):
        summary_name = summary_name[:-2] + ".pkl"
    return os.path.join(dirname, summary_name)


def load_summary(path):
    summary_path = prediction_to_summary_path(path)
    if not os.path.exists(summary_path):
        return {}
    with open(summary_path, "rb") as f:
        return pickle.load(f)


def agg(rows):
    if not rows:
        return {
            "n": 0,
            "accuracy": None,
            "mean_log_prob": None,
            "mean_binary_nll": None,
            "top1": None,
            "top5": None,
            "top10": None,
        }

    n = len(rows)
    accuracy_key = "binary_correct" if "binary_correct" in rows[0] else "correct"
    mean_binary_nll = None
    if "binary_nll_normalized" in rows[0]:
        mean_binary_nll = sum(float(r["binary_nll_normalized"]) for r in rows) / n

    return {
        "n": n,
        "accuracy": 100.0 * sum(float(r[accuracy_key]) for r in rows) / n,
        "mean_log_prob": sum(float(r["answer_logprob"]) for r in rows) / n,
        "mean_binary_nll": mean_binary_nll,
        "top1": 100.0 * sum(float(r["top_1_acc"]) for r in rows) / n,
        "top5": 100.0 * sum(float(r["top_5_acc"]) for r in rows) / n,
        "top10": 100.0 * sum(float(r["top_10_acc"]) for r in rows) / n,
    }


def fmt(value):
    if value is None:
        return "NA"
    return f"{value:.4f}"


def short_name(path):
    return path.replace(os.getcwd() + os.sep, "") if path.startswith(os.getcwd() + os.sep) else path


def parse_slice_from_path(path):
    match = re.search(r"-s(\d+)-n(all|\d+)\.p$", path)
    if not match:
        return 0, None

    start_index = int(match.group(1))
    max_examples = None if match.group(2) == "all" else int(match.group(2))
    return start_index, max_examples


def classify_row_split(row, local_ix, start_index, dev_size, test_size, summary):
    evaluation_split = row.get("evaluation_split")
    if evaluation_split in {"paper_dev", "laser_dev"}:
        return "dev"
    if evaluation_split in {"paper_test", "laser_test"}:
        return "test"

    declared_split = summary.get("args/fever_split") or summary.get("fever/returned_split")
    if declared_split in {"paper_dev", "laser_dev"}:
        return "dev"
    if declared_split in {"paper_test", "laser_test"}:
        return "test"

    split_name = row.get("paper_split")
    if split_name == "paper_dev":
        return "dev"
    if split_name == "paper_test":
        return "test"

    global_ix = row.get("dataset_global_ix")
    if global_ix is None:
        global_ix = start_index + local_ix

    if global_ix < dev_size:
        return "dev"
    if global_ix < dev_size + test_size:
        return "test"

    return "all"


def report_file(path, dev_size, test_size):
    with open(path, "rb") as f:
        preds = pickle.load(f)

    summary = load_summary(path)
    total = len(preds)
    start_index, max_examples = parse_slice_from_path(path)

    dev_rows = []
    test_rows = []
    other_rows = []
    for local_ix, row in enumerate(preds):
        split_name = classify_row_split(row, local_ix, start_index, dev_size, test_size, summary)
        if split_name == "dev":
            dev_rows.append(row)
        elif split_name == "test":
            test_rows.append(row)
        else:
            other_rows.append(row)

    dev_stats = agg(dev_rows)
    test_stats = agg(test_rows)
    all_stats = agg(preds)

    return {
        "path": path,
        "summary": summary,
        "total": total,
        "start_index": start_index,
        "max_examples": max_examples,
        "dev": dev_stats,
        "test": test_stats,
        "all": all_stats,
        "other": agg(other_rows),
    }


def print_detailed(report):
    print(f"\nFILE: {short_name(report['path'])}")
    print(f"TOTAL_EXAMPLES: {report['total']}")
    print(
        f"ALL   n={report['all']['n']} "
        f"acc={fmt(report['all']['accuracy'])} "
        f"logprob={fmt(report['all']['mean_log_prob'])} "
        f"binary_nll={fmt(report['all']['mean_binary_nll'])} "
        f"top1={fmt(report['all']['top1'])} "
        f"top5={fmt(report['all']['top5'])} "
        f"top10={fmt(report['all']['top10'])}"
    )
    print(
        f"DEV   n={report['dev']['n']} "
        f"acc={fmt(report['dev']['accuracy'])} "
        f"logprob={fmt(report['dev']['mean_log_prob'])} "
        f"binary_nll={fmt(report['dev']['mean_binary_nll'])} "
        f"top1={fmt(report['dev']['top1'])} "
        f"top5={fmt(report['dev']['top5'])} "
        f"top10={fmt(report['dev']['top10'])}"
    )
    print(
        f"TEST  n={report['test']['n']} "
        f"acc={fmt(report['test']['accuracy'])} "
        f"logprob={fmt(report['test']['mean_log_prob'])} "
        f"binary_nll={fmt(report['test']['mean_binary_nll'])} "
        f"top1={fmt(report['test']['top1'])} "
        f"top5={fmt(report['test']['top5'])} "
        f"top10={fmt(report['test']['top10'])}"
    )


def print_table(reports):
    header = (
        "run".ljust(70)
        + " total".rjust(8)
        + " dev_n".rjust(8)
        + " dev_acc".rjust(12)
        + " dev_bnll".rjust(12)
        + " test_n".rjust(8)
        + " test_acc".rjust(12)
        + " test_bnll".rjust(12)
    )
    print(header)
    print("-" * len(header))
    for report in reports:
        print(
            short_name(report["path"]).ljust(70)
            + str(report["total"]).rjust(8)
            + str(report["dev"]["n"]).rjust(8)
            + fmt(report["dev"]["accuracy"]).rjust(12)
            + fmt(report["dev"]["mean_binary_nll"]).rjust(12)
            + str(report["test"]["n"]).rjust(8)
            + fmt(report["test"]["accuracy"]).rjust(12)
            + fmt(report["test"]["mean_binary_nll"]).rjust(12)
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prediction_files", nargs="+")
    parser.add_argument("--dev-size", type=int, default=6510)
    parser.add_argument("--test-size", type=int, default=6576)
    parser.add_argument(
        "--sort",
        choices=["dev_acc", "dev_logprob", "dev_bnll", "test_acc", "test_logprob", "test_bnll", "path"],
        default="path",
    )
    args = parser.parse_args()

    dev_size, test_size = args.dev_size, args.test_size
    print(f"FEVER_SPLITS: paper_dev={dev_size} paper_test={test_size}")

    reports = [report_file(path, dev_size, test_size) for path in args.prediction_files]

    if args.sort == "dev_acc":
        reports.sort(key=lambda x: (-1e18 if x["dev"]["accuracy"] is None else -x["dev"]["accuracy"], x["path"]))
    elif args.sort == "dev_logprob":
        reports.sort(key=lambda x: (-1e18 if x["dev"]["mean_log_prob"] is None else -x["dev"]["mean_log_prob"], x["path"]))
    elif args.sort == "dev_bnll":
        reports.sort(key=lambda x: (1e18 if x["dev"]["mean_binary_nll"] is None else x["dev"]["mean_binary_nll"], x["path"]))
    elif args.sort == "test_acc":
        reports.sort(key=lambda x: (-1e18 if x["test"]["accuracy"] is None else -x["test"]["accuracy"], x["path"]))
    elif args.sort == "test_logprob":
        reports.sort(key=lambda x: (-1e18 if x["test"]["mean_log_prob"] is None else -x["test"]["mean_log_prob"], x["path"]))
    elif args.sort == "test_bnll":
        reports.sort(key=lambda x: (1e18 if x["test"]["mean_binary_nll"] is None else x["test"]["mean_binary_nll"], x["path"]))
    else:
        reports.sort(key=lambda x: x["path"])

    if len(reports) == 1:
        print_detailed(reports[0])
    else:
        print_table(reports)


if __name__ == "__main__":
    main()
