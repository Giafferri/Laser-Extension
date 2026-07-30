"""Generate publication figures from FEVER analysis exports."""

import argparse
import csv
import json
import math
import os
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


DEFAULT_CATEGORY_ORDER = ["lex", "syn", "inst", "slot", "verb"]
PALETTE = {
    "lex": "#1b9e77",
    "syn": "#1f78b4",
    "inst": "#ff8c42",
    "slot": "#d62728",
    "verb": "#808000",
    "original": "#111111",
}

LANG_TEXT = {
    "en": {
        "categories": {
            "lex": "Lexical",
            "syn": "Syntactic",
            "inst": "Instruction",
            "slot": "Answer Slot",
            "verb": "Verbosity",
            "original": "Original",
        },
        "baseline": "Baseline",
        "laser": "LASER",
        "accuracy": "Accuracy (%)",
        "gain": "LASER - Baseline (pts)",
        "category": "Category",
        "prompt": "Prompt",
        "divergence": "Normalized divergence from original",
        "mean_accuracy_title": "Fixed-Intervention Prompt Transfer: Mean Accuracy by Prompt Category",
        "mean_gain_title": "Fixed-Intervention Prompt Transfer: Mean LASER Gain by Prompt Category",
        "scatter_title": "Fixed-Intervention Prompt Transfer: Baseline vs LASER Accuracy",
        "ranked_gain_title": "Fixed-Intervention Prompt Transfer: Prompt-Level LASER Gain",
        "distribution_title": "Fixed-Intervention Prompt Transfer: Gain Distribution by Category",
        "divergence_title": "Fixed-Intervention Prompt Transfer: Divergence vs LASER Gain",
        "seed_title": "SVD Seed Sensitivity of Mean Prompt Transfer",
        "sign_title": "Prompt-Level Gain Sign Stability Across SVD Seeds",
        "nll_title": "Normalized Binary NLL: Baseline vs LASER",
        "claim_title": "Claim-Level Mean Transfer Gain Across the Fixed Prompt Suite",
        "binary_nll": "Normalized binary NLL",
        "claims": "Claims",
        "diag_label": "y = x",
        "reg_label": "Regression line",
        "original_label": "Original",
    },
    "fr": {
        "categories": {
            "lex": "Lexical",
            "syn": "Syntaxique",
            "inst": "Instruction",
            "slot": "Slot de reponse",
            "verb": "Verbosite",
            "original": "Original",
        },
        "baseline": "Baseline",
        "laser": "LASER",
        "accuracy": "Accuracy (%)",
        "gain": "LASER - Baseline (pts)",
        "category": "Categorie",
        "prompt": "Prompt",
        "divergence": "Divergence normalisee depuis l'original",
        "mean_accuracy_title": "Transfert de prompt a intervention figee : accuracy moyenne par categorie",
        "mean_gain_title": "Transfert de prompt a intervention figee : gain moyen LASER par categorie",
        "scatter_title": "Transfert de prompt a intervention figee : baseline vs LASER",
        "ranked_gain_title": "Transfert de prompt a intervention figee : gain LASER par prompt",
        "distribution_title": "Transfert de prompt a intervention figee : distribution des gains",
        "divergence_title": "Transfert de prompt a intervention figee : divergence vs gain LASER",
        "seed_title": "Sensibilite du transfert moyen aux seeds SVD",
        "sign_title": "Stabilite du signe du gain entre seeds SVD",
        "nll_title": "NLL binaire normalisee : baseline vs LASER",
        "claim_title": "Gain moyen par claim sur la suite fixe de prompts",
        "binary_nll": "NLL binaire normalisee",
        "claims": "Claims",
        "diag_label": "y = x",
        "reg_label": "Droite de regression",
        "original_label": "Original",
    },
}


def fail(message):
    raise SystemExit(f"ERROR: {message}")


def load_csv_rows(path):
    if not os.path.exists(path):
        fail(f"Missing required file: {path}")
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_optional_csv_rows(path):
    if not os.path.exists(path):
        return None
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_json(path):
    if not os.path.exists(path):
        fail(f"Missing required file: {path}")
    with open(path) as f:
        return json.load(f)


def validate_columns(rows, required_columns, label):
    if not rows:
        fail(f"{label} is empty")
    actual = set(rows[0].keys())
    missing = [column for column in required_columns if column not in actual]
    if missing:
        fail(f"{label} is missing required columns: {', '.join(missing)}")


def to_bool(value):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    fail(f"Could not parse boolean value: {value}")


def to_int(value, field_name):
    try:
        return int(value)
    except Exception as exc:
        fail(f"Could not parse integer field {field_name}: {value} ({exc})")


def to_float(value, field_name):
    try:
        return float(value)
    except Exception as exc:
        fail(f"Could not parse numeric field {field_name}: {value} ({exc})")


def to_optional_float(value, field_name):
    if value in {None, ""}:
        return None
    return to_float(value, field_name)


def parse_template_rows(rows):
    parsed = []
    for row in rows:
        parsed.append(
            {
                "template": row["template"],
                "category_id": row["category_id"],
                "category_label": row.get("category_label"),
                "variant_index": to_int(row["variant_index"], "variant_index"),
                "is_original": to_bool(row["is_original"]),
                "split": row["split"],
                "wrapper_char_len": to_float(row["wrapper_char_len"], "wrapper_char_len"),
                "wrapper_word_len": to_float(row["wrapper_word_len"], "wrapper_word_len"),
                "divergence_char_norm": to_float(row["divergence_char_norm"], "divergence_char_norm"),
                "divergence_token_jaccard": to_float(row["divergence_token_jaccard"], "divergence_token_jaccard"),
                "baseline_n": to_int(row["baseline_n"], "baseline_n"),
                "baseline_acc": to_float(row["baseline_acc"], "baseline_acc"),
                "baseline_logp": to_float(row["baseline_logp"], "baseline_logp"),
                "laser_n": to_int(row["laser_n"], "laser_n"),
                "laser_acc": to_float(row["laser_acc"], "laser_acc"),
                "laser_logp": to_float(row["laser_logp"], "laser_logp"),
                "baseline_vs_original_acc": to_float(row["baseline_vs_original_acc"], "baseline_vs_original_acc"),
                "laser_vs_original_baseline_acc": to_float(
                    row["laser_vs_original_baseline_acc"], "laser_vs_original_baseline_acc"
                ),
                "laser_vs_same_template_acc": to_float(
                    row["laser_vs_same_template_acc"], "laser_vs_same_template_acc"
                ),
                "baseline_binary_nll": to_optional_float(row.get("baseline_binary_nll"), "baseline_binary_nll"),
                "laser_binary_nll": to_optional_float(row.get("laser_binary_nll"), "laser_binary_nll"),
                "sign_stability": row.get("sign_stability"),
            }
        )
    return parsed


def parse_seed_rows(rows):
    return [
        {
            "seed": to_int(row["seed"], "seed"),
            "prompt_count": to_int(row["prompt_count"], "prompt_count"),
            "gain_mean_acc": to_float(row["gain_mean_acc"], "gain_mean_acc"),
            "gain_std_acc": to_float(row["gain_std_acc"], "gain_std_acc"),
        }
        for row in rows
    ]


def parse_claim_rows(rows):
    return [
        {
            "claim_id": to_int(row["claim_id"], "claim_id"),
            "gain": to_float(row["gain_prompt_seed_mean_accuracy"], "gain_prompt_seed_mean_accuracy"),
        }
        for row in rows
    ]


def parse_category_rows(rows):
    parsed = []
    for row in rows:
        parsed.append(
            {
                "category_id": row["category_id"],
                "category_label": row.get("category_label"),
                "n": to_int(row["n"], "n"),
                "baseline_mean_acc": to_float(row["baseline_mean_acc"], "baseline_mean_acc"),
                "baseline_std_acc": to_float(row["baseline_std_acc"], "baseline_std_acc"),
                "laser_mean_acc": to_float(row["laser_mean_acc"], "laser_mean_acc"),
                "laser_std_acc": to_float(row["laser_std_acc"], "laser_std_acc"),
                "gain_mean_acc": to_float(row["gain_mean_acc"], "gain_mean_acc"),
                "gain_std_acc": to_float(row["gain_std_acc"], "gain_std_acc"),
            }
        )
    return parsed


def localize_category(category_id, language):
    return LANG_TEXT[language]["categories"].get(category_id, category_id)


def category_sort_key(category_id):
    if category_id in DEFAULT_CATEGORY_ORDER:
        return (DEFAULT_CATEGORY_ORDER.index(category_id), category_id)
    if category_id == "original":
        return (len(DEFAULT_CATEGORY_ORDER), category_id)
    return (len(DEFAULT_CATEGORY_ORDER) + 1, category_id)


def validate_consistency(template_rows, category_rows, summary):
    paraphrase_rows = [row for row in template_rows if not row["is_original"]]
    if not paraphrase_rows:
        fail("No paraphrase rows found in template_metrics.csv")

    template_categories = {}
    for row in paraphrase_rows:
        template_categories.setdefault(row["category_id"], set()).add(row.get("category_label"))

    category_categories = {}
    for row in category_rows:
        category_categories[row["category_id"]] = row.get("category_label")

    if set(template_categories) != set(category_categories):
        fail(
            "Category ids are inconsistent between template_metrics.csv and category_metrics.csv: "
            f"template={sorted(template_categories)} category={sorted(category_categories)}"
        )

    for category_id, labels in template_categories.items():
        normalized_labels = {label for label in labels if label not in {None, ""}}
        category_label = category_categories.get(category_id)
        if normalized_labels and category_label not in normalized_labels:
            fail(
                f"Category label mismatch for {category_id}: template_metrics={sorted(normalized_labels)} "
                f"category_metrics={category_label}"
            )

    summary_count = summary.get("global_summary", {}).get("paraphrase_count")
    if summary_count is not None and summary_count != len(paraphrase_rows):
        fail(
            f"summary.json paraphrase_count={summary_count} does not match template_metrics paraphrase count={len(paraphrase_rows)}"
        )


def apply_style(style_name):
    if style_name == "paper":
        plt.rcParams.update(
            {
                "font.size": 11,
                "axes.titlesize": 13,
                "axes.labelsize": 11,
                "legend.fontsize": 10,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
                "figure.titlesize": 14,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "axes.grid": False,
            }
        )
        return

    if style_name == "presentation":
        plt.rcParams.update(
            {
                "font.size": 13,
                "axes.titlesize": 15,
                "axes.labelsize": 13,
                "legend.fontsize": 11,
                "xtick.labelsize": 11,
                "ytick.labelsize": 11,
                "figure.titlesize": 16,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "axes.grid": False,
            }
        )
        return

    fail(f"Unknown style: {style_name}")


def save_figure(fig, output_dir, basename, formats, dpi, transparent):
    saved_paths = []
    for fmt in formats:
        out_path = os.path.join(output_dir, f"{basename}.{fmt}")
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight", transparent=transparent)
        saved_paths.append(out_path)
        print(f"SAVED {out_path}")
    plt.close(fig)
    return saved_paths


def category_rows_in_order(category_rows):
    return sorted(category_rows, key=lambda row: category_sort_key(row["category_id"]))


def paraphrase_rows(template_rows):
    return [row for row in template_rows if not row["is_original"]]


def regression_line(xs, ys):
    if len(xs) < 2:
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    denom = sum((x - mean_x) ** 2 for x in xs)
    if math.isclose(denom, 0.0):
        return None
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denom
    intercept = mean_y - slope * mean_x
    return slope, intercept


def category_handles(language):
    return [
        Patch(facecolor=PALETTE[category_id], edgecolor="none", label=localize_category(category_id, language))
        for category_id in DEFAULT_CATEGORY_ORDER
    ] + [Patch(facecolor=PALETTE["original"], edgecolor="none", label=LANG_TEXT[language]["original_label"])]


def plot_category_mean_accuracy(category_rows, language):
    text = LANG_TEXT[language]
    ordered = category_rows_in_order(category_rows)
    labels = [localize_category(row["category_id"], language) for row in ordered]
    baseline = [row["baseline_mean_acc"] for row in ordered]
    baseline_std = [row["baseline_std_acc"] for row in ordered]
    laser = [row["laser_mean_acc"] for row in ordered]
    laser_std = [row["laser_std_acc"] for row in ordered]

    x = list(range(len(ordered)))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    ax.bar(
        [value - width / 2 for value in x],
        baseline,
        width=width,
        yerr=baseline_std,
        color="#bfbfbf",
        edgecolor="black",
        linewidth=0.6,
        capsize=4,
        label=text["baseline"],
    )
    ax.bar(
        [value + width / 2 for value in x],
        laser,
        width=width,
        yerr=laser_std,
        color="#4c78a8",
        edgecolor="black",
        linewidth=0.6,
        capsize=4,
        label=text["laser"],
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(text["accuracy"])
    ax.set_xlabel(text["category"])
    ax.set_title(text["mean_accuracy_title"])
    ax.legend(frameon=False)
    ax.set_ylim(min(baseline + laser) - 3.0, max(baseline + laser) + 4.5)
    fig.tight_layout()
    return fig


def plot_category_mean_gain(category_rows, language):
    text = LANG_TEXT[language]
    ordered = category_rows_in_order(category_rows)
    labels = [localize_category(row["category_id"], language) for row in ordered]
    gains = [row["gain_mean_acc"] for row in ordered]
    gain_std = [row["gain_std_acc"] for row in ordered]
    colors = [PALETTE[row["category_id"]] for row in ordered]

    x = list(range(len(ordered)))
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    ax.bar(
        x,
        gains,
        yerr=gain_std,
        color=colors,
        edgecolor="black",
        linewidth=0.6,
        capsize=4,
    )
    ax.axhline(0.0, color="#444444", linewidth=1.0, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(text["gain"])
    ax.set_xlabel(text["category"])
    ax.set_title(text["mean_gain_title"])
    fig.tight_layout()
    return fig


def plot_prompt_baseline_vs_laser_scatter(template_rows, language):
    text = LANG_TEXT[language]
    fig, ax = plt.subplots(figsize=(7.6, 7.0))

    grouped = {}
    for row in template_rows:
        grouped.setdefault(row["category_id"], []).append(row)

    legend_handles = []
    all_values = []
    for category_id in DEFAULT_CATEGORY_ORDER + ["original"]:
        rows = grouped.get(category_id, [])
        if not rows:
            continue
        xs = [row["baseline_acc"] for row in rows]
        ys = [row["laser_acc"] for row in rows]
        all_values.extend(xs)
        all_values.extend(ys)
        marker = "*" if category_id == "original" else "o"
        size = 170 if category_id == "original" else 62
        edge_width = 1.0 if category_id == "original" else 0.7
        ax.scatter(
            xs,
            ys,
            s=size,
            c=PALETTE[category_id],
            marker=marker,
            edgecolors="white",
            linewidths=edge_width,
            alpha=0.95,
            zorder=3 if category_id == "original" else 2,
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker=marker,
                linestyle="",
                markerfacecolor=PALETTE[category_id],
                markeredgecolor="white",
                markersize=11 if category_id == "original" else 8,
                label=localize_category(category_id, language),
            )
        )

    original_row = next((row for row in template_rows if row["is_original"]), None)
    if original_row is not None:
        ax.annotate(
            text["original_label"],
            (original_row["baseline_acc"], original_row["laser_acc"]),
            xytext=(8, -12),
            textcoords="offset points",
            fontsize=10,
            color=PALETTE["original"],
        )

    lo = min(all_values) - 1.0
    hi = max(all_values) + 1.0
    ax.plot([lo, hi], [lo, hi], color="#666666", linestyle="--", linewidth=1.2, label=text["diag_label"])
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(f"{text['baseline']} {text['accuracy']}")
    ax.set_ylabel(f"{text['laser']} {text['accuracy']}")
    ax.set_title(text["scatter_title"])
    ax.legend(handles=legend_handles + [Line2D([0, 1], [0, 1], color="#666666", linestyle="--", label=text["diag_label"])], frameon=False, loc="best")
    fig.tight_layout()
    return fig


def plot_prompt_gain_ranked(template_rows, language, sort_gain):
    text = LANG_TEXT[language]
    rows = paraphrase_rows(template_rows)
    reverse = sort_gain == "desc"
    rows = sorted(rows, key=lambda row: row["laser_vs_same_template_acc"], reverse=reverse)
    labels = [row["template"] for row in rows]
    gains = [row["laser_vs_same_template_acc"] for row in rows]
    colors = [PALETTE[row["category_id"]] for row in rows]
    positions = list(range(len(rows)))

    fig, ax = plt.subplots(figsize=(10.5, 12.8))
    ax.barh(positions, gains, color=colors, edgecolor="black", linewidth=0.4)
    ax.axvline(0.0, color="#444444", linewidth=1.0, linestyle="--")
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    if reverse:
        ax.invert_yaxis()
    ax.set_xlabel(text["gain"])
    ax.set_ylabel(text["prompt"])
    ax.set_title(text["ranked_gain_title"])
    ax.legend(handles=category_handles(language)[:-1], frameon=False, loc="lower right")
    fig.tight_layout()
    return fig


def plot_category_gain_distribution(template_rows, language):
    text = LANG_TEXT[language]
    rows = paraphrase_rows(template_rows)
    grouped = {category_id: [] for category_id in DEFAULT_CATEGORY_ORDER}
    for row in rows:
        grouped[row["category_id"]].append(row["laser_vs_same_template_acc"])

    labels = [localize_category(category_id, language) for category_id in DEFAULT_CATEGORY_ORDER]
    data = [grouped[category_id] for category_id in DEFAULT_CATEGORY_ORDER]
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    box = ax.boxplot(
        data,
        patch_artist=True,
        widths=0.58,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.2},
    )
    for patch, category_id in zip(box["boxes"], DEFAULT_CATEGORY_ORDER):
        patch.set_facecolor(PALETTE[category_id])
        patch.set_alpha(0.55)
        patch.set_edgecolor("black")

    rng = random.Random(0)
    for index, category_id in enumerate(DEFAULT_CATEGORY_ORDER, start=1):
        ys = grouped[category_id]
        xs = [index + rng.uniform(-0.14, 0.14) for _ in ys]
        ax.scatter(xs, ys, s=24, color=PALETTE[category_id], edgecolors="white", linewidths=0.4, alpha=0.9)

    ax.axhline(0.0, color="#444444", linewidth=1.0, linestyle="--")
    ax.set_xticks(list(range(1, len(labels) + 1)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(text["gain"])
    ax.set_xlabel(text["category"])
    ax.set_title(text["distribution_title"])
    fig.tight_layout()
    return fig


def top_and_bottom_rows(rows, count=3):
    ranked = sorted(rows, key=lambda row: row["laser_vs_same_template_acc"])
    bottom = ranked[:count]
    top = ranked[-count:]
    selected = []
    seen = set()
    for row in top + bottom:
        if row["template"] in seen:
            continue
        seen.add(row["template"])
        selected.append(row)
    return selected


def plot_divergence_vs_gain(template_rows, language, with_regression_line):
    text = LANG_TEXT[language]
    rows = paraphrase_rows(template_rows)
    fig, ax = plt.subplots(figsize=(8.0, 5.8))

    xs_all = []
    ys_all = []
    legend_handles = []
    for category_id in DEFAULT_CATEGORY_ORDER:
        cat_rows = [row for row in rows if row["category_id"] == category_id]
        if not cat_rows:
            continue
        xs = [row["divergence_char_norm"] for row in cat_rows]
        ys = [row["laser_vs_same_template_acc"] for row in cat_rows]
        xs_all.extend(xs)
        ys_all.extend(ys)
        ax.scatter(
            xs,
            ys,
            s=58,
            color=PALETTE[category_id],
            edgecolors="white",
            linewidths=0.6,
            alpha=0.92,
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markerfacecolor=PALETTE[category_id],
                markeredgecolor="white",
                markersize=8,
                label=localize_category(category_id, language),
            )
        )

    ax.axhline(0.0, color="#444444", linewidth=1.0, linestyle="--")

    if with_regression_line:
        coeffs = regression_line(xs_all, ys_all)
        if coeffs is not None:
            slope, intercept = coeffs
            x_min = min(xs_all)
            x_max = max(xs_all)
            x_line = [x_min, x_max]
            y_line = [slope * value + intercept for value in x_line]
            ax.plot(x_line, y_line, color="#333333", linewidth=1.3, linestyle="-", label=text["reg_label"])

    for row in top_and_bottom_rows(rows, count=3):
        ax.annotate(
            row["template"],
            (row["divergence_char_norm"], row["laser_vs_same_template_acc"]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=9,
        )

    ax.set_xlabel(text["divergence"])
    ax.set_ylabel(text["gain"])
    ax.set_title(text["divergence_title"])
    if with_regression_line:
        legend_handles.append(Line2D([0, 1], [0, 0], color="#333333", linewidth=1.3, label=text["reg_label"]))
    ax.legend(handles=legend_handles, frameon=False, loc="best")
    fig.tight_layout()
    return fig


def plot_seed_sensitivity(seed_rows, language):
    text = LANG_TEXT[language]
    ordered = sorted(seed_rows, key=lambda row: row["seed"])
    labels = [str(row["seed"]) for row in ordered]
    gains = [row["gain_mean_acc"] for row in ordered]
    spreads = [row["gain_std_acc"] for row in ordered]
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    ax.errorbar(
        labels,
        gains,
        yerr=spreads,
        fmt="o-",
        color="#1f78b4",
        ecolor="#7aa6c2",
        capsize=4,
        linewidth=1.5,
        markersize=7,
    )
    ax.axhline(0.0, color="#444444", linewidth=1.0, linestyle="--")
    ax.set_xlabel("SVD seed")
    ax.set_ylabel(text["gain"])
    ax.set_title(text["seed_title"])
    ax.text(
        0.99,
        0.02,
        "Error bars: SD across the 40 fixed prompts",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout()
    return fig


def plot_sign_stability(template_rows, language):
    text = LANG_TEXT[language]
    rows = paraphrase_rows(template_rows)
    order = ["positive", "negative", "mixed", "zero"]
    counts = [sum(row["sign_stability"] == label for row in rows) for label in order]
    colors = ["#1b9e77", "#d62728", "#ff8c42", "#808080"]
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    bars = ax.bar(order, counts, color=colors, edgecolor="black", linewidth=0.6)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, count + 0.25, str(count), ha="center", va="bottom")
    ax.set_ylabel("Prompts")
    ax.set_title(text["sign_title"])
    ax.set_ylim(0, max(counts + [1]) + 3)
    fig.tight_layout()
    return fig


def plot_binary_nll_scatter(template_rows, language):
    text = LANG_TEXT[language]
    fig, ax = plt.subplots(figsize=(7.2, 6.6))
    values = []
    for category_id in DEFAULT_CATEGORY_ORDER + ["original"]:
        rows = [row for row in template_rows if row["category_id"] == category_id]
        if not rows:
            continue
        xs = [row["baseline_binary_nll"] for row in rows]
        ys = [row["laser_binary_nll"] for row in rows]
        values.extend(xs + ys)
        ax.scatter(
            xs,
            ys,
            color=PALETTE[category_id],
            s=150 if category_id == "original" else 58,
            marker="*" if category_id == "original" else "o",
            edgecolors="white",
            linewidths=0.7,
            label=localize_category(category_id, language),
        )
    lo, hi = min(values) - 0.02, max(values) + 0.02
    ax.plot([lo, hi], [lo, hi], color="#666666", linestyle="--", linewidth=1.1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(f"{text['baseline']} {text['binary_nll']}")
    ax.set_ylabel(f"{text['laser']} {text['binary_nll']}")
    ax.set_title(text["nll_title"])
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    return fig


def plot_claim_gain_distribution(claim_rows, language):
    text = LANG_TEXT[language]
    gains = [row["gain"] for row in claim_rows]
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.hist(gains, bins=35, color="#4c78a8", edgecolor="white", linewidth=0.5, alpha=0.9)
    ax.axvline(0.0, color="#444444", linewidth=1.1, linestyle="--")
    mean_gain = sum(gains) / len(gains)
    ax.axvline(mean_gain, color="#d62728", linewidth=1.4, label=f"Mean = {mean_gain:.3f}")
    ax.set_xlabel(text["gain"])
    ax.set_ylabel(text["claims"])
    ax.set_title(text["claim_title"])
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def build_figures(template_rows, category_rows, seed_rows, claim_rows, args):
    figures = []
    figures.append(("01_category_mean_accuracy", plot_category_mean_accuracy(category_rows, args.language)))
    figures.append(("02_category_mean_gain", plot_category_mean_gain(category_rows, args.language)))
    figures.append(("03_prompt_baseline_vs_laser_scatter", plot_prompt_baseline_vs_laser_scatter(template_rows, args.language)))
    figures.append(("04_prompt_gain_ranked", plot_prompt_gain_ranked(template_rows, args.language, args.sort_gain)))
    figures.append(("05_category_gain_distribution", plot_category_gain_distribution(template_rows, args.language)))
    figures.append(
        (
            "06_divergence_vs_gain",
            plot_divergence_vs_gain(template_rows, args.language, args.with_regression_line),
        )
    )
    if seed_rows is not None:
        figures.append(("07_seed_sensitivity", plot_seed_sensitivity(seed_rows, args.language)))
        figures.append(("08_prompt_sign_stability", plot_sign_stability(template_rows, args.language)))
        if all(
            row["baseline_binary_nll"] is not None and row["laser_binary_nll"] is not None
            for row in template_rows
        ):
            figures.append(("09_binary_nll_scatter", plot_binary_nll_scatter(template_rows, args.language)))
    if claim_rows is not None:
        figures.append(("10_claim_gain_distribution", plot_claim_gain_distribution(claim_rows, args.language)))
    return figures


def main():
    parser = argparse.ArgumentParser(description="Generate FEVER paraphrase plots from exported analysis artifacts.")
    parser.add_argument("--analysis-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"])
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--language", choices=["en", "fr"], default="en")
    parser.add_argument("--style", choices=["paper", "presentation"], default="paper")
    parser.add_argument("--sort-gain", choices=["desc", "asc"], default="desc")
    parser.set_defaults(with_regression_line=True)
    parser.add_argument("--with-regression-line", dest="with_regression_line", action="store_true")
    parser.add_argument("--without-regression-line", dest="with_regression_line", action="store_false")
    parser.add_argument("--transparent", action="store_true", default=False)
    args = parser.parse_args()

    analysis_dir = os.path.abspath(args.analysis_dir)
    output_dir = os.path.abspath(args.output_dir or os.path.join(analysis_dir, "plots"))

    template_path = os.path.join(analysis_dir, "template_metrics.csv")
    category_path = os.path.join(analysis_dir, "category_metrics.csv")
    summary_path = os.path.join(analysis_dir, "summary.json")
    seed_path = os.path.join(analysis_dir, "seed_metrics.csv")
    claim_path = os.path.join(analysis_dir, "claim_metrics.csv")

    template_rows_raw = load_csv_rows(template_path)
    category_rows_raw = load_csv_rows(category_path)
    summary = load_json(summary_path)
    seed_rows_raw = load_optional_csv_rows(seed_path)
    claim_rows_raw = load_optional_csv_rows(claim_path)

    validate_columns(
        template_rows_raw,
        [
            "template",
            "category_id",
            "is_original",
            "baseline_acc",
            "laser_acc",
            "laser_vs_same_template_acc",
            "divergence_char_norm",
        ],
        "template_metrics.csv",
    )
    validate_columns(
        category_rows_raw,
        [
            "category_id",
            "baseline_mean_acc",
            "baseline_std_acc",
            "laser_mean_acc",
            "laser_std_acc",
            "gain_mean_acc",
            "gain_std_acc",
        ],
        "category_metrics.csv",
    )

    template_rows = parse_template_rows(template_rows_raw)
    category_rows = parse_category_rows(category_rows_raw)
    seed_rows = None
    claim_rows = None
    if seed_rows_raw is not None:
        validate_columns(seed_rows_raw, ["seed", "prompt_count", "gain_mean_acc", "gain_std_acc"], "seed_metrics.csv")
        seed_rows = parse_seed_rows(seed_rows_raw)
    if claim_rows_raw is not None:
        validate_columns(claim_rows_raw, ["claim_id", "gain_prompt_seed_mean_accuracy"], "claim_metrics.csv")
        claim_rows = parse_claim_rows(claim_rows_raw)
        if not claim_rows:
            fail("claim_metrics.csv is empty")
    validate_consistency(template_rows, category_rows, summary)

    os.makedirs(output_dir, exist_ok=True)
    apply_style(args.style)

    figures = build_figures(template_rows, category_rows, seed_rows, claim_rows, args)
    for basename, fig in figures:
        save_figure(fig, output_dir, basename, args.formats, args.dpi, args.transparent)


if __name__ == "__main__":
    main()
