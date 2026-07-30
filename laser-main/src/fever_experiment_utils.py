"""Provide shared utilities for the GPT-J/FEVER study."""

import math
import os
import pickle
import sys
from contextlib import contextmanager

import torch


FEVER_SPLIT_COMBINED = "combined"
FEVER_SPLIT_PAPER_DEV = "paper_dev"
FEVER_SPLIT_PAPER_TEST = "paper_test"
FEVER_SPLIT_LASER_DEV = "laser_dev"
FEVER_SPLIT_LASER_TEST = "laser_test"
FEVER_SPLIT_CHOICES = [
    FEVER_SPLIT_COMBINED,
    FEVER_SPLIT_PAPER_DEV,
    FEVER_SPLIT_PAPER_TEST,
    FEVER_SPLIT_LASER_DEV,
    FEVER_SPLIT_LASER_TEST,
]

FEVER_FILTERED_TOTAL_SIZE = 13086
FEVER_LASER_DEV_FRACTION = 0.20
FEVER_LASER_DEV_SIZE = int(FEVER_FILTERED_TOTAL_SIZE * FEVER_LASER_DEV_FRACTION)
FEVER_LASER_TEST_SIZE = FEVER_FILTERED_TOTAL_SIZE - FEVER_LASER_DEV_SIZE
FEVER_EXPECTED_PAPER_DEV_FINGERPRINT = "8acd78263cd33412"
FEVER_EXPECTED_PAPER_TEST_FINGERPRINT = "d130262df6139bc2"
GPTJ_FEVER_TRUE_TOKEN_ID = 2081
GPTJ_FEVER_FALSE_TOKEN_ID = 3991

INTERVENTION_SOURCE_CHOICES = [
    "unspecified",
    "published",
    "validation_candidate",
    "validation_selected",
]


def configure_utf8_stdio():
    """Configure UTF-8 standard streams when the platform supports it."""
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="backslashreplace")
        except (OSError, ValueError):
            pass


@contextmanager
def torch_rng_seed_scope(seed):
    """Temporarily set the PyTorch random seed and restore its state."""
    if seed is None:
        yield
        return

    cpu_rng_state = torch.get_rng_state()
    cuda_rng_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(seed)
    try:
        yield
    finally:
        torch.set_rng_state(cpu_rng_state)
        if cuda_rng_states is not None:
            torch.cuda.set_rng_state_all(cuda_rng_states)


def rate_to_rho(rate):
    if rate is None:
        return None
    rate = float(rate)
    if rate < 0.0 or rate > 10.0:
        raise ValueError(f"rate must be in [0, 10], found {rate}")
    return round(1.0 - 0.1 * rate, 10)


def rho_to_rate(rho):
    if rho is None:
        return None
    rho = float(rho)
    if rho < 0.0 or rho > 1.0:
        raise ValueError(f"rho must be in [0, 1], found {rho}")
    return round((1.0 - rho) / 0.1, 10)


def resolve_rate_and_rho(rate=None, rho=None, default_rate=1.0):
    if rate is not None and rho is not None:
        raise ValueError("rate and rho are mutually exclusive")

    if rate is None and rho is None:
        rate = float(default_rate)

    if rate is not None:
        resolved_rate = float(rate)
        resolved_rho = rate_to_rho(resolved_rate)
    else:
        resolved_rho = float(rho)
        resolved_rate = rho_to_rate(resolved_rho)

    return round(resolved_rate, 10), round(resolved_rho, 10)


def format_float_tag(value, decimals=6):
    return f"{float(value):.{decimals}f}".rstrip("0").rstrip(".")


def expected_fever_split_size(split):
    expected_sizes = {
        FEVER_SPLIT_COMBINED: FEVER_FILTERED_TOTAL_SIZE,
        FEVER_SPLIT_PAPER_DEV: 6510,
        FEVER_SPLIT_PAPER_TEST: 6576,
        FEVER_SPLIT_LASER_DEV: FEVER_LASER_DEV_SIZE,
        FEVER_SPLIT_LASER_TEST: FEVER_LASER_TEST_SIZE,
    }
    if split not in expected_sizes:
        raise KeyError(f"Unknown FEVER split: {split}")
    return expected_sizes[split]


def atomic_pickle_dump(value, path):
    """Write a pickle atomically to avoid leaving a partial result file."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    temporary_path = f"{path}.tmp-{os.getpid()}"
    try:
        with open(temporary_path, "wb") as handle:
            pickle.dump(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


def validate_binary_label_token_ids(tokenizer):
    true_token_ids = tokenizer(" true")["input_ids"]
    false_token_ids = tokenizer(" false")["input_ids"]

    if len(true_token_ids) != 1:
        raise ValueError(f'Expected " true" to map to one token for GPT-J, found {true_token_ids}')
    if len(false_token_ids) != 1:
        raise ValueError(f'Expected " false" to map to one token for GPT-J, found {false_token_ids}')

    return {
        "true_text": " true",
        "false_text": " false",
        "true_token_id": int(true_token_ids[0]),
        "false_token_id": int(false_token_ids[0]),
    }


def compute_binary_label_scores(last_token_logprob, gold_answer_ix, true_token_id, false_token_id):
    true_logprob = float(last_token_logprob[true_token_id].item())
    false_logprob = float(last_token_logprob[false_token_id].item())

    pair_logprobs = torch.stack(
        [last_token_logprob[false_token_id], last_token_logprob[true_token_id]],
        dim=0,
    )
    pair_log_denom = torch.logsumexp(pair_logprobs, dim=0)
    binary_false_logprob = float((pair_logprobs[0] - pair_log_denom).item())
    binary_true_logprob = float((pair_logprobs[1] - pair_log_denom).item())

    margin = binary_true_logprob - binary_false_logprob
    is_tie = margin == 0
    if margin > 0:
        pred_answer_ix = 1
        pred_answer = "true"
    else:
        # Match argmax([false, true]): an exact tie selects false.
        pred_answer_ix = 0
        pred_answer = "false"

    gold_binary_logprob = binary_true_logprob if gold_answer_ix == 1 else binary_false_logprob
    binary_nll = -gold_binary_logprob

    return {
        "true_logprob": true_logprob,
        "false_logprob": false_logprob,
        "binary_true_logprob": binary_true_logprob,
        "binary_false_logprob": binary_false_logprob,
        "binary_margin": margin,
        "binary_pred_label": pred_answer,
        "binary_pred_label_ix": pred_answer_ix,
        "binary_tie": is_tie,
        "binary_correct": pred_answer_ix == gold_answer_ix,
        "binary_nll": binary_nll,
        "binary_nll_normalized": binary_nll,
        "binary_label_pair_log_mass": float(pair_log_denom.item()),
        "answer_text": "true" if gold_answer_ix == 1 else "false",
        "answer_log_prob": true_logprob if gold_answer_ix == 1 else false_logprob,
    }


def _mean(values):
    if not values:
        return None
    return sum(values) / len(values)


def aggregate_binary_predictions(predictions):
    if not predictions:
        return {
            "binary_accuracy": None,
            "binary_nll": None,
            "binary_nll_normalized": None,
            "mean_raw_gold_token_logprob": None,
            "mean_binary_label_pair_log_mass": None,
            "top_1_accuracy": None,
            "top_5_accuracy": None,
            "top_10_accuracy": None,
        }

    binary_correct = [1.0 if row["binary_correct"] else 0.0 for row in predictions]
    binary_nll = [float(row["binary_nll"]) for row in predictions]
    binary_nll_normalized = [float(row["binary_nll_normalized"]) for row in predictions]
    raw_gold_logprob = [float(row["answer_logprob"]) for row in predictions]
    pair_log_mass = [float(row["binary_label_pair_log_mass"]) for row in predictions]
    top1 = [float(row["top_1_acc"]) for row in predictions]
    top5 = [float(row["top_5_acc"]) for row in predictions]
    top10 = [float(row["top_10_acc"]) for row in predictions]

    return {
        "binary_accuracy": 100.0 * _mean(binary_correct),
        "binary_nll": _mean(binary_nll),
        "binary_nll_normalized": _mean(binary_nll_normalized),
        "mean_raw_gold_token_logprob": _mean(raw_gold_logprob),
        "mean_binary_label_pair_log_mass": _mean(pair_log_mass),
        "top_1_accuracy": 100.0 * _mean(top1),
        "top_5_accuracy": 100.0 * _mean(top5),
        "top_10_accuracy": 100.0 * _mean(top10),
    }


def percentile(sorted_values, q):
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight
