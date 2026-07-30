"""Check the environment required by the GPT-J/FEVER study."""

import argparse
import json
import os
import platform
import sys
import time

import datasets
import numpy as np
import torch
import transformers
from transformers import AutoTokenizer

from dataset_utils.fever_study import FEVER
from fever_experiment_utils import (
    FEVER_EXPECTED_PAPER_DEV_FINGERPRINT,
    FEVER_EXPECTED_PAPER_TEST_FINGERPRINT,
    GPTJ_FEVER_FALSE_TOKEN_ID,
    GPTJ_FEVER_TRUE_TOKEN_ID,
    FEVER_FILTERED_TOTAL_SIZE,
    FEVER_LASER_DEV_SIZE,
    FEVER_LASER_TEST_SIZE,
    FEVER_SPLIT_COMBINED,
    FEVER_SPLIT_LASER_DEV,
    FEVER_SPLIT_LASER_TEST,
    rate_to_rho,
    torch_rng_seed_scope,
    validate_binary_label_token_ids,
)
from fever_prompting import list_fever_prompt_categories, list_fever_prompt_template_ids, validate_fever_prompt_specs
from laser.matrix_utils import do_low_rank


class ConsoleLogger:
    def log(self, message, also_stdout=True):
        if also_stdout:
            print(message)


def check_svd_determinism():
    generator = torch.Generator().manual_seed(1234)
    matrix = torch.randn(96, 64, generator=generator)
    with torch_rng_seed_scope(0):
        same_a = do_low_rank(matrix, 0.25).detach()
    with torch_rng_seed_scope(0):
        same_b = do_low_rank(matrix, 0.25).detach()
    with torch_rng_seed_scope(1):
        different = do_low_rank(matrix, 0.25).detach()
    if not torch.equal(same_a, same_b):
        raise RuntimeError("torch.svd_lowrank is not reproducible for repeated seed=0 in this environment")
    if torch.equal(same_a, different):
        raise RuntimeError("torch.svd_lowrank returned identical tensors for seed=0 and seed=1")
    return {
        "same_seed_identical": True,
        "different_seed_differs": True,
        "seed_0_vs_1_max_abs_diff": float(torch.max(torch.abs(same_a - different)).item()),
    }


def check_dataset():
    dataset_util = FEVER()
    splits = dataset_util.get_splits(ConsoleLogger())
    observed = {name: len(rows) for name, rows in splits.items()}
    expected = {
        FEVER_SPLIT_COMBINED: FEVER_FILTERED_TOTAL_SIZE,
        FEVER_SPLIT_LASER_DEV: FEVER_LASER_DEV_SIZE,
        FEVER_SPLIT_LASER_TEST: FEVER_LASER_TEST_SIZE,
    }
    for split_name, expected_size in expected.items():
        if observed.get(split_name) != expected_size:
            raise RuntimeError(
                f"FEVER split {split_name} has {observed.get(split_name)} rows; expected {expected_size}"
            )

    dev_ids = {row["global_ix"] for row in splits[FEVER_SPLIT_LASER_DEV]}
    test_ids = {row["global_ix"] for row in splits[FEVER_SPLIT_LASER_TEST]}
    if dev_ids & test_ids:
        raise RuntimeError("laser_dev and laser_test overlap")
    observed_fingerprints = {
        "paper_dev": dataset_util.metadata.get("dataset/paper_dev_fingerprint"),
        "paper_test": dataset_util.metadata.get("dataset/paper_test_fingerprint"),
    }
    expected_fingerprints = {
        "paper_dev": FEVER_EXPECTED_PAPER_DEV_FINGERPRINT,
        "paper_test": FEVER_EXPECTED_PAPER_TEST_FINGERPRINT,
    }
    if observed_fingerprints != expected_fingerprints:
        raise RuntimeError(
            "FEVER dataset fingerprints differ from the audited study dataset: "
            f"found={observed_fingerprints}, expected={expected_fingerprints}"
        )
    return {"sizes": observed, "metadata": dataset_util.metadata}


def check_prompt_catalog():
    template_ids = list_fever_prompt_template_ids()
    categories = list_fever_prompt_categories()
    errors = validate_fever_prompt_specs()
    if errors:
        raise RuntimeError(f"Prompt catalog lint failed: {errors}")
    if len(template_ids) != 41:
        raise RuntimeError(f"Expected 41 prompt templates, found {len(template_ids)}")
    counts = {}
    for template_id in template_ids[1:]:
        category_id = template_id.split("_", 1)[0]
        counts[category_id] = counts.get(category_id, 0) + 1
    expected_categories = {row["category_id"] for row in categories}
    if set(counts) != expected_categories or any(count != 8 for count in counts.values()):
        raise RuntimeError(f"Prompt categories are not balanced 5x8: {counts}")
    return {"template_count": len(template_ids), "paraphrase_count": len(template_ids) - 1, "category_counts": counts}


def main():
    parser = argparse.ArgumentParser(description="Preflight the GPT-J/FEVER study environment")
    parser.add_argument("--model-path", default="EleutherAI/gpt-j-6B")
    parser.add_argument("--device", choices=["cuda", "cpu", "mps", "auto"], default="cuda")
    parser.add_argument("--output", default=None)
    parser.add_argument("--skip-dataset", action="store_true")
    args = parser.parse_args()

    started = time.time()
    resolved_device = args.device
    if resolved_device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    if resolved_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")

    cuda_info = None
    if resolved_device == "cuda":
        properties = torch.cuda.get_device_properties(0)
        cuda_info = {
            "device_name": torch.cuda.get_device_name(0),
            "device_capability": list(torch.cuda.get_device_capability(0)),
            "total_memory_bytes": int(properties.total_memory),
            "torch_cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
        }
        if properties.total_memory < 20 * 1024**3:
            raise RuntimeError(f"At least 20 GiB GPU memory is required; found {properties.total_memory / 1024**3:.2f} GiB")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    label_tokens = validate_binary_label_token_ids(tokenizer)
    expected_label_ids = {
        "true_token_id": GPTJ_FEVER_TRUE_TOKEN_ID,
        "false_token_id": GPTJ_FEVER_FALSE_TOKEN_ID,
    }
    observed_label_ids = {key: label_tokens[key] for key in expected_label_ids}
    if observed_label_ids != expected_label_ids:
        raise RuntimeError(
            f"GPT-J binary label token ids differ from the audited interface: "
            f"found={observed_label_ids}, expected={expected_label_ids}"
        )

    if abs(rate_to_rho(9.9) - 0.01) > 1e-12:
        raise RuntimeError("LASER rate/rho mapping check failed")

    result = {
        "status": "ok",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": time.time() - started,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "datasets_version": datasets.__version__,
        "numpy_version": np.__version__,
        "resolved_device": resolved_device,
        "cuda": cuda_info,
        "binary_label_tokens": label_tokens,
        "rate_9_9_maps_to_rho": rate_to_rho(9.9),
        "prompt_catalog": check_prompt_catalog(),
        "svd_determinism": check_svd_determinism(),
        "fever_splits": None if args.skip_dataset else check_dataset(),
    }

    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output is not None:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        temporary = args.output + f".tmp-{os.getpid()}"
        try:
            with open(temporary, "w") as handle:
                json.dump(result, handle, indent=2, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, args.output)
        finally:
            if os.path.exists(temporary):
                os.remove(temporary)


if __name__ == "__main__":
    main()
