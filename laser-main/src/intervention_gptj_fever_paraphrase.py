"""Run one fixed-intervention prompt-transfer condition on FEVER."""

import argparse
import os
import platform
import time

import torch
from transformers import AutoTokenizer, GPTJForCausalLM

from dataset_utils.fever_study import FEVER
from fever_experiment_utils import (
    FEVER_SPLIT_CHOICES,
    INTERVENTION_SOURCE_CHOICES,
    configure_utf8_stdio,
    resolve_rate_and_rho,
)
from fever_prompting import list_fever_prompt_template_ids
from intervention_gptj_fever_study import (
    GPTJExperiment,
    create_fresh_logger,
    rate_tag,
    resolve_device,
    resolve_save_dir,
    run_file_suffix,
    run_slice_suffix,
    validate_prompt_registry_or_raise,
)


if __name__ == "__main__":
    configure_utf8_stdio()
    validate_prompt_registry_or_raise()

    parser = argparse.ArgumentParser(
        description="Process Arguments for fixed-intervention GPT-J FEVER prompt-transfer experiments"
    )

    rate_group = parser.add_mutually_exclusive_group()
    rate_group.add_argument("--rate", type=float, default=None, help="Legacy LASER rate parameter")
    rate_group.add_argument("--rho", type=float, default=None, help="Fraction of rank retained by LASER")

    parser.add_argument("--dtpts", type=int, default=22000, help="# samples per instruction")
    parser.add_argument("--batch_size", type=int, default=256, help="batch size for evaluation")
    parser.add_argument("--max_len", type=int, default=1, help="maximum length for generation")
    parser.add_argument("--k", type=int, default=10, help="top k for evaluation")
    parser.add_argument(
        "--intervention",
        type=str,
        default="rank-reduction",
        choices=["dropout", "rank-reduction"],
        help="what type of intervention to perform",
    )
    parser.add_argument(
        "--lname",
        type=str,
        default="None",
        choices=["k_proj", "q_proj", "v_proj", "out_proj", "fc_in", "fc_up", "fc_out", "None", "dont", "all", "mlp", "attn"],
        help="provided which type of parameters to effect",
    )
    parser.add_argument("--lnum", type=int, default=24, help="Layers to edit", choices=list(range(-1, 28)))
    parser.add_argument("--model_path", type=str, default="EleutherAI/gpt-j-6B", help="Place where model weights are stored")
    parser.add_argument("--revision", type=str, default="float16", help="Model revision to load")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"], help="Execution device")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed for deterministic LASER rank reduction")
    parser.add_argument("--fever-split", type=str, default="paper_test", choices=FEVER_SPLIT_CHOICES)
    parser.add_argument("--start_index", type=int, default=0, help="Start index within the selected FEVER split")
    parser.add_argument("--max_examples", type=int, default=None, help="Optional cap for a quick local smoke test")
    parser.add_argument(
        "--intervention-source",
        type=str,
        default="validation_selected",
        choices=INTERVENTION_SOURCE_CHOICES,
        help="How the intervention was chosen for this run",
    )
    parser.add_argument("--save-seed-in-path", action="store_true", help="Store seeded LASER runs under a seed-specific subdirectory")
    parser.add_argument("--home_dir", type=str, default="/mnt/data/iclr2024/fever/gptj_results", help="Directory where outputs are stored")
    parser.add_argument("--dataset_file", type=str, default="/mnt/data/counterfact", help="Unused legacy FEVER argument")
    parser.add_argument(
        "--prompt-template",
        type=str,
        default="original",
        choices=list_fever_prompt_template_ids(),
        help="Prompt template to evaluate",
    )

    args = parser.parse_args()
    args.rate, args.rho = resolve_rate_and_rho(rate=args.rate, rho=args.rho, default_rate=1.0)

    llm_name = "GPTJ"
    llm_path = args.model_path
    device = resolve_device(args.device)
    model_dtype = torch.float16 if device in ["cuda", "mps"] else torch.float32

    model_load_start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(llm_path)
    model = GPTJForCausalLM.from_pretrained(llm_path, revision=args.revision, torch_dtype=model_dtype)
    model_load_seconds = time.time() - model_load_start

    save_dir = resolve_save_dir(args.home_dir, llm_name, args, prompt_template=args.prompt_template)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    logger = create_fresh_logger(
        save_dir,
        f"{llm_name}-log-{args.lnum}-{args.lname}-{rate_tag(args)}-{args.prompt_template}{run_file_suffix(args)}.txt",
    )

    experiment = GPTJExperiment(save_dir=save_dir, logger=logger, prompt_template=args.prompt_template)
    experiment.device = device
    experiment.runtime_metadata = {
        "runtime/model_load_seconds": model_load_seconds,
        "runtime/python_version": platform.python_version(),
        "runtime/torch_version": torch.__version__,
        "runtime/transformers_model_class": type(model).__name__,
        "runtime/model_commit_hash": getattr(model.config, "_commit_hash", None),
        "runtime/tokenizer_commit_hash": tokenizer.init_kwargs.get("_commit_hash"),
    }

    logger.log("=" * 50)
    logger.log(f"Created a new Fixed-Intervention Prompt Transfer Experiment. Model {llm_name}")
    logger.log("=" * 50)
    for key, value in args.__dict__.items():
        logger.log(f">>>> Command line argument {key} => {value}")
    logger.log("=" * 50)
    logger.log(f"Resolved LASER mapping: rate={rate_tag(args)} <-> rho={args.rho}")

    dataset_util = FEVER()
    dataset = dataset_util.get_dataset(logger, split=args.fever_split)
    experiment.runtime_metadata.update(dataset_util.metadata)
    if args.start_index > 0:
        dataset = dataset[args.start_index:]
        logger.log(f"Sliced dataset from start_index={args.start_index}. Remaining examples: {len(dataset)}.")
    if args.max_examples is not None:
        dataset = dataset[:args.max_examples]
        logger.log(f"Truncated dataset to first {len(dataset)} examples for smoke testing.")

    experiment.intervene(model=model, tokenizer=tokenizer, dataset=dataset, args=args, llm_name=llm_name)
    logger.log("Experimented Completed.")
