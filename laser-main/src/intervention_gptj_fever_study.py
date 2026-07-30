"""Evaluate GPT-J on FEVER with the study-specific metrics and metadata."""

import argparse
import os
import platform
import time

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, GPTJForCausalLM

from dataset_utils.fever_study import FEVER
from fever_experiment_utils import (
    FEVER_SPLIT_CHOICES,
    FEVER_SPLIT_COMBINED,
    INTERVENTION_SOURCE_CHOICES,
    aggregate_binary_predictions,
    atomic_pickle_dump,
    compute_binary_label_scores,
    configure_utf8_stdio,
    format_float_tag,
    resolve_rate_and_rho,
    torch_rng_seed_scope,
    validate_binary_label_token_ids,
)
from fever_prompting import build_fever_prompt, get_fever_prompt_spec, validate_fever_prompt_specs
from laser.LaserWrapper import LaserWrapper
from study_utils.log_utils import Logger
from study_utils.metric_utils import ContextAnswerLogProb, DatasetMetrics
from study_utils.time_utils import Progress, elapsed_from_str


def run_slice_suffix(args):
    if args.start_index == 0 and args.max_examples is None:
        return ""
    max_examples = "all" if args.max_examples is None else str(args.max_examples)
    return f"-s{args.start_index}-n{max_examples}"


def fever_split_suffix(args):
    if args.fever_split == FEVER_SPLIT_COMBINED:
        return ""
    return f"-split{args.fever_split}"


def run_file_suffix(args):
    return f"{fever_split_suffix(args)}{run_slice_suffix(args)}"


def resolve_device(device_arg):
    if device_arg != "auto":
        return device_arg

    if torch.cuda.is_available():
        return "cuda"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def validate_prompt_registry_or_raise():
    validation_errors = validate_fever_prompt_specs()
    if not validation_errors:
        return

    details = []
    for template_id, issues in sorted(validation_errors.items()):
        details.append(f"{template_id}: {', '.join(issues)}")
    raise ValueError("Invalid FEVER prompt templates detected: " + " | ".join(details))


def prompt_summary_fields(template_id):
    prompt_spec = get_fever_prompt_spec(template_id)
    return {
        "prompt/template_id": prompt_spec["template_id"],
        "prompt/category_id": prompt_spec["category_id"],
        "prompt/category_label": prompt_spec["category_label"],
        "prompt/variant_index": prompt_spec["variant_index"],
        "prompt/is_original": prompt_spec["is_original"],
        "prompt/wrapper_char_len": prompt_spec["wrapper_char_len"],
        "prompt/wrapper_word_len": prompt_spec["wrapper_word_len"],
        "prompt/divergence_char_norm": prompt_spec["divergence_char_norm"],
        "prompt/divergence_token_jaccard": prompt_spec["divergence_token_jaccard"],
    }


def resolve_save_dir(home_dir, llm_name, args, prompt_template="original"):
    parts = [home_dir, llm_name, args.intervention, args.lname]
    if args.save_seed_in_path and args.seed is not None and args.lname != "dont":
        parts.extend(["seeds", f"seed_{args.seed}"])
    if prompt_template != "original":
        parts.extend(["prompts", prompt_template])
    return os.path.join(*parts)


def rate_tag(args):
    return format_float_tag(args.rate)


def count_predictions_by_split(predictions):
    counts = {}
    for row in predictions:
        split_name = row.get("paper_split", "unknown")
        counts[split_name] = counts.get(split_name, 0) + 1
    return counts


def create_fresh_logger(save_dir, fname):
    log_path = os.path.join(save_dir, fname)
    with open(log_path, "w", encoding="utf-8"):
        pass
    return Logger(save_dir=save_dir, fname=fname)


class GPTJExperiment:

    def __init__(self, save_dir, logger, prompt_template="original"):
        self.save_dir = save_dir
        self.logger = logger
        self.prompt_template = prompt_template
        self.progress = Progress(logger=logger)
        self.dataset_metric = DatasetMetrics(logger=logger)
        self.device = "cpu"
        self.runtime_metadata = {}

    def build_prompt(self, question):
        return build_fever_prompt(question, self.prompt_template)

    def intervene(self, model, tokenizer, dataset, args, llm_name):
        dataset_size = len(dataset)
        self.logger.log(
            f"Starting a new intervention with rate {rate_tag(args)} "
            f"(rho={args.rho}). Dataset size {dataset_size}. Batch size {args.batch_size}"
        )

        time_edit_start = time.time()
        with torch_rng_seed_scope(args.seed):
            model_edit = LaserWrapper.get_edited_model(
                model=model,
                lname=args.lname,
                lnum=args.lnum,
                rate=args.rate,
                intervention=args.intervention,
                logger=self.logger,
                in_place=True,
            )
        model_edit_seconds = time.time() - time_edit_start

        model_edit.to(self.device)
        model_edit.eval()
        self.logger.log(f"Edited and put model on {model_edit.device} in time {elapsed_from_str(time_edit_start)}")

        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        predictions = []
        self.dataset_metric.reset()
        self.progress.start()

        label_tokens = validate_binary_label_token_ids(tokenizer)
        self.logger.log(
            f'Validated GPT-J binary label tokens: " true"={label_tokens["true_token_id"]}, '
            f'" false"={label_tokens["false_token_id"]}'
        )

        prompt_token_lengths = []
        inference_start = time.time()
        for i in tqdm(range(dataset_size)):
            if (i - 1) % 100 == 0 and i > 1:
                self.dataset_metric.print()
                self.progress.print(ex_done=i, ex_left=(dataset_size - i))

            dataset_row = dataset[i]
            question = dataset_row["question"]
            answer_ix = dataset_row["answer"]
            prompted_question = self.build_prompt(question)
            assert answer_ix in [0, 1]

            inputs = tokenizer(prompted_question, return_tensors="pt").to(self.device)
            prompt_token_lengths.append(int(inputs.input_ids.shape[1]))

            with torch.no_grad():
                results = model_edit(inputs.input_ids)
                logits = results.logits[0]
                log_prob = torch.nn.functional.log_softmax(logits, dim=1)
                last_token_logprob = log_prob[-1]

                binary_scores = compute_binary_label_scores(
                    last_token_logprob=last_token_logprob,
                    gold_answer_ix=answer_ix,
                    true_token_id=label_tokens["true_token_id"],
                    false_token_id=label_tokens["false_token_id"],
                )

                sorted_logprob, sorted_indices = torch.sort(last_token_logprob, descending=True)
                top_k_logprob = sorted_logprob[:10].detach().cpu().numpy()
                top_k_indices = sorted_indices[:10].detach()
                top_k_tokens = list(tokenizer.batch_decode(top_k_indices))
                assert len(top_k_tokens) == 10

                answer = binary_scores["answer_text"]
                top_1_acc = float(answer.lower().strip() in [token.lower().strip() for token in top_k_tokens[:1]])
                top_5_acc = float(answer.lower().strip() in [token.lower().strip() for token in top_k_tokens[:5]])
                top_10_acc = float(answer.lower().strip() in [token.lower().strip() for token in top_k_tokens[:10]])

                selected_log_prob = log_prob[:-1, :]
                indices = inputs.input_ids[0, 1:].unsqueeze(1)
                selected_log_prob = torch.gather(selected_log_prob, index=indices, dim=1)
                question_log_prob = selected_log_prob.sum().item()
                total_log_prob = question_log_prob + binary_scores["answer_log_prob"]

                logprob_results = ContextAnswerLogProb(
                    total_log_prob=total_log_prob,
                    answer_log_prob=binary_scores["answer_log_prob"],
                    answer_len=1,
                )

            self.dataset_metric.accept(
                is_correct=binary_scores["binary_correct"],
                f1pr_score=None,
                log_prob_results=logprob_results,
                top_k_acc={1: top_1_acc, 5: top_5_acc, 10: top_10_acc},
            )

            if i % 10 == 0:
                print(
                    f"Question: {question} and gold answer {answer}. "
                    f"Predicted top 10 tokens {top_k_tokens}."
                )

            predictions.append(
                {
                    "ix": i,
                    "dataset_local_ix": i,
                    "dataset_global_ix": dataset_row.get("global_ix"),
                    "paper_split": dataset_row.get("paper_split"),
                    "paper_split_local_ix": dataset_row.get("paper_split_local_ix"),
                    "evaluation_split": dataset_row.get("evaluation_split", args.fever_split),
                    "evaluation_split_local_ix": dataset_row.get("evaluation_split_local_ix", i),
                    "question": question,
                    "prompted-question": prompted_question,
                    "gold-answer": answer,
                    "gold-answer-ix": answer_ix,
                    "generation": top_k_tokens[0],
                    "correct": binary_scores["binary_correct"],
                    "binary_correct": binary_scores["binary_correct"],
                    "true_logprob": binary_scores["true_logprob"],
                    "false_logprob": binary_scores["false_logprob"],
                    "binary_true_logprob": binary_scores["binary_true_logprob"],
                    "binary_false_logprob": binary_scores["binary_false_logprob"],
                    "binary_margin": binary_scores["binary_margin"],
                    "binary_pred_label": binary_scores["binary_pred_label"],
                    "binary_pred_label_ix": binary_scores["binary_pred_label_ix"],
                    "binary_tie": binary_scores["binary_tie"],
                    "binary_nll": binary_scores["binary_nll"],
                    "binary_nll_normalized": binary_scores["binary_nll_normalized"],
                    "binary_label_pair_log_mass": binary_scores["binary_label_pair_log_mass"],
                    "top_1_acc": top_1_acc,
                    "top_5_acc": top_5_acc,
                    "top_10_acc": top_10_acc,
                    "top_10_logprob": top_k_logprob,
                    "top_10_tokens": top_k_tokens,
                    "f1_score": None,
                    "precision": None,
                    "recall": None,
                    "case-sensitive": False,
                    "white-space-strip": True,
                    "total_logprob": total_log_prob,
                    "question_logprob": question_log_prob,
                    "answer_logprob": binary_scores["answer_log_prob"],
                    "answer_length": 1,
                    "question_answer_length": inputs.input_ids.shape[1] + 1,
                }
            )

        inference_seconds = time.time() - inference_start
        runtime_metadata = dict(self.runtime_metadata)
        runtime_metadata.update(
            {
                "runtime/model_edit_seconds": model_edit_seconds,
                "runtime/inference_seconds": inference_seconds,
                "runtime/examples_per_second": dataset_size / max(inference_seconds, 1e-12),
                "runtime/device": self.device,
                "runtime/model_dtype": str(next(model_edit.parameters()).dtype),
                "runtime/prompt_tokens_mean": sum(prompt_token_lengths) / max(len(prompt_token_lengths), 1),
                "runtime/prompt_tokens_min": min(prompt_token_lengths) if prompt_token_lengths else None,
                "runtime/prompt_tokens_max": max(prompt_token_lengths) if prompt_token_lengths else None,
            }
        )
        if self.device == "cuda":
            runtime_metadata.update(
                {
                    "runtime/cuda_device_name": torch.cuda.get_device_name(0),
                    "runtime/cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                    "runtime/cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                }
            )

        self.terminate_and_save(
            predictions,
            args=args,
            llm_name=llm_name,
            label_tokens=label_tokens,
            runtime_metadata=runtime_metadata,
        )

    def terminate_and_save(self, predictions, args, llm_name, label_tokens, runtime_metadata):
        self.logger.log("Saving results. Final Performance is given below:")
        self.dataset_metric.terminate()

        binary_summary = aggregate_binary_predictions(predictions)
        self.logger.log(
            "FEVER Binary Performance: "
            f"split={args.fever_split}, "
            f"binary_accuracy={binary_summary['binary_accuracy']}, "
            f"binary_nll_normalized={binary_summary['binary_nll_normalized']}, "
            f"mean_raw_gold_token_logprob={binary_summary['mean_raw_gold_token_logprob']}, "
            f"top1={binary_summary['top_1_accuracy']}, "
            f"top5={binary_summary['top_5_accuracy']}, "
            f"top10={binary_summary['top_10_accuracy']}"
        )
        self.dataset_metric.print()

        time_start = time.time()
        file_suffix = run_file_suffix(args)
        save_pred_fname = (
            f"{self.save_dir}/{llm_name}-predictions-{rate_tag(args)}-{args.dtpts}-{args.lnum}{file_suffix}.p"
        )

        atomic_pickle_dump(predictions, save_pred_fname)

        save_summary_fname = (
            f"{self.save_dir}/{llm_name}-result-summary-{rate_tag(args)}-{args.dtpts}-{args.lnum}{file_suffix}.pkl"
        )

        results = self.dataset_metric.agg_to_dict()
        for key, value in args.__dict__.items():
            results[f"args/{key}"] = value

        split_counts = count_predictions_by_split(predictions)
        results.update(prompt_summary_fields(self.prompt_template))
        results.update(
            {
                "fever/returned_split": args.fever_split,
                "fever/paper_dev_count": split_counts.get("paper_dev", 0),
                "fever/paper_test_count": split_counts.get("paper_test", 0),
                "metric/binary_accuracy": binary_summary["binary_accuracy"],
                "metric/binary_nll": binary_summary["binary_nll"],
                "metric/binary_nll_normalized": binary_summary["binary_nll_normalized"],
                "metric/mean_raw_gold_token_logprob": binary_summary["mean_raw_gold_token_logprob"],
                "metric/mean_binary_label_pair_log_mass": binary_summary["mean_binary_label_pair_log_mass"],
                "metric/top_1_accuracy": binary_summary["top_1_accuracy"],
                "metric/top_5_accuracy": binary_summary["top_5_accuracy"],
                "metric/top_10_accuracy": binary_summary["top_10_accuracy"],
                "labels/true_text": label_tokens["true_text"],
                "labels/false_text": label_tokens["false_text"],
                "labels/true_token_id": label_tokens["true_token_id"],
                "labels/false_token_id": label_tokens["false_token_id"],
                "intervention/rho": args.rho,
                "intervention/rate": args.rate,
            }
        )
        results.update(runtime_metadata)

        atomic_pickle_dump(results, save_summary_fname)

        self.logger.log(f"Time taken to store all results {elapsed_from_str(time_start)}")


if __name__ == "__main__":
    configure_utf8_stdio()
    validate_prompt_registry_or_raise()

    parser = argparse.ArgumentParser(description="Process Arguments for experiments with GPTJ LLM on FEVER")
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
    parser.add_argument("--fever-split", type=str, default=FEVER_SPLIT_COMBINED, choices=FEVER_SPLIT_CHOICES)
    parser.add_argument("--start_index", type=int, default=0, help="Start index within the selected FEVER split")
    parser.add_argument("--max_examples", type=int, default=None, help="Optional cap for a quick local smoke test")
    parser.add_argument(
        "--intervention-source",
        type=str,
        default="unspecified",
        choices=INTERVENTION_SOURCE_CHOICES,
        help="How the intervention was chosen for this run",
    )
    parser.add_argument("--save-seed-in-path", action="store_true", help="Store seeded LASER runs under a seed-specific subdirectory")
    parser.add_argument("--home_dir", type=str, default="/mnt/data/iclr2024/fever/gptj_results", help="Directory where outputs are stored")
    parser.add_argument("--dataset_file", type=str, default="/mnt/data/counterfact", help="Unused legacy FEVER argument")

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

    save_dir = resolve_save_dir(args.home_dir, llm_name, args, prompt_template="original")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    logger = create_fresh_logger(
        save_dir,
        f"{llm_name}-log-{args.lnum}-{args.lname}-{rate_tag(args)}{run_file_suffix(args)}.txt",
    )

    experiment = GPTJExperiment(save_dir=save_dir, logger=logger, prompt_template="original")
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
    logger.log(f"Created a new Experiment. Model {llm_name}")
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
