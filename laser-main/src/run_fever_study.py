"""Orchestrate the complete FEVER reproduction and prompt-transfer workflow."""

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import platform
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = SRC_DIR.parent.parent
ANALYSIS_DIR = WORKSPACE_DIR / "analysis"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from fever_experiment_utils import FEVER_LASER_DEV_SIZE, FEVER_LASER_TEST_SIZE, configure_utf8_stdio
from fever_reproduction_report import (
    build_descriptors,
    recursive_prediction_files,
    select_validation_winner,
)
from run_fever_paraphrase_suite import prediction_path, summary_path, validate_existing_output


PUBLISHED_INTERVENTION = {"lname": "fc_in", "lnum": 24, "rho": 0.01}
DEFAULT_SEEDS = [0, 1, 2, 3, 4]
SMOKE_EXAMPLES = 5


def display_command(command):
    if os.name == "nt":
        return subprocess.list2cmdline(command)
    return shlex.join(command)


def atomic_json_dump(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def run_logged(stage, command, log_dir, cwd=WORKSPACE_DIR):
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{stage}.log"
    print(f"\n===== {stage} =====")
    print(display_command(command))
    with log_path.open("a", encoding="utf-8", buffering=1) as log_handle:
        log_handle.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
        log_handle.write(display_command(command) + "\n")
        log_handle.flush()
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        try:
            for line in process.stdout:
                print(line, end="")
                log_handle.write(line)
            return_code = process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)
    return log_path


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_manifest():
    paths = sorted((WORKSPACE_DIR / "laser-main" / "src").rglob("*.py"))
    paths.extend(
        [
            ANALYSIS_DIR / "fever_report.py",
            ANALYSIS_DIR / "fever_reproduction_report.py",
            ANALYSIS_DIR / "fever_transfer_report.py",
            ANALYSIS_DIR / "plot_fever_paraphrase_results.py",
            ANALYSIS_DIR / "run_fever_transfer_plots.sh",
        ]
    )
    paths.extend(sorted((WORKSPACE_DIR / "tests").glob("*.py")))
    paths.extend(
        [
            WORKSPACE_DIR / "laser-main" / "requirements.txt",
            WORKSPACE_DIR / "requirements.txt",
            WORKSPACE_DIR / "REPRODUCE.md",
        ]
    )
    return {
        str(path.relative_to(WORKSPACE_DIR)): sha256_file(path)
        for path in paths
        if path.exists()
    }


def artifact_manifest(output_root):
    suffixes = {".csv", ".json", ".md", ".tex", ".png", ".pdf", ".p", ".pkl", ".txt", ".log"}
    return {
        str(path.relative_to(output_root)): {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(output_root.rglob("*"))
        if path.is_file()
        and path.suffix.lower() in suffixes
        and "manifests" not in path.relative_to(output_root).parts
        and path.name != "RUN_COMPLETE.json"
    }


def python_command(script, *arguments):
    return [sys.executable, str(script), *[str(argument) for argument in arguments]]


def select_winner(reproduction_root):
    paths = recursive_prediction_files(str(reproduction_root))
    descriptors = build_descriptors(paths, dev_size=FEVER_LASER_DEV_SIZE, test_size=FEVER_LASER_TEST_SIZE)
    winner = select_validation_winner(descriptors, validation_split="laser_dev")
    if winner is None:
        raise RuntimeError("No validation winner could be selected after the reproduction grid")
    return {
        "lname": winner["lname"],
        "lnum": int(winner["lnum"]),
        "rho": float(winner["rho"]),
        "rate": float(winner["rate"]),
        "seed": winner["seed"],
        "path": winner["path"],
    }


def selected_test_output_status(reproduction_root, winner, model_path, revision, device):
    output_path = prediction_path(
        home_dir=str(reproduction_root),
        template_id="original",
        lnum=winner["lnum"],
        lname=winner["lname"],
        rate=winner["rate"],
        dtpts=22000,
        fever_split="laser_test",
        start_index=0,
        max_examples=None,
        seed=0,
        seeded_path=True,
    )
    if not os.path.exists(output_path):
        return False, output_path, "missing prediction file"
    valid, reason = validate_existing_output(
        output_path,
        {
            "prediction_count": FEVER_LASER_TEST_SIZE,
            "template": "original",
            "fever_split": "laser_test",
            "lname": winner["lname"],
            "lnum": winner["lnum"],
            "rate": winner["rate"],
            "seed": 0,
            "model_path": model_path,
            "revision": revision,
            "device": device,
            "intervention_source": "validation_selected",
            "dtpts": 22000,
            "intervention": "rank-reduction",
            "require_provenance": True,
        },
    )
    return valid, output_path, reason


def validate_smoke_gate(
    smoke_root,
    expected_device="cuda",
    model_path="EleutherAI/gpt-j-6B",
    revision="float16",
):
    run_specs = [
        {
            "mode": "baseline",
            "lname": "dont",
            "lnum": 26,
            "rate": 9.9,
            "seed": None,
            "seeded_path": False,
            "intervention_source": "unspecified",
        },
        {
            "mode": "laser",
            "lname": "fc_in",
            "lnum": 24,
            "rate": 9.0,
            "seed": 0,
            "seeded_path": True,
            "intervention_source": "validation_selected",
        },
    ]
    validated = {}

    for spec in run_specs:
        output_path = prediction_path(
            home_dir=str(smoke_root),
            template_id="original",
            lnum=spec["lnum"],
            lname=spec["lname"],
            rate=spec["rate"],
            dtpts=22000,
            fever_split="laser_test",
            start_index=0,
            max_examples=SMOKE_EXAMPLES,
            seed=spec["seed"],
            seeded_path=spec["seeded_path"],
        )
        expected = {
            "prediction_count": SMOKE_EXAMPLES,
            "template": "original",
            "fever_split": "laser_test",
            "lname": spec["lname"],
            "lnum": spec["lnum"],
            "rate": spec["rate"],
            "seed": spec["seed"],
            "model_path": model_path,
            "revision": revision,
            "device": expected_device,
            "intervention_source": spec["intervention_source"],
            "dtpts": 22000,
            "intervention": "rank-reduction",
            "require_provenance": True,
        }
        valid, reason = validate_existing_output(output_path, expected)
        if not valid:
            raise RuntimeError(f"Smoke gate failed for {spec['mode']}: {reason}")

        with open(output_path, "rb") as handle:
            predictions = pickle.load(handle)
        with open(summary_path(output_path), "rb") as handle:
            summary = pickle.load(handle)

        required_row_fields = {
            "dataset_global_ix",
            "question",
            "prompted-question",
            "binary_correct",
            "binary_pred_label",
            "binary_true_logprob",
            "binary_false_logprob",
            "binary_margin",
            "binary_nll_normalized",
            "evaluation_split",
        }
        for index, row in enumerate(predictions):
            missing = sorted(required_row_fields - set(row))
            if missing:
                raise RuntimeError(
                    f"Smoke gate failed for {spec['mode']} row {index}: missing fields {missing}"
                )
            if row["evaluation_split"] != "laser_test":
                raise RuntimeError(
                    f"Smoke gate failed for {spec['mode']} row {index}: "
                    f"evaluation_split={row['evaluation_split']!r}"
                )
            if row["binary_pred_label"] not in {"true", "false"}:
                raise RuntimeError(
                    f"Smoke gate failed for {spec['mode']} row {index}: invalid binary prediction"
                )
            for field in [
                "binary_true_logprob",
                "binary_false_logprob",
                "binary_margin",
                "binary_nll_normalized",
            ]:
                if not math.isfinite(float(row[field])):
                    raise RuntimeError(
                        f"Smoke gate failed for {spec['mode']} row {index}: {field} is not finite"
                    )
            if float(row["binary_nll_normalized"]) < 0:
                raise RuntimeError(
                    f"Smoke gate failed for {spec['mode']} row {index}: binary NLL is negative"
                )

        binary_accuracy = float(summary.get("metric/binary_accuracy", float("nan")))
        binary_nll = float(summary.get("metric/binary_nll_normalized", float("nan")))
        if not math.isfinite(binary_accuracy) or not 0.0 <= binary_accuracy <= 100.0:
            raise RuntimeError(
                f"Smoke gate failed for {spec['mode']}: invalid binary accuracy {binary_accuracy}"
            )
        if not math.isfinite(binary_nll) or binary_nll < 0.0:
            raise RuntimeError(f"Smoke gate failed for {spec['mode']}: invalid binary NLL {binary_nll}")
        if summary.get("runtime/device") != expected_device:
            raise RuntimeError(
                f"Smoke gate failed for {spec['mode']}: runtime/device={summary.get('runtime/device')!r}, "
                f"expected {expected_device!r}"
            )
        true_token_id = summary.get("labels/true_token_id")
        false_token_id = summary.get("labels/false_token_id")
        if true_token_id is None or false_token_id is None or true_token_id == false_token_id:
            raise RuntimeError(f"Smoke gate failed for {spec['mode']}: invalid binary label token ids")
        if summary.get("labels/true_text") != " true" or summary.get("labels/false_text") != " false":
            raise RuntimeError(f"Smoke gate failed for {spec['mode']}: binary label texts are inconsistent")
        if expected_device == "cuda" and not summary.get("runtime/cuda_device_name"):
            raise RuntimeError(f"Smoke gate failed for {spec['mode']}: CUDA device metadata is missing")

        validated[spec["mode"]] = {
            "prediction_path": output_path,
            "summary_path": summary_path(output_path),
            "claim_ids": [row["dataset_global_ix"] for row in predictions],
            "questions": [row["question"] for row in predictions],
            "prompts": [row["prompted-question"] for row in predictions],
            "binary_accuracy": binary_accuracy,
            "binary_nll_normalized": binary_nll,
            "true_token_id": true_token_id,
            "false_token_id": false_token_id,
            "runtime_device": summary.get("runtime/device"),
            "cuda_device_name": summary.get("runtime/cuda_device_name"),
        }

    baseline = validated["baseline"]
    laser = validated["laser"]
    for field in ["claim_ids", "questions", "prompts", "true_token_id", "false_token_id"]:
        if baseline[field] != laser[field]:
            raise RuntimeError(f"Smoke gate failed: baseline/LASER mismatch for {field}")

    return {
        "status": "pass",
        "examples": SMOKE_EXAMPLES,
        "expected_device": expected_device,
        "baseline": baseline,
        "laser": laser,
    }


def count_csv_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def validate_artifacts(
    output_root,
    reproduction_analysis,
    transfer_analysis,
    plots_root,
    manifests_root,
    seeds,
    smoke_required,
):
    required_files = {
        "environment": manifests_root / "environment.json",
        "reproduction_csv": reproduction_analysis / "reproduction_summary.csv",
        "reproduction_json": reproduction_analysis / "reproduction_summary.json",
        "validation_grid": reproduction_analysis / "validation_grid.csv",
        "template_metrics": transfer_analysis / "template_metrics.csv",
        "template_seed_metrics": transfer_analysis / "template_seed_metrics.csv",
        "category_metrics": transfer_analysis / "category_metrics.csv",
        "seed_metrics": transfer_analysis / "seed_metrics.csv",
        "claim_metrics": transfer_analysis / "claim_metrics.csv",
        "claim_seed_metrics": transfer_analysis / "claim_seed_metrics.csv",
        "transfer_summary": transfer_analysis / "summary.json",
    }
    if smoke_required:
        required_files["smoke_validation"] = manifests_root / "smoke_validation.json"

    missing = [name for name, path in required_files.items() if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise RuntimeError(f"Artifact gate failed: missing or empty files {missing}")

    expected_csv_rows = {
        "reproduction_csv": 3,
        "validation_grid": 72,
        "template_metrics": 41,
        "template_seed_metrics": 41 * len(seeds),
        "category_metrics": 5,
        "seed_metrics": len(seeds),
        "claim_metrics": FEVER_LASER_TEST_SIZE,
        "claim_seed_metrics": FEVER_LASER_TEST_SIZE * len(seeds),
    }
    observed_csv_rows = {}
    for name, expected_count in expected_csv_rows.items():
        observed_count = count_csv_rows(required_files[name])
        observed_csv_rows[name] = observed_count
        if observed_count != expected_count:
            raise RuntimeError(
                f"Artifact gate failed: {name} has {observed_count} rows; expected {expected_count}"
            )

    with required_files["reproduction_json"].open(encoding="utf-8") as handle:
        reproduction = json.load(handle)
    if len(reproduction.get("summary_rows", [])) != 3 or len(reproduction.get("validation_grid", [])) != 72:
        raise RuntimeError("Artifact gate failed: reproduction JSON is incomplete")
    if reproduction.get("metadata", {}).get("validation_winner") is None:
        raise RuntimeError("Artifact gate failed: reproduction JSON has no validation winner")

    with required_files["transfer_summary"].open(encoding="utf-8") as handle:
        transfer = json.load(handle)
    transfer_metadata = transfer.get("metadata", {})
    transfer_global = transfer.get("global_summary", {})
    validated_configuration = transfer_metadata.get("validated_configuration", {})
    if validated_configuration.get("seeds") != sorted(seeds):
        raise RuntimeError("Artifact gate failed: transfer summary seeds are incomplete")
    if validated_configuration.get("template_count") != 41:
        raise RuntimeError("Artifact gate failed: transfer summary template count is not 41")
    if validated_configuration.get("claim_count") != FEVER_LASER_TEST_SIZE:
        raise RuntimeError("Artifact gate failed: transfer summary claim count is incorrect")
    if transfer_global.get("prompt_count") != 40:
        raise RuntimeError("Artifact gate failed: transfer global summary does not cover 40 paraphrases")
    if len(transfer.get("category_summary", [])) != 5 or len(transfer.get("seed_summary", [])) != len(seeds):
        raise RuntimeError("Artifact gate failed: transfer category/seed summaries are incomplete")

    expected_plot_basenames = [
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
    plot_paths = []
    for basename in expected_plot_basenames:
        for extension in ["png", "pdf"]:
            path = plots_root / f"{basename}.{extension}"
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(f"Artifact gate failed: missing or empty plot {path}")
            plot_paths.append(str(path))

    if smoke_required:
        with required_files["smoke_validation"].open(encoding="utf-8") as handle:
            smoke_validation = json.load(handle)
        if smoke_validation.get("status") != "pass":
            raise RuntimeError("Artifact gate failed: smoke validation did not pass")

    return {
        "status": "pass",
        "csv_rows": observed_csv_rows,
        "plot_count": len(plot_paths),
        "plots": plot_paths,
        "required_files": {name: str(path) for name, path in required_files.items()},
    }


def main():
    configure_utf8_stdio()
    parser = argparse.ArgumentParser(
        description="Run the complete paper-aligned GPT-J/FEVER reproduction and fixed-intervention transfer study"
    )
    parser.add_argument("--output-root", default=str(WORKSPACE_DIR / "results"))
    parser.add_argument("--device", choices=["cuda", "auto"], default="cuda")
    parser.add_argument("--model-path", default="EleutherAI/gpt-j-6B")
    parser.add_argument("--revision", default="float16")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--paper-reported-accuracy", type=float, default=56.2)
    parser.add_argument("--paper-reported-note", default="paper baseline 50.2; LASER 56.2")
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--minimum-free-gib", type=float, default=30.0)
    parser.add_argument("--plan-only", action="store_true", help="Validate arguments and print the study plan without inference")
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must not contain duplicates")
    if args.bootstrap_samples <= 0:
        parser.error("--bootstrap-samples must be positive")

    output_root = Path(args.output_root).resolve()
    reproduction_root = output_root / "reproduction_runs"
    transfer_root = output_root / "transfer_runs"
    smoke_root = output_root / "smoke"
    analysis_root = output_root / "analysis"
    reproduction_analysis = analysis_root / "reproduction"
    transfer_analysis = analysis_root / "transfer"
    plots_root = transfer_analysis / "plots"
    logs_root = output_root / "logs"
    manifests_root = output_root / "manifests"
    completion_marker = output_root / "RUN_COMPLETE.json"

    for directory in [reproduction_root, transfer_root, analysis_root, logs_root, manifests_root]:
        directory.mkdir(parents=True, exist_ok=True)

    if not args.plan_only:
        free_gib = shutil.disk_usage(output_root).free / 1024**3
        if free_gib < args.minimum_free_gib:
            raise RuntimeError(
                f"Only {free_gib:.2f} GiB are free on the output volume; "
                f"at least {args.minimum_free_gib:.2f} GiB are required before the study run."
            )
    if not args.plan_only and completion_marker.exists():
        completion_marker.unlink()

    started = time.time()
    manifest_path = manifests_root / "run_manifest.json"
    manifest = {
        "status": "running",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "workspace": str(WORKSPACE_DIR),
        "output_root": str(output_root),
        "host": {"platform": platform.platform(), "python": platform.python_version(), "cpu_count": os.cpu_count()},
        "protocol": {
            "validation_split": "laser_dev",
            "validation_claims": FEVER_LASER_DEV_SIZE,
            "test_split": "laser_test",
            "test_claims": FEVER_LASER_TEST_SIZE,
            "prompt_templates_total": 41,
            "paraphrases": 40,
            "seeds": args.seeds,
            "seed_protocol": {
                "paper_seed_reported": False,
                "validation_selection_seed": 0,
                "transfer_sensitivity_seeds": args.seeds,
                "rng_control_location": "external study runner around unmodified upstream LASER call",
            },
            "published_intervention": PUBLISHED_INTERVENTION,
            "validation_grid_runs": 72,
            "transfer_runs": 41 * (1 + len(args.seeds)),
            "binary_labels": [" true", " false"],
        },
        "source_sha256": source_manifest(),
    }
    atomic_json_dump(manifest, manifest_path)

    if args.plan_only:
        manifest.update(
            {
                "status": "plan_only",
                "planned_total_runs": 72 + 1 + 1 + 41 * (1 + len(args.seeds)) + (0 if args.skip_smoke else 2),
            }
        )
        atomic_json_dump(manifest, manifest_path)
        print(json.dumps(manifest["protocol"], indent=2, sort_keys=True))
        print(f"PLAN_ONLY manifest: {manifest_path}")
        return

    environment = os.environ.copy()
    environment.setdefault("PYTHONUTF8", "1")
    environment.setdefault("PYTHONIOENCODING", "utf-8")
    environment.setdefault("TOKENIZERS_PARALLELISM", "false")
    environment.setdefault("MPLCONFIGDIR", str(output_root / ".mplconfig"))
    environment.setdefault("HF_HOME", str(WORKSPACE_DIR / ".hf_home"))
    environment.setdefault("HF_DATASETS_CACHE", str(WORKSPACE_DIR / ".hf_datasets_cache"))
    os.environ.update(environment)

    try:
        run_logged(
            "01_preflight",
            python_command(
                SRC_DIR / "check_fever_environment.py",
                "--model-path", args.model_path,
                "--device", args.device,
                "--output", manifests_root / "environment.json",
            ),
            logs_root,
        )

        if not args.skip_smoke:
            run_logged(
                "02_cuda_smoke",
                python_command(
                    SRC_DIR / "run_fever_paraphrase_suite.py",
                    "--mode", "all",
                    "--templates", "original",
                    "--home-dir", smoke_root,
                    "--device", args.device,
                    "--fever-split", "laser_test",
                    "--max-examples", str(SMOKE_EXAMPLES),
                    "--seeds", "0",
                    "--baseline-rate", "9.9",
                    "--laser-lname", "fc_in",
                    "--laser-lnum", "24",
                    "--laser-rho", "0.10",
                    "--model_path", args.model_path,
                    "--revision", args.revision,
                ),
                logs_root,
            )
            smoke_validation = validate_smoke_gate(
                smoke_root,
                expected_device="cuda",
                model_path=args.model_path,
                revision=args.revision,
            )
            atomic_json_dump(smoke_validation, manifests_root / "smoke_validation.json")
            manifest["smoke_gate"] = smoke_validation
            atomic_json_dump(manifest, manifest_path)
            print(
                "SMOKE_GATE: PASS "
                f"examples={SMOKE_EXAMPLES} device={smoke_validation['expected_device']} "
                f"baseline_acc={smoke_validation['baseline']['binary_accuracy']:.4f} "
                f"laser_acc={smoke_validation['laser']['binary_accuracy']:.4f}"
            )

        run_logged(
            "03_validation_grid",
            python_command(
                SRC_DIR / "run_fever_reproduction_grid.py",
                "--mode", "grid",
                "--home-dir", reproduction_root,
                "--device", args.device,
                "--fever-split", "laser_dev",
                "--seed", "0",
                "--save-seed-in-path",
                "--model_path", args.model_path,
                "--revision", args.revision,
                "--keep-going",
            ),
            logs_root,
        )

        run_logged(
            "04_published_intervention_test",
            python_command(
                SRC_DIR / "run_fever_reproduction_grid.py",
                "--mode", "published",
                "--home-dir", reproduction_root,
                "--device", args.device,
                "--fever-split", "laser_test",
                "--seed", "0",
                "--save-seed-in-path",
                "--model_path", args.model_path,
                "--revision", args.revision,
            ),
            logs_root,
        )

        winner = select_winner(reproduction_root)
        manifest["validation_winner"] = winner
        atomic_json_dump(manifest, manifest_path)
        print(
            "VALIDATION_WINNER: "
            f"lname={winner['lname']} lnum={winner['lnum']} rho={winner['rho']} rate={winner['rate']}"
        )

        same_as_published = (
            winner["lname"] == PUBLISHED_INTERVENTION["lname"]
            and winner["lnum"] == PUBLISHED_INTERVENTION["lnum"]
            and abs(winner["rho"] - PUBLISHED_INTERVENTION["rho"]) < 1e-12
        )
        if not same_as_published:
            selected_complete, selected_path, selected_reason = selected_test_output_status(
                reproduction_root,
                winner,
                args.model_path,
                args.revision,
                "cuda",
            )
            if selected_complete:
                print(f"[skip] validation-selected test run is complete: {selected_path}")
            else:
                print(f"[repair] validation-selected test run: {selected_reason}")
                run_logged(
                    "05_validation_selected_test",
                    python_command(
                        SRC_DIR / "intervention_gptj_fever_study.py",
                        "--model_path", args.model_path,
                        "--revision", args.revision,
                        "--home_dir", reproduction_root,
                        "--device", args.device,
                        "--lname", winner["lname"],
                        "--lnum", winner["lnum"],
                        "--rho", winner["rho"],
                        "--fever-split", "laser_test",
                        "--intervention-source", "validation_selected",
                        "--seed", "0",
                        "--save-seed-in-path",
                    ),
                    logs_root,
                )
                selected_complete, selected_path, selected_reason = selected_test_output_status(
                    reproduction_root,
                    winner,
                    args.model_path,
                    args.revision,
                    "cuda",
                )
                if not selected_complete:
                    raise RuntimeError(
                        "Validation-selected test run completed but failed output validation: "
                        f"{selected_reason} ({selected_path})"
                    )

        run_logged(
            "06_reproduction_report",
            python_command(
                ANALYSIS_DIR / "fever_reproduction_report.py",
                "--results-root", reproduction_root,
                "--dev-size", FEVER_LASER_DEV_SIZE,
                "--test-size", FEVER_LASER_TEST_SIZE,
                "--validation-split", "laser_dev",
                "--test-split", "laser_test",
                "--paper-reported-accuracy", args.paper_reported_accuracy,
                "--paper-reported-note", args.paper_reported_note,
                "--export-dir", reproduction_analysis,
                "--require-complete-grid",
                "--expected-grid-runs", "72",
            ),
            logs_root,
        )

        run_logged(
            "07_transfer_suite",
            python_command(
                SRC_DIR / "run_fever_paraphrase_suite.py",
                "--mode", "all",
                "--home-dir", transfer_root,
                "--device", args.device,
                "--fever-split", "laser_test",
                "--seeds", *args.seeds,
                "--baseline-lname", "dont",
                "--baseline-lnum", "26",
                "--baseline-rate", "9.9",
                "--laser-lname", winner["lname"],
                "--laser-lnum", winner["lnum"],
                "--laser-rho", winner["rho"],
                "--intervention-source", "validation_selected",
                "--model_path", args.model_path,
                "--revision", args.revision,
                "--keep-going",
            ),
            logs_root,
        )

        run_logged(
            "08_transfer_report",
            python_command(
                ANALYSIS_DIR / "fever_transfer_report.py",
                "--baseline-root", transfer_root / "GPTJ" / "rank-reduction" / "dont",
                "--laser-root", transfer_root / "GPTJ" / "rank-reduction" / winner["lname"],
                "--dev-size", FEVER_LASER_DEV_SIZE,
                "--test-size", FEVER_LASER_TEST_SIZE,
                "--fever-split", "laser_test",
                "--bootstrap-samples", args.bootstrap_samples,
                "--bootstrap-seed", "0",
                "--expected-seeds", *args.seeds,
                "--require-complete-suite",
                "--export-dir", transfer_analysis,
            ),
            logs_root,
        )

        run_logged(
            "09_plots",
            python_command(
                ANALYSIS_DIR / "plot_fever_paraphrase_results.py",
                "--analysis-dir", transfer_analysis,
                "--output-dir", plots_root,
                "--formats", "png", "pdf",
                "--dpi", "220",
                "--language", "en",
                "--style", "paper",
                "--sort-gain", "desc",
                "--with-regression-line",
            ),
            logs_root,
        )

        artifact_validation = validate_artifacts(
            output_root=output_root,
            reproduction_analysis=reproduction_analysis,
            transfer_analysis=transfer_analysis,
            plots_root=plots_root,
            manifests_root=manifests_root,
            seeds=args.seeds,
            smoke_required=not args.skip_smoke,
        )
        atomic_json_dump(artifact_validation, manifests_root / "artifact_validation.json")
        manifest["artifact_gate"] = artifact_validation
        atomic_json_dump(manifest, manifest_path)
        print(
            "ARTIFACT_GATE: PASS "
            f"csv_exports={len(artifact_validation['csv_rows'])} "
            f"plots={artifact_validation['plot_count']}"
        )

        manifest.update(
            {
                "status": "complete",
                "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "elapsed_seconds": time.time() - started,
                "artifact_sha256": artifact_manifest(output_root),
            }
        )
        atomic_json_dump(manifest, manifest_path)
        atomic_json_dump(
            {
                "status": "complete",
                "output_root": str(output_root),
                "manifest": str(manifest_path),
                "plots": str(plots_root),
            },
            completion_marker,
        )
        print(f"\nRUN COMPLETE: {output_root}")
    except BaseException as exc:
        if completion_marker.exists():
            completion_marker.unlink()
        manifest.update(
            {
                "status": "failed",
                "failed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "elapsed_seconds": time.time() - started,
                "error": repr(exc),
            }
        )
        atomic_json_dump(manifest, manifest_path)
        raise


if __name__ == "__main__":
    main()
