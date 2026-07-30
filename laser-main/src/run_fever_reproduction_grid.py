"""Run the canonical-prompt FEVER intervention grid."""

import argparse
import os
import shlex
import subprocess
import sys

from fever_experiment_utils import FEVER_SPLIT_CHOICES, rho_to_rate
from run_fever_paraphrase_suite import expected_example_count, prediction_path, validate_existing_output


DEFAULT_LNAMES = ["fc_in", "fc_out", "q_proj", "k_proj", "v_proj", "out_proj"]
DEFAULT_LNUMS = [20, 22, 24, 26]
DEFAULT_RHOS = [0.10, 0.05, 0.01]


def build_command(script_path, args, lname, lnum, rho, intervention_source):
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
        "--rho",
        str(rho),
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
        "--fever-split",
        args.fever_split,
        "--intervention-source",
        intervention_source,
    ]
    if args.seed is not None:
        command.extend(["--seed", str(args.seed)])
    if args.save_seed_in_path:
        command.append("--save-seed-in-path")
    if args.max_examples is not None:
        command.extend(["--max_examples", str(args.max_examples)])
    return command


def main():
    parser = argparse.ArgumentParser(description="Run canonical GPT-J FEVER reproduction or validation-search grids")
    parser.add_argument("--mode", choices=["published", "grid"], default="grid")
    parser.add_argument("--home-dir", required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--fever-split", choices=FEVER_SPLIT_CHOICES, default="paper_dev")
    parser.add_argument("--model_path", default="EleutherAI/gpt-j-6B")
    parser.add_argument("--revision", default="float16")
    parser.add_argument("--dtpts", type=int, default=22000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_len", type=int, default=1)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--intervention", choices=["dropout", "rank-reduction"], default="rank-reduction")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--save-seed-in-path", action="store_true")
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--lnames", nargs="+", default=DEFAULT_LNAMES)
    parser.add_argument("--lnums", nargs="+", type=int, default=DEFAULT_LNUMS)
    parser.add_argument("--rhos", nargs="+", type=float, default=DEFAULT_RHOS)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-existing", dest="skip_existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    script_path = os.path.join(os.path.dirname(__file__), "intervention_gptj_fever_study.py")
    commands = []
    if args.mode == "published":
        commands.append(("fc_in", 24, 0.01, "published"))
    else:
        for lname in args.lnames:
            for lnum in args.lnums:
                for rho in args.rhos:
                    commands.append((lname, lnum, rho, "validation_candidate"))

    print(f"MODE: {args.mode}")
    print(f"FEVER_SPLIT: {args.fever_split}")
    print(f"TOTAL_RUNS_PLANNED: {len(commands)}")

    completed = 0
    skipped = 0
    failed = 0
    for lname, lnum, rho, intervention_source in commands:
        command = build_command(script_path, args, lname, lnum, rho, intervention_source)
        rate = rho_to_rate(rho)
        output_path = prediction_path(
            home_dir=args.home_dir,
            template_id="original",
            lnum=lnum,
            lname=lname,
            rate=rate,
            dtpts=args.dtpts,
            fever_split=args.fever_split,
            start_index=0,
            max_examples=args.max_examples,
            seed=args.seed,
            seeded_path=args.save_seed_in_path,
        )
        expected = {
            "prediction_count": expected_example_count(args.fever_split, 0, args.max_examples),
            "template": "original",
            "fever_split": args.fever_split,
            "lname": lname,
            "lnum": lnum,
            "rate": rate,
            "seed": args.seed,
            "model_path": args.model_path,
            "revision": args.revision,
            "device": args.device,
            "intervention_source": intervention_source,
            "dtpts": args.dtpts,
            "intervention": args.intervention,
            "require_provenance": True,
        }
        if os.path.exists(output_path) and not args.overwrite:
            valid, reason = validate_existing_output(output_path, expected)
            if valid and args.skip_existing:
                skipped += 1
                print(f"[skip] lname={lname} lnum={lnum} rho={rho} reason=complete")
                continue
            if valid:
                raise FileExistsError(f"Output already exists and overwrite is disabled: {output_path}")
            print(f"[repair] lname={lname} lnum={lnum} rho={rho} reason={reason}")

        print(f"[run] lname={lname} lnum={lnum} rho={rho}")
        print(shlex.join(command))
        if args.dry_run:
            continue
        try:
            subprocess.run(command, check=True, cwd=os.path.dirname(__file__))
            valid, validation_reason = validate_existing_output(output_path, expected)
            if not valid:
                raise RuntimeError(
                    f"Run completed but output validation failed for lname={lname} "
                    f"lnum={lnum} rho={rho}: {validation_reason}"
                )
            completed += 1
        except subprocess.CalledProcessError as exc:
            failed += 1
            print(f"[fail] lname={lname} lnum={lnum} rho={rho} exit_code={exc.returncode}")
            if not args.keep_going:
                raise

    print(f"DONE completed={completed} skipped={skipped} failed={failed} total={len(commands)}")
    if failed > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
