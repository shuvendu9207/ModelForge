#!/usr/bin/env python3
"""
ModelForge - Entry Point
Usage:
  python modelforge.py run    --dataset data/raw/data.csv --model random_forest
  python modelforge.py train  --dataset data/raw/data.csv --algorithm xgboost
  python modelforge.py compare
  python modelforge.py dashboard
"""

import argparse

from src.pipeline.pipeline import Pipeline
from src.tracking.tracker  import ExperimentTracker
from src.evaluation.metrics import Evaluator


def run(args):
    print()
    print("=" * 60)
    print("   ModelForge — ML Experiment Pipeline")
    print("=" * 60)

    pipeline = Pipeline(
        dataset_path=args.dataset,
        model_type=args.model,
        params={}
    )
    results = pipeline.execute()

    print()
    print("[*] Pipeline complete.")
    print(f"  Model    : {args.model}")
    print(f"  Accuracy : {results.get('accuracy', 'N/A'):.3f}")
    print(f"  F1 Score : {results.get('f1', 'N/A'):.3f}")
    print()

    tracker = ExperimentTracker()
    exp_id = tracker.log(
        dataset=args.dataset,
        model_type=args.model,
        params={},
        metrics=results,
    )
    print(f"[*] Experiment saved → ID: {exp_id}")


def compare(args):
    tracker = ExperimentTracker()
    experiments = tracker.load_all()

    print()
    print("=" * 60)
    print("   ModelForge — Experiment Comparison")
    print("=" * 60)
    print()
    print(f"  {'ID':<6} {'Model':<20} {'Accuracy':<12} {'F1':<12} {'Time'}")
    print("  " + "-" * 56)
    for exp in experiments:
        print(
            f"  {exp['id']:<6} {exp['model']:<20} "
            f"{exp['metrics'].get('accuracy','N/A'):.3f} "
            f"{exp['metrics'].get('f1','N/A'):.3f} "
            f"{exp.get('training_time','N/A')}"
        )
    print()


def dashboard(args):
    import subprocess
    print("[*] Launching ModelForge dashboard...")
    subprocess.run(["streamlit", "run", "dashboard/app.py"])


def main():
    parser = argparse.ArgumentParser(
        prog="modelforge",
        description="ModelForge End-to-End ML Platform"
    )
    sub = parser.add_subparsers(dest="command")

    # run
    rp = sub.add_parser("run", help="Run full ML pipeline")
    rp.add_argument("--dataset",  required=True, help="Path to dataset (CSV/JSON)")
    rp.add_argument("--model",    default="random_forest",
                    choices=["linear_regression", "logistic_regression",
                             "random_forest", "svm", "xgboost"])
    rp.add_argument("--params",   default=None, help="JSON string of hyperparameters")
    rp.add_argument("--report",   action="store_true", help="Save HTML report")

    # train
    tp = sub.add_parser("train", help="Train a model only")
    tp.add_argument("--dataset",   required=True)
    tp.add_argument("--algorithm", default="random_forest",
                    choices=["linear_regression", "logistic_regression",
                             "random_forest", "svm", "xgboost"])
    tp.add_argument("--output",    default="ml/models/model.pkl")

    # compare
    sub.add_parser("compare", help="Compare all tracked experiments")

    # dashboard
    sub.add_parser("dashboard", help="Launch the Streamlit dashboard")

    args = parser.parse_args()

    if args.command == "run":
        run(args)
    elif args.command == "train":
        from src.models.trainer import train
        train(dataset_path=args.dataset,
              algorithm=args.algorithm,
              output_path=args.output)
    elif args.command == "compare":
        compare(args)
    elif args.command == "dashboard":
        dashboard(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
