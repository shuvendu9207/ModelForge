"""
Model Evaluator
Computes and displays classification / regression metrics.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, mean_squared_error, r2_score
)


class Evaluator:

    def compute(self, y_true, y_pred, y_proba=None) -> dict:
        metrics = {}

        # Try classification metrics first
        try:
            metrics["accuracy"]  = round(accuracy_score(y_true, y_pred), 4)
            metrics["precision"] = round(precision_score(
                y_true, y_pred, average="weighted", zero_division=0), 4)
            metrics["recall"]    = round(recall_score(
                y_true, y_pred, average="weighted", zero_division=0), 4)
            metrics["f1"]        = round(f1_score(
                y_true, y_pred, average="weighted", zero_division=0), 4)

            if y_proba is not None:
                try:
                    if y_proba.shape[1] == 2:
                        metrics["roc_auc"] = round(
                            roc_auc_score(y_true, y_proba[:, 1]), 4)
                    else:
                        metrics["roc_auc"] = round(
                            roc_auc_score(y_true, y_proba, multi_class="ovr"), 4)
                except Exception:
                    pass

        except Exception:
            # Regression metrics fallback
            metrics["mse"]  = round(mean_squared_error(y_true, y_pred), 4)
            metrics["rmse"] = round(np.sqrt(metrics["mse"]), 4)
            metrics["r2"]   = round(r2_score(y_true, y_pred), 4)

        return metrics

    def print_report(self, metrics: dict):
        print()
        print("  ┌─────────────────────────────────┐")
        print("  │      Evaluation Results          │")
        print("  ├─────────────────────────────────┤")
        for key, val in metrics.items():
            print(f"  │  {key:<18} {val:<12} │")
        print("  └─────────────────────────────────┘")
        print()

    def confusion(self, y_true, y_pred):
        return confusion_matrix(y_true, y_pred)

    def classification_report(self, y_true, y_pred):
        return classification_report(y_true, y_pred)
