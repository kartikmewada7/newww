import argparse
import json
from collections import defaultdict
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from face_engine import (
    SIMILARITY_THRESHOLD,
    MIN_MATCH_MARGIN,
    predict_embedding,
)
from storage import (
    load_classifier,
    load_evaluation,
    load_memory,
    load_validation,
    save_evaluation_metrics,
)


def evaluate_split(
    eval_records: list[dict[str, Any]],
    training_records: list[dict[str, Any]],
    classifier: Any | None,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
    min_match_margin: float = MIN_MATCH_MARGIN,
) -> dict[str, Any]:
    """Evaluate face recognition on a given set of evaluation records.

    Distinguishes known classes, ambiguous reviews, and unknown rejections.
    Ambiguous/review predictions count as incorrect in standard Top-1 accuracy.
    """
    if not eval_records:
        return {"error": "No evaluation records provided.", "total_samples": 0}

    known_classes = sorted({r["label"] for r in training_records if r.get("label")})

    y_true: list[str] = []
    y_pred: list[str] = []
    raw_predictions: list[dict[str, Any]] = []

    status_counts: dict[str, int] = defaultdict(int)
    correct_count = 0
    unknown_samples = 0
    unknown_correctly_rejected = 0
    unknown_falsely_accepted = 0

    known_samples = 0
    known_correctly_matched = 0
    known_falsely_rejected = 0
    known_wrong_match = 0
    known_ambiguous = 0

    for row in eval_records:
        truth = row["label"]
        emb = row["embedding"]
        pred = predict_embedding(
            emb,
            training_records,
            classifier,
            similarity_threshold=similarity_threshold,
            min_match_margin=min_match_margin,
        )

        status = pred["status"]
        status_counts[status] += 1

        if status == "unknown":
            predicted_label = "__unknown__"
        elif status == "ambiguous":
            predicted_label = "__review__"
        else:
            predicted_label = pred["label"]

        y_true.append(truth)
        y_pred.append(predicted_label)

        is_correct = (truth == predicted_label)
        if is_correct:
            correct_count += 1

        is_unknown_ground_truth = (truth == "__unknown__")

        if is_unknown_ground_truth:
            unknown_samples += 1
            if status == "unknown":
                unknown_correctly_rejected += 1
            else:
                unknown_falsely_accepted += 1
        else:
            known_samples += 1
            if status == "matched" and predicted_label == truth:
                known_correctly_matched += 1
            elif status == "unknown":
                known_falsely_rejected += 1
            elif status == "ambiguous":
                known_ambiguous += 1
            else:
                known_wrong_match += 1

        raw_predictions.append({
            "expected": truth,
            "predicted": predicted_label,
            "status": status,
            "similarity": round(pred["similarity"], 4),
            "margin": round(pred["margin"], 4),
            "classifier_label": pred.get("classifier_label"),
            "correct": is_correct,
            "image_sha256": row.get("image_sha256"),
        })

    total_samples = len(eval_records)
    overall_accuracy = correct_count / total_samples if total_samples > 0 else 0.0

    # Metrics on known identities
    known_accuracy = (
        known_correctly_matched / known_samples if known_samples > 0 else 0.0
    )
    far = (
        unknown_falsely_accepted / unknown_samples if unknown_samples > 0 else 0.0
    )
    frr = (
        known_falsely_rejected / known_samples if known_samples > 0 else 0.0
    )

    # Scikit-learn classification report on known classes
    # We include all classes present in y_true and known_classes
    target_labels = sorted(set(y_true) | {p for p in y_pred if p not in ("__unknown__", "__review__")})
    
    # Calculate macro and weighted precision/recall/f1
    try:
        prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
        rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
        prec_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        rec_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    except Exception:
        prec_macro = rec_macro = f1_macro = 0.0
        prec_weighted = rec_weighted = f1_weighted = 0.0

    # Per-identity statistics
    per_class_stats: dict[str, dict[str, Any]] = {}
    for cls in sorted(set(y_true)):
        cls_indices = [i for i, t in enumerate(y_true) if t == cls]
        cls_total = len(cls_indices)
        cls_correct = sum(1 for i in cls_indices if y_pred[i] == cls)
        cls_ambiguous = sum(1 for i in cls_indices if y_pred[i] == "__review__")
        cls_unknown = sum(1 for i in cls_indices if y_pred[i] == "__unknown__")
        cls_wrong = cls_total - cls_correct - cls_ambiguous - cls_unknown

        per_class_stats[cls] = {
            "support": cls_total,
            "correct": cls_correct,
            "accuracy": cls_correct / cls_total if cls_total > 0 else 0.0,
            "ambiguous": cls_ambiguous,
            "rejected_as_unknown": cls_unknown,
            "wrong_identity": cls_wrong,
        }

    # Confusion matrix on known labels
    all_unique_labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=all_unique_labels).tolist()

    return {
        "total_samples": total_samples,
        "overall_accuracy": round(overall_accuracy, 4),
        "known_samples": known_samples,
        "known_accuracy": round(known_accuracy, 4),
        "precision_macro": round(float(prec_macro), 4),
        "recall_macro": round(float(rec_macro), 4),
        "f1_macro": round(float(f1_macro), 4),
        "precision_weighted": round(float(prec_weighted), 4),
        "recall_weighted": round(float(rec_weighted), 4),
        "f1_weighted": round(float(f1_weighted), 4),
        "unknown_samples": unknown_samples,
        "false_acceptance_rate": round(far, 4),
        "false_rejection_rate": round(frr, 4),
        "status_breakdown": dict(status_counts),
        "similarity_threshold": similarity_threshold,
        "min_match_margin": min_match_margin,
        "per_class_stats": per_class_stats,
        "confusion_matrix_labels": all_unique_labels,
        "confusion_matrix": cm,
        "predictions": raw_predictions,
    }


def sweep_thresholds(
    validation_records: list[dict[str, Any]],
    training_records: list[dict[str, Any]],
    classifier: Any | None,
    thresholds: list[float] | None = None,
    margins: list[float] | None = None,
) -> list[dict[str, Any]]:
    """Sweep similarity thresholds and margins on validation data to calibrate parameters."""
    if thresholds is None:
        thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    if margins is None:
        margins = [0.00, 0.03, 0.05, 0.08, 0.10]

    sweep_results = []
    for th in thresholds:
        for mg in margins:
            res = evaluate_split(
                validation_records,
                training_records,
                classifier,
                similarity_threshold=th,
                min_match_margin=mg,
            )
            sweep_results.append({
                "threshold": th,
                "margin": mg,
                "overall_accuracy": res["overall_accuracy"],
                "known_accuracy": res["known_accuracy"],
                "f1_macro": res["f1_macro"],
                "matched_count": res["status_breakdown"].get("matched", 0),
                "ambiguous_count": res["status_breakdown"].get("ambiguous", 0),
                "unknown_count": res["status_breakdown"].get("unknown", 0),
                "far": res["false_acceptance_rate"],
                "frr": res["false_rejection_rate"],
            })

    # Sort best by F1 macro, then overall accuracy
    sweep_results.sort(key=lambda x: (x["f1_macro"], x["overall_accuracy"]), reverse=True)
    return sweep_results


def print_evaluation_report(results: dict[str, Any], title: str = "Evaluation Report") -> None:
    """Print a clean, formatted terminal summary of evaluation results."""
    print(f"\n{'=' * 60}")
    print(f" {title.upper()} ")
    print(f"{'=' * 60}")
    print(f"Total evaluation samples : {results.get('total_samples', 0)}")
    print(f"Overall Top-1 Accuracy   : {results.get('overall_accuracy', 0):.2%}")
    print(f"Known-identity Accuracy  : {results.get('known_accuracy', 0):.2%}")
    print(f"Macro Precision / Recall : {results.get('precision_macro', 0):.4f} / {results.get('recall_macro', 0):.4f}")
    print(f"Macro F1 Score           : {results.get('f1_macro', 0):.4f}")
    print(f"Weighted F1 Score        : {results.get('f1_weighted', 0):.4f}")
    print(f"Similarity Threshold     : {results.get('similarity_threshold', 0)}")
    print(f"Match Margin Threshold   : {results.get('min_match_margin', 0)}")

    if results.get("unknown_samples", 0) > 0:
        print(f"Unknown samples          : {results.get('unknown_samples')}")
        print(f"False Acceptance Rate    : {results.get('false_acceptance_rate', 0):.2%}")
        print(f"False Rejection Rate     : {results.get('false_rejection_rate', 0):.2%}")

    print("\n--- Prediction Status Breakdown ---")
    for status, count in results.get("status_breakdown", {}).items():
        pct = count / results["total_samples"] if results["total_samples"] > 0 else 0
        print(f"  {status:<12}: {count:>4} ({pct:.1%})")

    per_class = results.get("per_class_stats", {})
    if per_class:
        print("\n--- Per-Identity Summary (Top 10 by support) ---")
        sorted_classes = sorted(per_class.items(), key=lambda x: x[1]["support"], reverse=True)[:10]
        print(f"{'Identity':<25} {'Support':>8} {'Correct':>8} {'Accuracy':>10}")
        print("-" * 55)
        for name, stats in sorted_classes:
            print(f"{name:<25} {stats['support']:>8} {stats['correct']:>8} {stats['accuracy']:>9.1%}")
    print(f"{'=' * 60}\n")


def evaluate_verification_pairs(
    pairs: list[dict[str, Any]],
    threshold: float = 0.45,
) -> dict[str, Any]:
    """Evaluate pairwise face verification on genuine and impostor pairs.

    For each pair, cosine similarity S = (e1 . e2) / (|e1| * |e2|).
    Prediction is True (Same) if S >= threshold, else False (Different).
    """
    if not pairs:
        return {"error": "No verification pairs provided.", "total_pairs": 0}

    y_true: list[bool] = []
    y_scores: list[float] = []
    y_pred: list[bool] = []
    pair_results: list[dict[str, Any]] = []

    true_accepts = 0   # TP: Genuine pair predicted as Same
    false_rejects = 0  # FN: Genuine pair predicted as Different
    true_rejects = 0   # TN: Impostor pair predicted as Different
    false_accepts = 0  # FP: Impostor pair predicted as Same

    for p in pairs:
        same = bool(p["same"])
        e1 = np.asarray(p["embedding1"], dtype=np.float32)
        e2 = np.asarray(p["embedding2"], dtype=np.float32)

        norm1 = np.linalg.norm(e1)
        norm2 = np.linalg.norm(e2)
        if norm1 > 0 and norm2 > 0:
            sim = float(np.dot(e1, e2) / (norm1 * norm2))
        else:
            sim = 0.0

        pred_same = bool(sim >= threshold)

        y_true.append(same)
        y_scores.append(float(sim))
        y_pred.append(pred_same)

        if same:
            if pred_same:
                true_accepts += 1
            else:
                false_rejects += 1
        else:
            if pred_same:
                false_accepts += 1
            else:
                true_rejects += 1

        pair_results.append({
            "name1": p.get("name1"),
            "image1": p.get("image1"),
            "name2": p.get("name2"),
            "image2": p.get("image2"),
            "ground_truth_same": bool(same),
            "predicted_same": bool(pred_same),
            "similarity": round(float(sim), 4),
            "correct": bool(pred_same == same),
        })

    total_pairs = len(pairs)
    genuine_count = true_accepts + false_rejects
    impostor_count = true_rejects + false_accepts

    accuracy = (true_accepts + true_rejects) / total_pairs if total_pairs > 0 else 0.0
    tar = true_accepts / genuine_count if genuine_count > 0 else 0.0
    frr = false_rejects / genuine_count if genuine_count > 0 else 0.0
    far = false_accepts / impostor_count if impostor_count > 0 else 0.0
    trr = true_rejects / impostor_count if impostor_count > 0 else 0.0

    precision = true_accepts / (true_accepts + false_accepts) if (true_accepts + false_accepts) > 0 else 0.0
    f1 = (2 * precision * tar) / (precision + tar) if (precision + tar) > 0 else 0.0

    try:
        auc = float(roc_auc_score(y_true, y_scores))
    except Exception:
        auc = 0.0

    return {
        "total_pairs": total_pairs,
        "genuine_pairs": genuine_count,
        "impostor_pairs": impostor_count,
        "threshold": round(threshold, 4),
        "accuracy": round(accuracy, 4),
        "tar": round(tar, 4),
        "frr": round(frr, 4),
        "far": round(far, 4),
        "trr": round(trr, 4),
        "precision": round(precision, 4),
        "f1": round(f1, 4),
        "roc_auc": round(auc, 4),
        "true_accepts": true_accepts,
        "false_rejects": false_rejects,
        "true_rejects": true_rejects,
        "false_accepts": false_accepts,
        "pairs": pair_results,
    }


def sweep_verification_thresholds(
    pairs: list[dict[str, Any]],
    thresholds: list[float] | None = None,
) -> list[dict[str, Any]]:
    """Sweep cosine similarity threshold to find optimal verification cutoff."""
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.linspace(0.10, 0.90, 81)]

    sweep_results = []
    for th in thresholds:
        res = evaluate_verification_pairs(pairs, threshold=th)
        sweep_results.append({
            "threshold": th,
            "accuracy": res["accuracy"],
            "tar": res["tar"],
            "far": res["far"],
            "frr": res["frr"],
            "f1": res["f1"],
            "precision": res["precision"],
            "roc_auc": res["roc_auc"],
        })

    sweep_results.sort(key=lambda x: (x["accuracy"], x["f1"]), reverse=True)
    return sweep_results


def print_verification_report(results: dict[str, Any], title: str = "LFW Verification Report") -> None:
    """Print a clean, formatted terminal summary of verification performance."""
    print(f"\n{'=' * 60}")
    print(f" {title.upper()} ")
    print(f"{'=' * 60}")
    print(f"Total verification pairs : {results.get('total_pairs', 0)}")
    print(f"  - Genuine pairs (Same) : {results.get('genuine_pairs', 0)}")
    print(f"  - Impostor pairs (Diff): {results.get('impostor_pairs', 0)}")
    print(f"Calibrated Threshold     : {results.get('threshold', 0):.4f}")
    print(f"Verification Accuracy    : {results.get('accuracy', 0):.2%}")
    print(f"True Accept Rate (TAR)   : {results.get('tar', 0):.2%} (True Positives: {results.get('true_accepts')})")
    print(f"False Reject Rate (FRR)  : {results.get('frr', 0):.2%} (False Negatives: {results.get('false_rejects')})")
    print(f"False Accept Rate (FAR)  : {results.get('far', 0):.2%} (False Positives: {results.get('false_accepts')})")
    print(f"True Reject Rate (TRR)   : {results.get('trr', 0):.2%} (True Negatives: {results.get('true_rejects')})")
    print(f"Precision / Recall / F1  : {results.get('precision', 0):.4f} / {results.get('tar', 0):.4f} / {results.get('f1', 0):.4f}")
    print(f"ROC-AUC Score            : {results.get('roc_auc', 0):.4f}")
    print(f"{'=' * 60}\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate face recognition classifier or verification benchmark.")
    parser.add_argument("--dataset", choices=["primary", "lfw"], default="primary", help="Dataset benchmark to evaluate")
    parser.add_argument("--protocol", choices=["classification", "verification"], default="classification", help="Evaluation protocol")
    parser.add_argument("--split", choices=["test", "val", "both"], default="test", help="Split to evaluate")
    parser.add_argument("--threshold", type=float, default=None, help="Cosine similarity threshold (default: auto-calibrated or 0.45)")
    parser.add_argument("--margin", type=float, default=MIN_MATCH_MARGIN, help="Minimum match margin")
    parser.add_argument("--sweep", action="store_true", help="Sweep thresholds on validation data")
    args = parser.parse_args()

    default_th = args.threshold if args.threshold is not None else SIMILARITY_THRESHOLD

    if args.protocol == "verification":
        from storage import (
            load_lfw_verification_pairs,
            load_lfw_verification_metrics,
            save_lfw_verification_metrics,
        )
        pairs_data = load_lfw_verification_pairs()
        dev_train = pairs_data.get("dev_train", [])
        dev_test = pairs_data.get("dev_test", [])

        print("=" * 60)
        print(" LFW PAIR VERIFICATION BENCHMARK EVALUATION ")
        print("=" * 60)
        print(f"Loaded {len(dev_train)} DevTrain (validation) pairs and {len(dev_test)} DevTest (held-out) pairs.")

        if args.sweep or (args.split in ("val", "both") and args.threshold is None):
            if not dev_train:
                print("No DevTrain pairs available for calibration sweep.")
                return
            print(f"\nCalibrating threshold on {len(dev_train)} validation pairs...")
            sweep = sweep_verification_thresholds(dev_train)
            print("\n--- Top 5 Calibrated Thresholds on Validation Pairs ---")
            for s in sweep[:5]:
                print(f"Th={s['threshold']:.2f} -> Acc={s['accuracy']:.2%}, TAR={s['tar']:.2%}, FAR={s['far']:.2%}, F1={s['f1']:.4f}, AUC={s['roc_auc']:.4f}")
            calibrated_threshold = sweep[0]["threshold"]
            print(f"\nSelected optimal threshold from validation: {calibrated_threshold:.2f} (Accuracy: {sweep[0]['accuracy']:.2%})")
            if args.sweep:
                return
        else:
            calibrated_threshold = default_th

        val_metrics = {}
        if args.split in ("val", "both"):
            val_metrics = evaluate_verification_pairs(dev_train, threshold=calibrated_threshold)
            print_verification_report(val_metrics, title="LFW DevTrain (Validation) Verification Report")

        test_metrics = {}
        if args.split in ("test", "both"):
            if not dev_test:
                print("No DevTest pairs found. Run import_lfw_dataset.py first.")
                return
            test_metrics = evaluate_verification_pairs(dev_test, threshold=calibrated_threshold)
            print_verification_report(test_metrics, title="LFW DevTest (Held-Out Test) Verification Report")

            benchmark_report = {
                "dataset": "LFW (lfw-deepfunneled)",
                "protocol": "pair_verification",
                "calibrated_threshold": calibrated_threshold,
                "val_metrics": val_metrics,
                "test_metrics": test_metrics,
            }
            save_lfw_verification_metrics(benchmark_report)
            print("  Saved verification benchmark metrics to data/lfw_verification_metrics.json")
        return

    # Standard / Open-set classification protocol
    if args.dataset == "lfw":
        from storage import (
            load_lfw_memory,
            load_lfw_validation,
            load_lfw_evaluation,
            load_lfw_classifier,
            save_lfw_evaluation_metrics,
        )
        memory = load_lfw_memory()
        clf, model_info = load_lfw_classifier()
        val_loader = load_lfw_validation
        eval_loader = load_lfw_evaluation
        metrics_saver = save_lfw_evaluation_metrics
        dataset_name = "LFW (lfw-deepfunneled)"
    else:
        memory = load_memory()
        clf, model_info = load_classifier()
        val_loader = load_validation
        eval_loader = load_evaluation
        metrics_saver = save_evaluation_metrics
        dataset_name = "Primary Kaggle Dataset"

    print(f"[{dataset_name}] Loaded classifier ({model_info.get('status')})")
    print(f"Training memory size: {len(memory)} samples, {len(model_info.get('classes', []))} classes")

    if args.sweep:
        val_records = val_loader()
        if not val_records:
            print(f"No validation records found for {dataset_name}.")
            return
        print(f"Sweeping thresholds on {len(val_records)} validation samples...")
        sweep = sweep_thresholds(val_records, memory, clf)
        print("\n--- Top 5 Parameter Configurations on Validation Set ---")
        for s in sweep[:5]:
            print(f"Th={s['threshold']:.2f}, Mg={s['margin']:.2f} -> Acc={s['overall_accuracy']:.1%}, F1={s['f1_macro']:.4f}, Matched={s['matched_count']}, Ambiguous={s['ambiguous_count']}, Unknown={s['unknown_count']}")
        return

    if args.split in ("val", "both"):
        val_records = val_loader()
        if val_records:
            val_results = evaluate_split(val_records, memory, clf, default_th, args.margin)
            print_evaluation_report(val_results, title=f"{dataset_name} - Validation Evaluation")
        else:
            print(f"No validation records found for {dataset_name}.")

    if args.split in ("test", "both"):
        eval_records = eval_loader()
        if eval_records:
            test_results = evaluate_split(eval_records, memory, clf, default_th, args.margin)
            print_evaluation_report(test_results, title=f"{dataset_name} - Held-Out Test Evaluation")
            metrics_saver(test_results)
        else:
            print(f"No test records found for {dataset_name}.")


if __name__ == "__main__":
    main()

