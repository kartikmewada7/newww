import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.svm import SVC

from face_engine import EMBEDDING_MODEL_VERSION

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
MEMORY_PATH = DATA_DIR / "memory.json"
VALIDATION_PATH = DATA_DIR / "validation.json"
EVALUATION_PATH = DATA_DIR / "evaluation.json"
FEEDBACK_PATH = DATA_DIR / "feedback_log.json"
MANIFEST_PATH = DATA_DIR / "kaggle_manifest.json"
METRICS_PATH = DATA_DIR / "evaluation_metrics.json"
CURRENT_MODEL_PATH = MODELS_DIR / "current.json"

# LFW Dataset Isolated Storage Paths
LFW_MEMORY_PATH = DATA_DIR / "lfw_memory.json"
LFW_VALIDATION_PATH = DATA_DIR / "lfw_validation.json"
LFW_EVALUATION_PATH = DATA_DIR / "lfw_evaluation.json"
LFW_MANIFEST_PATH = DATA_DIR / "lfw_manifest.json"
LFW_METRICS_PATH = DATA_DIR / "lfw_evaluation_metrics.json"
LFW_MODEL_INFO_PATH = MODELS_DIR / "lfw_model.json"
LFW_CLASSIFIER_PATH = MODELS_DIR / "lfw_classifier.joblib"
LFW_VERIFICATION_PAIRS_PATH = DATA_DIR / "lfw_verification_pairs.json"
LFW_VERIFICATION_METRICS_PATH = DATA_DIR / "lfw_verification_metrics.json"

DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.bool_, np.generic)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".tmp_{os.getpid()}")
    with temp.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, separators=(',', ':'), cls=NumpyEncoder)
    replaced = False
    for attempt in range(5):
        try:
            os.replace(temp, path)
            replaced = True
            break
        except PermissionError:
            import time
            time.sleep(0.05 * (attempt + 1))
    if not replaced:
        with path.open("w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, separators=(',', ':'), cls=NumpyEncoder)
        try:
            temp.unlink(missing_ok=True)
        except Exception:
            pass


def load_memory() -> list[dict[str, Any]]:
    return _read_json(MEMORY_PATH, {"records": []}).get("records", [])


def save_memory(records: list[dict[str, Any]]) -> None:
    _atomic_json(MEMORY_PATH, {
        "schema_version": 1,
        "embedding_model_version": EMBEDDING_MODEL_VERSION,
        "updated_at": utc_now(),
        "records": records,
    })


def load_validation() -> list[dict[str, Any]]:
    return _read_json(VALIDATION_PATH, {"records": []}).get("records", [])


def save_validation(records: list[dict[str, Any]]) -> None:
    _atomic_json(VALIDATION_PATH, {
        "schema_version": 1,
        "embedding_model_version": EMBEDDING_MODEL_VERSION,
        "updated_at": utc_now(),
        "records": records,
    })


def load_evaluation() -> list[dict[str, Any]]:
    return _read_json(EVALUATION_PATH, {"records": []}).get("records", [])


def save_evaluation(records: list[dict[str, Any]]) -> None:
    _atomic_json(EVALUATION_PATH, {
        "schema_version": 1,
        "embedding_model_version": EMBEDDING_MODEL_VERSION,
        "updated_at": utc_now(),
        "records": records,
    })


def load_manifest() -> list[dict[str, Any]]:
    return _read_json(MANIFEST_PATH, {"records": []}).get("records", [])


def save_manifest(records: list[dict[str, Any]]) -> None:
    _atomic_json(MANIFEST_PATH, {
        "schema_version": 1,
        "updated_at": utc_now(),
        "records": records,
    })


def load_evaluation_metrics() -> dict[str, Any]:
    return _read_json(METRICS_PATH, {})


def save_evaluation_metrics(metrics: dict[str, Any]) -> None:
    _atomic_json(METRICS_PATH, {
        **metrics,
        "updated_at": utc_now(),
    })


def append_feedback(entry: dict[str, Any]) -> None:
    data = _read_json(FEEDBACK_PATH, {"records": []})
    data["records"].append({**entry, "created_at": utc_now()})
    _atomic_json(FEEDBACK_PATH, data)


def get_current_model_info() -> dict[str, Any]:
    return _read_json(CURRENT_MODEL_PATH, {
        "version": 0,
        "artifact": None,
        "trained_at": None,
        "sample_count": 0,
        "classes": [],
        "status": "not_trained",
    })


def load_classifier():
    info = get_current_model_info()
    artifact = info.get("artifact")
    if not artifact:
        return None, info
    path = MODELS_DIR / artifact
    if not path.exists():
        return None, {**info, "status": "artifact_missing"}
    try:
        return joblib.load(path), info
    except Exception:
        return None, {**info, "status": "artifact_load_failed"}


def get_memory_summary() -> dict[str, Any]:
    """Return a detailed summary of memory records by source and identity."""
    records = load_memory()
    source_counts: dict[str, int] = {}
    identity_counts: dict[str, int] = {}

    for r in records:
        src = r.get("source", "unknown")
        lbl = r.get("label", "unknown")
        source_counts[src] = source_counts.get(src, 0) + 1
        identity_counts[lbl] = identity_counts.get(lbl, 0) + 1

    return {
        "total_records": len(records),
        "source_counts": source_counts,
        "unique_identities": len(identity_counts),
        "identity_counts": identity_counts,
    }


def reset_dataset_records(preserve_user_data: bool = True) -> dict[str, Any]:
    """Safely remove dataset-imported records without deleting user registrations.

    If preserve_user_data is True (default), records with source 'registration'
    or 'teacher_feedback' are strictly preserved in memory.json.
    Only records with source 'kaggle_dataset' (or benchmark imports) are removed.
    Validation and test evaluation sets from Kaggle dataset are also cleared.
    The classifier is retrained on the remaining verified user records if >= 2 classes exist.
    """
    memory = load_memory()
    if preserve_user_data:
        kept_memory = [r for r in memory if r.get("source") != "kaggle_dataset"]
        removed_count = len(memory) - len(kept_memory)
    else:
        kept_memory = []
        removed_count = len(memory)

    save_memory(kept_memory)

    # Filter evaluation records: preserve non-dataset evaluations if any
    eval_records = load_evaluation()
    kept_eval = [r for r in eval_records if r.get("source") != "kaggle_dataset"]
    save_evaluation(kept_eval)

    # Filter validation records
    val_records = load_validation()
    kept_val = [r for r in val_records if r.get("source") != "kaggle_dataset"]
    save_validation(kept_val)

    # Retrain classifier on remaining records
    new_info = train_and_version_classifier(kept_memory)

    return {
        "status": "dataset_reset_completed",
        "removed_memory_records": removed_count,
        "kept_memory_records": len(kept_memory),
        "kept_identities": sorted({r["label"] for r in kept_memory if r.get("label")}),
        "classifier_status": new_info.get("status"),
        "classifier_version": new_info.get("version"),
    }


def train_and_version_classifier(
    records: list[dict[str, Any]],
    metadata_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train an SVC classifier on usable memory records and save a versioned artifact.

    Evaluation or validation records must NEVER be passed to this function.
    Only verified training memory records with valid 512-D embeddings are used.
    """
    usable = [
        r for r in records
        if r.get("model_version") == EMBEDDING_MODEL_VERSION
        and r.get("label")
        and len(r.get("embedding", [])) == 512
    ]
    labels = sorted({r["label"] for r in usable})
    sample_count = len(usable)

    if len(labels) < 2:
        info = {
            "version": get_current_model_info().get("version", 0),
            "artifact": None,
            "trained_at": utc_now(),
            "sample_count": sample_count,
            "classes": labels,
            "status": "memory_only_need_two_identities",
        }
        if metadata_extra:
            info.update(metadata_extra)
        _atomic_json(CURRENT_MODEL_PATH, info)
        return info

    x = np.asarray([r["embedding"] for r in usable], dtype=np.float32)
    y = np.asarray([r["label"] for r in usable], dtype=str)

    clf = SVC(
        kernel="rbf",
        C=2.0,
        class_weight="balanced",
        decision_function_shape="ovr",
    )
    clf.fit(x, y)

    old_version = int(get_current_model_info().get("version", 0))
    new_version = old_version + 1
    artifact_name = f"classifier_v{new_version:04d}.joblib"
    artifact_path = MODELS_DIR / artifact_name
    temp_path = artifact_path.with_suffix(".joblib.tmp")
    joblib.dump(clf, temp_path)
    os.replace(temp_path, artifact_path)

    info = {
        "version": new_version,
        "artifact": artifact_name,
        "trained_at": utc_now(),
        "sample_count": sample_count,
        "classes": labels,
        "status": "trained",
        "embedding_model_version": EMBEDDING_MODEL_VERSION,
        "classifier": "sklearn.svm.SVC",
    }
    if metadata_extra:
        info.update(metadata_extra)

    _atomic_json(CURRENT_MODEL_PATH, info)
    return info


# ---------------------------------------------------------------------------
# LFW Dataset Isolated Storage & Model Management Functions
# ---------------------------------------------------------------------------

def load_lfw_memory() -> list[dict[str, Any]]:
    return _read_json(LFW_MEMORY_PATH, {"records": []}).get("records", [])


def save_lfw_memory(records: list[dict[str, Any]]) -> None:
    _atomic_json(LFW_MEMORY_PATH, {
        "schema_version": 1,
        "dataset": "LFW (lfw-deepfunneled)",
        "embedding_model_version": EMBEDDING_MODEL_VERSION,
        "updated_at": utc_now(),
        "records": records,
    })


def load_lfw_validation() -> list[dict[str, Any]]:
    return _read_json(LFW_VALIDATION_PATH, {"records": []}).get("records", [])


def save_lfw_validation(records: list[dict[str, Any]]) -> None:
    _atomic_json(LFW_VALIDATION_PATH, {
        "schema_version": 1,
        "dataset": "LFW (lfw-deepfunneled)",
        "embedding_model_version": EMBEDDING_MODEL_VERSION,
        "updated_at": utc_now(),
        "records": records,
    })


def load_lfw_evaluation() -> list[dict[str, Any]]:
    return _read_json(LFW_EVALUATION_PATH, {"records": []}).get("records", [])


def save_lfw_evaluation(records: list[dict[str, Any]]) -> None:
    _atomic_json(LFW_EVALUATION_PATH, {
        "schema_version": 1,
        "dataset": "LFW (lfw-deepfunneled)",
        "embedding_model_version": EMBEDDING_MODEL_VERSION,
        "updated_at": utc_now(),
        "records": records,
    })


def load_lfw_manifest() -> list[dict[str, Any]]:
    return _read_json(LFW_MANIFEST_PATH, {"records": []}).get("records", [])


def save_lfw_manifest(records: list[dict[str, Any]]) -> None:
    _atomic_json(LFW_MANIFEST_PATH, {
        "schema_version": 1,
        "dataset": "LFW (lfw-deepfunneled)",
        "updated_at": utc_now(),
        "records": records,
    })


def load_lfw_evaluation_metrics() -> dict[str, Any]:
    return _read_json(LFW_METRICS_PATH, {})


def save_lfw_evaluation_metrics(metrics: dict[str, Any]) -> None:
    _atomic_json(LFW_METRICS_PATH, {
        **metrics,
        "updated_at": utc_now(),
    })


def load_lfw_verification_pairs() -> dict[str, Any]:
    return _read_json(LFW_VERIFICATION_PAIRS_PATH, {"dev_train": [], "dev_test": []})


def save_lfw_verification_pairs(data: dict[str, Any]) -> None:
    _atomic_json(LFW_VERIFICATION_PAIRS_PATH, {
        "schema_version": 1,
        "dataset": "LFW (lfw-deepfunneled)",
        "updated_at": utc_now(),
        **data,
    })


def load_lfw_verification_metrics() -> dict[str, Any]:
    return _read_json(LFW_VERIFICATION_METRICS_PATH, {})


def save_lfw_verification_metrics(metrics: dict[str, Any]) -> None:
    _atomic_json(LFW_VERIFICATION_METRICS_PATH, {
        **metrics,
        "updated_at": utc_now(),
    })


def get_lfw_model_info() -> dict[str, Any]:
    return _read_json(LFW_MODEL_INFO_PATH, {
        "status": "not_trained",
        "dataset": "LFW",
        "sample_count": 0,
        "classes": [],
        "artifact": None,
    })


def load_lfw_classifier():
    info = get_lfw_model_info()
    if not LFW_CLASSIFIER_PATH.exists():
        return None, {**info, "status": "artifact_missing"}
    try:
        return joblib.load(LFW_CLASSIFIER_PATH), info
    except Exception:
        return None, {**info, "status": "artifact_load_failed"}


def train_and_save_lfw_classifier(
    records: list[dict[str, Any]],
    metadata_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train an SVC classifier exclusively for the LFW experiment.

    Saves directly to models/lfw_classifier.joblib and models/lfw_model.json.
    Does NOT touch models/current.json or the primary classifier.
    """
    usable = [
        r for r in records
        if r.get("model_version") == EMBEDDING_MODEL_VERSION
        and r.get("label")
        and len(r.get("embedding", [])) == 512
    ]
    labels = sorted({r["label"] for r in usable})
    sample_count = len(usable)

    if len(labels) < 2:
        info = {
            "artifact": None,
            "trained_at": utc_now(),
            "sample_count": sample_count,
            "classes": labels,
            "status": "need_at_least_two_identities",
            "dataset": "LFW",
            "embedding_model_version": EMBEDDING_MODEL_VERSION,
        }
        if metadata_extra:
            info.update(metadata_extra)
        _atomic_json(LFW_MODEL_INFO_PATH, info)
        return info

    x = np.asarray([r["embedding"] for r in usable], dtype=np.float32)
    y = np.asarray([r["label"] for r in usable], dtype=str)

    clf = SVC(
        kernel="rbf",
        C=2.0,
        class_weight="balanced",
        decision_function_shape="ovr",
    )
    clf.fit(x, y)

    temp_path = LFW_CLASSIFIER_PATH.with_suffix(".joblib.tmp")
    joblib.dump(clf, temp_path)
    os.replace(temp_path, LFW_CLASSIFIER_PATH)

    info = {
        "artifact": LFW_CLASSIFIER_PATH.name,
        "trained_at": utc_now(),
        "sample_count": sample_count,
        "classes": labels,
        "status": "trained",
        "dataset": "LFW",
        "embedding_model_version": EMBEDDING_MODEL_VERSION,
        "classifier": "sklearn.svm.SVC",
    }
    if metadata_extra:
        info.update(metadata_extra)

    _atomic_json(LFW_MODEL_INFO_PATH, info)
    return info

