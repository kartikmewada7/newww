import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort
from insightface.app import FaceAnalysis

MODEL_PACK = "buffalo_l"
EMBEDDING_MODEL_VERSION = "insightface-buffalo_l-512-v1"
SIMILARITY_THRESHOLD = 0.45  # Initial baseline; calibrate with held-out images.
MIN_MATCH_MARGIN = 0.05      # Initial baseline; calibrate with held-out images.
MIN_FACE_SIZE = 40


def configure_nvidia_dll_paths() -> list[str]:
    """Configure DLL search paths for Windows Python environments.

    On Windows, Python 3.8+ does not search sys.path or site-packages for
    dependent DLLs by default unless os.add_dll_directory() is called.
    This resolves the missing cublasLt64_13.dll error when onnxruntime-gpu
    is paired with pip-installed nvidia packages.
    """
    configured: list[str] = []
    if sys.platform != "win32":
        return configured

    # Check directories under site-packages/nvidia and onnxruntime
    search_roots = [Path(p) for p in sys.path if Path(p).is_dir()]
    for root in search_roots:
        nvidia_root = root / "nvidia"
        if nvidia_root.is_dir():
            for dll in nvidia_root.rglob("*.dll"):
                parent_str = str(dll.parent.resolve())
                if parent_str not in configured:
                    try:
                        os.add_dll_directory(parent_str)
                    except Exception:
                        pass
                    configured.append(parent_str)

    if configured:
        os.environ["PATH"] = (
            os.pathsep.join(configured) + os.pathsep + os.environ.get("PATH", "")
        )

    if hasattr(ort, "preload_dlls"):
        try:
            ort.preload_dlls(directory="")
        except Exception:
            pass

    return configured


@lru_cache(maxsize=1)
def get_face_app() -> FaceAnalysis:
    """Initialize and return the cached InsightFace FaceAnalysis pipeline.

    Attempts GPU execution first via CUDAExecutionProvider. If CUDA fails to
    load or initialize, gracefully falls back to CPUExecutionProvider.
    """
    configure_nvidia_dll_paths()

    available = ort.get_available_providers()
    use_cuda = "CUDAExecutionProvider" in available

    if use_cuda:
        try:
            app = FaceAnalysis(
                name=MODEL_PACK,
                allowed_modules=["detection", "recognition"],
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            app.prepare(ctx_id=0, det_size=(640, 640))
            return app
        except Exception as exc:
            print(f"[FaceEngine] CUDA initialization failed ({exc}); falling back to CPU.")

    app = FaceAnalysis(
        name=MODEL_PACK,
        allowed_modules=["detection", "recognition"],
        providers=["CPUExecutionProvider"],
    )
    app.prepare(ctx_id=-1, det_size=(640, 640))
    return app


def get_active_providers() -> list[str]:
    """Inspect the detection model's active ONNX Runtime session providers."""
    try:
        app = get_face_app()
        det_model = app.models.get("detection")
        if det_model and hasattr(det_model, "session") and det_model.session is not None:
            return list(det_model.session.get_providers())
    except Exception:
        pass
    return ["CPUExecutionProvider"]


def runtime_info() -> dict[str, Any]:
    """Return diagnostic information about the runtime and model session."""
    available = ort.get_available_providers()
    active = get_active_providers()
    gpu_active = "CUDAExecutionProvider" in active

    return {
        "onnxruntime_version": ort.__version__,
        "available_providers": available,
        "active_providers": active,
        "cuda_available": "CUDAExecutionProvider" in available,
        "gpu_active": gpu_active,
        "device": "GPU (CUDAExecutionProvider)" if gpu_active else "CPU (CPUExecutionProvider)",
        "embedding_model_version": EMBEDDING_MODEL_VERSION,
        "model_pack": MODEL_PACK,
    }


def decode_image(image_bytes: bytes) -> np.ndarray:
    if not image_bytes:
        raise ValueError("The uploaded image is empty.")
    array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode this image. Use JPG or PNG.")
    return image


def extract_faces(image_bytes: bytes) -> list[dict[str, Any]]:
    """Detect faces and extract 512-D normalized embeddings from image bytes."""
    image = decode_image(image_bytes)
    faces = get_face_app().get(image)
    results = []

    for face in faces:
        x1, y1, x2, y2 = [int(v) for v in face.bbox]
        if min(x2 - x1, y2 - y1) < MIN_FACE_SIZE:
            continue

        embedding = np.asarray(face.normed_embedding, dtype=np.float32)
        if embedding.shape != (512,):
            raise ValueError(
                f"Expected a 512-D embedding; received {embedding.shape}. "
                "Check that the selected model pack is compatible."
            )

        results.append({
            "bbox": [x1, y1, x2, y2],
            "embedding": embedding.tolist(),
            "det_score": float(getattr(face, "det_score", 0.0)),
            "model_version": EMBEDDING_MODEL_VERSION,
        })

    return results


def cosine_similarity(a: list[float], b: list[float]) -> float:
    va = np.asarray(a, dtype=np.float32)
    vb = np.asarray(b, dtype=np.float32)
    if va.shape != (512,) or vb.shape != (512,):
        raise ValueError("Embedding dimensions do not match this model.")
    denominator = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denominator <= 1e-12:
        return -1.0
    return float(np.dot(va, vb) / denominator)


def predict_embedding(
    embedding: list[float],
    memory_records: list[dict[str, Any]],
    classifier: Any | None,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
    min_match_margin: float = MIN_MATCH_MARGIN,
) -> dict[str, Any]:
    """Predict identity from embedding using similarity gating and classifier consensus."""
    records = [
        row for row in memory_records
        if row.get("model_version") == EMBEDDING_MODEL_VERSION
        and row.get("label")
    ]
    if not records:
        return {
            "label": "Unknown",
            "status": "unknown",
            "similarity": -1.0,
            "margin": 0.0,
            "classifier_label": None,
        }

    # Fast vectorized similarity calculation across all verified examples
    v = np.asarray(embedding, dtype=np.float32)
    v_norm = float(np.linalg.norm(v))
    if v_norm <= 1e-12:
        return {
            "label": "Unknown",
            "status": "unknown",
            "similarity": -1.0,
            "margin": 0.0,
            "classifier_label": None,
        }
    v_unit = v / v_norm

    embs = np.asarray([r["embedding"] for r in records], dtype=np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms <= 1e-12] = 1.0
    sims = np.dot(embs / norms, v_unit)

    class_scores: dict[str, float] = {}
    for idx, row in enumerate(records):
        lbl = row["label"]
        s = float(sims[idx])
        if lbl not in class_scores or s > class_scores[lbl]:
            class_scores[lbl] = s

    ranked = sorted(class_scores.items(), key=lambda item: item[1], reverse=True)
    best_label, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else -1.0
    margin = best_score - second_score

    if best_score < similarity_threshold:
        return {
            "label": "Unknown",
            "status": "unknown",
            "similarity": best_score,
            "margin": margin,
            "classifier_label": None,
        }

    if len(ranked) > 1 and margin < min_match_margin:
        return {
            "label": "Review",
            "status": "ambiguous",
            "similarity": best_score,
            "margin": margin,
            "classifier_label": None,
        }

    classifier_label = None
    if classifier is not None:
        try:
            classifier_label = str(classifier.predict(
                np.asarray(embedding, dtype=np.float32).reshape(1, -1)
            )[0])
        except Exception:
            classifier_label = None

    # Classifier participates in the decision, but the similarity gate
    # prevents a classifier from forcing a low-similarity unknown face to a name.
    if classifier_label is not None and classifier_label != best_label:
        return {
            "label": "Review",
            "status": "ambiguous",
            "similarity": best_score,
            "margin": margin,
            "classifier_label": classifier_label,
        }

    chosen = classifier_label if classifier_label is not None else best_label
    return {
        "label": chosen,
        "status": "matched",
        "similarity": best_score,
        "margin": margin,
        "classifier_label": classifier_label,
    }
