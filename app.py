import hashlib
from datetime import datetime, timezone

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from face_engine import (
    EMBEDDING_MODEL_VERSION,
    decode_image,
    extract_faces,
    predict_embedding,
    runtime_info,
)
from storage import (
    append_feedback,
    get_current_model_info,
    get_memory_summary,
    load_classifier,
    load_evaluation,
    load_evaluation_metrics,
    load_manifest,
    load_memory,
    load_validation,
    reset_dataset_records,
    save_evaluation,
    save_memory,
    sha256_bytes,
    train_and_version_classifier,
    utc_now,
)
from evaluation import evaluate_split

st.set_page_config(
    page_title="Face Recognition Learning Lab",
    page_icon="🧠",
    layout="wide",
)

st.title("Face Recognition Learning Lab")
st.caption(
    "Standalone prototype · local embeddings · teacher feedback · "
    "versioned classifier · held-out evaluation · Kaggle benchmark"
)

info = runtime_info()
mem_summary = get_memory_summary()
current_info = get_current_model_info()

with st.sidebar:
    st.header("Runtime Environment")
    st.write(f"ONNX Runtime: `{info['onnxruntime_version']}`")
    st.write(f"Device: **{info['device']}**")
    st.write(f"CUDA provider available: **{info['cuda_available']}**")
    st.write(f"GPU session active: **{info['gpu_active']}**")
    st.write(f"Embedding model: `{info['embedding_model_version']}`")

    if info["gpu_active"]:
        st.success("CUDA execution provider is active on GPU.")
    elif info["cuda_available"]:
        st.warning("CUDA is reported available, but model session is running on CPU.")
    else:
        st.info("Running on CPUExecutionProvider (CPU fallback).")

    st.markdown("---")
    st.header("Classifier Model")
    st.write(f"Version: **v{current_info.get('version', 0):04d}**")
    st.write(f"Status: `{current_info.get('status', 'unknown')}`")
    st.write(f"Classes: **{len(current_info.get('classes', []))}**")
    st.write(f"Training samples: **{current_info.get('sample_count', 0)}**")

    st.markdown("---")
    st.header("Memory Breakdown")
    for src, cnt in mem_summary["source_counts"].items():
        st.write(f"• `{src}`: {cnt}")
    st.write(f"Total verified samples: **{mem_summary['total_records']}**")
    st.write(f"Total identities: **{mem_summary['unique_identities']}**")

memory = load_memory()
names = sorted({r["label"] for r in memory if r.get("label")})
tabs = st.tabs([
    "1 · Register / Add Memory",
    "2 · Group Photo + Feedback",
    "3 · Held-out Evaluation",
    "4 · Model Versions & Dataset",
])


def image_to_rgb(image_bytes: bytes):
    bgr = decode_image(image_bytes)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def draw_results(image_bytes: bytes, face_rows: list[dict]):
    bgr = decode_image(image_bytes)
    for row in face_rows:
        x1, y1, x2, y2 = row["bbox"]
        status = row.get("status", "unknown")
        label = row.get("display_label", row.get("label", "Unknown"))
        color = (0, 170, 0) if status == "matched" else (0, 150, 255)
        cv2.rectangle(bgr, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            bgr, str(label)[:36], (x1, max(22, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA
        )
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


with tabs[0]:
    st.subheader("Register student examples")
    st.info(
        "Use only images you have permission to process. Registration photos "
        "are training memory. Do not use held-out evaluation captures here."
    )
    existing_names = sorted(set(names))
    selected_mode = st.radio(
        "Student", ["Add a new identity", "Add examples to an existing identity"],
        horizontal=True,
    )
    if selected_mode == "Add examples to an existing identity" and existing_names:
        label = st.selectbox("Existing identity", existing_names, key="existing_label")
    else:
        label = st.text_input("New student label", key="new_label").strip()

    uploads = st.file_uploader(
        "Upload individual face photos (one face per image)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="registration_uploads",
    )
    st.caption(
        "For a useful initial prototype, capture multiple images per person "
        "under different lighting/angles. Avoid near-duplicate frames."
    )

    if st.button("Save verified examples and train", type="primary"):
        label = label.strip()
        if not label or not uploads:
            st.error("Enter/select an identity and upload at least one image.")
        else:
            current_memory = load_memory()
            existing = {r["label"].casefold() for r in current_memory}
            if selected_mode == "Add a new identity" and label.casefold() in existing:
                st.error("That identity already exists. Choose add examples instead.")
            else:
                new_records = []
                seen_hashes = {r.get("image_sha256") for r in current_memory}
                try:
                    for upload in uploads:
                        raw = upload.getvalue()
                        digest = sha256_bytes(raw)
                        if digest in seen_hashes:
                            raise ValueError(
                                f"{upload.name}: exact duplicate image already stored."
                            )
                        faces = extract_faces(raw)
                        if len(faces) != 1:
                            raise ValueError(
                                f"{upload.name}: expected exactly one usable face, "
                                f"found {len(faces)}."
                            )
                        new_records.append({
                            "label": label,
                            "embedding": faces[0]["embedding"],
                            "model_version": EMBEDDING_MODEL_VERSION,
                            "source": "registration",
                            "image_sha256": digest,
                            "created_at": utc_now(),
                        })
                        seen_hashes.add(digest)

                    save_memory(current_memory + new_records)
                    new_info = train_and_version_classifier(
                        load_memory()
                    )
                    st.success(
                        f"Saved {len(new_records)} example(s) for {label}. "
                        f"Classifier status: {new_info['status']}."
                    )
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not save registration examples: {exc}")

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.metric("Stored training examples", len(load_memory()))
    with col_m2:
        st.metric("Registered identities", len({
            r["label"] for r in load_memory() if r.get("label")
        }))


with tabs[1]:
    st.subheader("Recognize a group photo and correct mistakes")
    st.write(
        "Predictions are suggestions only. A teacher must confirm the identity "
        "before the example is added to training memory."
    )
    group_upload = st.file_uploader(
        "Upload group photo", type=["jpg", "jpeg", "png"], key="group_upload"
    )
    if st.button("Analyze group photo", type="primary"):
        if not group_upload:
            st.error("Upload a group photo first.")
        else:
            try:
                raw = group_upload.getvalue()
                faces = extract_faces(raw)
                current_memory = load_memory()
                clf, model_info = load_classifier()
                analyzed = []
                for face in faces:
                    pred = predict_embedding(
                        face["embedding"], current_memory, clf
                    )
                    analyzed.append({
                        **face,
                        "predicted": pred["label"],
                        "status": pred["status"],
                        "similarity": pred["similarity"],
                        "margin": pred["margin"],
                        "classifier_label": pred.get("classifier_label"),
                    })
                st.session_state["pending_group"] = {
                    "sha256": sha256_bytes(raw),
                    "image_bytes": raw,
                    "faces": analyzed,
                    "created_at": utc_now(),
                    "classifier_version": model_info.get("version", 0),
                }
            except Exception as exc:
                st.error(f"Recognition failed: {exc}")

    pending = st.session_state.get("pending_group")
    if pending:
        faces = pending["faces"]
        st.image(
            draw_results(
                pending["image_bytes"],
                [{
                    **f,
                    "display_label": f["predicted"],
                    "status": f["status"],
                } for f in faces],
            ),
            caption="Current predictions (not attendance)",
            use_container_width=True,
        )
        if not faces:
            st.warning("No usable faces detected. Try a clearer/larger photo.")
        else:
            st.markdown("#### Teacher review")
            st.caption(
                "Select the correct identity for each face. Choose Unknown if "
                "the person is not in the registered set. Unknown examples are "
                "logged but are not added as a known identity."
            )
            choices = ["Unknown"] + sorted(set(names))
            for i, face in enumerate(faces):
                x1, y1, x2, y2 = face["bbox"]
                st.markdown(
                    f"**Face {i+1}** · predicted: `{face['predicted']}` · "
                    f"status: `{face['status']}` · "
                    f"similarity: `{face['similarity']:.3f}` · "
                    f"box: `{[x1, y1, x2, y2]}`"
                )
                default = face["predicted"]
                if default not in choices:
                    default = "Unknown"
                st.selectbox(
                    f"Correct identity for face {i+1}",
                    choices,
                    index=choices.index(default),
                    key=f"feedback_choice_{pending['sha256']}_{i}",
                )

            if st.button("Save teacher feedback and retrain", type="primary"):
                current_memory = load_memory()
                new_records = []
                feedback_count = 0
                existing_hashes = {
                    r.get("image_sha256") for r in current_memory
                }
                for i, face in enumerate(faces):
                    corrected = st.session_state[
                        f"feedback_choice_{pending['sha256']}_{i}"
                    ]
                    predicted = face["predicted"]
                    append_feedback({
                        "group_image_sha256": pending["sha256"],
                        "face_index": i,
                        "predicted_label": predicted,
                        "corrected_label": corrected,
                        "similarity": face["similarity"],
                        "classifier_version_used": pending["classifier_version"],
                        "model_version": EMBEDDING_MODEL_VERSION,
                    })
                    feedback_count += 1
                    if corrected == "Unknown":
                        continue

                    face_digest = hashlib.sha256(
                        (
                            pending["sha256"] + ":" + str(i)
                        ).encode("utf-8")
                    ).hexdigest()
                    if face_digest in existing_hashes:
                        continue
                    new_records.append({
                        "label": corrected,
                        "embedding": face["embedding"],
                        "model_version": EMBEDDING_MODEL_VERSION,
                        "source": "teacher_feedback",
                        "image_sha256": face_digest,
                        "created_at": utc_now(),
                    })
                    existing_hashes.add(face_digest)

                save_memory(current_memory + new_records)
                version_info = train_and_version_classifier(load_memory())
                st.session_state.pop("pending_group", None)
                st.success(
                    f"Logged {feedback_count} review(s), added "
                    f"{len(new_records)} verified known example(s). "
                    f"Classifier status: {version_info['status']}."
                )
                st.rerun()


with tabs[2]:
    st.subheader("Held-out evaluation")
    st.warning(
        "Evaluation images are kept separate from training memory and are never "
        "used by the training function. For a valid estimate, use distinct "
        "captures taken at different times/angles, not the registration images."
    )

    eval_split_choice = st.radio(
        "Dataset Split to Inspect",
        ["Held-out Test Set (evaluation.json)", "Validation Set (validation.json)"],
        horizontal=True,
    )
    if "Test" in eval_split_choice:
        active_eval_records = load_evaluation()
        split_title = "Held-out Test"
    else:
        active_eval_records = load_validation()
        split_title = "Validation"

    st.write(f"Loaded **{len(active_eval_records)}** examples in {split_title} split.")

    col_btn1, col_btn2 = st.columns([2, 3])
    with col_btn1:
        eval_now = st.button(f"Evaluate current classifier on {split_title}", type="primary")

    if eval_now and active_eval_records:
        clf, model_info = load_classifier()
        train_records = load_memory()
        with st.spinner("Evaluating classifier across held-out split..."):
            res = evaluate_split(active_eval_records, train_records, clf)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Overall Accuracy", f"{res['overall_accuracy']:.1%}")
        m2.metric("Known-Identity Accuracy", f"{res['known_accuracy']:.1%}")
        m3.metric("Macro F1", f"{res['f1_macro']:.3f}")
        m4.metric("Weighted F1", f"{res['f1_weighted']:.3f}")

        st.markdown("#### Status Breakdown")
        b1, b2, b3 = st.columns(3)
        b1.metric("Matched (Accepted)", res["status_breakdown"].get("matched", 0))
        b2.metric("Ambiguous (Flagged for Review)", res["status_breakdown"].get("ambiguous", 0))
        b3.metric("Unknown (Rejected)", res["status_breakdown"].get("unknown", 0))

        if res.get("unknown_samples", 0) > 0:
            st.markdown("#### Open-Set Rejection Metrics")
            u1, u2 = st.columns(2)
            u1.metric("False Acceptance Rate (FAR)", f"{res['false_acceptance_rate']:.1%}")
            u2.metric("False Rejection Rate (FRR)", f"{res['false_rejection_rate']:.1%}")

        st.markdown("#### Prediction Details")
        filter_mode = st.selectbox(
            "Filter rows",
            ["All", "Errors Only", "Ambiguous / Review Only", "Matched Only"],
            key="eval_filter_mode",
        )
        preds = res["predictions"]
        if filter_mode == "Errors Only":
            preds = [p for p in preds if not p["correct"]]
        elif filter_mode == "Ambiguous / Review Only":
            preds = [p for p in preds if p["status"] == "ambiguous"]
        elif filter_mode == "Matched Only":
            preds = [p for p in preds if p["status"] == "matched"]

        st.dataframe(
            [
                {
                    "Expected": p["expected"],
                    "Predicted": p["predicted"],
                    "Status": p["status"],
                    "Similarity": p["similarity"],
                    "Margin": p["margin"],
                    "Classifier Label": p["classifier_label"],
                    "Correct": p["correct"],
                }
                for p in preds
            ],
            use_container_width=True,
        )

        st.markdown("#### Per-Identity Accuracy")
        per_cls = res.get("per_class_stats", {})
        if per_cls:
            st.dataframe(
                [
                    {
                        "Identity": k,
                        "Support": v["support"],
                        "Correct": v["correct"],
                        "Accuracy": f"{v['accuracy']:.1%}",
                        "Ambiguous": v["ambiguous"],
                        "Rejected Unknown": v["rejected_as_unknown"],
                        "Wrong Identity": v["wrong_identity"],
                    }
                    for k, v in sorted(per_cls.items(), key=lambda x: x[0])
                ],
                use_container_width=True,
            )

    st.markdown("---")
    st.markdown("#### Add Manual Held-Out Sample")
    current_memory = load_memory()
    current_names = sorted({r["label"] for r in current_memory if r.get("label")})
    eval_upload = st.file_uploader(
        "Upload one held-out image (one face per image)",
        type=["jpg", "jpeg", "png"],
        key="eval_upload_single",
    )
    expected_choices = ["Unknown"] + current_names
    expected = st.selectbox(
        "Ground-truth identity for this image",
        expected_choices,
        key="eval_expected_single",
    )
    if st.button("Add manual held-out example"):
        if not eval_upload:
            st.error("Upload a held-out image.")
        else:
            raw = eval_upload.getvalue()
            digest = sha256_bytes(raw)
            training_hashes = {
                r.get("image_sha256") for r in current_memory
            }
            eval_records = load_evaluation()
            eval_hashes = {r.get("image_sha256") for r in eval_records}
            try:
                if digest in training_hashes:
                    raise ValueError(
                        "This exact image is in training memory. "
                        "Choose a different capture."
                    )
                if digest in eval_hashes:
                    raise ValueError("This image is already in held-out evaluation.")
                faces = extract_faces(raw)
                if len(faces) != 1:
                    raise ValueError(
                        f"Expected exactly one usable face, found {len(faces)}."
                    )
                eval_records.append({
                    "label": "__unknown__" if expected == "Unknown" else expected,
                    "embedding": faces[0]["embedding"],
                    "model_version": EMBEDDING_MODEL_VERSION,
                    "image_sha256": digest,
                    "source": "manual_evaluation",
                    "created_at": utc_now(),
                })
                save_evaluation(eval_records)
                st.success("Held-out example saved. It was not used for training.")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))


with tabs[3]:
    st.subheader("Model versions and dataset management")
    current_info = get_current_model_info()

    st.markdown("#### Current Classifier Metadata")
    st.json(current_info)

    st.markdown("#### Kaggle Dataset Benchmark Status")
    manifest = load_manifest()
    metrics = load_evaluation_metrics()

    if manifest:
        total_m = len(manifest)
        success_m = sum(1 for m in manifest if m.get("status") == "success")
        skipped_m = sum(1 for m in manifest if m.get("status") == "skipped")
        err_m = sum(1 for m in manifest if m.get("status") == "error")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Manifest Total", total_m)
        c2.metric("Valid Faces", success_m)
        c3.metric("Skipped", skipped_m)
        c4.metric("Errors", err_m)

        if metrics:
            st.markdown("##### Last Saved Benchmark Results")
            st.write(f"Model Version: **v{metrics.get('model_version', 0):04d}**")
            st.write(f"Trained at: `{metrics.get('trained_at')}`")
            test_m = metrics.get("test_metrics", {})
            if test_m:
                st.write(f"Top-1 Accuracy: **{test_m.get('overall_accuracy', 0):.2%}**")
                st.write(f"Macro F1 Score: **{test_m.get('f1_macro', 0):.4f}**")
    else:
        st.info("Kaggle dataset has not been imported yet. Run `python import_kaggle_dataset.py`.")

    st.markdown("---")
    st.markdown("#### Training Examples by Source")
    records = load_memory()
    if records:
        src_summary = get_memory_summary()
        st.write(f"Total training memory records: **{len(records)}** across **{src_summary['unique_identities']}** identities.")
        st.dataframe(
            [
                {
                    "identity": row.get("label"),
                    "source": row.get("source"),
                    "model_version": row.get("model_version"),
                    "created_at": row.get("created_at"),
                }
                for row in records[:200]
            ],
            use_container_width=True,
        )
        if len(records) > 200:
            st.caption(f"Showing first 200 of {len(records)} records.")
    else:
        st.info("No training examples stored yet.")

    st.markdown("---")
    st.markdown("#### Safe Dataset Operations")
    st.warning("Operations here manage benchmark dataset records while preserving manual student registrations.")

    col_act1, col_act2 = st.columns(2)
    with col_act1:
        if st.button("Retrain Classifier on Current Memory"):
            new_info = train_and_version_classifier(load_memory())
            st.success(f"Retrained classifier into v{new_info['version']:04d}. Status: {new_info['status']}.")
            st.rerun()

    with col_act2:
        confirm_reset = st.checkbox("Confirm: Remove Kaggle records only (keeps user registrations)")
        if st.button("Reset Kaggle Dataset Records", type="secondary", disabled=not confirm_reset):
            res = reset_dataset_records(preserve_user_data=True)
            st.success(
                f"Removed {res['removed_memory_records']} Kaggle records. "
                f"Preserved {res['kept_memory_records']} manual records ({res['kept_identities']})."
            )
            st.rerun()

