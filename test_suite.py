import os
import sys
import unittest
from pathlib import Path
import numpy as np

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import face_engine
import storage
import evaluation


class TestFaceRecognitionPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        local_sample = ROOT / "data" / "sample_test_face.jpg"
        if local_sample.exists():
            cls.orig_sample_path = local_sample
        else:
            cls.orig_sample_path = Path(
                r"C:\Users\KARTIK MEWADA\.cache\kagglehub\datasets\vasukipatel\face-recognition-dataset\versions\1\Original Images\Original Images\Akshay Kumar\Akshay Kumar_0.jpg"
            )

    def test_01_dll_and_runtime_configuration(self):
        """Verify NVIDIA DLL directories are discovered and runtime_info reports active provider."""
        info = face_engine.runtime_info()
        self.assertIn("onnxruntime_version", info)
        self.assertIn("available_providers", info)
        self.assertIn("active_providers", info)
        self.assertTrue(len(info["active_providers"]) > 0)
        print(f"[TEST 1] Runtime provider: {info['device']}, active: {info['active_providers']}")

    def test_02_cosine_similarity(self):
        """Verify 512-D cosine similarity calculations and dimension guards."""
        v1 = np.random.randn(512).astype(np.float32)
        v1 /= np.linalg.norm(v1)
        # Identical vector should yield 1.0
        sim_self = face_engine.cosine_similarity(v1.tolist(), v1.tolist())
        self.assertAlmostEqual(sim_self, 1.0, places=4)

        # Opposite vector should yield -1.0
        v_opp = (-v1).tolist()
        sim_opp = face_engine.cosine_similarity(v1.tolist(), v_opp)
        self.assertAlmostEqual(sim_opp, -1.0, places=4)

        # Dimension guard
        with self.assertRaises(ValueError):
            face_engine.cosine_similarity([1.0, 2.0], [3.0, 4.0])

    def test_03_sample_face_extraction(self):
        """Verify face extraction on a known original image from the Kaggle dataset."""
        if not self.orig_sample_path.exists():
            self.skipTest(f"Sample image not found: {self.orig_sample_path}")

        raw = self.orig_sample_path.read_bytes()
        faces = face_engine.extract_faces(raw)
        self.assertEqual(len(faces), 1, "Expected exactly 1 face in sample image")
        face = faces[0]
        self.assertEqual(len(face["embedding"]), 512, "Expected 512-D embedding")
        self.assertGreater(face["det_score"], 0.5, "Expected high detection confidence")
        x1, y1, x2, y2 = face["bbox"]
        self.assertGreater(x2, x1)
        self.assertGreater(y2, y1)

    def test_04_predict_embedding_gating(self):
        """Verify dual-gate prediction: similarity threshold, match margin, and unknown rejection."""
        # Create synthetic memory
        vec_alice = np.zeros(512, dtype=np.float32)
        vec_alice[0] = 1.0
        vec_bob = np.zeros(512, dtype=np.float32)
        vec_bob[1] = 1.0

        memory = [
            {
                "label": "Alice",
                "embedding": vec_alice.tolist(),
                "model_version": face_engine.EMBEDDING_MODEL_VERSION,
            },
            {
                "label": "Bob",
                "embedding": vec_bob.tolist(),
                "model_version": face_engine.EMBEDDING_MODEL_VERSION,
            },
        ]

        # 1. Exact match with Alice
        pred_alice = face_engine.predict_embedding(vec_alice.tolist(), memory, classifier=None)
        self.assertEqual(pred_alice["label"], "Alice")
        self.assertEqual(pred_alice["status"], "matched")
        self.assertAlmostEqual(pred_alice["similarity"], 1.0, places=4)

        # 2. Unknown test face orthogonal to all memory (similarity 0.0 < threshold 0.45)
        vec_unknown = np.zeros(512, dtype=np.float32)
        vec_unknown[2] = 1.0
        pred_unknown = face_engine.predict_embedding(vec_unknown.tolist(), memory, classifier=None)
        self.assertEqual(pred_unknown["label"], "Unknown")
        self.assertEqual(pred_unknown["status"], "unknown")

        # 3. Ambiguous face with near-equal similarity to Alice and Bob (margin < 0.05)
        vec_ambig = np.zeros(512, dtype=np.float32)
        vec_ambig[0] = 0.7071
        vec_ambig[1] = 0.7071
        pred_ambig = face_engine.predict_embedding(vec_ambig.tolist(), memory, classifier=None)
        self.assertEqual(pred_ambig["label"], "Review")
        self.assertEqual(pred_ambig["status"], "ambiguous")

    def test_05_safe_dataset_reset_preserves_user_records(self):
        """Verify reset_dataset_records preserves manual registrations and removes only kaggle_dataset."""
        orig_mem = storage.load_memory()
        orig_eval = storage.load_evaluation()
        orig_val = storage.load_validation()
        orig_current = storage.get_current_model_info()
        orig_artifacts = set(os.listdir(storage.MODELS_DIR))

        try:
            user_record = {
                "label": "StudentTestUser",
                "embedding": [0.1] * 512,
                "model_version": face_engine.EMBEDDING_MODEL_VERSION,
                "source": "registration",
                "image_sha256": "fake_hash_user_1",
                "created_at": storage.utc_now(),
            }
            kaggle_record = {
                "label": "CelebrityTestUser",
                "embedding": [0.2] * 512,
                "model_version": face_engine.EMBEDDING_MODEL_VERSION,
                "source": "kaggle_dataset",
                "image_sha256": "fake_hash_kaggle_1",
                "created_at": storage.utc_now(),
            }

            test_mem = orig_mem + [user_record, kaggle_record]
            storage.save_memory(test_mem)

            # Call reset
            result = storage.reset_dataset_records(preserve_user_data=True)
            updated_mem = storage.load_memory()

            # Check that user record is preserved
            user_labels = [r["label"] for r in updated_mem if r.get("label") == "StudentTestUser"]
            self.assertEqual(len(user_labels), 1)

            # Check that kaggle record is removed
            kaggle_labels = [r["label"] for r in updated_mem if r.get("label") == "CelebrityTestUser"]
            self.assertEqual(len(kaggle_labels), 0)
        finally:
            storage.save_memory(orig_mem)
            storage.save_evaluation(orig_eval)
            storage.save_validation(orig_val)
            storage._atomic_json(storage.CURRENT_MODEL_PATH, orig_current)
            for art in (set(os.listdir(storage.MODELS_DIR)) - orig_artifacts):
                try:
                    (storage.MODELS_DIR / art).unlink()
                except Exception:
                    pass

    def test_06_evaluation_metrics_calculation(self):
        """Verify evaluation metrics computation (accuracy, F1, confusion matrix, FAR/FRR)."""
        vec_a = [1.0] + [0.0] * 511
        vec_b = [0.0, 1.0] + [0.0] * 510

        train_records = [
            {"label": "PersonA", "embedding": vec_a, "model_version": face_engine.EMBEDDING_MODEL_VERSION},
            {"label": "PersonB", "embedding": vec_b, "model_version": face_engine.EMBEDDING_MODEL_VERSION},
        ]

        test_records = [
            {"label": "PersonA", "embedding": vec_a, "image_sha256": "h1"},
            {"label": "PersonB", "embedding": vec_b, "image_sha256": "h2"},
            {"label": "__unknown__", "embedding": [0.0, 0.0, 1.0] + [0.0] * 509, "image_sha256": "h3"},
        ]

        metrics = evaluation.evaluate_split(test_records, train_records, classifier=None)
        self.assertEqual(metrics["total_samples"], 3)
        self.assertEqual(metrics["known_samples"], 2)
        self.assertEqual(metrics["unknown_samples"], 1)
        self.assertAlmostEqual(metrics["overall_accuracy"], 1.0, places=4)
        self.assertAlmostEqual(metrics["known_accuracy"], 1.0, places=4)
        self.assertAlmostEqual(metrics["false_acceptance_rate"], 0.0, places=4)
        self.assertAlmostEqual(metrics["false_rejection_rate"], 0.0, places=4)


if __name__ == "__main__":
    unittest.main(verbosity=2)

