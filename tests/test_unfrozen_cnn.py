"""Automated unit test suite for Stage 30: Baseline Unfrozen CNN Experiment."""

import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn

from baseline.train_unfrozen_cnn import (
    pad_to_square,
    SentenceCropDataset,
    build_unfrozen_model,
    evaluate_dataset,
    INDEX_CSV,
    SENTENCE_DATASET_DIR,
)
from sklearn.model_selection import StratifiedGroupKFold, train_test_split


class TestUnfrozenCNN(unittest.TestCase):
    """Test suite verifying components of the unfrozen CNN pipeline."""

    def test_pad_to_square(self):
        """Verify proportional aspect ratio padding onto square canvas."""
        # Wide rectangle (like a sentence crop: 1600 x 200)
        img_wide = Image.new("RGB", (1600, 200), (128, 128, 128))
        padded = pad_to_square(img_wide, target_size=224)
        self.assertEqual(padded.size, (224, 224))

        # Tall rectangle
        img_tall = Image.new("RGB", (200, 1600), (128, 128, 128))
        padded_tall = pad_to_square(img_tall, target_size=224)
        self.assertEqual(padded_tall.size, (224, 224))

    def test_build_unfrozen_densenet201(self):
        """Verify DenseNet201 architecture, head replacement, and unfrozen parameters."""
        model, total_params = build_unfrozen_model(model_name="densenet201", pretrained=False)
        self.assertIsInstance(model, nn.Module)
        self.assertGreater(total_params, 18_000_000)

        # Verify all parameters have requires_grad = True
        for name, param in model.named_parameters():
            self.assertTrue(param.requires_grad, f"Parameter {name} was not unfrozen!")

        # Verify forward pass on dummy batch
        dummy_input = torch.randn(2, 3, 224, 224)
        logits = model(dummy_input).squeeze(-1)
        self.assertEqual(logits.shape, (2,))

        # Verify gradient backward
        loss = logits.sum()
        loss.backward()
        for name, param in model.named_parameters():
            self.assertIsNotNone(param.grad, f"Gradient missing for {name}")

    def test_dataset_splits_leakage_contrast(self):
        """Verify Run 1A creates student leakage while Run 1B guarantees zero student leakage."""
        self.assertTrue(INDEX_CSV.exists(), f"Index CSV missing at {INDEX_CSV}")
        df = pd.read_csv(INDEX_CSV)
        self.assertEqual(len(df), 1145)

        # Run 1A: Crop split ignoring student_id
        train_df_1a, test_df_1a = train_test_split(
            df,
            test_size=0.20,
            stratify=df["label"],
            random_state=42,
        )
        overlap_1a = set(train_df_1a["student_id"]).intersection(set(test_df_1a["student_id"]))
        test_students_1a = set(test_df_1a["student_id"])
        # With 1145 crops across 215 students (~5 crops/student), random 80/20 must cause >85% student overlap
        self.assertGreater(len(overlap_1a), 100, "Run 1A must produce massive student overlap!")
        self.assertGreater(len(overlap_1a) / len(test_students_1a), 0.85)

        # Run 1B: StratifiedGroupKFold strictly by student_id
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        splits = list(sgkf.split(df, y=df["label"], groups=df["student_id"]))
        train_idx, test_idx = splits[0]
        train_df_1b = df.iloc[train_idx]
        test_df_1b = df.iloc[test_idx]

        overlap_1b = set(train_df_1b["student_id"]).intersection(set(test_df_1b["student_id"]))
        self.assertEqual(len(overlap_1b), 0, "Run 1B must have exactly 0 student overlap!")
        self.assertEqual(len(train_df_1b) + len(test_df_1b), 1145)

    def test_sentence_crop_dataset_loading(self):
        """Verify PyTorch dataset loads real crops and applies augmentation."""
        df = pd.read_csv(INDEX_CSV).head(10)
        ds_train = SentenceCropDataset(df, SENTENCE_DATASET_DIR, is_train=True)
        ds_eval = SentenceCropDataset(df, SENTENCE_DATASET_DIR, is_train=False)

        sample_train = ds_train[0]
        sample_eval = ds_eval[0]

        self.assertEqual(sample_train["image"].shape, (3, 224, 224))
        self.assertEqual(sample_eval["image"].shape, (3, 224, 224))
        self.assertIn(sample_train["label"].item(), [0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
