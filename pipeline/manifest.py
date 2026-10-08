"""Manifest generation pipeline for Workstream A.

Consolidates School A (Future Gen) and School B (Kanyashala) handwritten data into
data/manifest.csv with ground truth labels and verified image paths.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_school_a_manifest(
    labels_csv_path: str,
    master_mapping_path: str,
    raw_dir: str,
    positives_dict: Dict[int, List[int]],
) -> pd.DataFrame:
    """Build manifest rows for School A (Future Gen)."""
    df_labels = pd.read_csv(labels_csv_path)

    # Load master mapping if available for match_status
    if os.path.exists(master_mapping_path):
        df_master = pd.read_csv(master_mapping_path)
        master_subset = df_master[["student_id", "match_status"]].drop_duplicates(
            subset=["student_id"]
        )
        df_merged = pd.merge(df_labels, master_subset, on="student_id", how="left")
    else:
        df_merged = df_labels.copy()
        df_merged["match_status"] = "PAPER_ONLY"

    rows: List[Dict[str, Any]] = []
    base_raw = Path(raw_dir)

    for _, row in df_merged.iterrows():
        student_id = str(row["student_id"]).strip()
        grade = int(row["grade"])
        section = str(row["section"]).strip() if pd.notna(row["section"]) else "A"
        roll_num = int(row["roll_number"]) if pd.notna(row["roll_number"]) else -1
        rel_path = str(row["relative_path"]).replace("\\", "/")
        filename = str(row["filename"]).strip()

        # Image existence validation
        full_img_path = base_raw / rel_path
        if not full_img_path.exists():
            # Try alternate path under grade folder
            candidates = list(base_raw.glob(f"**/{filename}"))
            if candidates:
                full_img_path = candidates[0]
                rel_path = full_img_path.relative_to(base_raw).as_posix()
            else:
                raise FileNotFoundError(
                    f"Image not found on disk: {full_img_path} for student {student_id}"
                )

        # Ground truth label: 1 if in positives dictionary, else 0
        target_rolls = positives_dict.get(grade, [])
        label = 1 if roll_num in target_rolls else 0

        # Label source
        match_status = str(row.get("match_status", "PAPER_ONLY"))
        if match_status == "MATCHED":
            label_source = "paper+stylus"
        else:
            label_source = "paper-only"

        # Manifest relative path from repository root
        repo_rel_path = (Path(raw_dir) / rel_path).as_posix()

        rows.append(
            {
                "student_id": student_id,
                "school": "school_a",
                "school_name": "Future Gen",
                "grade": grade,
                "section": section,
                "roll_number": roll_num,
                "label": label,
                "label_source": label_source,
                "match_status": match_status,
                "image_filename": filename,
                "relative_image_path": repo_rel_path,
                "is_held_out": False,
            }
        )

    return pd.DataFrame(rows)


def build_school_b_manifest(raw_dir: str) -> pd.DataFrame:
    """Build manifest rows for School B (Kanyashala) held-out test cohort."""
    base_raw = Path(raw_dir)
    rows: List[Dict[str, Any]] = []

    if not base_raw.exists():
        print(f"Warning: School B directory not found at {raw_dir}")
        return pd.DataFrame()

    for file_path in sorted(base_raw.glob("**/*")):
        if file_path.is_file() and file_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            rel_to_raw = file_path.relative_to(base_raw).as_posix()
            parent_folder = file_path.parent.name  # e.g., "Grade 4"

            # Parse grade from folder name
            grade_match = re.search(r"\d+", parent_folder)
            grade = int(grade_match.group(0)) if grade_match else -1

            stem = file_path.stem
            student_id = f"SCHB_G{grade}_{stem}"
            repo_rel_path = (Path(raw_dir) / rel_to_raw).as_posix()

            rows.append(
                {
                    "student_id": student_id,
                    "school": "school_b",
                    "school_name": "Kanyashala",
                    "grade": grade,
                    "section": "unassigned",
                    "roll_number": -1,
                    "label": float("nan"),  # Held-out, label-free
                    "label_source": "held_out_external_test",
                    "match_status": "HELD_OUT",
                    "image_filename": file_path.name,
                    "relative_image_path": repo_rel_path,
                    "is_held_out": True,
                }
            )

    return pd.DataFrame(rows)


def generate_manifest(config_path: str = "configs/config.yaml") -> pd.DataFrame:
    """Generate and write the combined manifest.csv."""
    cfg = load_config(config_path)

    positives = cfg["labels"]["school_a_positives"]
    # Ensure keys in positives are ints
    positives = {int(k): list(v) for k, v in positives.items()}

    df_a = build_school_a_manifest(
        labels_csv_path=cfg["paths"]["school_a_labels_csv"],
        master_mapping_path=cfg["paths"]["master_mapping_csv"],
        raw_dir=cfg["paths"]["school_a_raw_dir"],
        positives_dict=positives,
    )

    df_b = build_school_b_manifest(raw_dir=cfg["paths"]["school_b_raw_dir"])

    df_manifest = pd.concat([df_a, df_b], ignore_index=True)

    out_csv = cfg["paths"]["manifest_csv"]
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df_manifest.to_csv(out_csv, index=False)
    print(f"Manifest successfully generated at: {out_csv}")
    print(f"Total rows: {len(df_manifest)}")
    print(f"School A (Future Gen) records: {len(df_a)}")
    print(f"  - Positives: {int((df_a['label'] == 1).sum())}")
    print(f"  - Negatives: {int((df_a['label'] == 0).sum())}")
    print(f"School B (Kanyashala) records: {len(df_b)} (Held-out)")

    return df_manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate manifest.csv for Workstream A")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config.yaml")
    args = parser.parse_args()
    generate_manifest(args.config)
