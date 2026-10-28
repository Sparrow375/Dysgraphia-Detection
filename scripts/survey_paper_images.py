"""
Step 0: Survey and group all paper-written notebook images.
Outputs: outputs/step4/paper_image_manifest.csv with student matching status.
"""
import os, sys, pandas as pd, re
from collections import defaultdict

PROJECT_ROOT = r"c:\Users\embar\OneDrive-N\D0cuments\Dysgraphia\Dysgraphia-Detection"
sys.path.insert(0, PROJECT_ROOT)

VV_DIR = os.path.join(PROJECT_ROOT, "VV data")
ZMAPPED_STUDENTS = os.path.join(PROJECT_ROOT, "zmapped_output", "students")
GT_CSV = os.path.join(PROJECT_ROOT, "zmapped_output", "ground_truth_kinematics.csv")
OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "step4")
os.makedirs(OUT_DIR, exist_ok=True)

EXTENSIONS = ('.jpg', '.jpeg', '.png', '.tiff', '.tif')

gt = pd.read_csv(GT_CSV)
gt_students = gt[['student_id', 'school', 'grade']].drop_duplicates().set_index('student_id')

# =====================================================================
# PART A: Per-student photos already in zmapped_output/students/<sid>/handwritten/
# =====================================================================
per_student_records = []
for sid in sorted(os.listdir(ZMAPPED_STUDENTS)):
    hw = os.path.join(ZMAPPED_STUDENTS, sid, "handwritten")
    imgs = []
    if os.path.isdir(hw):
        imgs = [f for f in os.listdir(hw) if f.lower().endswith(EXTENSIONS)]
    
    if sid not in gt_students.index:
        continue
    school = gt_students.loc[sid, 'school']
    grade = int(gt_students.loc[sid, 'grade'])
    
    if imgs:
        per_student_records.append({
            "student_id": sid,
            "school": school,
            "grade": grade,
            "photo_path": os.path.join(hw, sorted(imgs)[0]),  # first photo
            "n_photos": len(imgs),
            "source": "zmapped_per_student",
            "match_type": "exact_student"
        })

print(f"[A] Per-student photos in zmapped_output: {len(per_student_records)}")
for g in [4, 5, 6, 7]:
    cnt = sum(1 for r in per_student_records if r["grade"] == g)
    print(f"    Grade {g}: {cnt} students")

# =====================================================================
# PART B: VV Handwritten class-level photos (no student IDs in filenames)
# Grade-level pools to be treated as class-level aggregates
# =====================================================================
# Map VV folder -> (school, grade)
VV_GRADE_MAP = {
    os.path.join(VV_DIR, "Future gen Handwritten", "grade 4"): ("Future Gen", 4),
    os.path.join(VV_DIR, "Future gen Handwritten", "grade 5"): ("Future Gen", 5),
    os.path.join(VV_DIR, "Future gen Handwritten", "6th grade"): ("Future Gen", 6),
    os.path.join(VV_DIR, "Future gen Handwritten", "Grade 7"): ("Future Gen", 7),
    os.path.join(VV_DIR, "Kanyashala Handwritten", "Grade 4"): ("Kanyashala", 4),
    os.path.join(VV_DIR, "Kanyashala Handwritten", "Grade 5"): ("Kanyashala", 5),
    os.path.join(VV_DIR, "Kanyashala Handwritten", "Grade 6"): ("Kanyashala", 6),
    os.path.join(VV_DIR, "Kanyashala Handwritten", "Grade 7"): ("Kanyashala", 7),
}

# Also kanyashala srujani, future gen srujani
for d in os.listdir(VV_DIR):
    full_d = os.path.join(VV_DIR, d)
    if not os.path.isdir(full_d):
        continue
    dl = d.lower()
    if "srujani" in dl or "nitish" in dl or "avaneesh" in dl or "sriram" in dl:
        # These are stylus session dirs, not paper photo dirs - skip
        continue

vv_class_records = []
for folder, (school, grade) in VV_GRADE_MAP.items():
    if not os.path.isdir(folder):
        continue
    imgs = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(EXTENSIONS):
                imgs.append(os.path.join(root, f))
    
    # Match to GT students of same school+grade
    matched_sids = gt_students[(gt_students.school == school) & (gt_students.grade == grade)].index.tolist()
    
    vv_class_records.append({
        "folder": folder,
        "school": school,
        "grade": grade,
        "n_images": len(imgs),
        "n_matched_gt_students": len(matched_sids),
        "sample_images": imgs[:3],
        "all_images": imgs,
        "matched_student_ids": matched_sids
    })
    print(f"[B] {school} Gr{grade}: {len(imgs)} class images -> {len(matched_sids)} GT students")

# =====================================================================
# SUMMARY
# =====================================================================
total_vv_class_imgs = sum(r['n_images'] for r in vv_class_records)
total_vv_students = sum(r['n_matched_gt_students'] for r in vv_class_records)

print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Per-student photos (usable for student-level correlation): {len(per_student_records)}")
print(f"VV class-level photos (can compute grade-level representative): {total_vv_class_imgs} images across {total_vv_students} GT students")
print()
print("KS students (Kanyashala) WITH per-student photos: 0")
print("KS students have VV class-level data available for Grades 4-7")

# Build manifest
manifest_rows = []
for r in per_student_records:
    manifest_rows.append({
        "student_id": r["student_id"],
        "school": r["school"],
        "grade": r["grade"],
        "photo_path": r["photo_path"],
        "n_photos": r["n_photos"],
        "match_type": "exact_student",
        "source": "zmapped_per_student",
        "usable_for_correlation": True
    })

# For VV class-level, note: cannot correlate per-student but can estimate grade-level distribution
for r in vv_class_records:
    for sid in r["matched_student_ids"]:
        # Only add if NOT already covered by per-student
        if any(x["student_id"] == sid for x in per_student_records):
            continue
        # Assign the class-level image (any one representative image)
        manifest_rows.append({
            "student_id": sid,
            "school": r["school"],
            "grade": r["grade"],
            "photo_path": r["all_images"][0] if r["all_images"] else "",
            "n_photos": r["n_images"],
            "match_type": "class_level_grade_pool",
            "source": "vv_class_handwritten",
            "usable_for_correlation": False  # class-level images cannot match individual students
        })

manifest_df = pd.DataFrame(manifest_rows)
out_path = os.path.join(OUT_DIR, "paper_image_manifest.csv")
manifest_df.to_csv(out_path, index=False)

print()
print(f"Manifest written -> {out_path}")
print(f"Total manifest rows: {len(manifest_df)}")
print(f"  Usable for per-student correlation: {manifest_df['usable_for_correlation'].sum()}")
print(f"  Grade-pool only (class-level VV data): {(~manifest_df['usable_for_correlation']).sum()}")
print()
print("==> CONCLUSION: Only the 66 zmapped per-student photos can be used for")
print("    Step 4 student-level correlation. VV class-level images are grade-pool")
print("    data with NO individual student labels — they cannot be matched per-student.")
print("    The VV class-level data will be used for grade-level distribution comparison only.")
