# Deep Crop Accuracy Audit for Missed Positives (False Negatives)

### SCHB_G8_IMG_20261007_104901 (School: school_b, Grade: 8, Tool: pencil)
- **Predictions**: DINOv2 = `0.071`, CLIP = `0.492`, Dual-ViT Rank = `0.581`, Tabular = `0.325`
- **Total Sentences Detected**: 4 (Expected: 6)
- **Crop Image Paths**: `data/processed/school_b/SCHB_G8_IMG_20261007_104901/`
- **Detected Crop & Segmentation Issues**:
  - **Missing Tasks**: Only 4/6 tasks segmented (2 missing on page).
  - **Bottom Cutoff**: sentence_02_copy_english ink touches bottom border (64 px). Descenders may be cut off.
  - **Right Cutoff**: sentence_02_copy_english ink touches right border (21 px). Last word may be clipped.
  - **Low Word Count**: sentence_02_copy_english has only 2 words detected.
  - **Left Cutoff**: sentence_03_dictated_hindi ink touches left border (21 px). First word may be clipped.
  - **Right Cutoff**: sentence_03_dictated_hindi ink touches right border (21 px). Last word may be clipped.
  - **Low Word Count**: sentence_03_dictated_hindi has only 2 words detected.
  - **Bottom Cutoff**: sentence_06_own_english ink touches bottom border (49 px). Descenders may be cut off.
  - **Left Cutoff**: sentence_06_own_english ink touches left border (23 px). First word may be clipped.
  - **Right Cutoff**: sentence_06_own_english ink touches right border (202 px). Last word may be clipped.
  - **Unsegmented Ink Below Protocol**: 154352 ink pixels below y=1222. Child may have written below the detected boxes!

### SCHB_G4_IMG_20261007_100335 (School: school_b, Grade: 4, Tool: pen)
- **Predictions**: DINOv2 = `0.112`, CLIP = `0.234`, Dual-ViT Rank = `0.428`, Tabular = `0.462`
- **Total Sentences Detected**: 4 (Expected: 6)
- **Crop Image Paths**: `data/processed/school_b/SCHB_G4_IMG_20261007_100335/`
- **Detected Crop & Segmentation Issues**:
  - **Missing Tasks**: Only 4/6 tasks segmented (2 missing on page).
  - **Overlap**: sentence_02_copy_english and sentence_03_dictated_hindi vertically overlap by 17px.
  - **Bottom Cutoff**: sentence_01_copy_hindi ink touches bottom border (794 px). Descenders may be cut off.
  - **Left Cutoff**: sentence_01_copy_hindi ink touches left border (21 px). First word may be clipped.
  - **Right Cutoff**: sentence_01_copy_hindi ink touches right border (24 px). Last word may be clipped.
  - **Left Cutoff**: sentence_02_copy_english ink touches left border (283 px). First word may be clipped.
  - **Right Cutoff**: sentence_02_copy_english ink touches right border (21 px). Last word may be clipped.
  - **Bottom Cutoff**: sentence_03_dictated_hindi ink touches bottom border (186 px). Descenders may be cut off.
  - **Left Cutoff**: sentence_03_dictated_hindi ink touches left border (24 px). First word may be clipped.
  - **Right Cutoff**: sentence_03_dictated_hindi ink touches right border (21 px). Last word may be clipped.
  - **Left Cutoff**: sentence_04_dictated_english ink touches left border (179 px). First word may be clipped.
  - **Abnormally Short Box**: sentence_04_dictated_english has height 99px (risk of clipping).
  - **Unsegmented Ink Below Protocol**: 199103 ink pixels below y=1067. Child may have written below the detected boxes!

### G5_A_Roll07 (School: school_a, Grade: 5, Tool: pencil)
- **Predictions**: DINOv2 = `0.126`, CLIP = `0.134`, Dual-ViT Rank = `0.344`, Tabular = `0.548`
- **Total Sentences Detected**: 6 (Expected: 6)
- **Crop Image Paths**: `data/processed/school_a/G5_A_Roll07/`
- **Detected Crop & Segmentation Issues**:
  - **Overlap**: sentence_02_copy_english and sentence_03_dictated_hindi vertically overlap by 35px.
  - **Bottom Cutoff**: sentence_01_copy_hindi ink touches bottom border (300 px). Descenders may be cut off.
  - **Top Cutoff**: sentence_02_copy_english ink touches top border (1180 px). Ascenders/matras may be cut off.
  - **Bottom Cutoff**: sentence_02_copy_english ink touches bottom border (193 px). Descenders may be cut off.
  - **Top Cutoff**: sentence_03_dictated_hindi ink touches top border (177 px). Ascenders/matras may be cut off.
  - **Bottom Cutoff**: sentence_03_dictated_hindi ink touches bottom border (2220 px). Descenders may be cut off.
  - **Low Word Count**: sentence_03_dictated_hindi has only 2 words detected.
  - **Bottom Cutoff**: sentence_04_dictated_english ink touches bottom border (141 px). Descenders may be cut off.
  - **Bottom Cutoff**: sentence_05_own_hindi ink touches bottom border (1124 px). Descenders may be cut off.
  - **Low Word Count**: sentence_05_own_hindi has only 2 words detected.
  - **Unsegmented Ink Below Protocol**: 42422 ink pixels below y=1523. Child may have written below the detected boxes!

### G3_A_Roll19 (School: school_a, Grade: 3, Tool: pencil)
- **Predictions**: DINOv2 = `0.223`, CLIP = `0.635`, Dual-ViT Rank = `0.733`, Tabular = `0.324`
- **Total Sentences Detected**: 4 (Expected: 4)
- **Crop Image Paths**: `data/processed/school_a/G3_A_Roll19/`
- **Detected Crop & Segmentation Issues**:
  - **Overlap**: sentence_02_copy_english and sentence_03_dictated_hindi vertically overlap by 34px.
  - **Bottom Cutoff**: sentence_02_copy_english ink touches bottom border (206 px). Descenders may be cut off.
  - **Left Cutoff**: sentence_02_copy_english ink touches left border (118 px). First word may be clipped.
  - **Top Cutoff**: sentence_03_dictated_hindi ink touches top border (65 px). Ascenders/matras may be cut off.
  - **Left Cutoff**: sentence_03_dictated_hindi ink touches left border (24 px). First word may be clipped.
  - **Top Cutoff**: sentence_04_dictated_english ink touches top border (94 px). Ascenders/matras may be cut off.
  - **Left Cutoff**: sentence_04_dictated_english ink touches left border (21 px). First word may be clipped.
  - **Unsegmented Ink Below Protocol**: 156548 ink pixels below y=1015. Child may have written below the detected boxes!

### G3_A_Roll12 (School: school_a, Grade: 3, Tool: pencil)
- **Predictions**: DINOv2 = `0.241`, CLIP = `0.605`, Dual-ViT Rank = `0.735`, Tabular = `0.660`
- **Total Sentences Detected**: 4 (Expected: 4)
- **Crop Image Paths**: `data/processed/school_a/G3_A_Roll12/`
- **Detected Crop & Segmentation Issues**:
  - **Overlap**: sentence_02_copy_english and sentence_03_dictated_hindi vertically overlap by 45px.
  - **Left Cutoff**: sentence_01_copy_hindi ink touches left border (78 px). First word may be clipped.
  - **Bottom Cutoff**: sentence_02_copy_english ink touches bottom border (172 px). Descenders may be cut off.
  - **Left Cutoff**: sentence_02_copy_english ink touches left border (62 px). First word may be clipped.
  - **Top Cutoff**: sentence_03_dictated_hindi ink touches top border (1699 px). Ascenders/matras may be cut off.
  - **Bottom Cutoff**: sentence_03_dictated_hindi ink touches bottom border (704 px). Descenders may be cut off.
  - **Left Cutoff**: sentence_03_dictated_hindi ink touches left border (180 px). First word may be clipped.
  - **Right Cutoff**: sentence_03_dictated_hindi ink touches right border (84 px). Last word may be clipped.
  - **Left Cutoff**: sentence_04_dictated_english ink touches left border (110 px). First word may be clipped.
  - **Right Cutoff**: sentence_04_dictated_english ink touches right border (65 px). Last word may be clipped.
  - **Unsegmented Ink Below Protocol**: 160157 ink pixels below y=1112. Child may have written below the detected boxes!

### G7_A_Roll06 (School: school_a, Grade: 7, Tool: pencil)
- **Predictions**: DINOv2 = `0.254`, CLIP = `0.451`, Dual-ViT Rank = `0.665`, Tabular = `0.401`
- **Total Sentences Detected**: 6 (Expected: 6)
- **Crop Image Paths**: `data/processed/school_a/G7_A_Roll06/`
- **Detected Crop & Segmentation Issues**:
  - **Overlap**: sentence_02_copy_english and sentence_03_dictated_hindi vertically overlap by 41px.
  - **Overlap**: sentence_04_dictated_english and sentence_05_own_hindi vertically overlap by 25px.
  - **Right Cutoff**: sentence_01_copy_hindi ink touches right border (21 px). Last word may be clipped.
  - **Bottom Cutoff**: sentence_02_copy_english ink touches bottom border (433 px). Descenders may be cut off.
  - **Left Cutoff**: sentence_02_copy_english ink touches left border (173 px). First word may be clipped.
  - **Right Cutoff**: sentence_02_copy_english ink touches right border (59 px). Last word may be clipped.
  - **Abnormally Tall Box**: sentence_02_copy_english has height 653px (may encompass multiple ruled lines or extra whitespace).
  - **Left Cutoff**: sentence_03_dictated_hindi ink touches left border (152 px). First word may be clipped.
  - **Right Cutoff**: sentence_03_dictated_hindi ink touches right border (34 px). Last word may be clipped.
  - **Top Cutoff**: sentence_04_dictated_english ink touches top border (491 px). Ascenders/matras may be cut off.
  - **Right Cutoff**: sentence_04_dictated_english ink touches right border (21 px). Last word may be clipped.
  - **Top Cutoff**: sentence_05_own_hindi ink touches top border (1057 px). Ascenders/matras may be cut off.
  - **Bottom Cutoff**: sentence_05_own_hindi ink touches bottom border (735 px). Descenders may be cut off.
  - **Left Cutoff**: sentence_05_own_hindi ink touches left border (139 px). First word may be clipped.
  - **Right Cutoff**: sentence_05_own_hindi ink touches right border (24 px). Last word may be clipped.
  - **Top Cutoff**: sentence_06_own_english ink touches top border (759 px). Ascenders/matras may be cut off.
  - **Left Cutoff**: sentence_06_own_english ink touches left border (100 px). First word may be clipped.
  - **Right Cutoff**: sentence_06_own_english ink touches right border (42 px). Last word may be clipped.
  - **Unsegmented Ink Below Protocol**: 74554 ink pixels below y=2254. Child may have written below the detected boxes!

### SCHB_G7_IMG_20261007_105422 (School: school_b, Grade: 7, Tool: pencil)
- **Predictions**: DINOv2 = `0.342`, CLIP = `0.448`, Dual-ViT Rank = `0.700`, Tabular = `0.253`
- **Total Sentences Detected**: 6 (Expected: 6)
- **Crop Image Paths**: `data/processed/school_b/SCHB_G7_IMG_20261007_105422/`
- **Detected Crop & Segmentation Issues**:
  - **Overlap**: sentence_04_dictated_english and sentence_05_own_hindi vertically overlap by 15px.
  - **Bottom Cutoff**: sentence_01_copy_hindi ink touches bottom border (1449 px). Descenders may be cut off.
  - **Left Cutoff**: sentence_01_copy_hindi ink touches left border (27 px). First word may be clipped.
  - **Right Cutoff**: sentence_01_copy_hindi ink touches right border (71 px). Last word may be clipped.
  - **Left Cutoff**: sentence_02_copy_english ink touches left border (30 px). First word may be clipped.
  - **Low Word Count**: sentence_02_copy_english has only 2 words detected.
  - **Bottom Cutoff**: sentence_03_dictated_hindi ink touches bottom border (141 px). Descenders may be cut off.
  - **Left Cutoff**: sentence_03_dictated_hindi ink touches left border (44 px). First word may be clipped.
  - **Right Cutoff**: sentence_03_dictated_hindi ink touches right border (22 px). Last word may be clipped.
  - **Left Cutoff**: sentence_04_dictated_english ink touches left border (30 px). First word may be clipped.
  - **Right Cutoff**: sentence_04_dictated_english ink touches right border (69 px). Last word may be clipped.
  - **Bottom Cutoff**: sentence_05_own_hindi ink touches bottom border (2042 px). Descenders may be cut off.
  - **Right Cutoff**: sentence_05_own_hindi ink touches right border (21 px). Last word may be clipped.
  - **Left Cutoff**: sentence_06_own_english ink touches left border (24 px). First word may be clipped.
  - **Right Cutoff**: sentence_06_own_english ink touches right border (21 px). Last word may be clipped.
  - **Unsegmented Ink Below Protocol**: 100277 ink pixels below y=1764. Child may have written below the detected boxes!
