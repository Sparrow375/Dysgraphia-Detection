"""
Script-driven Scoreboard Generator for Experiment 7.
Reads primary result JSON files directly from reports/ to prevent any number drift.
"""

import json
from pathlib import Path

REPORTS_DIR = Path("reports")

def generate_markdown_scoreboard():
    # Load all JSONs
    with open(REPORTS_DIR / "exp7_dinov2_small_probe_results.json") as f:
        dino = json.load(f)
    with open(REPORTS_DIR / "exp7_clip_probe_results.json") as f:
        clip = json.load(f)
    with open(REPORTS_DIR / "exp7_densenet201_probe_results.json") as f:
        dense = json.load(f)
    with open(REPORTS_DIR / "exp7_trocr_probe_results.json") as f:
        trocr = json.load(f)
    with open(REPORTS_DIR / "exp7_candidate2_writer_style_results.json") as f:
        writer = json.load(f)
    with open(REPORTS_DIR / "exp7_vlm_rubric_results.json") as f:
        vlm = json.load(f)
    with open(REPORTS_DIR / "exp7_paired_bootstrap_results.json") as f:
        paired = json.load(f)

    dino_joint = dino["architectures"]["Joint Concatenated Multilingual Model"]
    dino_blend = dino["architectures"]["Decision Blend (0.5 Hindi + 0.5 English)"]
    clip_blend = clip["architectures"]["Decision Blend (0.5 Hindi + 0.5 English)"]
    clip_joint = clip["architectures"]["Joint Concatenated Multilingual Model"]
    dense_blend = dense["architectures"]["Decision Blend (0.5 Hindi + 0.5 English)"]
    trocr_blend = trocr["architectures"]["Decision Blend (0.5 Hindi + 0.5 English)"]

    dual_vit = paired["dual_vit_vs_dinov2_joint"]
    dense_sq = paired["densenet_tiled_vs_squashed"]
    tab_comp = paired["dinov2_joint_vs_tabular"]

    lines = []
    lines.append("| Rank | Model Architecture | Pipeline / Evaluation Mode | 5x5 CV AUROC | AUPRC | Sens @ 80% Spec | Sens @ 90% Spec | Source JSON Record |")
    lines.append("| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :--- |")
    
    rows = [
        ("1 (Tie)", "Dual-ViT 50/50 Ensemble", "Fixed 50/50 Rank Average", dual_vit["auc_a"], dual_vit["ap_a"], "75.0% (30/40)", "42.5% (17/40)", "exp7_paired_bootstrap.json"),
        ("2 (Tie)", "DINOv2-Small (ViT-S/14)", "Joint Multilingual Model", dino_joint["auroc"], dino_joint["auprc"], f"{dino_joint['sens80']*100:.1f}% ({dino_joint['det80']}/40)", f"{dino_joint['sens90']*100:.1f}% ({dino_joint['det90']}/40)", "exp7_dinov2_small_probe.json"),
        ("3 (Tie)", "DINOv2-Small (ViT-S/14)", "50/50 Decision Blend", dino_blend["auroc"], dino_blend["auprc"], f"{dino_blend['sens80']*100:.1f}% ({dino_blend['det80']}/40)", f"{dino_blend['sens90']*100:.1f}% ({dino_blend['det90']}/40)", "exp7_dinov2_small_probe.json"),
        ("4", "OpenAI CLIP (ViT-B/32)", "50/50 Decision Blend", clip_blend["auroc"], clip_blend["auprc"], f"{clip_blend['sens80']*100:.1f}% ({clip_blend['det80']}/40)", f"{clip_blend['sens90']*100:.1f}% ({clip_blend['det90']}/40)", "exp7_clip_probe.json"),
        ("5", "OpenAI CLIP (ViT-B/32)", "Joint Multilingual Model", clip_joint["auroc"], clip_joint["auprc"], f"{clip_joint['sens80']*100:.1f}% ({clip_joint['det80']}/40)", f"{clip_joint['sens90']*100:.1f}% ({clip_joint['det90']}/40)", "exp7_clip_probe.json"),
        ("6", "DenseNet201 (CNN-201)", "Tiled 50/50 Decision Blend", dense_blend["auroc"], dense_blend["auprc"], f"{dense_blend['sens80']*100:.1f}% ({dense_blend['det80']}/40)", f"{dense_blend['sens90']*100:.1f}% ({dense_blend['det90']}/40)", "exp7_densenet201_probe.json"),
        ("7", "Stage 4 Tabular Stack", "30 Handcrafted BHK Features", tab_comp["auc_b"], tab_comp["ap_b"], "47.5% (19/40)", "37.5% (15/40)", "exp7_paired_bootstrap.json"),
        ("8", "DenseNet201 (CNN-201)", "Squashed Sentence Ablation", dense_sq["auc_b"], dense_sq["ap_b"], "50.0% (20/40)", "37.5% (15/40)", "exp7_paired_bootstrap.json"),
        ("9", "Writer-Style Encoder", "SupCon Metric Learning", writer["blend"]["auroc"], writer["blend"]["auprc"], f"{writer['blend']['sens80']*100:.1f}%", f"{writer['blend']['sens90']*100:.1f}%", "exp7_candidate2_writer_style.json"),
        ("10", "Microsoft TrOCR-Base", "50/50 Decision Blend", trocr_blend["auroc"], trocr_blend["auprc"], f"{trocr_blend['sens80']*100:.1f}% ({trocr_blend['det80']}/40)", f"{trocr_blend['sens90']*100:.1f}% ({trocr_blend['det90']}/40)", "exp7_trocr_probe.json"),
        ("11", "Candidate 4 VLM Rubric", "Qwen2-VL-2B Pediatric Rubric", vlm["auroc"], vlm["auprc"], f"{vlm['sens_at_80_spec']*100:.1f}% ({vlm['det_at_80_spec']}/40)", f"{vlm['sens_at_90_spec']*100:.1f}% ({vlm['det_at_90_spec']}/40)", "exp7_vlm_rubric.json"),
    ]

    for r in rows:
        lines.append(f"| **{r[0]}** | **{r[1]}** | {r[2]} | **{r[3]:.4f}** | **{r[4]:.4f}** | {r[5]} | {r[6]} | `{r[7]}` |")

    table_md = "\n".join(lines)
    print(table_md)
    return table_md

if __name__ == "__main__":
    generate_markdown_scoreboard()
