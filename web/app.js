document.addEventListener("DOMContentLoaded", () => {
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("file-input");
  const previewContainer = document.getElementById("preview-container");
  const previewImg = document.getElementById("preview-img");
  const clearBtn = document.getElementById("clear-btn");
  const analyzeBtn = document.getElementById("analyze-btn");
  const btnText = document.getElementById("btn-text");
  const btnSpinner = document.getElementById("btn-spinner");

  const resultsPlaceholder = document.getElementById("results-placeholder");
  const resultsContainer = document.getElementById("results-container");

  const badgeCard = document.getElementById("screening-badge-card");
  const badgeTitle = document.getElementById("badge-title");
  const badgeDesc = document.getElementById("badge-desc");
  const badgeScoreVal = document.getElementById("badge-score-val");

  const overlayImg = document.getElementById("overlay-img");
  const kinematicsPlotImg = document.getElementById("kinematics-plot-img");
  const inkSkeletonImg = document.getElementById("ink-skeleton-img");
  const tableBody = document.getElementById("features-table-body");
  const riskFactorsList = document.getElementById("risk-factors-list");

  let currentFile = null;

  // Tab switching
  const tabButtons = document.querySelectorAll(".tab-btn");
  const tabPanes = document.querySelectorAll(".tab-pane");

  tabButtons.forEach(btn => {
    btn.addEventListener("click", () => {
      tabButtons.forEach(b => b.classList.remove("active"));
      tabPanes.forEach(p => p.classList.remove("active"));

      btn.classList.add("active");
      const target = document.getElementById(btn.getAttribute("data-tab"));
      if (target) target.classList.add("active");
    });
  });

  // Dropzone click -> file input
  dropzone.addEventListener("click", () => fileInput.click());

  // Drag & drop
  ["dragenter", "dragover"].forEach(evt => {
    dropzone.addEventListener(evt, e => {
      e.preventDefault();
      dropzone.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach(evt => {
    dropzone.addEventListener(evt, e => {
      e.preventDefault();
      dropzone.classList.remove("dragover");
    });
  });

  dropzone.addEventListener("drop", e => {
    const files = e.dataTransfer.files;
    if (files.length > 0) handleFile(files[0]);
  });

  fileInput.addEventListener("change", e => {
    if (e.target.files.length > 0) handleFile(e.target.files[0]);
  });

  // Clipboard paste (Ctrl+V)
  window.addEventListener("paste", e => {
    const items = (e.clipboardData || e.originalEvent.clipboardData).items;
    for (let item of items) {
      if (item.type.indexOf("image") !== -1) {
        const file = item.getAsFile();
        handleFile(file);
        break;
      }
    }
  });

  function handleFile(file) {
    if (!file || !file.type.startsWith("image/")) {
      alert("Please upload a valid handwriting image file (JPG, PNG, WEBP).");
      return;
    }
    currentFile = file;
    const reader = new FileReader();
    reader.onload = e => {
      previewImg.src = e.target.result;
      dropzone.style.display = "none";
      previewContainer.style.display = "block";
      analyzeBtn.disabled = false;
    };
    reader.readAsDataURL(file);
  }

  // Clear file
  clearBtn.addEventListener("click", () => {
    currentFile = null;
    fileInput.value = "";
    previewImg.src = "";
    previewContainer.style.display = "none";
    dropzone.style.display = "block";
    analyzeBtn.disabled = true;
  });

  // Quick Demo Pills
  document.querySelectorAll(".demo-pill").forEach(pill => {
    pill.addEventListener("click", async () => {
      const demoType = pill.getAttribute("data-type");
      pill.style.opacity = "0.5";
      try {
        const res = await fetch(`/api/demo?type=${demoType}`);
        if (!res.ok) throw new Error("Demo sample not found");
        const blob = await res.blob();
        const file = new File([blob], `${demoType}.jpg`, { type: blob.type });
        handleFile(file);
        // Automatically trigger analysis
        setTimeout(() => analyzeBtn.click(), 200);
      } catch (err) {
        alert("Could not load demo sample: " + err.message);
      } finally {
        pill.style.opacity = "1";
      }
    });
  });

  // Analyze Click
  analyzeBtn.addEventListener("click", async () => {
    if (!currentFile) return;

    analyzeBtn.disabled = true;
    btnText.textContent = "Extracting 20D Features & Kinematics...";
    btnSpinner.style.display = "inline-block";

    const formData = new FormData();
    formData.append("image", currentFile);

    try {
      const response = await fetch("/api/analyze", {
        method: "POST",
        body: formData
      });

      if (!response.ok) {
        const errJson = await response.json();
        throw new Error(errJson.error || "Analysis failed");
      }

      const data = await response.json();
      renderResults(data);
    } catch (err) {
      alert("Analysis error: " + err.message);
    } finally {
      analyzeBtn.disabled = false;
      btnText.textContent = "🔍 Analyze Handwriting Sample";
      btnSpinner.style.display = "none";
    }
  });

  function renderResults(data) {
    resultsPlaceholder.style.display = "none";
    resultsContainer.style.display = "block";

    // 1. Badge & Verdict
    const verdict = data.verdict;
    badgeScoreVal.textContent = `${verdict.risk_score_percent}%`;
    badgeTitle.textContent = verdict.screening_badge;

    badgeCard.className = `screening-badge-card badge-${verdict.status}`;

    if (verdict.status === "typical") {
      badgeDesc.textContent = "Handwriting exhibits consistent letter sizing, fluid kinematic velocity curves, and minimal NVI hesitations.";
    } else if (verdict.status === "borderline") {
      badgeDesc.textContent = "Mild irregularities detected in baseline wander, spacing entropy, or stroke transitions. Recommend teacher monitoring.";
    } else {
      badgeDesc.textContent = "Significant motor dysfluency, elevated velocity inversions (NVI), or letter size inconsistency detected. Recommended for formal evaluation.";
    }

    // 2. Visualizations
    overlayImg.src = data.visuals.overlay_b64;
    kinematicsPlotImg.src = data.visuals.kinematics_plot_b64;
    inkSkeletonImg.src = data.visuals.ink_skeleton_b64;

    // 3. Risk Factors
    riskFactorsList.innerHTML = "";
    (verdict.contributing_risk_factors || []).forEach(rf => {
      const li = document.createElement("li");
      li.textContent = rf;
      riskFactorsList.appendChild(li);
    });

    // 4. Populate 20D Table
    tableBody.innerHTML = "";

    const rows = [
      // Spatial BHK
      { domain: "Spatial BHK", name: "BHK #1 & #8: Letter Size CoV", val: data.features.bhk.size_covariance_score, norm: "≤ 1.90 (Inconsistency)" },
      { domain: "Spatial BHK", name: "BHK #9: Relative Height Ratio (IQR)", val: data.features.bhk.height_iqr_ratio, norm: "1.20 - 1.85 (Proportionality)" },
      { domain: "Spatial BHK", name: "BHK #3: Baseline Drift & Wander", val: data.features.bhk.baseline_drift_score, norm: "≤ 0.85 (Slope + RMSE)" },
      { domain: "Spatial BHK", name: "BHK #4: Spacing Entropy", val: data.features.bhk.spacing_entropy, norm: "≤ 0.70 (Rhythm chaos)" },
      { domain: "Spatial BHK", name: "BHK #6: Pressure Proxy CoV (Width)", val: data.features.bhk.stroke_width_cv, norm: "≤ 0.35 (Force stability)" },
      { domain: "Spatial BHK", name: "BHK #7: Telescoping Character Overlap", val: data.features.bhk.telescoping_score + "%", norm: "≤ 24.0% (Collisions)" },
      { domain: "Spatial BHK", name: "BHK #5: Acute Turns & Jaggedness", val: data.features.bhk.acute_turns_score, norm: "≤ 0.40 (Tremor turns)" },
      { domain: "Spatial BHK", name: "BHK #2: Left Margin Alignment Drift", val: data.features.bhk.left_margin_score, norm: "≤ 0.25 (|dy/dx| slope)" },
      { domain: "Spatial BHK", name: "BHK #13: Inter-Line Collisions", val: data.features.bhk.line_collision_score, norm: "≤ 0.15 (Line collisions)" },

      // Kinematics
      { domain: "Kinematics", name: "NVI per Stroke (Scale-Invariant Fluency)", val: data.features.kinematics.nvi_per_stroke, norm: "1.2 - 2.4 (Reversals/stroke)" },
      { domain: "Kinematics", name: "NVI per 100px Arc Length", val: data.features.kinematics.nvi_per_100px, norm: "0.8 - 1.6 (Hesitation density)" },
      { domain: "Kinematics", name: "NVI Rate (Hesitations / sec)", val: data.features.kinematics.nvi_rate, norm: "1.5 - 3.5 Hz" },
      { domain: "Kinematics", name: "Flash & Hogan Dimensionless Jerk", val: data.features.kinematics.dimensionless_jerk, norm: "≤ 2.50 (Movement jerkiness)" },
      { domain: "Kinematics", name: "4-8 Hz Neuromuscular Tremor Index", val: (data.features.kinematics.tremor_index_4_8hz * 100).toFixed(1) + "%", norm: "≤ 16.0% (Involuntary power)" },
      { domain: "Kinematics", name: "Velocity Skewness (Deceleration Tail)", val: data.features.kinematics.velocity_skewness, norm: "0.05 to 0.45 (Right skew)" },
      { domain: "Kinematics", name: "Mean Reconstructed Velocity", val: data.features.kinematics.mean_velocity, norm: "35 - 65 px/s" },
      { domain: "Kinematics", name: "Peak Reconstructed Velocity", val: data.features.kinematics.peak_velocity, norm: "70 - 150 px/s" },
      { domain: "Kinematics", name: "Pen Lift Count (Stroke Count)", val: data.features.kinematics.pen_lift_count, norm: "Varies by sentence length" },
      { domain: "Kinematics", name: "Mean Stroke Arc Length", val: data.features.kinematics.mean_stroke_length + " px", norm: "12 - 35 px" },
      { domain: "Kinematics", name: "Total Velocity Inversions", val: data.features.kinematics.total_nvi, norm: "Total local speed dips" }
    ];

    rows.forEach(r => {
      const tr = document.createElement("tr");
      const valStr = typeof r.val === "number" ? r.val.toFixed(3) : r.val;
      tr.innerHTML = `
        <td><span style="color:${r.domain === 'Spatial BHK' ? '#38bdf8' : '#34d399'}; font-weight:600;">${r.domain}</span></td>
        <td><strong>${r.name}</strong></td>
        <td><span class="val-badge">${valStr}</span></td>
        <td style="color:#94a3b8; font-size:12px;">${r.norm}</td>
      `;
      tableBody.appendChild(tr);
    });
  }
});
