document.addEventListener("DOMContentLoaded", () => {
  // Elements
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

  // Stat Badges (8 Key Biomarkers)
  const statStrokes = document.getElementById("stat-strokes");
  const statStrokeLen = document.getElementById("stat-stroke-len");
  const statInkWidth = document.getElementById("stat-ink-width");
  const statPressure = document.getElementById("stat-pressure");
  const statSpeed = document.getElementById("stat-speed");
  const statNvi = document.getElementById("stat-nvi");
  const statTremor = document.getElementById("stat-tremor");
  const statSizeCov = document.getElementById("stat-size-cov");

  // Canvas Viewport & HUD
  const canvasViewport = document.getElementById("canvas-viewport");
  const writingCanvas = document.getElementById("writing-canvas");
  const pointHud = document.getElementById("point-hud");
  const hudV = document.getElementById("hud-v");
  const hudA = document.getElementById("hud-a");
  const hudK = document.getElementById("hud-k");
  const hudWidth = document.getElementById("hud-width");
  const hudPressure = document.getElementById("hud-pressure");
  const hudStroke = document.getElementById("hud-stroke");
  const hudStrokeLen = document.getElementById("hud-stroke-len");
  const hudArcPos = document.getElementById("hud-arc-pos");
  const hudStatus = document.getElementById("hud-status");
  const telemetryText = document.getElementById("telemetry-text");

  // Heatmap Metric Buttons & Legend
  const btnModeVelocity = document.getElementById("btn-mode-velocity");
  const btnModePressure = document.getElementById("btn-mode-pressure");
  const legendMetricLabel = document.getElementById("legend-metric-label");
  const legendScales = document.getElementById("legend-scales");
  let currentHeatmapMode = "velocity"; // "velocity" or "pressure"

  // Canvas Layer Toggles
  const toggleHeatmap = document.getElementById("toggle-heatmap");
  const toggleBoxes = document.getElementById("toggle-boxes");
  const toggleBaselines = document.getElementById("toggle-baselines");
  const toggleNvi = document.getElementById("toggle-nvi");

  // Waveform Elements & Toggles
  const waveformCanvas = document.getElementById("waveform-canvas");
  const scrubberInfo = document.getElementById("waveform-scrubber-info");
  const toggleWaveV = document.getElementById("toggle-wave-v");
  const toggleWaveP = document.getElementById("toggle-wave-p");
  const toggleWaveA = document.getElementById("toggle-wave-a");
  const toggleWaveJ = document.getElementById("toggle-wave-j");
  const toggleWaveK = document.getElementById("toggle-wave-k");
  const toggleWaveNvi = document.getElementById("toggle-wave-nvi");

  // Strokes Tab KPIs & Table
  const kpiTotalStrokes = document.getElementById("kpi-total-strokes");
  const kpiTotalArc = document.getElementById("kpi-total-arc");
  const kpiMeanLen = document.getElementById("kpi-mean-len");
  const kpiMeanWidth = document.getElementById("kpi-mean-width");
  const kpiMeanPress = document.getElementById("kpi-mean-press");
  const kpiSubStrokes = document.getElementById("kpi-sub-strokes");
  const kpiSubArc = document.getElementById("kpi-sub-arc");
  const kpiSubLen = document.getElementById("kpi-sub-len");
  const kpiSubWidth = document.getElementById("kpi-sub-width");
  const kpiSubPress = document.getElementById("kpi-sub-press");
  const strokesTableBody = document.getElementById("strokes-table-body");

  // Table Body & Export
  const tableBody = document.getElementById("features-table-body");
  const btnExportJson = document.getElementById("btn-export-json");

  // State
  let currentFile = null;
  let analysisData = null;
  let loadedBaseImage = null;
  let hoveredPoint = null;
  let selectedStrokeId = null;

  const strokeColors = [
    "#38bdf8", "#a78bfa", "#f43f5e", "#fbbf24", "#34d399",
    "#60a5fa", "#f472b6", "#fb923c", "#4ade80", "#22d3ee",
    "#818cf8", "#e879f9", "#f87171", "#facc15", "#2dd4bf",
    "#c084fc", "#fb7185", "#38bdf8", "#a3e635", "#e2e8f0"
  ];

  // Tab switching
  const tabButtons = document.querySelectorAll(".tab-btn");
  const tabPanes = document.querySelectorAll(".tab-pane");

  tabButtons.forEach(btn => {
    btn.addEventListener("click", () => {
      tabButtons.forEach(b => b.classList.remove("active"));
      tabPanes.forEach(p => p.classList.remove("active"));

      btn.classList.add("active");
      const target = document.getElementById(btn.getAttribute("data-tab"));
      if (target) {
        target.classList.add("active");
        if (target.id === "tab-waveforms") {
          setTimeout(renderWaveforms, 50);
        } else if (target.id === "tab-interactive") {
          setTimeout(drawWritingCanvas, 50);
        } else if (target.id === "tab-compare-gt") {
          runGtComparison();
        }
      }
    });
  });

  // Stages Navigation
  const stageBtns = document.querySelectorAll(".stage-nav-btn");
  const stagePanels = document.querySelectorAll(".stage-panel");

  stageBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      stageBtns.forEach(b => b.classList.remove("active"));
      stagePanels.forEach(p => p.classList.remove("active"));

      btn.classList.add("active");
      const stageNum = btn.getAttribute("data-stage");
      const panel = document.getElementById(`stage-panel-${stageNum}`);
      if (panel) panel.classList.add("active");
    });
  });

  // Dropzone click -> file input (clicks on the dropzone surface itself)
  dropzone.addEventListener("click", (e) => {
    // Don't re-fire if the click came from the + button (it has its own handler)
    if (!e.target.closest(".dropzone-plus-btn")) fileInput.click();
  });

  // The big + button — primary upload entry point
  const dropzonePlus = document.getElementById("dropzone-plus");
  if (dropzonePlus) {
    dropzonePlus.addEventListener("click", (e) => {
      e.stopPropagation(); // prevent bubbling to dropzone
      fileInput.click();
    });
  }

  // "Change Photo" floating button — swap photo without clearing result
  const changePhotoBtn = document.getElementById("change-photo-btn");
  if (changePhotoBtn) {
    changePhotoBtn.addEventListener("click", () => fileInput.click());
  }

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
    analysisData = null;
    loadedBaseImage = null;
    hoveredPoint = null;
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
        setTimeout(() => analyzeBtn.click(), 250);
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

      analysisData = await response.json();

      // Preload image for canvas
      const img = new Image();
      img.onload = () => {
        loadedBaseImage = img;
        renderResults(analysisData);
      };
      img.src = previewImg.src;

    } catch (err) {
      alert("Analysis error: " + err.message);
    } finally {
      analyzeBtn.disabled = false;
      btnText.textContent = "🔍 Extract 20D Multimodal Features";
      btnSpinner.style.display = "none";
    }
  });

  // Heatmap Metric Mode Toggles
  if (btnModeVelocity && btnModePressure) {
    btnModeVelocity.addEventListener("click", () => {
      btnModeVelocity.classList.add("active");
      btnModePressure.classList.remove("active");
      currentHeatmapMode = "velocity";
      if (legendMetricLabel) legendMetricLabel.textContent = "Instantaneous Velocity v(s):";
      if (legendScales) {
        legendScales.innerHTML = `
          <span>Slow / Tight Corner (&lt;0.5 H_med/s)</span>
          <span>Cruising Velocity (~1.5 H_med/s)</span>
          <span>Rapid Ballistic (&gt;3.0 H_med/s)</span>
        `;
      }
      drawWritingCanvas();
    });

    btnModePressure.addEventListener("click", () => {
      btnModePressure.classList.add("active");
      btnModeVelocity.classList.remove("active");
      currentHeatmapMode = "pressure";
      if (legendMetricLabel) legendMetricLabel.textContent = "Optical Stylus Pressure Proxy P(s) = 2·EDT / H_med:";
      if (legendScales) {
        legendScales.innerHTML = `
          <span>Light / Thin Nib (&lt;0.08 W/H_med)</span>
          <span>Moderate Down-Force (~0.18 W/H_med)</span>
          <span>Heavy Grip / Broad Nib (&gt;0.30 W/H_med)</span>
        `;
      }
      drawWritingCanvas();
    });
  }

  // Layer Toggles Event Listeners
  [toggleHeatmap, toggleBoxes, toggleBaselines, toggleNvi].forEach(cb => {
    if (cb) {
      cb.addEventListener("change", () => {
        drawWritingCanvas();
      });
    }
  });

  // Waveform Toggles Event Listeners
  [toggleWaveV, toggleWaveP, toggleWaveA, toggleWaveJ, toggleWaveK, toggleWaveNvi].forEach(cb => {
    if (cb) {
      cb.addEventListener("change", () => {
        renderWaveforms();
      });
    }
  });

  if (btnExportJson) {
    btnExportJson.addEventListener("click", () => {
      if (!analysisData || !analysisData.export_payload) {
        alert("No feature extraction payload available to export.");
        return;
      }
      const payload = analysisData.export_payload;
      const jsonStr = JSON.stringify(payload, null, 2);
      const blob = new Blob([jsonStr], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      const sid = payload.sample_id || "sample";
      a.href = url;
      a.download = `dysgraphia_features_${sid}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    });
  }

  // Populate Results
  function renderResults(data) {
    resultsPlaceholder.style.display = "none";
    resultsContainer.style.display = "block";
    if (btnExportJson) btnExportJson.style.display = "inline-flex";

    const s = data.summary || {};
    const hMed = data.image_scale_metadata?.h_med_px || 25.0;

    // 1. Top Stat Badges (8 Key Biomarkers)
    if (statStrokes) statStrokes.textContent = s.strokes_count ?? 0;
    if (statStrokeLen) statStrokeLen.innerHTML = `${s.mean_stroke_length_px ?? 0} <span class="stat-unit">px</span>`;
    if (statInkWidth) statInkWidth.innerHTML = `${s.mean_ink_width_px ?? 0} <span class="stat-unit">px</span>`;
    if (statPressure) statPressure.innerHTML = `${s.mean_pressure_proxy ?? 0.00} <span class="stat-unit">W/H_med</span>`;
    if (statSpeed) statSpeed.innerHTML = `${s.mean_velocity ?? 0.0} <span class="stat-unit">H_med/s</span>`;
    if (statNvi) statNvi.textContent = s.nvi_per_stroke ?? 0.0;
    if (statTremor) statTremor.textContent = `${s.spatial_roughness_percent ?? s.tremor_percent ?? 0}%`;
    if (statSizeCov) statSizeCov.textContent = s.size_covariance ?? 0.0;

    // 2. Populate Recovered Strokes Overview & Table
    if (kpiTotalStrokes) kpiTotalStrokes.textContent = s.strokes_count ?? 0;
    if (kpiTotalArc) kpiTotalArc.innerHTML = `${s.total_arc_length_px ?? 0} <span class="kpi-unit">px</span>`;
    if (kpiSubArc) kpiSubArc.textContent = `${((s.total_arc_length_px || 0) / Math.max(hMed, 1)).toFixed(1)} H_med total`;
    if (kpiMeanLen) kpiMeanLen.innerHTML = `${s.mean_stroke_length_px ?? 0} <span class="kpi-unit">px</span>`;
    if (kpiSubLen) kpiSubLen.textContent = `${s.mean_stroke_length_hmed ?? 0} H_med / stroke`;
    if (kpiMeanWidth) kpiMeanWidth.innerHTML = `${s.mean_ink_width_px ?? 0} <span class="kpi-unit">px</span>`;
    if (kpiSubWidth) kpiSubWidth.textContent = `CoV: ${s.ink_width_cv ?? 0} (EDT)`;
    if (kpiMeanPress) kpiMeanPress.textContent = s.mean_pressure_proxy ?? "0.000";
    if (kpiSubPress) kpiSubPress.textContent = `Std: ±${s.pressure_proxy_std ?? 0} W/H_med`;

    // Populate Strokes Table
    if (strokesTableBody) {
      strokesTableBody.innerHTML = "";
      const strokesList = data.strokes_data || [];
      strokesList.forEach((stk, idx) => {
        const tr = document.createElement("tr");
        const color = strokeColors[idx % strokeColors.length];
        const confPct = Math.round((stk.confidence || 0.8) * 100);

        tr.innerHTML = `
          <td><span class="stroke-color-dot" style="background:${color};"></span><strong>#${idx + 1}</strong></td>
          <td>${stk.point_count} pts</td>
          <td><strong>${stk.arc_length_px}</strong> px</td>
          <td><span class="val-badge">${stk.arc_length_h_med}</span></td>
          <td>${stk.mean_ink_width_px ?? (s.mean_ink_width_px ?? 0)} px</td>
          <td>${stk.mean_pressure_proxy ?? (s.mean_pressure_proxy ?? 0)}</td>
          <td>${stk.estimated_mean_velocity_h_med_per_s ?? stk.mean_velocity ?? 0}</td>
          <td>${stk.estimated_peak_velocity_h_med_per_s ?? stk.peak_velocity ?? 0}</td>
          <td>${stk.direction_deg ?? 0}°</td>
          <td><span style="font-size:11px; font-weight:600; color:#38bdf8;">${confPct}%</span></td>
          <td><button type="button" class="btn-inspect-stroke" data-stroke="${idx + 1}">Inspect 🔍</button></td>
        `;

        tr.addEventListener("click", () => {
          document.querySelectorAll("#strokes-table tbody tr").forEach(r => r.classList.remove("selected-stroke"));
          tr.classList.add("selected-stroke");
          selectedStrokeId = idx + 1;
          const tabBtnInteractive = document.querySelector('.tab-btn[data-tab="tab-interactive"]');
          if (tabBtnInteractive) tabBtnInteractive.click();
          drawWritingCanvas();
        });

        strokesTableBody.appendChild(tr);
      });
    }

    // 3. Draw Interactive Handwriting Canvas
    drawWritingCanvas();

    // 4. Render Togglable Waveforms
    renderWaveforms();

    // 5. Update Stage Walkthrough Images
    if (data.stages) {
      document.getElementById("stage-img-1").src = data.stages.stage1_binarization;
      document.getElementById("stage-img-2").src = data.stages.stage2_skeleton;
      document.getElementById("stage-img-3").src = data.stages.stage3_bhk;
      document.getElementById("stage-img-4").src = data.stages.stage4_strokes;
      document.getElementById("stage-img-5").src = data.stages.stage5_heatmap;
    }

    // 6. Populate 20D Biomarker Table
    tableBody.innerHTML = "";
    (data.biomarkers_20d || []).forEach(bm => {
      const tr = document.createElement("tr");
      let domainBadge = `<span style="color:#38bdf8; font-weight:600;">Spatial BHK</span>`;
      if (bm.domain.includes("Kinematics")) {
        domainBadge = `<span style="color:#34d399; font-weight:600;">Kinematics</span>`;
      } else if (bm.domain.includes("Pressure")) {
        domainBadge = `<span style="color:#a78bfa; font-weight:600;">Pressure &amp; Ink</span>`;
      }

      const valStr = typeof bm.val === "number" ? bm.val.toFixed(3) : bm.val;
      const unitStr = bm.unit ? ` <span style="color:#94a3b8; font-size:11px;">${bm.unit}</span>` : "";

      const confVal = typeof bm.confidence === "number" ? Math.round(bm.confidence * 100) : 100;
      let confBadgeColor = "#10b981";
      if (confVal < 40) confBadgeColor = "#f43f5e";
      else if (confVal < 70) confBadgeColor = "#f59e0b";
      const confBadge = `<span style="display:inline-block; padding:2px 8px; border-radius:12px; font-size:11px; font-weight:600; background:rgba(255,255,255,0.08); color:${confBadgeColor}; border:1px solid ${confBadgeColor}44;">${confVal}%</span>`;

      tr.innerHTML = `
        <td>${domainBadge}</td>
        <td><strong>${bm.name}</strong></td>
        <td><span class="val-badge">${valStr}</span>${unitStr}</td>
        <td>${confBadge}</td>
        <td><span class="formula-code">${bm.formula}</span></td>
        <td style="color:#cbd5e1; font-size:12px;">${bm.meaning}</td>
      `;
      tableBody.appendChild(tr);
    });
  }

  // Velocity to Color (Turbo-like vibrant colormap)
  function getVelocityColor(v, vMin = 15, vMax = 85) {
    const t = Math.max(0, Math.min(1, (v - vMin) / (vMax - vMin)));
    let r, g, b;
    if (t < 0.25) {
      const u = t / 0.25;
      r = Math.round(59 + u * (6 - 59));
      g = Math.round(130 + u * (182 - 130));
      b = Math.round(246 + u * (212 - 246));
    } else if (t < 0.5) {
      const u = (t - 0.25) / 0.25;
      r = Math.round(6 + u * (16 - 6));
      g = Math.round(182 + u * (185 - 182));
      b = Math.round(212 + u * (129 - 212));
    } else if (t < 0.75) {
      const u = (t - 0.5) / 0.25;
      r = Math.round(16 + u * (250 - 16));
      g = Math.round(185 + u * (204 - 185));
      b = Math.round(129 + u * (21 - 129));
    } else {
      const u = (t - 0.75) / 0.25;
      r = Math.round(250 + u * (239 - 250));
      g = Math.round(204 + u * (68 - 204));
      b = Math.round(21 + u * (68 - 21));
    }
    return `rgb(${r},${g},${b})`;
  }

  // Pressure / Stroke Width to Color (Violet -> Cyan -> Emerald -> Yellow -> Bright Orange/Red)
  function getPressureColor(p, pMin = 0.05, pMax = 0.35) {
    const t = Math.max(0, Math.min(1, (p - pMin) / (pMax - pMin || 0.01)));
    let r, g, b;
    if (t < 0.25) {
      const u = t / 0.25;
      r = Math.round(147 + u * (6 - 147));
      g = Math.round(51 + u * (182 - 51));
      b = Math.round(234 + u * (212 - 234));
    } else if (t < 0.5) {
      const u = (t - 0.25) / 0.25;
      r = Math.round(6 + u * (16 - 6));
      g = Math.round(182 + u * (185 - 182));
      b = Math.round(212 + u * (129 - 212));
    } else if (t < 0.75) {
      const u = (t - 0.5) / 0.25;
      r = Math.round(16 + u * (250 - 16));
      g = Math.round(185 + u * (204 - 185));
      b = Math.round(129 + u * (21 - 129));
    } else {
      const u = (t - 0.75) / 0.25;
      r = Math.round(250 + u * (239 - 250));
      g = Math.round(204 + u * (68 - 204));
      b = Math.round(21 + u * (68 - 21));
    }
    return `rgb(${r},${g},${b})`;
  }

  // Draw Interactive Handwriting Canvas
  function drawWritingCanvas() {
    if (!loadedBaseImage || !analysisData) return;

    const ctx = writingCanvas.getContext("2d");
    const origW = analysisData.dimensions.width;
    const origH = analysisData.dimensions.height;

    writingCanvas.width = origW;
    writingCanvas.height = origH;

    // 1. Draw base handwriting image
    ctx.drawImage(loadedBaseImage, 0, 0, origW, origH);

    // 2. Draw BHK Letter Bounding Boxes
    if (toggleBoxes && toggleBoxes.checked && analysisData.spatial_boxes) {
      ctx.lineWidth = 1.8;
      ctx.strokeStyle = "rgba(16, 185, 129, 0.85)";
      ctx.fillStyle = "rgba(16, 185, 129, 0.12)";

      analysisData.spatial_boxes.forEach(b => {
        const xMin = b[0], yMin = b[1], xMax = b[2], yMax = b[3], cx = b[4], cy = b[5];
        ctx.strokeRect(xMin, yMin, xMax - xMin, yMax - yMin);
        ctx.fillRect(xMin, yMin, xMax - xMin, yMax - yMin);

        // Centroid dot
        ctx.beginPath();
        ctx.arc(cx, cy, 2.5, 0, 2 * Math.PI);
        ctx.fillStyle = "#38bdf8";
        ctx.fill();
      });
    }

    // 3. Draw Fitted Baselines
    if (toggleBaselines && toggleBaselines.checked && analysisData.spatial_baselines) {
      ctx.lineWidth = 2.2;
      ctx.strokeStyle = "rgba(244, 63, 94, 0.9)";
      analysisData.spatial_baselines.forEach(bl => {
        ctx.beginPath();
        ctx.moveTo(bl[0], bl[1]);
        ctx.lineTo(bl[2], bl[3]);
        ctx.stroke();
      });
    }

    // 4. Draw Heatmap Strokes (Velocity or Pressure)
    const pts = analysisData.point_kinematics || [];
    if (toggleHeatmap && toggleHeatmap.checked && pts.length > 1) {
      ctx.lineWidth = 2.8;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";

      for (let i = 0; i < pts.length - 1; i++) {
        const p1 = pts[i];
        const p2 = pts[i + 1];

        // Only draw segment if same stroke and physically connected
        if (p1[5] === p2[5]) {
          const dx = p2[0] - p1[0];
          const dy = p2[1] - p1[1];
          if (dx * dx + dy * dy < 250) {
            ctx.beginPath();
            const color = (currentHeatmapMode === "pressure")
              ? getPressureColor(((p1[8] || 0.15) + (p2[8] || 0.15)) / 2)
              : getVelocityColor((p1[2] + p2[2]) / 2);
            ctx.strokeStyle = color;
            ctx.moveTo(p1[0], p1[1]);
            ctx.lineTo(p2[0], p2[1]);
            ctx.stroke();
          }
        }
      }
    }

    // 5. Draw NVI Hesitations
    if (toggleNvi && toggleNvi.checked && pts.length > 0) {
      pts.forEach(p => {
        if (p[6] === 1) {
          ctx.beginPath();
          ctx.arc(p[0], p[1], 4, 0, 2 * Math.PI);
          ctx.fillStyle = "rgba(245, 158, 11, 0.9)";
          ctx.strokeStyle = "#ffffff";
          ctx.lineWidth = 1.2;
          ctx.fill();
          ctx.stroke();
        }
      });
    }

    // 6. Highlight Selected Stroke from Strokes Table
    if (selectedStrokeId !== null && pts.length > 1) {
      ctx.save();
      ctx.lineWidth = 5.0;
      ctx.strokeStyle = "#38bdf8";
      ctx.shadowColor = "#38bdf8";
      ctx.shadowBlur = 16;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";

      for (let i = 0; i < pts.length - 1; i++) {
        const p1 = pts[i];
        const p2 = pts[i + 1];
        if (p1[5] === selectedStrokeId && p2[5] === selectedStrokeId) {
          const dx = p2[0] - p1[0];
          const dy = p2[1] - p1[1];
          if (dx * dx + dy * dy < 250) {
            ctx.beginPath();
            ctx.moveTo(p1[0], p1[1]);
            ctx.lineTo(p2[0], p2[1]);
            ctx.stroke();
          }
        }
      }
      ctx.restore();
    }

    // 7. Draw Hover Reticle if pointing at a stroke
    if (hoveredPoint) {
      ctx.save();
      const hx = hoveredPoint[0];
      const hy = hoveredPoint[1];

      // Pulsing outer ring
      ctx.beginPath();
      ctx.arc(hx, hy, 10, 0, 2 * Math.PI);
      ctx.strokeStyle = "#38bdf8";
      ctx.lineWidth = 2.5;
      ctx.shadowColor = "#38bdf8";
      ctx.shadowBlur = 14;
      ctx.stroke();

      // Center crosshair dot
      ctx.beginPath();
      ctx.arc(hx, hy, 3.5, 0, 2 * Math.PI);
      ctx.fillStyle = "#ffffff";
      ctx.fill();

      // Short crosshair lines
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(hx - 14, hy); ctx.lineTo(hx - 6, hy);
      ctx.moveTo(hx + 6, hy);  ctx.lineTo(hx + 14, hy);
      ctx.moveTo(hx, hy - 14); ctx.lineTo(hx, hy - 6);
      ctx.moveTo(hx, hy + 6);  ctx.lineTo(hx, hy + 14);
      ctx.stroke();
      ctx.restore();
    }
  }

  // Pointer Hover Tracking on Handwriting Canvas
  writingCanvas.addEventListener("mousemove", e => {
    if (!analysisData || !analysisData.point_kinematics) return;

    const rect = writingCanvas.getBoundingClientRect();
    const scaleX = writingCanvas.width / rect.width;
    const scaleY = writingCanvas.height / rect.height;

    const mouseImgX = (e.clientX - rect.left) * scaleX;
    const mouseImgY = (e.clientY - rect.top) * scaleY;

    // Nearest Neighbor search in point_kinematics:
    // [x, y, v, a, k, stroke_id, is_nvi, ink_w, press, total_len_px, total_len_hmed, arc_pos]
    const pts = analysisData.point_kinematics;
    let closestPt = null;
    let minDistSq = 32 * 32;

    for (let i = 0; i < pts.length; i++) {
      const p = pts[i];
      const dSq = (p[0] - mouseImgX) ** 2 + (p[1] - mouseImgY) ** 2;
      if (dSq < minDistSq) {
        minDistSq = dSq;
        closestPt = p;
      }
    }

    if (closestPt) {
      hoveredPoint = closestPt;
      drawWritingCanvas();

      // Position floating HUD tooltip near cursor
      const hudX = e.clientX - rect.left + 18;
      const hudY = e.clientY - rect.top + 18;
      pointHud.style.left = `${Math.min(hudX, rect.width - 240)}px`;
      pointHud.style.top = `${Math.min(hudY, rect.height - 240)}px`;
      pointHud.style.display = "block";

      const v = closestPt[2];
      const a = closestPt[3];
      const k = closestPt[4];
      const stkId = closestPt[5];
      const isNvi = closestPt[6] === 1;
      const inkW = closestPt[7] !== undefined ? closestPt[7] : 2.0;
      const press = closestPt[8] !== undefined ? closestPt[8] : 0.15;
      const stkLenPx = closestPt[9] !== undefined ? closestPt[9] : 0.0;
      const stkLenHmed = closestPt[10] !== undefined ? closestPt[10] : 0.0;
      const arcPos = closestPt[11] !== undefined ? closestPt[11] : 0.0;

      hudV.textContent = `${v} H_med/s`;
      hudA.textContent = `${a > 0 ? "+" : ""}${a} H_med/s²`;
      hudK.textContent = `${k} (κ·H_med)`;
      if (hudWidth) hudWidth.textContent = `${inkW} px`;
      if (hudPressure) hudPressure.textContent = `${press} W/H_med`;
      hudStroke.textContent = `Stroke #${stkId} of ${analysisData.summary?.strokes_count || 1}`;
      if (hudStrokeLen) hudStrokeLen.textContent = `${stkLenPx} px (${stkLenHmed} H_med)`;
      if (hudArcPos) hudArcPos.textContent = `${arcPos} px (${stkLenPx > 0 ? Math.round(arcPos / stkLenPx * 100) : 0}%)`;

      let motionType = "Moderate Cruise";
      if (v < 0.6) motionType = "Corner Deceleration";
      else if (v > 2.5) motionType = "Rapid Ballistic Burst";

      hudStatus.innerHTML = isNvi
        ? `<span style="color:#f59e0b; font-weight:700;">⚠️ Velocity Inversion (NVI)</span>`
        : `<span style="color:#34d399; font-weight:600;">🟢 ${motionType}</span>`;

      // Update fixed Telemetry Bar
      telemetryText.innerHTML = `
        <strong>📍 Position:</strong> (${Math.round(closestPt[0])}, ${Math.round(closestPt[1])}) &nbsp;|&nbsp;
        <strong>⚡ Speed:</strong> <span style="color:#38bdf8; font-weight:700;">${v} H_med/s</span> &nbsp;|&nbsp;
        <strong>🖋️ Ink Width:</strong> <span style="color:#a78bfa; font-weight:700;">${inkW} px</span> &nbsp;|&nbsp;
        <strong>⬇️ Pressure:</strong> <span style="color:#10b981; font-weight:700;">${press} W/H_med</span> &nbsp;|&nbsp;
        <strong>📏 Stroke #${stkId}:</strong> ${stkLenPx} px (${stkLenHmed} H_med)
        ${isNvi ? ' &nbsp;|&nbsp; <span style="color:#f59e0b; font-weight:700;">⚠️ Hesitation (NVI)</span>' : ''}
      `;
    } else {
      if (hoveredPoint !== null) {
        hoveredPoint = null;
        drawWritingCanvas();
        pointHud.style.display = "none";
        telemetryText.innerHTML = `
          Hover or point your cursor anywhere on the handwriting above to inspect instantaneous velocity, acceleration, ink width, and pressure at that exact point in real time.
        `;
      }
    }
  });

  writingCanvas.addEventListener("mouseleave", () => {
    hoveredPoint = null;
    drawWritingCanvas();
    pointHud.style.display = "none";
    telemetryText.innerHTML = `
      Hover or point your cursor anywhere on the handwriting above to inspect instantaneous velocity, acceleration, ink width, and pressure at that exact point in real time.
    `;
  });

  // Render Togglable Waveforms Canvas
  function renderWaveforms() {
    if (!analysisData || !analysisData.waveform) return;

    const wf = analysisData.waveform;
    const time = wf.time || [];
    const n = time.length;
    if (n < 2) return;

    const ctx = waveformCanvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const rect = waveformCanvas.getBoundingClientRect();
    const w = rect.width;
    const h = 260;

    waveformCanvas.width = w * dpr;
    waveformCanvas.height = h * dpr;
    ctx.scale(dpr, dpr);

    // Canvas background (light theme)
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, w, h);

    const padL = 50, padR = 25, padT = 30, padB = 40;
    const plotW = w - padL - padR;
    const plotH = h - padT - padB;

    // Draw Subtle Grid
    ctx.strokeStyle = "rgba(0, 0, 0, 0.06)";
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = padT + (plotH / 4) * i;
      ctx.beginPath();
      ctx.moveTo(padL, y);
      ctx.lineTo(w - padR, y);
      ctx.stroke();
    }

    // Time Axis Ticks
    const maxT = time[n - 1] || 1.0;
    ctx.fillStyle = "#64748b";
    ctx.font = "10px JetBrains Mono, monospace";
    ctx.textAlign = "center";
    for (let i = 0; i <= 5; i++) {
      const tVal = (maxT / 5) * i;
      const x = padL + (plotW / 5) * i;
      ctx.fillText(`${tVal.toFixed(1)}s`, x, h - padB + 18);
    }
    ctx.fillText("Execution Time (seconds)", padL + plotW / 2, h - 8);

    // Helpers to normalize and draw curves
    function drawCurve(arr, color, lw = 1.8) {
      if (!arr || arr.length === 0) return;
      const minVal = Math.min(...arr);
      const maxVal = Math.max(...arr);
      const range = maxVal - minVal || 1.0;

      ctx.beginPath();
      ctx.strokeStyle = color;
      ctx.lineWidth = lw;
      for (let i = 0; i < n; i++) {
        const x = padL + (i / (n - 1)) * plotW;
        const normY = (arr[i] - minVal) / range;
        const y = padT + plotH - normY * plotH;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }

    // 1. Velocity v(t) - Cyan
    if (toggleWaveV && toggleWaveV.checked && wf.velocity) {
      drawCurve(wf.velocity, "#38bdf8", 2.2);
    }

    // 2. Pressure Proxy P(t) - Emerald
    if (toggleWaveP && toggleWaveP.checked && wf.pressure) {
      drawCurve(wf.pressure, "#10b981", 2.0);
    }

    // 3. Acceleration a(t) - Purple
    if (toggleWaveA && toggleWaveA.checked && wf.acceleration) {
      drawCurve(wf.acceleration, "#a78bfa", 1.6);
    }

    // 4. Jerk j(t) - Rose
    if (toggleWaveJ && toggleWaveJ.checked && wf.jerk) {
      drawCurve(wf.jerk, "#f43f5e", 1.2);
    }

    // 5. Curvature κ(t) - Amber
    if (toggleWaveK && toggleWaveK.checked && wf.curvature) {
      drawCurve(wf.curvature, "#fbbf24", 1.4);
    }

    // 6. NVI Markers (Peaks & Troughs)
    if (toggleWaveNvi && toggleWaveNvi.checked && toggleWaveV && toggleWaveV.checked && wf.velocity) {
      const vArr = wf.velocity;
      const minV = Math.min(...vArr);
      const maxV = Math.max(...vArr);
      const range = maxV - minV || 1.0;

      // Green Peaks
      (wf.peaks || []).forEach(idx => {
        if (idx < n) {
          const x = padL + (idx / (n - 1)) * plotW;
          const y = padT + plotH - ((vArr[idx] - minV) / range) * plotH;
          ctx.beginPath();
          ctx.arc(x, y, 4, 0, 2 * Math.PI);
          ctx.fillStyle = "#10b981";
          ctx.fill();
        }
      });

      // Orange Troughs (Hesitations)
      (wf.troughs || []).forEach(idx => {
        if (idx < n) {
          const x = padL + (idx / (n - 1)) * plotW;
          const y = padT + plotH - ((vArr[idx] - minV) / range) * plotH;
          ctx.beginPath();
          ctx.arc(x, y, 4, 0, 2 * Math.PI);
          ctx.fillStyle = "#f59e0b";
          ctx.fill();
        }
      });
    }

    scrubberInfo.textContent = `Hover over the waveform to scrub across continuous time series (${n} sampled points).`;
  }

  // Waveform Scrubber
  waveformCanvas.addEventListener("mousemove", e => {
    if (!analysisData || !analysisData.waveform) return;

    const wf = analysisData.waveform;
    const time = wf.time || [];
    const n = time.length;
    if (n < 2) return;

    const rect = waveformCanvas.getBoundingClientRect();
    const padL = 50, padR = 25;
    const plotW = rect.width - padL - padR;

    const mouseX = e.clientX - rect.left;
    if (mouseX < padL || mouseX > rect.width - padR) return;

    const frac = Math.max(0, Math.min(1, (mouseX - padL) / plotW));
    const idx = Math.min(n - 1, Math.floor(frac * n));

    const curT = time[idx];
    const curV = wf.velocity ? wf.velocity[idx] : 0;
    const curP = wf.pressure ? wf.pressure[idx] : 0;
    const curA = wf.acceleration ? wf.acceleration[idx] : 0;
    const curJ = wf.jerk ? wf.jerk[idx] : 0;
    const curK = wf.curvature ? wf.curvature[idx] : 0;

    scrubberInfo.innerHTML = `
      <strong>Time:</strong> ${curT}s &nbsp;|&nbsp;
      <span style="color:#38bdf8;"><strong>v(t):</strong> ${curV} H_med/s</span> &nbsp;|&nbsp;
      <span style="color:#10b981;"><strong>P(t):</strong> ${curP} W/H_med</span> &nbsp;|&nbsp;
      <span style="color:#a78bfa;"><strong>a(t):</strong> ${curA} H_med/s²</span> &nbsp;|&nbsp;
      <span style="color:#f43f5e;"><strong>j(t):</strong> ${curJ} H_med/s³</span> &nbsp;|&nbsp;
      <span style="color:#fbbf24;"><strong>κ(t):</strong> ${curK}</span>
    `;
  });

  // --- Ground Truth Trajectory Alignment & Audit ---
  const gtCsvSelect = document.getElementById("gt-csv-select");
  const btnRunGtCompare = document.getElementById("btn-run-gt-compare");
  const gtStatCleaning = document.getElementById("gt-stat-cleaning");
  const gtStatSpikeFix = document.getElementById("gt-stat-spike-fix");
  const gtStatHmed = document.getElementById("gt-stat-hmed");
  const gtStatComparable = document.getElementById("gt-stat-comparable");
  const gtTbody = document.getElementById("gt-comparison-tbody");

  async function runGtComparison() {
    if (!gtTbody) return;
    gtTbody.innerHTML = `<tr><td colspan="7" style="text-align:center; padding:30px; color:var(--text-secondary);">⏳ Computing ground truth trajectory alignment and cleaning timestamps...</td></tr>`;

    try {
      const csvName = gtCsvSelect ? gtCsvSelect.value : "kinematics_1790656668.csv";
      const payload = {
        csv_name: csvName,
        features: analysisData && analysisData.features ? analysisData.features : null
      };

      const resp = await fetch("/api/compare_gt", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const data = await resp.json();
      if (data.error) {
        gtTbody.innerHTML = `<tr><td colspan="7" style="text-align:center; padding:20px; color:var(--accent-rose);">❌ ${data.error}</td></tr>`;
        return;
      }

      // Update Summary Badges
      const meta = data.metadata;
      if (gtStatCleaning) {
        const dropPct = meta.raw_points > 0 ? Math.round(meta.dropped_points / meta.raw_points * 100) : 0;
        gtStatCleaning.textContent = `${meta.dropped_points.toLocaleString()} (${dropPct}%)`;
      }
      if (gtStatSpikeFix) {
        const rawM = (meta.raw_max_velocity_px_s / 1e6).toFixed(2);
        const cleanK = (meta.clean_max_velocity_px_s / 1e3).toFixed(1);
        gtStatSpikeFix.textContent = `${rawM}M → ${cleanK}k px/s`;
      }
      if (gtStatHmed) {
        gtStatHmed.textContent = `${meta.gt_h_med_px.toFixed(2)} px`;
      }
      if (gtStatComparable) {
        gtStatComparable.textContent = `${meta.comparable_count} / ${meta.total_features}`;
      }

      // Render Table
      gtTbody.innerHTML = "";
      data.features.forEach(f => {
        const tr = document.createElement("tr");
        tr.style.borderBottom = "1px solid var(--border-color)";

        const gValStr = f.gt_value !== null ? Number(f.gt_value).toFixed(4) : `<span style="color:var(--text-muted); font-style:italic;">N/A (No GT)</span>`;
        const pValStr = f.predicted_value !== null ? Number(f.predicted_value).toFixed(4) : `<span style="color:var(--text-muted); font-style:italic;">None</span>`;

        let diffBadge = "";
        let statusBadge = "";

        if (f.comparable) {
          const err = f.error_pct;
          let badgeColor = "var(--accent-emerald)";
          if (err > 75) badgeColor = "var(--accent-rose)";
          else if (err > 25) badgeColor = "var(--accent-amber)";
          diffBadge = `<span class="val-badge" style="background:${badgeColor}22; color:${badgeColor}; font-weight:700;">${err !== null ? err.toFixed(1) + "%" : "—"}</span>`;
          statusBadge = `<span style="color:var(--accent-emerald); font-weight:600;">✓ Directly Comparable</span> <span style="font-size:12px; color:var(--text-secondary); display:block;">Aligned in ${f.units}</span>`;
        } else {
          diffBadge = `<span style="color:var(--text-muted);">—</span>`;
          statusBadge = `<span style="color:var(--accent-amber); font-weight:600;">⚠️ Not Comparable</span> <span style="font-size:12px; color:var(--text-secondary); display:block; margin-top:2px;">${f.status_description}</span>`;
        }

        tr.innerHTML = `
          <td style="padding: 10px 14px; font-weight:600; color:var(--text-primary);">${f.feature}</td>
          <td style="padding: 10px 14px;"><span class="val-badge" style="font-size:11px;">${f.category}</span></td>
          <td style="padding: 10px 14px; font-family:var(--font-mono); font-size:12px; color:var(--text-secondary);">${f.units}</td>
          <td style="padding: 10px 14px; text-align:right; font-family:var(--font-mono); font-size:12px; font-weight:600; color:var(--text-primary);">${gValStr}</td>
          <td style="padding: 10px 14px; text-align:right; font-family:var(--font-mono); font-size:12px; font-weight:600; color:var(--accent-blue);">${pValStr}</td>
          <td style="padding: 10px 14px; text-align:right;">${diffBadge}</td>
          <td style="padding: 10px 14px;">${statusBadge}</td>
        `;
        gtTbody.appendChild(tr);
      });
    } catch (err) {
      gtTbody.innerHTML = `<tr><td colspan="7" style="text-align:center; padding:20px; color:var(--accent-rose);">❌ Connection error: ${err.message}</td></tr>`;
    }
  }

  if (btnRunGtCompare) btnRunGtCompare.addEventListener("click", runGtComparison);
  if (gtCsvSelect) gtCsvSelect.addEventListener("change", runGtComparison);
});
