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

  // Stat Badges
  const statStrokes = document.getElementById("stat-strokes");
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
  const hudStroke = document.getElementById("hud-stroke");
  const hudStatus = document.getElementById("hud-status");
  const telemetryText = document.getElementById("telemetry-text");

  // Canvas Layer Toggles
  const toggleHeatmap = document.getElementById("toggle-heatmap");
  const toggleBoxes = document.getElementById("toggle-boxes");
  const toggleBaselines = document.getElementById("toggle-baselines");
  const toggleNvi = document.getElementById("toggle-nvi");

  // Waveform Elements & Toggles
  const waveformCanvas = document.getElementById("waveform-canvas");
  const scrubberInfo = document.getElementById("waveform-scrubber-info");
  const toggleWaveV = document.getElementById("toggle-wave-v");
  const toggleWaveA = document.getElementById("toggle-wave-a");
  const toggleWaveJ = document.getElementById("toggle-wave-j");
  const toggleWaveK = document.getElementById("toggle-wave-k");
  const toggleWaveNvi = document.getElementById("toggle-wave-nvi");

  // Table Body
  const tableBody = document.getElementById("features-table-body");

  // State
  let currentFile = null;
  let analysisData = null;
  let loadedBaseImage = null;
  let hoveredPoint = null;

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

  // Layer Toggles Event Listeners
  [toggleHeatmap, toggleBoxes, toggleBaselines, toggleNvi].forEach(cb => {
    cb.addEventListener("change", () => {
      drawWritingCanvas();
    });
  });

  // Waveform Toggles Event Listeners
  [toggleWaveV, toggleWaveA, toggleWaveJ, toggleWaveK, toggleWaveNvi].forEach(cb => {
    cb.addEventListener("change", () => {
      renderWaveforms();
    });
  });

  // Populate Results
  function renderResults(data) {
    resultsPlaceholder.style.display = "none";
    resultsContainer.style.display = "block";

    // 1. Top Stat Badges
    const s = data.summary;
    statStrokes.textContent = s.strokes_count;
    statSpeed.innerHTML = `${s.mean_velocity} <span class="stat-unit">px/s</span>`;
    statNvi.textContent = s.nvi_per_stroke;
    statTremor.textContent = `${s.tremor_percent}%`;
    statSizeCov.textContent = s.size_covariance;

    // 2. Draw Interactive Handwriting Canvas
    drawWritingCanvas();

    // 3. Render Togglable Waveforms
    renderWaveforms();

    // 4. Update Stage Walkthrough Images
    document.getElementById("stage-img-1").src = data.stages.stage1_binarization;
    document.getElementById("stage-img-2").src = data.stages.stage2_skeleton;
    document.getElementById("stage-img-3").src = data.stages.stage3_bhk;
    document.getElementById("stage-img-4").src = data.stages.stage4_strokes;
    document.getElementById("stage-img-5").src = data.stages.stage5_heatmap;

    // 5. Populate 20D Biomarker Table
    tableBody.innerHTML = "";
    (data.biomarkers_20d || []).forEach(bm => {
      const tr = document.createElement("tr");
      const isBhk = bm.domain.includes("BHK");
      const domainBadge = isBhk
        ? `<span style="color:#38bdf8; font-weight:600;">Spatial BHK</span>`
        : `<span style="color:#34d399; font-weight:600;">Kinematics</span>`;

      const valStr = typeof bm.val === "number" ? bm.val.toFixed(3) : bm.val;
      const unitStr = bm.unit ? ` <span style="color:#94a3b8; font-size:11px;">${bm.unit}</span>` : "";

      tr.innerHTML = `
        <td>${domainBadge}</td>
        <td><strong>${bm.name}</strong></td>
        <td><span class="val-badge">${valStr}</span>${unitStr}</td>
        <td><span class="formula-code">${bm.formula}</span></td>
        <td style="color:#cbd5e1; font-size:12px;">${bm.meaning}</td>
      `;
      tableBody.appendChild(tr);
    });
  }

  // Velocity to Color (Turbo-like vibrant colormap)
  function getVelocityColor(v, vMin = 15, vMax = 85) {
    const t = Math.max(0, Math.min(1, (v - vMin) / (vMax - vMin)));
    // Blue (0) -> Cyan (0.25) -> Green (0.5) -> Yellow (0.75) -> Red (1.0)
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
    if (toggleBoxes.checked && analysisData.spatial_boxes) {
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
    if (toggleBaselines.checked && analysisData.spatial_baselines) {
      ctx.lineWidth = 2.2;
      ctx.strokeStyle = "rgba(244, 63, 94, 0.9)";
      analysisData.spatial_baselines.forEach(bl => {
        ctx.beginPath();
        ctx.moveTo(bl[0], bl[1]);
        ctx.lineTo(bl[2], bl[3]);
        ctx.stroke();
      });
    }

    // 4. Draw Velocity Heatmap Strokes
    const pts = analysisData.point_kinematics || [];
    if (toggleHeatmap.checked && pts.length > 1) {
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
            ctx.strokeStyle = getVelocityColor((p1[2] + p2[2]) / 2);
            ctx.moveTo(p1[0], p1[1]);
            ctx.lineTo(p2[0], p2[1]);
            ctx.stroke();
          }
        }
      }
    }

    // 5. Draw NVI Hesitations
    if (toggleNvi.checked && pts.length > 0) {
      pts.forEach(p => {
        if (p[6] === 1) {
          // NVI marker: glowing amber dot
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

    // 6. Draw Hover Reticle if pointing at a stroke
    if (hoveredPoint) {
      ctx.save();
      const hx = hoveredPoint[0];
      const hy = hoveredPoint[1];
      const hv = hoveredPoint[2];

      // Pulsing outer ring
      ctx.beginPath();
      ctx.arc(hx, hy, 9, 0, 2 * Math.PI);
      ctx.strokeStyle = "#38bdf8";
      ctx.lineWidth = 2.5;
      ctx.shadowColor = "#38bdf8";
      ctx.shadowBlur = 12;
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

    // Nearest Neighbor search in point_kinematics: [x, y, v, a, k, stroke_id, is_nvi]
    const pts = analysisData.point_kinematics;
    let closestPt = null;
    let minDistSq = 32 * 32; // 32px search radius in image space

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
      pointHud.style.left = `${Math.min(hudX, rect.width - 220)}px`;
      pointHud.style.top = `${Math.min(hudY, rect.height - 180)}px`;
      pointHud.style.display = "block";

      const v = closestPt[2];
      const a = closestPt[3];
      const k = closestPt[4];
      const stkId = closestPt[5];
      const isNvi = closestPt[6] === 1;

      hudV.textContent = `${v} px/s`;
      hudA.textContent = `${a > 0 ? "+" : ""}${a} px/s²`;
      hudK.textContent = `${k} rad/px`;
      hudStroke.textContent = `Stroke #${stkId}`;

      let motionType = "Moderate Cruise";
      if (v < 22) motionType = "Corner Deceleration";
      else if (v > 65) motionType = "Rapid Ballistic Burst";

      hudStatus.innerHTML = isNvi
        ? `<span style="color:#f59e0b; font-weight:700;">⚠️ Velocity Inversion (NVI)</span>`
        : `<span style="color:#34d399; font-weight:600;">🟢 ${motionType}</span>`;

      // Update fixed Telemetry Bar
      telemetryText.innerHTML = `
        <strong>📍 Position:</strong> (${Math.round(closestPt[0])}, ${Math.round(closestPt[1])}) &nbsp;|&nbsp;
        <strong>⚡ Speed:</strong> <span style="color:#38bdf8; font-weight:700;">${v} px/s</span> &nbsp;|&nbsp;
        <strong>📈 Accel:</strong> ${a > 0 ? "+" : ""}${a} px/s² &nbsp;|&nbsp;
        <strong>🔄 Curvature κ:</strong> ${k} rad/px &nbsp;|&nbsp;
        <strong>✏️ Program:</strong> Stroke #${stkId}
        ${isNvi ? ' &nbsp;|&nbsp; <span style="color:#f59e0b; font-weight:700;">⚠️ Hesitation (NVI)</span>' : ''}
      `;
    } else {
      if (hoveredPoint !== null) {
        hoveredPoint = null;
        drawWritingCanvas();
        pointHud.style.display = "none";
        telemetryText.innerHTML = `
          Hover or point your cursor anywhere on the handwriting above to inspect instantaneous velocity, acceleration, and curvature at that exact point in real time.
        `;
      }
    }
  });

  writingCanvas.addEventListener("mouseleave", () => {
    hoveredPoint = null;
    drawWritingCanvas();
    pointHud.style.display = "none";
    telemetryText.innerHTML = `
      Hover or point your cursor anywhere on the handwriting above to inspect instantaneous velocity, acceleration, and curvature at that exact point in real time.
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

    // Dark canvas background
    ctx.fillStyle = "#111827";
    ctx.fillRect(0, 0, w, h);

    const padL = 50, padR = 25, padT = 30, padB = 40;
    const plotW = w - padL - padR;
    const plotH = h - padT - padB;

    // Draw Subtle Grid
    ctx.strokeStyle = "rgba(255, 255, 255, 0.08)";
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
    if (toggleWaveV.checked && wf.velocity) {
      drawCurve(wf.velocity, "#38bdf8", 2.2);
    }

    // 2. Acceleration a(t) - Purple
    if (toggleWaveA.checked && wf.acceleration) {
      drawCurve(wf.acceleration, "#a78bfa", 1.6);
    }

    // 3. Jerk j(t) - Rose
    if (toggleWaveJ.checked && wf.jerk) {
      drawCurve(wf.jerk, "#f43f5e", 1.2);
    }

    // 4. Curvature κ(t) - Amber
    if (toggleWaveK.checked && wf.curvature) {
      drawCurve(wf.curvature, "#fbbf24", 1.4);
    }

    // 5. NVI Markers (Peaks & Troughs)
    if (toggleWaveNvi.checked && toggleWaveV.checked && wf.velocity) {
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
    const curA = wf.acceleration ? wf.acceleration[idx] : 0;
    const curJ = wf.jerk ? wf.jerk[idx] : 0;
    const curK = wf.curvature ? wf.curvature[idx] : 0;

    scrubberInfo.innerHTML = `
      <strong>Time:</strong> ${curT}s &nbsp;|&nbsp;
      <span style="color:#38bdf8;"><strong>v(t):</strong> ${curV} px/s</span> &nbsp;|&nbsp;
      <span style="color:#a78bfa;"><strong>a(t):</strong> ${curA} px/s²</span> &nbsp;|&nbsp;
      <span style="color:#f43f5e;"><strong>j(t):</strong> ${curJ}</span> &nbsp;|&nbsp;
      <span style="color:#fbbf24;"><strong>κ(t):</strong> ${curK} rad/px</span>
    `;
  });
});
