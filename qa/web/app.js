// Dysgraphia QA & Segment Editor — Interactive Application Logic

(function() {
    'use strict';

    // Application State
    const state = {
        students: [],
        filteredStudents: [],
        currentStudentId: null,
        currentStudentData: null,
        selectedTaskIndex: -1,
        selectedScript: 'devanagari',
        
        currentTool: 'select', // 'select' | 'draw'
        zoom: 1.0,
        pan: { x: 0, y: 0 },
        isPanning: false,
        panStart: { x: 0, y: 0 },
        
        dragState: null, // { type: 'move'|'resize', handle: '', startX, startY, initBbox: [x,y,w,h] }
        drawState: null, // { isDrawing: false, startX: 0, startY: 0, currentBox: null }
        
        showOverlayDebug: false,
        gradeFilter: 'all',
        statusFilter: 'all',

        hasUnsavedChanges: false,
        isAddingNewTask: false,
    };

    const standardTasks = [
        { id: 'sentence_01_copy_hindi', name: 'copy_hindi', script: 'devanagari' },
        { id: 'sentence_02_copy_english', name: 'copy_english', script: 'latin' },
        { id: 'sentence_03_dictated_hindi', name: 'dictated_hindi', script: 'devanagari' },
        { id: 'sentence_04_dictated_english', name: 'dictated_english', script: 'latin' },
        { id: 'sentence_05_own_hindi', name: 'own_hindi', script: 'devanagari' },
        { id: 'sentence_06_own_english', name: 'own_english', script: 'latin' },
    ];

    function getNextAvailableTaskInfo() {
        const sentences = state.currentStudentData?.sentences || [];
        const existingIds = new Set(sentences.map(s => s.task_id));
        const existingNames = new Set(sentences.map(s => s.task_name));

        // Find first unused standard task in sequence (tasks 1 through 6)
        for (const st of standardTasks) {
            if (!existingIds.has(st.id) && !existingNames.has(st.name)) {
                return { ...st };
            }
        }

        // If standard 1..6 already exist, assign next custom sequence number
        const nextNum = sentences.length + 1;
        const numStr = String(nextNum).padStart(2, '0');
        const lastTask = sentences.slice(-1)[0];
        const nextScript = lastTask ? (lastTask.script === 'devanagari' ? 'latin' : 'devanagari') : 'devanagari';
        return {
            id: `sentence_${numStr}_custom`,
            name: `custom_task_${numStr}`,
            script: nextScript,
        };
    }

    let autoSaveTimeout = null;

    // DOM Elements
    const el = {
        gradePills: document.getElementById('gradePills'),
        statusFilter: document.getElementById('statusFilter'),
        studentSelect: document.getElementById('studentSelect'),
        prevBtn: document.getElementById('prevBtn'),
        nextBtn: document.getElementById('nextBtn'),
        verifyBtn: document.getElementById('verifyBtn'),
        helpBtn: document.getElementById('helpBtn'),
        helpModal: document.getElementById('helpModal'),
        closeHelpBtn: document.getElementById('closeHelpBtn'),
        
        toolSelect: document.getElementById('toolSelect'),
        toolDraw: document.getElementById('toolDraw'),
        btnZoomIn: document.getElementById('btnZoomIn'),
        btnZoomOut: document.getElementById('btnZoomOut'),
        btnFitScreen: document.getElementById('btnFitScreen'),
        btnResetZoom: document.getElementById('btnResetZoom'),
        zoomLabel: document.getElementById('zoomLabel'),
        toggleOverlay: document.getElementById('toggleOverlay'),
        canvasStatus: document.getElementById('canvasStatus'),
        
        viewport: document.getElementById('viewport'),
        world: document.getElementById('world'),
        pageImg: document.getElementById('pageImg'),
        svgOverlay: document.getElementById('svgOverlay'),
        
        metaStudentId: document.getElementById('metaStudentId'),
        metaBadge: document.getElementById('metaBadge'),
        metaGrade: document.getElementById('metaGrade'),
        metaRoll: document.getElementById('metaRoll'),
        metaTasksCount: document.getElementById('metaTasksCount'),
        metaWordsCount: document.getElementById('metaWordsCount'),
        studentNotes: document.getElementById('studentNotes'),
        
        editorCard: document.getElementById('editorCard'),
        editorStatus: document.getElementById('editorStatus'),
        editTaskName: document.getElementById('editTaskName'),
        btnScriptDev: document.getElementById('btnScriptDev'),
        btnScriptLat: document.getElementById('btnScriptLat'),
        boxX: document.getElementById('boxX'),
        boxY: document.getElementById('boxY'),
        boxW: document.getElementById('boxW'),
        boxH: document.getElementById('boxH'),
        btnSaveRecrop: document.getElementById('btnSaveRecrop'),
        btnDeleteTask: document.getElementById('btnDeleteTask'),
        
        taskCountBadge: document.getElementById('taskCountBadge'),
        btnAddSentence: document.getElementById('btnAddSentence'),
        tasksList: document.getElementById('tasksList'),
        toast: document.getElementById('toast'),
    };

    // --- Init ---
    async function init() {
        bindEvents();
        await loadStudents();
    }

    // --- Event Listeners ---
    function bindEvents() {
        // Grade Pills
        el.gradePills.addEventListener('click', (e) => {
            const btn = e.target.closest('.pill');
            if (!btn) return;
            el.gradePills.querySelectorAll('.pill').forEach(p => p.classList.remove('active'));
            btn.classList.add('active');
            state.gradeFilter = btn.dataset.grade;
            applyFilters();
        });

        // Status Filter
        el.statusFilter.addEventListener('change', () => {
            state.statusFilter = el.statusFilter.value;
            applyFilters();
        });

        // Student Dropdown & Prev/Next (with auto-save)
        el.studentSelect.addEventListener('change', async () => {
            if (el.studentSelect.value) {
                await beforeNavigate();
                loadStudent(el.studentSelect.value);
            }
        });
        el.prevBtn.addEventListener('click', navigatePrev);
        el.nextBtn.addEventListener('click', navigateNext);

        // Verification & Notes
        el.verifyBtn.addEventListener('click', toggleVerification);
        el.studentNotes.addEventListener('blur', saveStudentNotes);

        // Shortcuts Modal
        el.helpBtn.addEventListener('click', () => el.helpModal.classList.remove('hidden'));
        el.closeHelpBtn.addEventListener('click', () => el.helpModal.classList.add('hidden'));
        el.helpModal.addEventListener('click', (e) => {
            if (e.target === el.helpModal) el.helpModal.classList.add('hidden');
        });

        // Tools
        el.toolSelect.addEventListener('click', () => setTool('select'));
        el.toolDraw.addEventListener('click', toggleDrawTool);

        // Zoom & Pan
        el.btnZoomIn.addEventListener('click', () => zoomAtCenter(1.25));
        el.btnZoomOut.addEventListener('click', () => zoomAtCenter(0.8));
        el.btnResetZoom.addEventListener('click', resetZoom);
        el.btnFitScreen.addEventListener('click', fitToScreen);
        el.toggleOverlay.addEventListener('change', toggleDebugOverlay);

        // Canvas Viewport Events (Mouse Pan & Zoom)
        el.viewport.addEventListener('wheel', onViewportWheel, { passive: false });
        el.viewport.addEventListener('mousedown', onViewportMouseDown);
        window.addEventListener('mousemove', onWindowMouseMove);
        window.addEventListener('mouseup', onWindowMouseUp);

        // Editor Form
        el.btnScriptDev.addEventListener('click', () => setEditorScript('devanagari'));
        el.btnScriptLat.addEventListener('click', () => setEditorScript('latin'));
        [el.boxX, el.boxY, el.boxW, el.boxH].forEach(input => {
            input.addEventListener('input', onBboxInputsChange);
        });
        el.editTaskName.addEventListener('change', onTaskNameChange);
        el.btnSaveRecrop.addEventListener('click', () => saveAndRecropActiveTask(false));
        el.btnDeleteTask.addEventListener('click', deleteActiveTask);
        el.btnAddSentence.addEventListener('click', startAddNewTask);

        // Global Keyboard Shortcuts
        window.addEventListener('keydown', onKeyDown);
    }

    // --- Student Data Loading ---
    async function loadStudents() {
        try {
            el.canvasStatus.textContent = 'Loading cohort...';
            const res = await fetch('/api/students');
            const data = await res.json();
            state.students = data.students || [];
            applyFilters();
            if (state.filteredStudents.length > 0) {
                loadStudent(state.filteredStudents[0].student_id);
            }
        } catch (err) {
            showToast('Failed to load students: ' + err.message, 'error');
            el.canvasStatus.textContent = 'Error loading cohort';
        }
    }

    function applyFilters() {
        let filtered = [...state.students];

        if (state.gradeFilter !== 'all') {
            const g = parseInt(state.gradeFilter, 10);
            filtered = filtered.filter(s => s.grade === g);
        }

        if (state.statusFilter === 'needs_review') {
            filtered = filtered.filter(s => !s.verified);
        } else if (state.statusFilter === 'verified') {
            filtered = filtered.filter(s => s.verified);
        } else if (state.statusFilter === 'positive') {
            filtered = filtered.filter(s => s.label === 1);
        }

        state.filteredStudents = filtered;
        populateStudentSelect();
    }

    function populateStudentSelect() {
        el.studentSelect.innerHTML = '';
        if (state.filteredStudents.length === 0) {
            el.studentSelect.innerHTML = '<option value="">No students matching filter</option>';
            return;
        }

        state.filteredStudents.forEach((s) => {
            const opt = document.createElement('option');
            opt.value = s.student_id;
            const posBadge = s.label === 1 ? ' [At-Risk]' : '';
            const verBadge = s.verified ? ' [✓]' : '';
            opt.textContent = `${s.student_id} (G${s.grade} R${s.roll_number})${posBadge}${verBadge}`;
            el.studentSelect.appendChild(opt);
        });

        if (state.currentStudentId) {
            el.studentSelect.value = state.currentStudentId;
        }
    }

    async function loadStudent(studentId) {
        state.currentStudentId = studentId;
        state.hasUnsavedChanges = false;
        el.studentSelect.value = studentId;
        el.canvasStatus.textContent = `Loading ${studentId}...`;

        try {
            const res = await fetch(`/api/student/${studentId}`);
            const data = await res.json();
            if (data.error) throw new Error(data.error);

            state.currentStudentData = data;
            state.selectedTaskIndex = -1;

            updateStudentMetaUI(data);
            setupPageCanvas(data);
            renderTaskCards();
            updateEditorUI();

            el.canvasStatus.textContent = `Viewing ${studentId} (${data.image_width}×${data.image_height})`;
        } catch (err) {
            showToast(`Error loading ${studentId}: ${err.message}`, 'error');
            el.canvasStatus.textContent = 'Error loading student';
        }
    }

    function updateStudentMetaUI(data) {
        const studentInfo = state.students.find(s => s.student_id === data.student_id) || {};
        el.metaStudentId.textContent = data.student_id;
        el.metaGrade.textContent = studentInfo.grade ?? '-';
        el.metaRoll.textContent = studentInfo.roll_number ?? '-';
        el.metaTasksCount.textContent = data.sentences.length;
        
        const totalWords = data.sentences.reduce((acc, s) => acc + (s.word_count || 0), 0);
        el.metaWordsCount.textContent = totalWords;
        el.studentNotes.value = data.notes || '';

        // Badge
        if (studentInfo.label === 1) {
            el.metaBadge.className = 'badge badge-pos';
            el.metaBadge.textContent = 'At-Risk (Positive)';
        } else {
            el.metaBadge.className = 'badge badge-neg';
            el.metaBadge.textContent = 'Typical (Negative)';
        }

        // Verify button state
        if (data.verified) {
            el.verifyBtn.classList.add('is-verified');
            el.verifyBtn.innerHTML = '<span class="icon">&#10004;</span> Verified';
        } else {
            el.verifyBtn.classList.remove('is-verified');
            el.verifyBtn.innerHTML = '<span class="icon">&#10003;</span> Mark Verified';
        }
    }

    function setupPageCanvas(data) {
        const imgSrc = state.showOverlayDebug ? data.overlay_url : data.page_url;
        el.pageImg.src = imgSrc;
        
        el.pageImg.onload = () => {
            const w = el.pageImg.naturalWidth;
            const h = el.pageImg.naturalHeight;
            el.svgOverlay.setAttribute('viewBox', `0 0 ${w} ${h}`);
            el.svgOverlay.style.width = `${w}px`;
            el.svgOverlay.style.height = `${h}px`;
            
            fitToScreen();
            renderSvgBoxes();
        };
    }

    // --- SVG Overlay Rendering ---
    function renderSvgBoxes() {
        el.svgOverlay.innerHTML = '';
        if (!state.currentStudentData) return;

        const sentences = state.currentStudentData.sentences || [];

        sentences.forEach((s, idx) => {
            const [x, y, w, h] = s.crop_bbox;
            const isSelected = (idx === state.selectedTaskIndex);
            const script = s.script || 'devanagari';

            const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
            g.dataset.index = idx;

            // Bounding rectangle
            const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
            rect.setAttribute('x', x);
            rect.setAttribute('y', y);
            rect.setAttribute('width', w);
            rect.setAttribute('height', h);
            rect.setAttribute('class', `bbox-rect ${script} ${isSelected ? 'active' : ''}`);
            rect.dataset.index = idx;
            g.appendChild(rect);

            // Text Label
            const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            text.setAttribute('x', x + 10);
            text.setAttribute('y', Math.max(22, y - 6));
            text.setAttribute('class', 'bbox-label');
            text.textContent = `${s.task_name} (${script}, ${s.word_count || 0}w)`;
            g.appendChild(text);

            // If selected, render 8 resize handles with generous hitbox
            if (isSelected) {
                // Keep handles a constant 14-16 screen pixels regardless of zoom level
                const handleSize = Math.max(12, 14 / state.zoom);
                const half = handleSize / 2;
                const hitboxSize = Math.max(26, 30 / state.zoom);
                const hitHalf = hitboxSize / 2;

                const handles = [
                    { id: 'nw', cx: x, cy: y, cursor: 'nwse-resize' },
                    { id: 'n',  cx: x + w/2, cy: y, cursor: 'ns-resize' },
                    { id: 'ne', cx: x + w, cy: y, cursor: 'nesw-resize' },
                    { id: 'e',  cx: x + w, cy: y + h/2, cursor: 'ew-resize' },
                    { id: 'se', cx: x + w, cy: y + h, cursor: 'nwse-resize' },
                    { id: 's',  cx: x + w/2, cy: y + h, cursor: 'ns-resize' },
                    { id: 'sw', cx: x, cy: y + h, cursor: 'nesw-resize' },
                    { id: 'w',  cx: x, cy: y + h/2, cursor: 'ew-resize' },
                ];

                handles.forEach(hd => {
                    // Transparent generous hitbox
                    const hitRect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
                    hitRect.setAttribute('x', hd.cx - hitHalf);
                    hitRect.setAttribute('y', hd.cy - hitHalf);
                    hitRect.setAttribute('width', hitboxSize);
                    hitRect.setAttribute('height', hitboxSize);
                    hitRect.setAttribute('class', 'resize-handle-hitbox');
                    hitRect.style.cursor = hd.cursor;
                    hitRect.dataset.handle = hd.id;
                    hitRect.dataset.index = idx;
                    g.appendChild(hitRect);

                    // Visible handle square
                    const hRect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
                    hRect.setAttribute('x', hd.cx - half);
                    hRect.setAttribute('y', hd.cy - half);
                    hRect.setAttribute('width', handleSize);
                    hRect.setAttribute('height', handleSize);
                    hRect.setAttribute('class', 'resize-handle');
                    hRect.style.cursor = hd.cursor;
                    hRect.dataset.handle = hd.id;
                    hRect.dataset.index = idx;
                    g.appendChild(hRect);
                });
            }

            el.svgOverlay.appendChild(g);
        });
    }

    // --- Task Cards List in Sidebar ---
    function renderTaskCards() {
        el.tasksList.innerHTML = '';
        if (!state.currentStudentData) return;

        const sentences = state.currentStudentData.sentences || [];
        el.taskCountBadge.textContent = sentences.length;
        if (el.metaTasksCount) el.metaTasksCount.textContent = sentences.length;

        sentences.forEach((s, idx) => {
            const card = document.createElement('div');
            const isSelected = (idx === state.selectedTaskIndex);
            card.className = `task-item-card ${isSelected ? 'selected' : ''}`;
            card.dataset.index = idx;

            const scriptColor = s.script === 'devanagari' ? '#10b981' : '#f59e0b';

            card.innerHTML = `
                <div class="task-card-header">
                    <span class="task-name-label">${s.task_name}</span>
                    <span class="badge" style="background: rgba(255,255,255,0.08); color: ${scriptColor}; border: 1px solid ${scriptColor};">
                        ${s.script} • ${s.word_count || 0} words
                    </span>
                </div>
                <div class="task-crop-container">
                    ${s.crop_url ? `<img src="${s.crop_url}" class="task-crop-img" alt="${s.task_name}"/>` : `<div style="padding: 12px; color: var(--text-muted, #94a3b8); font-size: 12px; text-align: center;">Saving & cropping...</div>`}
                </div>
            `;

            card.addEventListener('click', () => {
                selectTask(idx);
            });

            el.tasksList.appendChild(card);
        });
    }

    function selectTask(index) {
        state.isAddingNewTask = false;
        state.selectedTaskIndex = index;
        renderSvgBoxes();
        renderTaskCards();
        updateEditorUI();
    }

    function updateEditorUI() {
        if (state.selectedTaskIndex < 0 || !state.currentStudentData || !state.currentStudentData.sentences[state.selectedTaskIndex]) {
            el.editorStatus.textContent = state.isAddingNewTask ? 'Drawing new task box...' : 'Select a box on canvas';
            el.btnSaveRecrop.disabled = true;
            el.btnDeleteTask.disabled = true;
            el.boxX.value = '';
            el.boxY.value = '';
            el.boxW.value = '';
            el.boxH.value = '';
            return;
        }

        const task = state.currentStudentData.sentences[state.selectedTaskIndex];
        el.editorStatus.textContent = `Editing: ${task.task_name}`;
        el.btnSaveRecrop.disabled = false;
        el.btnDeleteTask.disabled = false;

        // Populate Form
        setDropdownByValue(el.editTaskName, task.task_name);
        setEditorScript(task.script || 'devanagari', false);

        const [x, y, w, h] = task.crop_bbox;
        el.boxX.value = Math.round(x);
        el.boxY.value = Math.round(y);
        el.boxW.value = Math.round(w);
        el.boxH.value = Math.round(h);
    }

    function setDropdownByValue(select, val) {
        let found = false;
        for (let opt of select.options) {
            if (opt.value === val || opt.text === val) {
                select.value = opt.value;
                found = true;
                break;
            }
        }
        if (!found) {
            select.value = 'custom';
        }
    }

    function setEditorScript(script, triggerSave = true) {
        state.selectedScript = script;
        if (script === 'devanagari') {
            el.btnScriptDev.classList.add('active');
            el.btnScriptLat.classList.remove('active');
        } else {
            el.btnScriptDev.classList.remove('active');
            el.btnScriptLat.classList.add('active');
        }
        if (state.selectedTaskIndex >= 0 && state.currentStudentData) {
            state.currentStudentData.sentences[state.selectedTaskIndex].script = script;
            renderSvgBoxes();
            renderTaskCards();
            if (triggerSave) {
                triggerAutoSave(150);
            }
        }
    }

    function onBboxInputsChange() {
        if (state.selectedTaskIndex < 0 || !state.currentStudentData) return;
        const x = parseInt(el.boxX.value, 10) || 0;
        const y = parseInt(el.boxY.value, 10) || 0;
        const w = parseInt(el.boxW.value, 10) || 10;
        const h = parseInt(el.boxH.value, 10) || 10;

        state.currentStudentData.sentences[state.selectedTaskIndex].crop_bbox = [x, y, w, h];
        renderSvgBoxes();
        triggerAutoSave(400);
    }

    function onTaskNameChange() {
        if (state.selectedTaskIndex < 0 || !state.currentStudentData) return;
        const name = el.editTaskName.value;
        state.currentStudentData.sentences[state.selectedTaskIndex].task_name = name;
        renderSvgBoxes();
        renderTaskCards();
        triggerAutoSave(150);
    }

    // --- Auto-Save Scheduler ---
    function triggerAutoSave(delayMs = 250) {
        state.hasUnsavedChanges = true;
        el.canvasStatus.textContent = 'Auto-saving...';
        if (autoSaveTimeout) clearTimeout(autoSaveTimeout);
        autoSaveTimeout = setTimeout(async () => {
            await saveAndRecropActiveTask(true);
        }, delayMs);
    }

    // --- Save & Re-Crop API Call ---
    async function saveAndRecropActiveTask(silent = false) {
        if (state.selectedTaskIndex < 0 || !state.currentStudentData) return;
        const task = state.currentStudentData.sentences[state.selectedTaskIndex];
        const studentId = state.currentStudentData.student_id;
        if (!task) return;

        const payload = {
            task_id: task.task_id,
            task_name: el.editTaskName.value || task.task_name,
            script: state.selectedScript,
            bbox: [
                parseInt(el.boxX.value, 10) || task.crop_bbox[0],
                parseInt(el.boxY.value, 10) || task.crop_bbox[1],
                parseInt(el.boxW.value, 10) || task.crop_bbox[2],
                parseInt(el.boxH.value, 10) || task.crop_bbox[3],
            ],
        };

        el.canvasStatus.textContent = `Auto-saving ${task.task_name}...`;

        try {
            const res = await fetch(`/api/student/${studentId}/update_sentence`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });
            const data = await res.json();
            if (data.error) throw new Error(data.error);

            // Update local state
            task.crop_bbox = data.crop_bbox;
            task.word_count = data.word_count;
            task.crop_url = data.crop_url;
            task.is_manually_adjusted = true;
            state.currentStudentData.overlay_url = data.overlay_url;
            state.hasUnsavedChanges = false;

            if (state.showOverlayDebug) {
                el.pageImg.src = data.overlay_url;
            }

            renderSvgBoxes();
            renderTaskCards();
            if (!silent) {
                showToast(`✓ ${task.task_name} saved (${data.word_count} words)`, 'success');
            }
            el.canvasStatus.textContent = 'All changes auto-saved ✓';
        } catch (err) {
            showToast(`Save failed: ${err.message}`, 'error');
            el.canvasStatus.textContent = 'Auto-save error';
        }
    }

    // --- Immediate Delete (No popup blocking) ---
    async function deleteActiveTask() {
        if (state.selectedTaskIndex < 0 || !state.currentStudentData) return;
        const task = state.currentStudentData.sentences[state.selectedTaskIndex];
        const studentId = state.currentStudentData.student_id;
        if (!task) return;

        const taskName = task.task_name;
        const taskId = task.task_id;

        // Cancel pending auto-save
        if (autoSaveTimeout) clearTimeout(autoSaveTimeout);
        state.hasUnsavedChanges = false;

        // Immediate removal from local state for instant UI response
        state.currentStudentData.sentences.splice(state.selectedTaskIndex, 1);
        state.selectedTaskIndex = -1;
        renderSvgBoxes();
        renderTaskCards();
        updateEditorUI();
        el.canvasStatus.textContent = `Deleted ${taskName}`;

        try {
            const res = await fetch(`/api/student/${studentId}/sentence/${taskId}`, {
                method: 'DELETE',
            });
            const data = await res.json();
            if (data.error) throw new Error(data.error);

            if (data.overlay_url) {
                state.currentStudentData.overlay_url = data.overlay_url;
                if (state.showOverlayDebug) {
                    el.pageImg.src = data.overlay_url;
                }
            }
            showToast(`✓ Deleted ${taskName}`, 'success');
        } catch (err) {
            showToast(`Delete failed: ${err.message}`, 'error');
        }
    }

    async function toggleVerification() {
        if (!state.currentStudentData) return;
        const studentId = state.currentStudentData.student_id;
        const newStatus = !state.currentStudentData.verified;

        try {
            const res = await fetch(`/api/student/${studentId}/verify`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ verified: newStatus, notes: el.studentNotes.value }),
            });
            const data = await res.json();
            if (data.error) throw new Error(data.error);

            state.currentStudentData.verified = newStatus;
            const item = state.students.find(s => s.student_id === studentId);
            if (item) item.verified = newStatus;

            updateStudentMetaUI(state.currentStudentData);
            populateStudentSelect();
            showToast(newStatus ? '✓ Sheet marked as VERIFIED' : 'Verification removed', 'success');
        } catch (err) {
            showToast(`Verification error: ${err.message}`, 'error');
        }
    }

    async function saveStudentNotes() {
        if (!state.currentStudentData) return;
        const studentId = state.currentStudentData.student_id;
        try {
            await fetch(`/api/student/${studentId}/verify`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    verified: state.currentStudentData.verified,
                    notes: el.studentNotes.value,
                }),
            });
        } catch (_) {}
    }

    // --- Interactive Mouse Handlers on Canvas ---
    function onViewportMouseDown(e) {
        // Space or Middle mouse -> Pan
        if (e.button === 1 || e.code === 'Space' || e.target === el.viewport || e.target === el.pageImg) {
            if (state.currentTool !== 'draw') {
                startPanning(e);
                return;
            }
        }

        const point = clientToPageCoord(e.clientX, e.clientY);

        // Tool: Draw New Box
        if (state.currentTool === 'draw') {
            state.drawState = {
                isDrawing: true,
                startX: point.x,
                startY: point.y,
            };
            return;
        }

        // Handle Resize Handle click (visible handle or generous hitbox)
        if (e.target.classList.contains('resize-handle') || e.target.classList.contains('resize-handle-hitbox')) {
            const handle = e.target.dataset.handle;
            const idx = parseInt(e.target.dataset.index, 10);
            selectTask(idx);
            const bbox = [...state.currentStudentData.sentences[idx].crop_bbox];
            state.dragState = {
                type: 'resize',
                handle: handle,
                startX: point.x,
                startY: point.y,
                initBbox: bbox,
            };
            e.stopPropagation();
            return;
        }

        // Handle Box click / drag
        if (e.target.classList.contains('bbox-rect')) {
            const idx = parseInt(e.target.dataset.index, 10);
            selectTask(idx);
            const bbox = [...state.currentStudentData.sentences[idx].crop_bbox];
            state.dragState = {
                type: 'move',
                startX: point.x,
                startY: point.y,
                initBbox: bbox,
            };
            e.stopPropagation();
            return;
        }

        // Click on background -> start panning
        startPanning(e);
    }

    function onWindowMouseMove(e) {
        if (state.isPanning) {
            const dx = e.clientX - state.panStart.x;
            const dy = e.clientY - state.panStart.y;
            state.pan.x += dx;
            state.pan.y += dy;
            state.panStart = { x: e.clientX, y: e.clientY };
            updateTransform();
            return;
        }

        const point = clientToPageCoord(e.clientX, e.clientY);

        // Handle Box Move / Resize
        if (state.dragState && state.selectedTaskIndex >= 0) {
            const task = state.currentStudentData.sentences[state.selectedTaskIndex];
            const dx = point.x - state.dragState.startX;
            const dy = point.y - state.dragState.startY;
            let [x, y, w, h] = state.dragState.initBbox;

            if (state.dragState.type === 'move') {
                x = Math.max(0, x + dx);
                y = Math.max(0, y + dy);
            } else if (state.dragState.type === 'resize') {
                const hd = state.dragState.handle;
                if (hd.includes('w')) {
                    const newW = w - dx;
                    if (newW > 20) { x += dx; w = newW; }
                }
                if (hd.includes('e')) {
                    w = Math.max(20, w + dx);
                }
                if (hd.includes('n')) {
                    const newH = h - dy;
                    if (newH > 20) { y += dy; h = newH; }
                }
                if (hd.includes('s')) {
                    h = Math.max(20, h + dy);
                }
            }

            task.crop_bbox = [Math.round(x), Math.round(y), Math.round(w), Math.round(h)];
            el.boxX.value = Math.round(x);
            el.boxY.value = Math.round(y);
            el.boxW.value = Math.round(w);
            el.boxH.value = Math.round(h);
            renderSvgBoxes();
            return;
        }

        // Handle Draw Box Preview
        if (state.drawState && state.drawState.isDrawing) {
            const x1 = Math.min(state.drawState.startX, point.x);
            const y1 = Math.min(state.drawState.startY, point.y);
            const w = Math.abs(point.x - state.drawState.startX);
            const h = Math.abs(point.y - state.drawState.startY);

            let preview = document.getElementById('drawPreviewBox');
            if (!preview) {
                preview = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
                preview.id = 'drawPreviewBox';
                preview.setAttribute('class', 'draw-preview');
                el.svgOverlay.appendChild(preview);
            }
            preview.setAttribute('x', x1);
            preview.setAttribute('y', y1);
            preview.setAttribute('width', w);
            preview.setAttribute('height', h);
        }
    }

    function onWindowMouseUp(e) {
        if (state.isPanning) {
            state.isPanning = false;
            el.viewport.classList.remove('dragging');
        }

        if (state.dragState) {
            state.dragState = null;
            // Auto-save immediately when mouse releases box/handle
            triggerAutoSave(150);
        }

        if (state.drawState && state.drawState.isDrawing) {
            const point = clientToPageCoord(e.clientX, e.clientY);
            const x = Math.round(Math.min(state.drawState.startX, point.x));
            const y = Math.round(Math.min(state.drawState.startY, point.y));
            const w = Math.round(Math.abs(point.x - state.drawState.startX));
            const h = Math.round(Math.abs(point.y - state.drawState.startY));

            // Remove preview
            const preview = document.getElementById('drawPreviewBox');
            if (preview) preview.remove();
            state.drawState = null;

            if (w > 30 && h > 20) {
                finalizeDrawnBox(x, y, w, h);
            } else {
                state.isAddingNewTask = false;
            }
            setTool('select');
        }
    }

    function finalizeDrawnBox(x, y, w, h) {
        if (!state.currentStudentData) return;

        // If adding a new task OR no task is currently selected
        if (state.isAddingNewTask || state.selectedTaskIndex < 0) {
            state.isAddingNewTask = false;
            const taskInfo = getNextAvailableTaskInfo();
            const newTask = {
                task_id: taskInfo.id,
                task_name: taskInfo.name,
                script: taskInfo.script,
                crop_bbox: [x, y, w, h],
                word_count: 0,
                crop_url: '',
                is_manually_adjusted: true,
            };
            state.currentStudentData.sentences.push(newTask);
            const newIndex = state.currentStudentData.sentences.length - 1;
            selectTask(newIndex);
            showToast(`Added new task: ${taskInfo.id}`, 'success');
            triggerAutoSave(100);
            return;
        }

        // Otherwise (only if explicitly redrawing an existing selected task)
        const task = state.currentStudentData.sentences[state.selectedTaskIndex];
        task.crop_bbox = [x, y, w, h];
        task.is_manually_adjusted = true;
        el.boxX.value = x;
        el.boxY.value = y;
        el.boxW.value = w;
        el.boxH.value = h;
        renderSvgBoxes();
        triggerAutoSave(150);
    }

    function startPanning(e) {
        state.isPanning = true;
        state.panStart = { x: e.clientX, y: e.clientY };
        el.viewport.classList.add('dragging');
    }

    function onViewportWheel(e) {
        e.preventDefault();
        const factor = e.deltaY < 0 ? 1.15 : 0.85;
        const rect = el.viewport.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;

        const newZoom = Math.min(4.0, Math.max(0.15, state.zoom * factor));
        state.pan.x = mouseX - (mouseX - state.pan.x) * (newZoom / state.zoom);
        state.pan.y = mouseY - (mouseY - state.pan.y) * (newZoom / state.zoom);
        state.zoom = newZoom;

        updateTransform();
    }

    function zoomAtCenter(factor) {
        const rect = el.viewport.getBoundingClientRect();
        const cx = rect.width / 2;
        const cy = rect.height / 2;
        const newZoom = Math.min(4.0, Math.max(0.15, state.zoom * factor));
        state.pan.x = cx - (cx - state.pan.x) * (newZoom / state.zoom);
        state.pan.y = cy - (cy - state.pan.y) * (newZoom / state.zoom);
        state.zoom = newZoom;
        updateTransform();
    }

    function resetZoom() {
        state.zoom = 1.0;
        state.pan = { x: 40, y: 40 };
        updateTransform();
    }

    function fitToScreen() {
        if (!el.pageImg.naturalWidth) return;
        const vRect = el.viewport.getBoundingClientRect();
        const pad = 40;
        const scaleW = (vRect.width - pad) / el.pageImg.naturalWidth;
        const scaleH = (vRect.height - pad) / el.pageImg.naturalHeight;
        state.zoom = Math.min(scaleW, scaleH);
        state.pan.x = (vRect.width - el.pageImg.naturalWidth * state.zoom) / 2;
        state.pan.y = (vRect.height - el.pageImg.naturalHeight * state.zoom) / 2;
        updateTransform();
    }

    function updateTransform() {
        el.world.style.transform = `translate(${state.pan.x}px, ${state.pan.y}px) scale(${state.zoom})`;
        el.zoomLabel.textContent = `${Math.round(state.zoom * 100)}%`;
        if (state.selectedTaskIndex >= 0) {
            renderSvgBoxes();
        }
    }

    function clientToPageCoord(clientX, clientY) {
        const vRect = el.viewport.getBoundingClientRect();
        const x = (clientX - vRect.left - state.pan.x) / state.zoom;
        const y = (clientY - vRect.top - state.pan.y) / state.zoom;
        return { x: Math.round(x), y: Math.round(y) };
    }

    function toggleDebugOverlay() {
        state.showOverlayDebug = el.toggleOverlay.checked;
        if (state.currentStudentData) {
            el.pageImg.src = state.showOverlayDebug ? state.currentStudentData.overlay_url : state.currentStudentData.page_url;
        }
    }

    function setTool(tool) {
        state.currentTool = tool;
        if (tool === 'select') {
            state.isAddingNewTask = false;
            el.toolSelect.classList.add('active');
            el.toolDraw.classList.remove('active');
            el.viewport.classList.remove('drawing');
            el.canvasStatus.textContent = 'Select & Drag mode';
        } else {
            el.toolSelect.classList.remove('active');
            el.toolDraw.classList.add('active');
            el.viewport.classList.add('drawing');
            el.canvasStatus.textContent = 'Draw mode: Click and drag on image to define task box';
        }
    }

    async function startAddNewTask() {
        if (!state.currentStudentData) return;
        if (state.hasUnsavedChanges && state.selectedTaskIndex >= 0) {
            if (autoSaveTimeout) clearTimeout(autoSaveTimeout);
            await saveAndRecropActiveTask(true);
        }
        state.isAddingNewTask = true;
        state.selectedTaskIndex = -1;
        renderSvgBoxes();
        renderTaskCards();
        updateEditorUI();
        setTool('draw');
        el.canvasStatus.textContent = 'Draw mode: Click & drag on sheet to define new task box';
        showToast('Click and drag on the sheet to define the new task box', 'info');
    }

    async function toggleDrawTool() {
        if (state.currentTool === 'draw') {
            state.isAddingNewTask = false;
            setTool('select');
        } else {
            await startAddNewTask();
        }
    }

    // --- Navigation (with auto-save before moving) ---
    async function beforeNavigate() {
        if (state.hasUnsavedChanges && state.selectedTaskIndex >= 0) {
            if (autoSaveTimeout) clearTimeout(autoSaveTimeout);
            await saveAndRecropActiveTask(true);
        }
    }

    async function navigatePrev() {
        await beforeNavigate();
        const curIdx = state.filteredStudents.findIndex(s => s.student_id === state.currentStudentId);
        if (curIdx > 0) {
            loadStudent(state.filteredStudents[curIdx - 1].student_id);
        } else if (state.filteredStudents.length > 0) {
            loadStudent(state.filteredStudents[state.filteredStudents.length - 1].student_id);
        }
    }

    async function navigateNext() {
        await beforeNavigate();
        const curIdx = state.filteredStudents.findIndex(s => s.student_id === state.currentStudentId);
        if (curIdx >= 0 && curIdx < state.filteredStudents.length - 1) {
            loadStudent(state.filteredStudents[curIdx + 1].student_id);
        } else if (state.filteredStudents.length > 0) {
            loadStudent(state.filteredStudents[0].student_id);
        }
    }

    function onKeyDown(e) {
        if (['INPUT', 'SELECT', 'TEXTAREA'].includes(e.target.tagName)) return;

        // Delete key deletes selected box
        if (e.key === 'Delete' || e.key === 'Backspace') {
            if (state.selectedTaskIndex >= 0) {
                e.preventDefault();
                deleteActiveTask();
                return;
            }
        }

        if (e.key === '[' || e.key === 'ArrowLeft') {
            e.preventDefault();
            navigatePrev();
        } else if (e.key === ']' || e.key === 'ArrowRight') {
            e.preventDefault();
            navigateNext();
        } else if (e.key === 'v' || e.key === 'V') {
            e.preventDefault();
            toggleVerification();
        } else if (e.key === 'd' || e.key === 'D') {
            e.preventDefault();
            toggleDrawTool();
        } else if (e.key === 'f' || e.key === 'F') {
            e.preventDefault();
            fitToScreen();
        } else if (e.key === '0') {
            e.preventDefault();
            resetZoom();
        } else if (e.key === 'Escape') {
            e.preventDefault();
            state.isAddingNewTask = false;
            state.selectedTaskIndex = -1;
            setTool('select');
            renderSvgBoxes();
            renderTaskCards();
            updateEditorUI();
        } else if ((e.ctrlKey || e.metaKey) && (e.key === 's' || e.key === 'S')) {
            e.preventDefault();
            saveAndRecropActiveTask(false);
        }
    }

    // --- Toast Notifications ---
    let toastTimeout = null;
    function showToast(message, type = 'info') {
        el.toast.textContent = message;
        el.toast.className = `toast toast-${type}`;
        el.toast.classList.remove('hidden');

        if (toastTimeout) clearTimeout(toastTimeout);
        toastTimeout = setTimeout(() => {
            el.toast.classList.add('hidden');
        }, 3200);
    }

    // Launch App
    document.addEventListener('DOMContentLoaded', init);
})();
