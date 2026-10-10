# Task List — Phase 1 Interactive Review & Manual Adjustment Web App

## Active Sprint: Interactive Verification & Manual Adjustment Tool

### Objectives
1. Build a local web application allowing fast visual inspection and manual bounding-box adjustment / recropping for any sheet.
2. Provide an interactive canvas over the deskewed page with draggable/resizable bounding box handles and freehand rectangle drawing.
3. Live backend API to re-crop the grayscale image, update JSON schemas, extract words/baselines, re-render `overlay_debug.png`, and mark sheets as verified.
4. Seamless navigation across all 115 School A students with search, filters (grade, label, status), and keyboard shortcuts.

### Tasks
- [x] 1. Design and implement Python backend (`qa/app.py`) with `aiohttp`:
  - `GET /api/students`: Full student directory with review flags
  - `GET /api/student/{student_id}`: Page dimensions, ruling lines, sentence bounding boxes, crop paths
  - `POST /api/student/{student_id}/update_sentence`: Live crop update, JSON sync, and overlay regeneration
  - `POST /api/student/{student_id}/verify`: Persist verification status
  - Static file serving for processed images and frontend assets
- [x] 2. Build modern, responsive single-page web app (`qa/web/index.html`, `qa/web/style.css`, `qa/web/app.js`):
  - Glassmorphic dark UI with Google Fonts (Inter/Outfit)
  - Interactive Canvas with zoom, pan, bounding box dragging, handle resizing, and new box drawing
  - Sentence cards panel with lossless crop previews, script toggles (Hindi / English), and task assignment
  - Navigation bar with grade filters, status indicators, and keyboard navigation (`[` / `]` or arrow keys)
- [x] 3. Test backend endpoints with automated unit / API test (all passed)
- [x] 4. Launch web app server and test in browser using `browser_subagent` (live on `http://127.0.0.1:8090/`)
- [x] 5. Add interactive refinements based on user testing:
  - Instant task deletion via Delete/Backspace hotkeys and UI button without modal blockers
  - Automatic background auto-saving on handle release, bbox coordinate input, and student navigation
  - Handle resize anti-flicker fix (removed CSS hover scale and added 30px touch hitboxes)
  - Dedicated "+ Add Task" button flow creating distinct protocol tasks (`sentence_01`..`06` or custom) without overwriting selection
- [x] 6. Update `context.md` with interactive review tool architecture and auto-save / add-task mechanisms
- [x] 7. Git commit and push to `origin/baseline-v1`
