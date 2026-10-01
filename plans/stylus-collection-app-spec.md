# Stylus Data Collection App — Build Spec (v3)

For the school visit: **~150 students, Grades 3–7, ~4.5 h in school, 1–2 stylus devices, Hindi + English.**
Changes from v2: **Marathi → Hindi**, **Grades 3–7 only (5 classes of ~30)**, throughput recalculated, full app spec added.
v2's schedule and sentences are outdated; this file replaces them for the stylus side.

> **Assumption to confirm:** the stylus device is an **S-Pen phone/tablet running the page in Chrome**, and the **laptop is the server + dashboard** on its own Wi-Fi hotspot. The same page also works on a pen-enabled laptop (it uses standard Pointer Events).

---

## 1. Architecture (keep it boring)

```
[ S-Pen device A ]──┐                      ┌─ data/sessions/*.json  (one file per child)
[ S-Pen device B ]──┼─ Wi-Fi hotspot ─ [ Laptop: Python server ] ─ data/index.csv
[ (optional) ]──────┘     (no internet)     └─ /dash  (live dashboard in laptop browser)
```

- **Front end:** one static `index.html` + vanilla JS. No framework, no build step, no CDN (everything inline or served locally, including the Hindi font).
- **Back end:** Python 3 + FastAPI (or Flask) + uvicorn. Flat JSON files plus one CSV index. No database.
- **Offline-first:** every device keeps its own copy (IndexedDB) until the server acknowledges it. If Wi-Fi drops, collection continues.
- **Laptop hotspot:** Windows "Mobile hotspot" or a cheap travel router. Devices open `http://<laptop-ip>:8000`. Plain http is fine for pen capture.
- **Secure-context caveat:** Wake Lock needs HTTPS. Instead set the device to stay awake (§3). Fullscreen and orientation lock work over http.

Repo layout:

```
collector/
  server.py            # API + static files + dashboard
  static/index.html    # the device app
  static/dash.html
  static/fonts/        # NotoSansDevanagari + Latin font, bundled locally
  config.json          # tasks, tiers, stimuli, ruling, caps
  tools/make_roster.py # IDs, seeded random halves, queue order
  tools/render_png.py  # offline: JSON → PNG per task
  tools/qc_report.py   # offline: batch quality report
  data/                # sessions/, index.csv, backup/ (mirrored to a USB drive)
```

---

## 2. Pre-flight spike test (do this first, 30–45 min)

Build a 50-line page that logs pen events, then check on **the exact device(s)** you will use. Do not build the rest until this passes.

| Check | Pass condition |
|---|---|
| `pointerType === "pen"` for S-Pen, `"touch"` for finger/palm | Both distinguishable |
| Effective sample rate (samples ÷ pen-down seconds, using coalesced events) | ≥ 100 Hz; log the real value |
| Pressure varies while writing (not constant) | std dev > 0.01 |
| `tiltX`, `tiltY` present | Non-zero values appear |
| **Hover** events (pen near screen, not touching) fire in Chrome | If not, try Samsung Internet; else fall back to **pen-up gap time** as in-air time |
| Palm resting on screen while pen writes | No stray strokes in pen log |
| Timestamps monotonic | No reversals |
| Latency feels natural | Ink appears under the nib with no visible lag |
| Hindi conjuncts render correctly (क्ष, त्र, ज्ञ, श्र, क्रि, र्ष) | All shaped properly |
| Kill the tab mid-task, reopen | Resume from last completed task |

Also measure **physical screen size** with a ruler (for mm calibration, §6).

---

## 3. Device setup checklist (Samsung/Android)

- Screen timeout max; **Developer options → Stay awake** (while charging); keep on charger/power bank.
- Auto-rotate off, **landscape locked**; brightness fixed ~70%; Do Not Disturb on; notifications off.
- **S Pen settings:** turn off Air actions/Air command, "screen off memo", and any handwriting-to-text conversion.
- **Pin the app** (Android screen pinning) and use Chrome fullscreen so children cannot exit or trigger edge gestures.
- Disable Chrome translate/autofill/passwords; clear cache before the day.
- Keep 8 mm blank margins at screen edges (edge-gesture area).
- Stand or non-slip mat; child-height desk/chair; wipe between children.
- A tablet (11"+ with S Pen) is better than a phone if you can borrow one: more natural writing size.

---

## 4. Tasks, timing and tiers

Timer for every task starts at the **child's first pen-down**, not when the screen appears. This keeps speed measures clean and records reaction time separately.

| ID | Task | Cap | Ends when |
|---|---|---|---|
| **S1** | Graphomotor strip (loops, zigzag, spiral) | 35 s | Done ✓, or cap |
| **S2** | **Hindi copy**, comfortable | 75 s | Done ✓, or cap |
| **S3** | **English copy**, comfortable | 45 s | Done ✓, or cap |

| Tier | Tasks | Avg slot* | Use when |
|---|---|---|---|
| 2 | S1 + S2 + S3 | ≈ 2.4 min | Default (2 devices) |
| 1 | S1 + S2 | ≈ 1.6 min | Behind schedule |
| 0 | S2 only | ≈ 1.1 min | 1 device, or far behind |

\*Estimates including 15–20 s changeover. The app logs real slot times; replace these with measured averages after the pilot.

**Stylus coverage** (window 220 min × 85% efficiency ÷ slot):

| Devices | Tier 2 | Tier 1 | Tier 0 |
|---|---|---|---|
| 2 | ~156 | ~234 | ~340 |
| 1 | ~78 | ~117 | ~170 |

Tier is **set centrally from the laptop dashboard** and picked up by devices at the start of the next child.

Not on stylus (paper only): fast copy and free writing. Add a `tasks` entry in `config.json` if you later want them; each adds ≈ 0.7–1.2 min to the slot.

---

## 5. Stimuli (Grades 3–7)

> **Drafts.** Have a Hindi and an English teacher check every sentence, or swap in a line from each grade's own textbook. Store text as **Unicode NFC**, end Hindi sentences with the **Devanagari danda ।** (U+0964). Within a grade, everyone copies the same sentence.

### 5.1 Copy sentences (S2, S3 on stylus; same text on paper)

| Grade | Hindi (S2) | English (S3) |
|---|---|---|
| 3 | हमारे स्कूल में एक सुंदर बगीचा है। | My mother cooks food in the kitchen. |
| 4 | रविवार को हम सब क्रिकेट खेलते हैं। | On Sundays we play cricket with our friends. |
| 5 | हमारे शिक्षक हर दिन नई कहानी सुनाते हैं। | Our teacher tells us a new story every day. |
| 6 | वर्षा ऋतु में नदियाँ पानी से भर जाती हैं। | Rain falls from the clouds and fills the rivers and lakes. |
| 7 | भारत एक विशाल देश है, जहाँ अनेक भाषाएँ बोली जाती हैं। | The quick brown fox jumps over the lazy dog. |

Word counts: Hindi 6 / 7 / 8 / 9 / 11; English 7 / 8 / 9 / 11 / 9.
Each Hindi sentence has the headline (*shirorekha*), vowel signs and at least one conjunct or special sign (स्कू, क्रि, क्ष, र्ष/ऋ, chandrabindu). The grade 7 English sentence is a pangram (all 26 letters).

### 5.2 Graphomotor strip (S1, identical for all grades)

One canvas, three zones (analyse by y/x region later):
- **A. Loops:** a row of continuous loops (ℓℓℓ) between two guide lines; the first 2 loops printed in light gray as a starter. Child continues to the end of the line.
- **B. Zigzag:** same, with an up-down zigzag.
- **C. Spiral:** a start dot at the centre of a 30 mm square; child draws a spiral outward, as far as the square.

This task is the **cross-grade anchor**, because the sentences differ by grade.

### 5.3 Free-writing prompts (paper only)

| Grade | Hindi prompt | English prompt |
|---|---|---|
| 3 | अपने पसंदीदा खेल के बारे में एक वाक्य लिखो। | Write one sentence about your favourite food. |
| 4 | अपने पसंदीदा खाने के बारे में एक–दो वाक्य लिखो। | Write one or two sentences about your favourite game. |
| 5 | अपने पसंदीदा त्योहार के बारे में दो वाक्य लिखो। | Write two sentences about your favourite festival. |
| 6 | अपनी पसंदीदा जगह के बारे में दो–तीन वाक्य लिखो। | Write two or three sentences about a place you like. |
| 7 | अपने सबसे अच्छे दोस्त के बारे में तीन वाक्य लिखो। | Write three sentences about your best friend. |

Tell children spelling does not matter; cross out instead of erasing. Spelling is not scored.

### 5.4 Spoken instructions (operator reads; the same text is on screen in small type)

| Task | Hindi | English |
|---|---|---|
| S1 | लाइन के अंत तक पैटर्न बनाओ। फिर बिंदु से शुरू करके गोल-गोल घुमाते हुए बाहर की ओर बनाओ। | Continue the patterns to the end of the line, then draw a spiral outward from the dot. |
| S2 | यह वाक्य सुंदर और साफ़ लिखो, जैसे कक्षा में लिखते हो। | Copy this sentence neatly, like you do in class. |
| S3 | अब इस English वाक्य को उसी तरह लिखो। | Now copy this sentence the same way. |
| All | जब हो जाए, ऊपर ✓ दबाओ। | When you finish, tap ✓ at the top. |

Opening line for every child: "हम अपना ऐप जाँच रहे हैं, तुम्हारी परीक्षा नहीं है।" ("We're testing our app, not you.") No comments on handwriting quality.

---

## 6. Screen layout and drawing rules

Landscape, roughly 16 × 7 cm on a phone (**measure yours**). Layout in **millimetres**, converted using a per-device calibration.

| Zone | Height | Content |
|---|---|---|
| Top bar | 8 mm | 3 progress dots; **Done ✓** button at top-right (pen-tap only, 14 × 8 mm) |
| Stimulus band | 12 mm | Copy sentence on light gray; omit on S1 |
| Writing area | rest (~45 mm) | Ruled lines, **8 mm spacing** (replace with measured notebook spacing); start-point dot at the left of line 1 |

- 8 mm blank margin on left and right.
- Hindi stimulus glyph height ≈ 6 mm; English x-height ≈ 4 mm. Use a plain, textbook-like font (Noto Sans Devanagari + a simple Latin sans), **bundled locally**.
- Copy sentences should fit **1–2 lines** at that size; grade 7 Hindi may need 2.
- **No eraser, no undo, no clear button for the child.** Crossing out is data. The operator can redo a whole task.
- **Ink rendering is cosmetic:** constant ~1.2 mm line, optional slight pressure width. Raw samples are the data.
- Canvas: `getContext('2d', {desynchronized: true, alpha: false})`, size scaled by `devicePixelRatio`, no heavy work in the pointer handler.
- Guide lines: light gray, 0.3 mm. Same ruling and spacing as the paper sheet for that grade.
- **Calibration screen** (once per device): draw a 100 mm bar, operator measures it with a ruler and enters the real length, giving `mm_per_css_px_x/y`. Store in the device profile and in every session.

---

## 7. Capture specification

Use **Pointer Events**, accept `pointerType === "pen"` only. Finger or palm touches are ignored for drawing; if a finger touches the writing area, show a small "Please use the pen / कृपया पेन से लिखो" banner.

CSS: `touch-action: none; user-select: none; -webkit-touch-callout: none; overscroll-behavior: none;` and `oncontextmenu = e => e.preventDefault()`.

**Per sample, store:** `t` (ms since task's first pen-down, from `event.timeStamp`), `x`, `y` (float CSS px → also mm), `pressure` (0–1), `tiltX`, `tiltY`, `twist` (if present), `state` (D = pen down, H = hover), `buttons` (barrel button bit).

```js
canvas.addEventListener('pointerdown', e => {
  if (e.pointerType !== 'pen') return showPenBanner();
  if (performance.now() < acceptInputAfter) return;     // transition mask
  canvas.setPointerCapture(e.pointerId);
  if (!task.t0) startTaskClock(e.timeStamp);             // cap timer starts here
  beginStroke();
  record(e, 'D');
});
canvas.addEventListener('pointermove', e => {
  if (e.pointerType !== 'pen') return;
  const down = (e.buttons & 1) === 1;
  const evs = e.getCoalescedEvents ? e.getCoalescedEvents() : [e]; // ALL hardware samples
  for (const c of evs) record(c, down ? 'D' : 'H');
  if (down) drawSegments(evs);
});
canvas.addEventListener('pointerup',     e => { if (e.pointerType === 'pen') endStroke(e); });
canvas.addEventListener('pointercancel', e => { if (e.pointerType === 'pen') endStroke(e); });
```

Implementation rules:
- Preallocate typed arrays per task (e.g. 8 channels × 20,000 samples) and push by index. No DOM updates, no network calls, no `JSON.stringify` while the child is writing.
- Record **stroke boundaries** (stroke id, start/end sample index) and **pen-up gaps**; hover samples go in a separate stream.
- Log session metadata: `navigator.userAgent`, screen size, `devicePixelRatio`, orientation, estimated sample rate per task, battery level if available.
- Task clock: cap enforced by `setTimeout` from first pen-down; auto-advance at cap; reaction time = first pen-down − screen shown; record `end_reason`: `cap | done | operator | error`.
- **Transition mask:** after any screen change, ignore pen input for 1.2 s so a stray stroke cannot land on the new screen.
- If the child has not touched down after 12 s, gently pulse the start dot; do not auto-skip.
- Pen events during the end-of-cap transition: finish the current stroke into the old task, then switch.

---

## 8. Session flow (per child, ≈ 2.4 min)

```
IDLE ──[operator taps START, next child shown on screen]──▶
 S1 screen (instructions bilingual, small) ─▶ pen-down starts clock ─▶ Done/cap ─▶ mask 1.2 s
 S2 screen ───────────────────────────────────────────────────────────▶ Done/cap ─▶ mask 1.2 s
 S3 screen (Tier 2 only) ─────────────────────────────────────────────▶ Done/cap
 ─▶ AUTOSAVE locally ─▶ QC check ─▶ background upload ─▶ IDLE showing NEXT UP
```

| Seconds | Step |
|---|---|
| 0–10 | Child sits; operator checks sticker ID against "Next up" on screen, taps **START** |
| 10–45 | S1 (≤ 35 s) |
| 45–120 | S2 (≤ 75 s, typically 45–60 s) |
| 120–165 | S3 (≤ 45 s) |
| 165–175 | Autosave, upload in background, **next child is already seated on deck** |

**Operator controls (finger input, outside the child's canvas):** START, Skip child, Redo this task, Flag (left-handed, broken pen, child upset), Notes, Next child. Keep it to a few large buttons.

**Operator rules:** read the script exactly; never correct handwriting; never say test or score; if a child is upset, stop and flag.

---

## 9. Roster and queue

`tools/make_roster.py` generates (with a fixed random seed, saved with the data):

- `child_id` like `G5-017` (grade + running number), no names anywhere.
- `grade`, `class`, `half` (A/B, ~15 each), `order` (`stylus_first` or `paper_first`), `dominant_hand` if collected.
- **Single shared queue** per class half. Whichever device is free calls `GET /api/next?device=A`; the server **claims** that child (timeout 3 min). This balances load automatically between 1 or 2 devices.
- **Skip** returns the child to the end of the queue. Absentees are marked `absent` and never shown.
- Each child wears a **sticker with the ID** (printed with the paper sheet); the operator confirms visually.

Counterbalancing: Half A (stylus-first) is queued at the start of the class slot, Half B (paper-first) after their paper session. The dashboard shows which group the runner should send next.

---

## 10. Data format

One JSON file per child: `data/sessions/G5-017__<session_id>.json`. Samples are stored as **column arrays** (compact, easy to load with pandas/NumPy).

```json
{
  "schema": "1.0",
  "app_version": "0.3.1",
  "session_id": "uuid",
  "child_id": "G5-017",
  "grade": 5, "class": "5", "order": "stylus_first", "hand": "R",
  "tier": 2, "device_id": "A", "operator": "op1",
  "device": {"ua": "...", "screen_css_px": [1280, 580], "dpr": 3.0,
             "mm_per_css_px": [0.1253, 0.1253], "orientation": "landscape"},
  "tasks": [{
    "task_id": "S2", "lang": "hi",
    "target_text": "हमारे शिक्षक हर दिन नई कहानी सुनाते हैं।",
    "shown_at_epoch_ms": 1790000000000,
    "reaction_ms": 1840, "duration_ms": 52310, "cap_ms": 75000,
    "end_reason": "done", "est_rate_hz": 238,
    "layout": {"line_spacing_mm": 8, "stimulus_h_mm": 12, "margin_mm": 8},
    "strokes": [[0, 143], [144, 291]],
    "t":  [0, 4, 8],
    "x":  [101.2, 101.9, 102.6],
    "y":  [210.4, 210.1, 209.8],
    "p":  [0.12, 0.35, 0.41],
    "tx": [12, 12, 13], "ty": [-5, -5, -4],
    "state": "DDD",
    "hover": {"t": [], "x": [], "y": [], "tx": [], "ty": []},
    "thumb_png_b64": "..."
  }],
  "flags": ["left_handed"],
  "notes": "",
  "qc": {"ok": true, "warnings": []},
  "sha256": "hash of the above (excluding this field)"
}
```

- `x`, `y` stored in **mm** (and raw CSS px kept alongside, or recoverable from the calibration).
- Each upload returns an acknowledgement with the hash; the device deletes its local copy only after the server confirms the same hash.
- `data/index.csv`: one row per session (child_id, grade, order, tier, device, start time, slot seconds, qc status, hash).
- **Rendering PNGs is done offline** by `render_png.py` from the raw samples, in two variants (constant-width centreline; pressure-weighted width), at ≈ 300 dpi equivalent. Do not render in the app. Tiny thumbnails sent with each session are only for the dashboard.

---

## 11. Laptop server and dashboard

**Endpoints**

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/config` | tasks, tiers, stimuli, current tier |
| GET | `/api/next?device=A` | claim next child in queue |
| POST | `/api/skip` | requeue a child (reason) |
| POST | `/api/session` | save one child's session; returns ack + hash |
| POST | `/api/heartbeat` | device battery, sync backlog, slot times |
| POST | `/api/tier` | set tier centrally |
| GET | `/api/status` | counts, pace, ETA, recommendation |

Server rules: write each file to a temp name then rename (atomic); also copy to `data/backup/` and mirror that folder to a USB stick every few minutes (a simple robocopy/rsync loop); reject duplicates by `session_id`; never overwrite an existing child's file (new session gets a new suffix).

**Dashboard (`/dash`) shows**
- Grid of all child IDs, colour coded: waiting / on stylus now / done / skipped / absent / QC warning.
- Rolling average slot time per device, number done, number left, **projected finish time**.
- **Checkpoint panel:** minutes ahead/behind plan; button **Switch to Tier 1** (applies to all devices at next child).
- Per-device status: battery, last heartbeat, unsynced sessions.
- Last 8 thumbnails so you can spot a dead pen or blank screens immediately.
- Quick actions: mark absent, requeue, add note.

---

## 12. Automatic quality checks (flag, don't block)

| Flag | Rule (tune on the pilot) |
|---|---|
| `no_strokes` | < 2 strokes in a copy task |
| `too_few_samples` | pen-down samples < 150 in S2 |
| `low_rate` | estimated rate < 80 Hz |
| `no_pressure` | pressure std dev < 0.005 |
| `time_reversal` | non-monotonic timestamps |
| `out_of_area` | > 10% of ink outside the writing area |
| `long_idle` | > 50% of the task idle before cap |
| `palm_or_finger` | many touch events during the task |
| `no_hover` | zero hover samples in the session (device check) |

At the end of each child, the operator sees a green tick or an amber "Redo S2?" with a one-tap redo. Redo only if time allows (and only during the buffer for later classes). `tools/qc_report.py` produces the same checks for the whole dataset afterward.

---

## 13. Time-saving features (summary)

1. **Queue-driven** — no typing IDs; one START tap.
2. **Child Done ✓ button** plus caps, so nobody waits out the full time.
3. **Auto-advance** between tasks with a 1.2 s mask.
4. **Clock starts at first pen-down**, so no operator timing.
5. **On-deck seating** — next child waits beside the device; changeover ≤ 15 s.
6. **Work-stealing queue** — either device takes the next child automatically.
7. **Central tier switch** from the laptop; no touching devices.
8. **Background upload and autosave** — no waiting for saves.
9. **Instant QC and thumbnails** — catch broken recordings during the session.
10. **No in-app rendering or analysis** — all heavy work offline later.

---

## 14. Failure handling

| Problem | Response |
|---|---|
| Wi-Fi drops | Keep collecting; "unsynced N" badge; retry every 10 s; "Export unsynced" button |
| Browser tab crashes | Reopen: app restores the roster and resumes at the last completed task |
| Pen stops registering | Swap pen; flag session; redo S2 if time |
| Low battery (< 20%) | Banner on the operator screen; swap to charger/power bank |
| Child uses a finger | Banner asks for the pen; no ink |
| Child upset or refuses | Stop; flag; no pressure; log as `declined` |
| Wrong child at the device | Skip → requeue; the correct ID is claimed |
| Laptop dies | Devices keep their local copies; export all to files and merge later |

---

## 15. Paper side (summary, Grades 3–7)

All tasks on one A4 ruled sheet per child, pre-printed with QR/ID, the grade's sentences above the lines (near-point copying), and a 5 cm scale bar. Same sentences as the stylus.

| Task | Cap |
|---|---|
| P1 graphomotor strip | 45 s |
| P2 Hindi copy, comfortable | 90 s |
| P3 Hindi copy, fast | 40 s |
| P4 English copy | 60 s |
| P5 Hindi free writing | 90 s |
| P6 English free writing | 75 s |

At every time-out the child draws `|` after their last word. A half-class session takes ≈ 12 minutes. Use the same fresh pencil (G3–4) or ballpoint (G5–7) for everyone.

---

## 16. Day schedule (5 classes of ~30, 2 devices)

Each class is split into random halves of ~15. One half goes to the stylus lane while the other does paper in the classroom, then they swap.
Stylus: 15 × 2.4 min ÷ 2 devices ≈ 18 min per half → **≈ 36 min per class**.

| Time | Activity |
|---|---|
| 0:00–0:25 | Setup: hotspot, server, devices, calibration, sheets, half-lists |
| 0:30–1:06 | Class 1 (live pilot; choose a mid grade, e.g. 5) |
| 1:06–1:42 | Class 2 |
| 1:42–2:18 | Class 3 → **Checkpoint** |
| 2:18–2:54 | Class 4 |
| 2:54–3:30 | Class 5 |
| 3:30–4:10 | Buffer: absentees, redos, notebook photos |
| 4:10–4:30 | Backup, count files vs roster, debrief |

Class order: mix grades (e.g. 5, 3, 7, 4, 6) so time of day is not confounded with grade.
**Checkpoint rule (at 2:18):** ≤ 10 min behind → keep Tier 2; more than 10 min behind → **Tier 1**; more than 25 min behind → Tier 1 plus drop redos.
Crew: 2 stylus operators, 2 paper facilitators, 1 runner (+ you on the laptop dashboard). With fewer people, teachers run the paper script from a card.

---

## 17. Build order (so you always have something working)

| Step | Deliverable | Est. |
|---|---|---|
| 1 | Spike test (§2) on the real device | 1 h |
| 2 | Canvas + capture + export one task to JSON | 3 h |
| 3 | Task flow, timers, Done button, masks, S1–S3 screens, Hindi/English text | 4 h |
| 4 | Server: `/api/session`, flat files, index.csv, backup loop | 2 h |
| 5 | Roster + queue + work-stealing + operator screen | 3 h |
| 6 | Dashboard, tier switch, pace/ETA | 3 h |
| 7 | QC flags, offline sync queue, crash recovery | 3 h |
| 8 | **Full dress rehearsal**: 10 people × the real flow with a stopwatch | 1 h |

Skip: logins, user accounts, fancy charts, in-app rendering, analysis. Use rehearsal timings to update the slot-time assumptions in §4.

---

## 18. Open items to confirm

- [ ] Which stylus device(s): phone or tablet, and how many (1 or 2)
- [ ] School's actual notebook ruling per grade (replace 8 mm default)
- [ ] Hindi and English sentences checked by teachers
- [ ] Hotspot approach (laptop vs router) and who sits at the laptop
- [ ] Hover events work in your browser (spike test), otherwise pen-up gaps only
- [ ] Parental consent and school approval in place; anonymous IDs only; encrypt the laptop disk
