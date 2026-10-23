# Spatial Droplet Desktop Automation Framework — Enterprise Research Audit & Master Engineering Plan (`Planner.md`)

**Document Role:** Master Research Audit, Root-Cause Forensic Analysis, and Precision Engineering Blueprint  
**Target Workspaces:** `c:\Users\NANDHA K\Documents\antigravity\amazing-bose` & `E:\autonew`  
**Evaluation Baseline:** OSWorld-G 100-Sample Visual Grounding Benchmark (`42.0%` Strict / `45.0%` Tolerant Hits)  

---

## 0. Enterprise-Grade Rephrased Prompts (Part A, Part B, Part C)

To standardize this initiative for enterprise R&D, quantitative engineering, and human-computer interaction (HCI) teams, the operational mandates are formalized below:

### Enterprise Prompt — PART A (Principal Research Auditor & Architecture Supervisor)
> *"Act as a Principal AI Research Scientist and Systems Architecture Supervisor. Conduct a forensic end-to-end code and data audit of the Spatial Droplet GUI Automation Framework across all 100 OSWorld-G benchmark samples:  
> 1. **Zero-Inflation Integrity Audit**: Inspect all perception, candidate ranking, VLM prompting, and evaluation modules to verify the complete absence of ground-truth leakage, coordinate hardcoding, or benchmark-specific shortcuts behind the 45/100 baseline score.  
> 2. **Quantitative Root-Cause Failure Analysis**: Decompose the 55 missed samples across the perception-to-reasoning pipeline by measuring Oracle Detector Recall (`FullBleedLatticeSprayer`) versus Shortlist Manifest Recall (`DiagonalDropletReasoner`) and VLM Selection Precision. Identify every algorithmic bottleneck, silent exception, resolution compression artifact, and coordinate-space mismatch.  
> 3. **Product & Innovation Viability Assessment**: Evaluate the commercial and technical viability of the three-tier Spatial Droplet paradigm (Win32 OS Fast-Path, Zero-API Deterministic Droplet Engine, and Set-of-Marks VLM Trajectory Engine with `cv2.absdiff` closed-loop verification) for public and enterprise desktop automation.  
> 4. **High-Leverage Accuracy Innovations**: Formulate generalizable, zero-hardcoding algorithmic upgrades—including Multi-Stage Coarse-to-Fine ROI Zoom Grounding, Task-Aware Candidate Retention, Unoccluded High-Resolution Set-of-Marks Rendering, and Sub-Token Character/Slider Interpolation—to close the gap between the 45% VLM score and the 99% detector oracle ceiling.  
> 5. **Module-Level Engineering Specification**: Document exact file-by-file remediation steps across `sprayer.py`, `reasoner.py`, `grid_resolver.py`, `trajectory_engine.py`, `actuator.py`, `voice/`, and `chat_widget.py` inside a canonical `Planner.md` blueprint."*

### Enterprise Prompt — PART B (Lead Quantitative Systems & Audio/VLM Engineer)
> *"Act as a Staff Quantitative Systems Engineer specializing in low-latency multimodal inference and real-time audio signal processing. Execute the 6 High-Quality Architectural Constraints defined in `Planner.md`:  
> 1. **Constraint-Driven Grounding Implementation**: Eliminate the 31-sample candidate pruning bottleneck, fix the `screenshot_pro_benchmark.py` base64 re-encoding exception, upgrade VLM Set-of-Marks resolution from `640px` to `1280px` with two-stage ROI crop zoom refinement, and unify `target_point` coordinate unscaling.  
> 2. **Token-Disciplined Deep Reasoning**: Bound internal VLM chain-of-thought (`<think>`) tokens to concise, high-signal spatial elimination so inference latency drops significantly while grounding accuracy increases.  
> 3. **End-to-End Voice Pipeline Remediation & Automated Verification**: Diagnose and resolve microphone capture, silence detection, and instruction dispatch failures in `spatial_droplet/voice/` and `chat_widget.py`. Implement adaptive Voice Activity Detection (Auto-VAD end-of-speech auto-stop), Automatic Gain Control (AGC peak normalization to `-3 dBFS`), hardware sample-rate resampling (`44.1/48kHz -> 16kHz`), prompt de-hallucination, and race-free Tkinter `Entry` submission. Validate via repeated automated multi-utterance transcription and dispatch tests.  
> 4. **Post-Implementation Forensic Verification**: Re-verify all modules, run quantitative benchmark validation to confirm accuracy gains, and enforce 100% SHA-256 byte parity across workspaces."*

### Enterprise Prompt — PART C (Principal Design Technologist & Modern UI Architect)
> *"Act as a Principal UI/UX Design Technologist. Redesign and polish `spatial_droplet/ui/chat_widget.py` to deliver an Anthropic-grade metallic-glass floating copilot interface:  
> 1. **Toolbar Decluttering & Layout Rebalancing**: Remove the legacy `+` icon on the bottom-left toolbar and realign the mode controls cleanly.  
> 2. **Decoupled Interactive Mode Pills**: Replace the monolithic segmented container with three independent, floating rounded pill boxes (`[Chat]`, `[Quick]`, `[Droplets]`), each featuring smooth delayed hover interpolation, subtle border illumination, and state-aware active styling.  
> 3. **Tactile Depth Send Button & Hotkey-First Voice Mic Button**: Engineer a multi-layered beveled Send button with smooth hover transition timing, alongside a modern vector Microphone button operating as a 1-click / hotkey toggle (`Alt+V` / `F8`, no hold required) with live pulsing audio-ring feedback.  
> 4. **Anthropic Editorial Typography**: Apply Anthropic's signature typography stack (`Styrene A`, `Copernicus`, `Charter`, `Georgia`, `Segoe UI Variable Display`) with warm alabaster (`#F4F3EE`) primary text and muted stone (`#9C9A92`) secondary metadata.  
> 5. **Humanized Robot Pet Mascot with Specular Metallic Chassis**: Enhance `RobotMascotAnimator` with natural human micro-saccades, expressive upper/lower eyelid curvature, subtle head-tilt kinematics, warm emotional glow cues, and a multi-stop metallic-glass outer shell reflection.  
> 6. **Brushed Metallic-Glass Bar & Smooth Animated Model Selector Popover**: Upgrade the main card to a brushed gunmetal/obsidian metallic-glass surface (`#18181B` to `#121214` gradient with specular top refraction). Replace the overlapping `⚡` model text with a sleek capsule button (`Model Name  Tier  ^`) that smoothly animates open a floating glassy popover menu matching the reference specification (header `Model`, badge pills `New`/`Live`, active `✓` checkmark, chevron `>`, and `Select Model Ctrl+/` footer)."*

---

## 1. PART A — Forensic Audit & Root-Cause Analysis of the 45/100 Benchmark Score

### 1.1 Integrity Audit: Is Anything Hardcoded or Artificially Inflated?
**Verdict: 100% Genuine — Zero Hardcoding, Zero Ground-Truth Leakage.**

A line-by-line audit of [`screenshot_pro_benchmark.py`](file:///c:/Users/NANDHA%20K/Documents/antigravity/amazing-bose/screenshot_pro_benchmark.py), [`spatial_droplet/vision/sprayer.py`](file:///c:/Users/NANDHA%20K/Documents/antigravity/amazing-bose/spatial_droplet/vision/sprayer.py), and [`spatial_droplet/reasoning/reasoner.py`](file:///c:/Users/NANDHA%20K/Documents/antigravity/amazing-bose/spatial_droplet/reasoning/reasoner.py) confirms:
1. **Strict Ground-Truth Isolation**: In `screenshot_pro_benchmark.py` (lines 448–476), `gt_box = convert_gt_box(s["box_coordinates"])` is never passed to `sprayer.spray(img)`, `predict_vanilla()`, or `predict_with_spatial()`. Ground-truth coordinates are accessed exclusively *after* prediction to compute `is_point_in_box` and Euclidean distance.
2. **Coordinate-Agnostic Candidate Scoring**: In `reasoner.py` (lines 690–1128), `relevance_score(d)` ranks droplets strictly by OCR text overlap, icon detector archetype labels, container ownership, and relative geometric alignment among detected siblings—never by hardcoded pixel coordinates or sample IDs.
3. **Genuine Multimodal Execution**: Every single one of the 42 strict hits and 45 tolerant hits was produced by live candidate detection and model inference.

---

### 1.2 Quantitative Root-Cause Analysis: Why Only 45 Hits Out of 100?

To determine whether the 55 misses originated in **Computer Vision Detection (`sprayer.py`)**, **Candidate Shortlisting (`reasoner.py`)**, or **VLM Visual Selection**, we executed an Oracle Recall Audit (`scratch/measure_recall_100.py`) across all 100 OSWorld-G benchmark samples:

| Pipeline Stage | Strict Recall (`0.0 px` to GT Box) | Tolerant Recall (`<= 24.0 px` to GT Box) | Samples Lost at Stage |
| :--- | :---: | :---: | :---: |
| **Stage 1: Raw `FullBleedLatticeSprayer` Output** | **89.0%** (89 / 100) | **99.0%** (99 / 100) | Only **1** sample missed (`#83`) |
| **Stage 2: `reasoner.py` 96-Candidate Manifest** | **58.0%** (58 / 100) | **68.0%** (68 / 100) | **31** valid targets pruned out! |
| **Stage 3: Final VLM Selection (`benchmark_matrix`)** | **42.0%** (42 / 100) | **45.0%** (45 / 100) | **23** targets mis-selected by VLM |

#### The 5 Exact Technical Causes of the 55 Misses:

1. **Bottleneck #1 — Candidate Pruning Eliminated 31 Valid Targets Before the VLM Ever Saw Them (56.4% of all failures!)**:
   - `sprayer.py` successfully placed a droplet directly inside or within 24px of the Ground Truth target on **99 out of 100 screenshots**!
   - However, dense desktop screenshots contain **350 to 615 UI droplets**.
   - In `reasoner.py` (line 529 & line 1181), `max_ui_nodes = 130` and `cand_budget = 96` pruned out **31 true target droplets** (`Samples 32, 37, 42, 43, 47, 48, 54, 57, 58, 59, 60, 62, 69, 70, 72, 76, 77, 78, 79, 80, 81, 84, 86, 88, 89, 91, 92, 93, 94, 95, 98`).
   - *Why were they pruned?* Because `relevance_score(d)` penalized generic visual icons (`score -= 150` on line 831) while awarding `+300` to `+600` points to random static text lines that happened to share a common word with the prompt (e.g., `"window"`, `"slide"`, `"value"`, `"color"`, `"row"`). Consequently, unlabeled toolbar icons (GIMP Bucket Fill `#42`, Unified Transform `#47`, Magic Wand `#48`, LibreOffice Underline `#37`, Zoom Slider `#60/#79/#80`) were pushed below rank 96 and completely excluded from the Set-of-Marks image and text manifest!

2. **Bottleneck #2 — Silent `cv2.imencode` TypeError in `screenshot_pro_benchmark.py` (Lines 229–262)**:
   - In `screenshot_pro_benchmark.py`:
     ```python
     som_img, _, _ = reasoner._render_unoccluded_som(img, droplets[:50])
     _, buf = cv2.imencode(".jpg", som_img if som_img is not None else img, [cv2.IMWRITE_JPEG_QUALITY, 75])
     ```
   - `reasoner._render_unoccluded_som()` returns a **base64 data URI string** (`"data:image/jpeg;base64,..."`), NOT a `numpy.ndarray`!
   - Passing that string to `cv2.imencode()` threw a `cv2.error` on 100% of samples, which was silently caught by `except Exception: pass` (line 260) and fell back to `reasoner.decide(..., provider="groq", target_model="qwen/qwen3.8-27b")`.
   - That is why `kaggle_qwen_spatial` and `spatial_qwen27b` had **90/100 identical `(pred_x, pred_y)` predictions** and identical `42% / 45%` accuracy!

3. **Bottleneck #3 — Aggressive `640×360` Downscaling & Badge Clutter (`max_image_side=640, jpeg_quality=65`)**:
   - In `reasoner.py` (lines 1397 & 1813), `1920×1080` screenshots were downscaled by **3×** to `640×360` at JPEG quality 65.
   - At `640×360`, a `16×16` toolbar icon shrinks to **`5×5` pixels**, while 96 numbered Set-of-Marks badges (`[132]`) overlap across dense toolbars (such as GIMP's left tool palette and LibreOffice's top ribbon), obscuring the neighboring icons and causing the VLM to repeatedly pick the topmost visible badge in the cluster.

4. **Bottleneck #4 — Ignored `target_point` Coordinates & Missing Scale Division in `reasoner.py` (Line 2265)**:
   - The system prompt explicitly instructs the VLM to output `{"target_dot_id": <int|null>, "target_point": [<x>,<y>]|null}` (line 2206).
   - However, the parser at line 2265 only checked `decision.get("coordinate") or decision.get("point")` — completely ignoring `decision.get("target_point")`! When the VLM output `target_dot_id: null` with an exact `target_point: [x, y]`, `reasoner.py` ignored the point and fell back to `all_candidates[0]`.

5. **Bottleneck #5 — Over-Eager `TabularGridResolver` Pre-emption & Sub-Word Target Offsets (`Samples 44, 72, 76, 77, 83, 88, 92`)**:
   - `TabularGridResolver` triggered on instructions like `"Click on the product of row 12"` (`#77`) or `"Click to select row 12"` (`#88`), matching arbitrary OCR numbers `"12"` inside table cells instead of the left-most spreadsheet row header (`x < 60`) or column header.
   - For character-level instructions (`#44: letter "t" in "virtual"`, `#54: letter "M" in "Mission"`, `#72: letter "o" in "for"`, `#83: exclamation mark of "Big news!"`), the droplet center is at the center of the whole word box rather than interpolated to the specific character offset inside the word bounding box.

---

## 2. PART A.3 — End-to-End Project Scope & Innovation Validation

The **Spatial Droplet Framework** is an enterprise-ready desktop automation architecture with three distinct competitive advantages over conventional pixel-predicting computer-use agents:

1. **Tier 0 — `< 25ms` Deterministic Win32 OS Fast-Path Router (`IntentScopeRouter`)**:
   - Handles app launching, window switching, maximizing/minimizing/closing, system hotkeys, and direct URL/folder opening via native OS APIs without VLM latency or mis-click risk.
2. **Tier 1 — `< 100ms` Zero-API Pure Droplet Engine**:
   - Combines WinRT Neural OCR, non-blocking Windows UIA tree caching, and CLAHE contour/icon detection to execute explicit label/icon clicks locally at `$0.00` compute cost.
3. **Tier 2 — Closed-Loop Set-of-Marks Multimodal VLM + `cv2.absdiff` Verification**:
   - Converts continuous `1920×1080` coordinate regression (where vanilla Qwen 27B gets `0.0%` strict hits) into discrete grounded landmark selection, backed by `120×120px` pre/post-click pixel diff verification and automatic micro-nudge retry.

**Conclusion on Viability**: With the 5 bottlenecks above resolved, the framework's 99% detector recall can be directly unlocked by the VLM, making it highly suitable for both public power-users and private enterprise RPA deployments.

---

## 3. PART B — The 6 High-Quality Architectural Constraints & Precision Methodologies

To systematically upgrade grounding accuracy and fix the voice subsystem, Part B enforces the following **6 High-Quality Engineering Constraints**:

### Constraint 1: High-Recall Stratified Candidate Manifest & Zero-Pruning Rescue (`reasoner.py`)
- **Methodology**:
  - Increase `max_ui_nodes` from `130` to `300` and `cand_budget` / `max_vlm_candidates` to `140`.
  - Remove the `-150` penalty on visual icons when the user instruction implies clicking a tool, icon, button, checkbox, arrow, or slider.
  - Add **Spatial Grid Cell Stratification** (`6×4` screen grid = 24 zones) and **Dense Toolbar/Sidebar Protection** so every toolbar icon in GIMP / LibreOffice / VS Code / Chrome is preserved in the candidate manifest.
  - Upgrade `SpatialGeometryEngine` character-offset interpolation so requests targeting a specific character or punctuation mark (`letter "t" in "virtual"`, `exclamation mark of "Big news!"`, `letter "o" of "for"`) interpolate the exact horizontal character position within the OCR word bounding box!

### Constraint 2: Two-Stage Coarse-to-Fine High-Resolution SoM & ROI Zoom Refinement (`reasoner.py` & `screenshot_pro_benchmark.py`)
- **Methodology**:
  - Fix `screenshot_pro_benchmark.py` `predict_with_spatial("kaggle_qwen")` so it delegates directly to `reasoner.decide(..., provider="ngrok")` instead of calling `cv2.imencode` on a base64 string.
  - Increase `_render_unoccluded_som` resolution from `max_image_side=640, jpeg_quality=65` to `max_image_side=1280, jpeg_quality=85` so `16×16` toolbar icons remain crisp and readable.
  - Implement **Stage-2 ROI Crop Zoom Refinement**: When the Stage-1 prediction lands in a dense cluster (>= 3 candidates within 80px) or when confidence/evidence is ambiguous, automatically crop a `1:1` native-resolution `480×480` ROI patch around the predicted region, render crisp non-overlapping micro-badges on all local sprayer droplets in that ROI, and run a fast Stage-2 zoom verification pass!

### Constraint 3: Unified Coordinate Unscaling & Nearest-Droplet Point Snapping (`reasoner.py`)
- **Methodology**:
  - Parse `target_point`, `point`, `coordinate`, and `[x, y]` across both the `ngrok` and `groq` branches.
  - Properly unscale VLM-returned coordinates from image/normalized space (`[0..1000]` or `scale_vlm`) back to native screen pixels (`sw_screen × sh_screen`) and snap to the tightest enclosing or nearest (`<= 36px`) sprayer droplet from the *full* unpruned `key_nodes` list (`99%` oracle pool).

### Constraint 4: Precision Tabular & Slider Geometry Guards (`grid_resolver.py` & `reasoner.py`)
- **Methodology**:
  - Restrict `TabularGridResolver` so row headers (`Row 12`, `Row 25`, `31st row`) only match left-margin row-index cells (`x < 110`) and column headers (`Column F`, `Column B`) only match top-margin column-letter cells, preventing false-positive numeric matches inside data cells.
  - Enhance status-bar zoom slider detection (`25%`, `30%`, `80%`, `100%`, `maximum` of zoom slider at bottom-right `y > sh - 60, x > sw * 0.75`) so fractional slider positions (`p%`) interpolate cleanly along the detected zoom trackbar bounds.

### Constraint 5: End-to-End Voice Capture, Auto-VAD, AGC Normalization & Race-Free Dispatch (`spatial_droplet/voice/` & `chat_widget.py`)
- **Methodology**:
  - **Native Sample Rate & Resampling (`audio_recorder.py`)**: Query the default input device's native sample rate if `16000 Hz` fails, and cleanly resample to `16000 Hz` mono `int16`.
  - **Automatic Voice Activity Detection (`audio_recorder.py` & `voice_manager.py`)**: Track rolling frame RMS energy during recording. Once speech onset is detected (`rms > threshold`), automatically stop recording after `1.2 seconds` of trailing silence (Auto-VAD) so the user never has to hold the mic button or wait 30 seconds! Also still allow instant manual toggle-off via Mic click or `Alt+V` / `F8`.
  - **Automatic Gain Control (AGC) & Silence Trimming (`transcriber.py` / `voice_manager.py`)**: Trim leading/trailing dead air and peak-normalize quiet voice waveforms to `-3 dBFS` (`~23,000` int16 peak) before WAV encoding so quiet laptop microphones transcribe with 100% clarity.
  - **De-Hallucinated Whisper Prompt (`transcriber.py`)**: Replace the long command-list prompt (which caused Whisper continuation hallucinations) with a minimal vocabulary glossary hint (`"GUI automation command."`).
  - **Race-Free UI Instruction Dispatch (`chat_widget.py` & `app.py`)**: Update `set_voice_recording()` to update the mic button and mascot emotion *in-place* on the canvas without calling `canvas.delete("all")` or destroying `self.entry`, ensuring `set_voice_text(text, auto_submit=True)` reliably populates and dispatches 100% of dictated instructions!

### Constraint 6: Token-Disciplined Quantitative Deliberation & Repeated Verification
- **Methodology**:
  - Bound VLM prompts to concise structured JSON output (`max_tokens=320`, concise 1-sentence rationale) to cut inference latency by >50% and prevent token truncation.
  - Run automated multi-utterance synthetic speech + live Whisper API transcription tests and re-evaluate the benchmark suite to verify concrete accuracy improvements.

---

## 4. PART C — UI/UX Specification (Anthropic Metallic-Glass & Model Popover)

All visual upgrades are consolidated in [`spatial_droplet/ui/chat_widget.py`](file:///c:/Users/NANDHA%20K/Documents/antigravity/amazing-bose/spatial_droplet/ui/chat_widget.py):
1. **Remove `+` Icon**: Eliminate `_render_plus_button` and left-align the mode pills at `x = 22`.
2. **Three Separate Pill Boxes (`[Chat]`, `[Quick]`, `[Droplets]`)**:
   - Render each mode option as its own standalone rounded capsule (`radius=11`, `height=26px`, `8px` gap between pills) with independent metallic-glass fill (`#202024` idle, `#2E2E34` hover, `#32323A` active with specular border) and smooth `120ms` delayed hover color interpolation.
3. **Modern Depth Send Button & Toggle Mic Button**:
   - **Send Button**: Multi-layered 3D depth capsule (`#D96548` terracotta core, `#E87B60` top specular bevel, `#A8462E` bottom shadow rim, smooth hover brighten transition, crisp vector arrow).
   - **Mic Button**: Sleek standalone rounded pill/circle with vector condenser mic + hotkey badge (`Alt+V`), smooth hover elevation, and animated concentric pulse rings during active toggle/Auto-VAD listening.
4. **Anthropic Typography**:
   - Primary font stack: `("Styrene A", "Copernicus", "Georgia", "Segoe UI Variable Display", "Segoe UI")` with warm alabaster `#F4F3EE` text and `#9C9A92` muted labels.
5. **Humanized Interactive Robot Mascot (`RobotMascotAnimator`)**:
   - Add human-like micro-saccades, expressive upper/lower eyelid arcs, warm coral cheek glow in `happy` state, subtle head-tilt angle interpolation, and a high-gloss metallic-glass specular outer coating on the helmet shell.
6. **Metallic-Glass Coated Chat Bar & Smooth Animated Model Selector Popover (Matching Reference Image)**:
   - Main bar rendered with a multi-stop brushed dark gunmetal metallic gradient (`#1E1E22` -> `#141417` -> `#18181C`), dual-layer glass refraction border (`#383840` outer, `#FFFFFF22` top specular highlight).
   - **Model Trigger Pill**: Clean rounded capsule at the bottom toolbar displaying `Qwen 2.5 VL  Medium  ^` (primary model name in `#689DF6` soft periwinkle blue, tier in `#8E8E93` muted grey, and crisp chevron `^`) with zero overlapping `⚡` icons.
   - **Smooth Animated Model Popover Window**: Clicking the model capsule (or pressing `Ctrl+/`) smoothly opens (`150ms` alpha/slide ease-in) a borderless floating dark glassy popover card matching the uploaded reference image:
     - Top header: `Model` in `#8A8A8E`
     - Selectable rows with hover highlight (`#2A2A2E`), model title, tier subtitle (`Medium` / `Dual-T4` / `Rotator`), rounded badge pills (`New`, `Notice ⓘ`), active checkmark `✓` or right chevron `>`,
     - Bottom separator line and `Select Model  Ctrl+/` footer row.
