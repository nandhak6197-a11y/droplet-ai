# Test Infrastructure & E2E Verification Specification

## Overview
This document defines the comprehensive test infrastructure, testing methodology, and opaque-box test suites for the **Screen-Aware Deep Self-Reasoning Spatial Droplet Engine** in `c:\Users\NANDHA K\Documents\antigravity\amazing-bose`.

The test infrastructure validates all requirements (**R1–R4**) and the **8 core architectural features** across a rigorous 4-tier testing hierarchy. All tests run completely offline with zero external network or hardware display dependencies using deterministic mocks and stubs.

---

## 1. Feature Inventory Mapping

| Feature ID | Feature Name | Requirement | Core Module | Test Suite Location |
|:---:|:---|:---:|:---|:---|
| **F1** | Window-Relative Visual Hierarchy & Zones | R1 | `spatial_droplet.reasoning.reasoner`, `spatial_droplet.vision.screen_parser` | `scratch/test_e2e_screen_aware.py` |
| **F2** | Topological Set-of-Marks 2D Graph | R1 | `spatial_droplet.reasoning.reasoner` | `scratch/test_e2e_screen_aware.py` |
| **F3** | 5-Stage CoT Deliberation Trace | R1 | `spatial_droplet.reasoning.reasoner`, `spatial_droplet.core.models`, `spatial_droplet.core.memory` | `scratch/test_deliberation_action_parser.py`, `scratch/test_e2e_screen_aware.py` |
| **F4** | Multimodal Reasoner Integration (Gemini Pro) | R2 | `spatial_droplet.reasoning.providers.registry`, `openai_provider`, `gemini_provider` | `scratch/test_e2e_screen_aware.py` |
| **F5** | Strict Prompt Token Budget Guard (<5,000 tokens) | R2 | `spatial_droplet.reasoning.token_guard`, `reasoner.py` | `scratch/test_e2e_screen_aware.py` |
| **F6** | Deterministic Canonical Action Parser (11 Actions) | R3 | `spatial_droplet.reasoning.reasoner`, `spatial_droplet.engine.trajectory_engine` | `scratch/test_deliberation_action_parser.py` |
| **F7** | PrecisionActuator & Visual Settle Verification | R3 | `spatial_droplet.platform.actuator`, `spatial_droplet.vision.settle_sensor`, `spatial_droplet.verification.action_verifier` | `scratch/test_e2e_screen_aware.py`, `scratch/test_deliberation_action_parser.py` |
| **F8** | Test Infrastructure & Mirror Parity | R4 | `scratch/sync_and_verify.py`, root test harness | `scratch/test_e2e_screen_aware.py` |

---

## 2. Four-Tier Testing Methodology

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       4-TIER TESTING ARCHITECTURE                           │
├─────────────────────────────────────────────────────────────────────────────┤
│  TIER 4: REAL-WORLD DESKTOP APPLICATION SCENARIOS                           │
│  • Multi-turn modal dialog flows (Open Folder / Save As commit priority)    │
│  • Window management lifecycle (Restore -> Type & Enter -> Minimize)        │
│  • Stagnation recovery & auto-pivot (Breaking <=2 unverified turns)         │
├─────────────────────────────────────────────────────────────────────────────┤
│  TIER 3: PAIRWISE & CROSS-FEATURE COMBINATIONS                              │
│  • F1 + F2: Window-relative hierarchy + 2D cardinal relative hints          │
│  • F2 + F4: Set-of-Marks high-res image overlay + Multimodal VLM dispatch   │
│  • F3 + F6: 5-Stage CoT markdown fences + 11 canonical action extraction    │
│  • F5 + F6: Prompt token compaction (<5,000) preserving canonical actions  │
│  • F6 + F7: Canonical actions dispatch to PrecisionActuator & Win32 events  │
│  • F7 + F3: Loop guard detecting 2-turn ping-pong -> In-turn auto-pivot     │
├─────────────────────────────────────────────────────────────────────────────┤
│  TIER 2: BOUNDARY & CORNER CASES                                            │
│  • Empty/Null node manifests, 0x0 display frames, single isolated droplet   │
│  • Coordinates on window boundaries, negative coords, out-of-bounds clicks  │
│  • Missing quotes, trailing commas, unclosed JSON blocks, conversational CoT│
│  • Non-existent HWNDs, missing keys, invalid hotkey chords, zero-motion     │
│  • Extreme node counts (60–144 droplets) testing compaction limits          │
├─────────────────────────────────────────────────────────────────────────────┤
│  TIER 1: FEATURE COVERAGE (HAPPY PATH & CONTRACTS)                          │
│  • F1: 5 functional screen zones ([HEADER/NAV], [SIDEBAR], etc.)            │
│  • F2: SoM non-occluding 1px outlines, border-pinned badges, hollow rings   │
│  • F3: 5-Stage CoT deliberation trace schema and accessors                  │
│  • F4: Gemini Pro OpenAI-compatible VLM provider payload structure          │
│  • F5: Token budget estimation and safety headroom (<5,000 tokens)          │
│  • F6: 11 Canonical actions parsed and normalized deterministically        │
│  • F7: PrecisionActuator mouse, text, hotkey, and window operations         │
│  • F8: SHA-256 byte parity verification between amazing-bose and E:\autonew  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Test Suites Structure

### 3.1 `scratch/test_deliberation_action_parser.py`
Dedicated to **Feature 3 (5-Stage CoT Deliberation)** and **Feature 6 (11 Canonical Actions Parser)**:
- **Canonical Action Normalization**:
  Extracts and maps all 11 canonical actions: `click`, `double_click`, `type_and_enter`, `clear_and_type`, `launch_app`, `open_path`, `switch_window`, `minimize`, `maximize`, `hotkey`, `stop`.
- **Deliberation Schema & Backward Compatibility**:
  5-Stage CoT attributes:
  1. `user_intent_and_success_criteria`
  2. `screen_state_and_visual_hierarchy`
  3. `mistake_critique_and_loop_diagnosis`
  4. `end_to_end_master_plan`
  5. `tactical_action_justification`
  Bidirectional synchronization with legacy fields (`intent_analysis`, `observation_analysis`, `critique_of_previous_step`, `strategy_plan`, `candidate_evaluation`).
- **Dot-ID Recovery & Normalization**:
  Handles int, float, string formats (`"15"`, `"Dot [15]"`, `"dot_15"`, `"null"`, `None`).
- **Fault-Tolerant Markdown & Conversational Parsing**:
  Strips `<think>...</think>` tags, extracts JSON blocks embedded in arbitrary markdown and prose, handles trailing commas, and falls back to regex key-value extraction.
- **Zero Passive Deliberation Invariant**:
  Rejects passive text responses; ensures every turn returns an actionable grounded desktop command.

### 3.2 `scratch/test_e2e_screen_aware.py`
Dedicated to **Features 1, 2, 4, 5, 7, 8** and **Tier 4 Desktop Application Workflows**:
- **Window-Relative Visual Hierarchy & Zones (F1)**:
  Classifies coordinates relative to active window bounds (`win_rect`) and elevated modal sheets (`modal_rect`).
- **Topological Set-of-Marks 2D Graph & Rendering (F2)**:
  Constructs 2D cardinal relative connectivity hints (`Right of Dot [X]`, `Left of Dot [Y]`, `near='...'`, `in '...'`).
  Renders non-occluding 1px bounding boxes, 2px hollow micro-anchors, and border-pinned badges in base64 JPEG format.
- **Multimodal Reasoner Provider (F4)**:
  Validates OpenAI-compatible multimodal endpoint payload generation with high-resolution image URLs and structured system instructions.
- **Strict Token Budget Guard (F5)**:
  Guarantees prompt tokens remain strictly under 5,000 tokens even with large droplet manifests (up to 144 candidates).
- **PrecisionActuator & Closed-Loop Verification (F7)**:
  Mocks hardware Win32 SendInput calls; verifies native window minimize/maximize execution, smart clipboard injection, and `cv2.absdiff` motion settle sensing.
- **Autonomous Trajectory Loop Guard & Multi-Turn Scenarios (F7, Tier 4)**:
  Tests multi-step autonomous desktop task execution, detecting stagnation/ping-pong within <=2 unverified turns, executing in-turn auto-pivot, and prioritizing modal commit actions.
- **Mirror Parity & Offline Invariants (F8)**:
  Verifies recursive SHA-256 byte comparison logic against mirror targets (`E:\autonew`).

---

## 4. Test Execution & Verification

### Running the Test Suites
The canonical command to execute the E2E and Deliberation Parser test suites is:

```powershell
py -3.10 -m pytest scratch/test_e2e_screen_aware.py scratch/test_deliberation_action_parser.py -o asyncio_mode=auto -v
```

### Running the Full Regression Suite
To run all core regression suites alongside the new test suites:

```powershell
py -3.10 -m pytest scratch/test_e2e_screen_aware.py scratch/test_deliberation_action_parser.py scratch/test_production_pillars.py scratch/test_phase3_generalized.py scratch/test_deep_cot_reasoner.py scratch/test_root_cause_and_drift_fixes.py scratch/test_loop_guard_auto_pivot.py scratch/test_window_actions_and_token_safety.py scratch/test_grounded_reasoning.py -o asyncio_mode=auto
```

### Execution Environment Constraints
1. **Python Version**: Python 3.10.11 (`C:\Users\NANDHA K\AppData\Local\Programs\Python\Python310\python.exe`).
2. **Pytest Version**: `pytest 8.3.5`, `pytest-asyncio 0.24.0`.
3. **Execution Mode**: Complete offline isolation (all external LLM API calls and Win32 hardware input calls stubbed/mocked).
4. **Encoding**: Windows UTF-8 stdout reconfiguration included in test runners.
