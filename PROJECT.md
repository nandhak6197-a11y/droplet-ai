# Project: Screen-Aware Deep Self-Reasoning Spatial Droplet Engine

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       DUAL-TIER HYBRID REASONING PIPELINE                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  TIER 1: REAL-TIME PERCEPTION & SPATIAL TOPOLOGY                            │
│  ┌───────────────────────┐  ┌──────────────────────┐  ┌──────────────────┐  │
│  │ FastScreenGrabber     │  │ FullBleedLattice     │  │ ScreenState      │  │
│  │ Direct GDI Capture    │─>│ Sprayer Consensus    │─>│ Parser & Win32   │  │
│  │ (1920x1080 Frame)     │  │ (Lattice Nodes)      │  │ Window Sensor    │  │
│  └───────────────────────┘  └──────────────────────┘  └──────────────────┘  │
│                                    │                            │           │
│                                    v                            v           │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ Dynamic Visual Hierarchy & Topological Layout Graph Builder           │  │
│  │ • Window-Relative Hierarchy: [HEADER/NAV], [SIDEBAR], [WORKSPACE],    │  │
│  │   [FOOTER] (Taskbar vs App Status Bar), [MODAL/POPUP] Precedence      │  │
│  │ • 2D Relative Connectivity Hints (Left/Right/Above/Below)             │  │
│  │ • 1px archetype outlines, border-pinned badges, hollow micro-rings    │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                    │                                        │
│                                    v                                        │
│  TIER 2: COGNITIVE 5-STAGE COT DELIBERATION & ACTION EXTRACTION             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ Grounding Provider Dispatch (Gemini Pro VLM / 25-Key Groq Pool)       │  │
│  │ • Strict <5,000 Token Compaction Guard (3-Tier Dynamic Compaction)    │  │
│  │ • High-Res SoM Screenshot Overlay (JPEG Base64)                       │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                    │                                        │
│                                    v                                        │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ Deterministic Deliberation-to-Action Parser (11 Canonical Actions)    │  │
│  │ 1. user_intent_and_success_criteria                                   │  │
│  │ 2. screen_state_and_visual_hierarchy                                  │  │
│  │ 3. mistake_critique_and_loop_diagnosis                                │  │
│  │ 4. end_to_end_master_plan (Done, Active, Pending)                     │  │
│  │ 5. tactical_action_justification                                      │  │
│  │ Zero tolerance for passive text termination -> 100% grounded actions   │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                    │                                        │
│                                    v                                        │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ PrecisionActuator Execution & Closed-Loop Verification                │  │
│  │ • Hardware Win32 SendInput (<0.5ms latency)                           │  │
│  │ • Closed-loop settle verification via cv2.absdiff & window title diff  │  │
│  │ • Loop Guard breaking stagnation/ping-pong within <= 2 turns          │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                    │                                        │
│                                    v                                        │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ Automated Regression Testing & SHA-256 Mirror Parity                  │  │
│  │ • 100% pass across 67+ regression tests (pytest -o asyncio_mode=auto) │  │
│  │ • 100% bit-for-bit SHA-256 byte parity with E:\autonew                │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Feature Inventory
Every feature from the Survey phase appears here with its assigned milestone.
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Window-Relative Visual Hierarchy & Zones | Dynamic zone classification ([HEADER/NAV], [SIDEBAR], [WORKSPACE], [FOOTER], [MODAL/POPUP]) relative to active foreground window frame (`win_rect`) | M1 | Survey 1 & R1 |
| 2 | Topological Set-of-Marks 2D Graph | Non-occluding SoM overlay with 1px bounds, border-pinned badges, 2px hollow rings, and 2D cardinal connectivity hints (Right/Left/Above/Below) | M1 | Survey 1 & R1 |
| 3 | 5-Stage CoT Deliberation Trace | Strict 5-stage CoT deliberation schema enforced in prompts, dataclass models, memory milestone tracker, and normalizer | M1 | Survey 1 & R1 |
| 4 | Multimodal Reasoner Integration (Gemini Pro) | Gemini Pro multimodal VLM integration via official Google OpenAI-compatible endpoints (`/v1beta/openai/`) alongside 25-key Groq pool | M2 | Survey 2 & R2 |
| 5 | High-Res SoM Screenshot Overlay | High-resolution Set-of-Marks image payload attached to multimodal reasoner alongside topological layout graph | M2 | Survey 2 & R2 |
| 6 | Strict Prompt Token Budget Guard (<5,000 tokens) | Proactive token budget guard enforcing <5,000 tokens with 3-tier dynamic compaction (candidate pruning, graph condensation, history compaction) | M2 | Survey 2 & R2 |
| 7 | Deterministic Canonical Action Parser | Resilient parser deterministically extracting 11 canonical actions (click, double_click, type_and_enter, clear_and_type, launch_app, open_path, switch_window, minimize, maximize, hotkey, stop) with 0 passive text turns | M3 | Survey 2 & R3 |
| 8 | PrecisionActuator & Window Management | Sub-ms SendInput execution, native minimize/maximize/switch_window without coordinate guesswork | M3 | Survey 2 & R3 |
| 9 | Closed-Loop Visual Settle & Loop Guard | cv2.absdiff motion delta, semantic window title tracking, and loop guard breaking stagnation/ping-pong within <= 2 consecutive turns | M3 | Survey 2 & R3 |
| 10 | Pytest Configuration & Test Health | Canonical `pytest.ini` with `testpaths = scratch`, `asyncio_mode = auto`, and ignore filters for standalone live benchmark scripts; 100% pass on 67+ tests | M4 | Survey 3 & R4 |
| 11 | SHA-256 Bit-for-Bit Mirror Parity Automation | Recursive SHA-256 sync and verification tool guaranteeing 100% byte parity between `amazing-bose` and `E:\autonew` across all modified files | M4 | Survey 3 & R4 |
| 12 | E2E Opaque-Box Test Suite Verification | Requirement-driven 4-tier test suite verifying all R1-R4 acceptance criteria, followed by adversarial hardening | M5 (Final) | Survey 3 & R4 |

## Milestones

| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Screen-Aware Spatial Perception & 5-Stage CoT | `spatial_droplet/reasoning/reasoner.py` (zones, 2D graph, CoT prompt/clean), `spatial_droplet/vision/screen_parser.py`, `spatial_droplet/core/models.py`, `spatial_droplet/core/memory.py` | none | DONE |
| M2 | Multimodal Reasoner (Gemini Pro) & Token Budget Guard | `spatial_droplet/reasoning/providers/registry.py`, `openai_provider.py`, `gemini_provider.py`, `spatial_droplet/reasoning/token_guard.py`, `reasoner.py` provider dispatch | M1 | DONE |
| M3 | Canonical Action Parser & Closed-Loop Verification | `spatial_droplet/reasoning/reasoner.py` (canonical normalizer), `spatial_droplet/platform/actuator.py`, `spatial_droplet/verification/action_verifier.py`, `spatial_droplet/engine/trajectory_engine.py` (loop guard <=2 turns) | M2 | DONE |
| M4 | Test Infrastructure & Mirror Parity Automation | `pytest.ini`, `scratch/sync_and_verify.py` (generalized recursive sync) | M3 | DONE |
| M5 | Final Verification & Adversarial Hardening | Pass 100% of 67+ regression tests, pass 100% new E2E screen-aware test suites, Tier 5 adversarial hardening, 100% SHA-256 byte parity with `E:\autonew` | M4, E2E Track | DONE |

## Interface Contracts

### M1 ↔ M2 (Perception & Spatial Graph ↔ Multimodal Reasoner & Token Guard)
- `_build_spatial_layout_graph(nodes, sw, sh, modal_rect=None, win_rect=None) -> str`: Produces structured 5-zone topological graph text with 2D relative neighbor hints (Right/Left/Above/Below).
- `_render_unoccluded_som(frame, nodes, sw, sh, max_image_side=1280) -> str`: Returns base64 data URL JPEG.
- `TokenBudgetGuard.enforce(system_instruction, user_content, nodes, max_tokens=4800) -> tuple[str, str, list[DiagonalDroplet]]`: Counts tokens using tiktoken (fallback 4 chars/token); if >=4800, compacts nodes, graph, and history to guarantee <5,000 total tokens.

### M2 ↔ M3 (Multimodal Reasoner ↔ Deliberation-to-Action Parser)
- Provider output string (raw LLM response, markdown fenced JSON, or conversational CoT) -> `_clean_decision(raw_decision) -> DecisionResult`:
  - Enforces 5-stage CoT properties: `user_intent_and_success_criteria`, `screen_state_and_visual_hierarchy`, `mistake_critique_and_loop_diagnosis`, `end_to_end_master_plan`, `tactical_action_justification`.
  - Normalizes `action` deterministically into one of 11 canonical actions:
    `["click", "double_click", "type_and_enter", "clear_and_type", "launch_app", "open_path", "switch_window", "minimize", "maximize", "hotkey", "stop"]`.
  - Guarantees `target_dot_id` is an integer or None (with dot-id recovery from deliberation text).
  - Guarantees 0 turns terminate in passive text.

### M3 ↔ M4 & Trajectory Engine (Parser & Actuator ↔ Verification & Loop Guard)
- `PrecisionActuator.execute(droplet, action, input_text, ...)`: Direct SendInput and window action dispatch.
- `ActionVerifier.verify_action(pre_snapshot, droplet, action) -> VerificationResult`: Evaluates `cv2.absdiff(pre.roi, post_roi)` and `post_title != pre.title`.
- `AutonomousTrajectoryEngine`: Detects stagnation (`unverified_turns >= 2`), immediately triggers loop auto-pivot or dialog commit escape.

## Code Layout
- `spatial_droplet/core/`: `models.py` (ScreenAwarenessContext, TrajectoryStep, DecisionResult), `memory.py` (EpisodicTrajectoryMemory).
- `spatial_droplet/vision/`: `screen_parser.py` (ScreenStateParser, window-relative modal & zone extraction), `settle_sensor.py` (AdaptiveSettleSensor).
- `spatial_droplet/reasoning/`:
  - `reasoner.py` (DiagonalDropletReasoner, spatial layout graph, CoT prompts, canonical action cleaner).
  - `token_guard.py` (TokenBudgetGuard, 3-tier dynamic compaction).
  - `groq_rotator.py` (25-key rotation pool).
  - `providers/`: `registry.py`, `openai_provider.py`, `gemini_provider.py`, `groq_provider.py`.
- `spatial_droplet/platform/`: `actuator.py` (PrecisionActuator), `window.py` (Win32 window context & management).
- `spatial_droplet/verification/`: `action_verifier.py` (cv2.absdiff & semantic title verification).
- `spatial_droplet/engine/`: `trajectory_engine.py` (AutonomousTrajectoryEngine, loop guard <= 2 turns).
- Root / tooling: `pytest.ini`, `scratch/sync_and_verify.py`.
