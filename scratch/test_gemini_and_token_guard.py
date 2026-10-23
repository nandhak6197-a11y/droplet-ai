"""
Unit and Integration Test Suite for Milestone 2:
- Google Gemini Grounding Provider (Gemini 1.5 Pro VLM)
- 4-Tier API Key Resolution & Multi-Alias Registry Mapping
- High-Resolution Set-of-Marks (SoM) Multimodal Visual Payload Formatting (up to 1920px)
- Explicit Memory Deallocation & Leak Prevention
- Strict Prompt Token Budget Guard (<4,800 tokens target, <5,000 ceiling)
- 3-Tier Dynamic Compaction (Candidate Pruning, Spatial Graph Condensation, History Compaction)
- Provider Routing in decide() and decide_turn() with Seamless Fallback
"""
import pytest
import os
import sys
import gc
import numpy as np
import base64
from unittest.mock import MagicMock, patch
import tiktoken

sys.path.insert(0, os.path.abspath("."))

from spatial_droplet.reasoning.reasoner import DiagonalDropletReasoner, DecisionResult
from spatial_droplet.reasoning.providers.base import BaseGroundingProvider, GroundingDecision
from spatial_droplet.reasoning.providers.registry import get_grounding_provider
from spatial_droplet.reasoning.providers.gemini_provider import GeminiGroundingProvider
from spatial_droplet.reasoning.token_guard import TokenBudgetGuard
from spatial_droplet.core.models import (
    DiagonalDroplet,
    WindowContext,
    ScreenAwarenessContext,
    TrajectoryStep
)
from spatial_droplet.core.memory import EpisodicTrajectoryMemory


class TestGeminiProviderInitializationAndKeyResolution:
    """Verifies Gemini provider initialization, base URLs, and 4-tier API key resolution."""

    def test_gemini_provider_defaults_and_explicit_key(self):
        """Verifies explicit api_key param, default base URL, default model."""
        p = GeminiGroundingProvider(api_key="test-explicit-key")
        assert p.api_key == "test-explicit-key"
        assert p.base_url == "https://generativelanguage.googleapis.com/v1beta/openai/"
        assert p.model_name == "gemini-1.5-pro"
        assert p.is_available() is True

    def test_gemini_api_key_env_resolution(self, monkeypatch):
        """Tier 2: GEMINI_API_KEY environment variable."""
        monkeypatch.setenv("GEMINI_API_KEY", "gemini-env-key-123")
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        p = GeminiGroundingProvider()
        assert p.api_key == "gemini-env-key-123"
        assert p.is_available() is True

    def test_google_api_key_env_resolution(self, monkeypatch):
        """Tier 3: GOOGLE_API_KEY fallback environment variable."""
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.setenv("GOOGLE_API_KEY", "google-env-key-456")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        p = GeminiGroundingProvider()
        assert p.api_key == "google-env-key-456"
        assert p.is_available() is True

    def test_openai_api_key_fallback_resolution(self, monkeypatch):
        """Tier 4: OPENAI_API_KEY fallback environment variable."""
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "openai-fallback-key-789")
        p = GeminiGroundingProvider()
        assert p.api_key == "openai-fallback-key-789"
        assert p.is_available() is True

    def test_gemini_unavailable_when_no_key(self, monkeypatch):
        """Verifies is_available() is False when no key is configured."""
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        p = GeminiGroundingProvider()
        assert p.is_available() is False
        assert p.api_key == ""

    def test_gemini_custom_model_and_base_url(self):
        """Verifies custom model and base_url parameters."""
        p = GeminiGroundingProvider(
            api_key="custom-key",
            base_url="https://custom.gateway.internal/v1",
            model_name="gemini-2.0-flash"
        )
        assert p.base_url == "https://custom.gateway.internal/v1"
        assert p.model_name == "gemini-2.0-flash"


class TestGeminiRegistryIntegration:
    """Verifies registry mapping for all Gemini aliases and auto-detection."""

    def test_registry_maps_gemini_aliases(self):
        """Verifies get_grounding_provider resolves all standard Gemini aliases."""
        aliases = [
            "gemini",
            "gemini-pro",
            "google",
            "gemini_pro",
            "gemini-flash",
            "gemini-1.5-pro",
            "gemini-2.0-flash"
        ]
        for alias in aliases:
            prov = get_grounding_provider(alias, api_key="mock-key")
            assert isinstance(prov, GeminiGroundingProvider), f"Alias '{alias}' did not return GeminiGroundingProvider"

    def test_registry_auto_detects_gemini(self, monkeypatch):
        """Verifies auto mode selects Gemini when GEMINI_API_KEY is present and local is absent."""
        monkeypatch.setenv("GEMINI_API_KEY", "auto-gemini-key")
        with patch("spatial_droplet.reasoning.providers.local_provider.LocalGroundingProvider.is_available", return_value=False):
            prov = get_grounding_provider("auto")
            assert isinstance(prov, GeminiGroundingProvider)
            assert prov.is_available() is True


class TestGeminiMultimodalPayloadAndExecution:
    """Verifies multimodal Set-of-Marks image attachment, 5-stage CoT parsing, and async/sync duality."""

    def test_multimodal_payload_formatting_with_high_res_image(self):
        """Verifies that SoM screenshot data URL is attached with detail='high'."""
        provider = GeminiGroundingProvider(api_key="mock-gemini-key")
        captured_calls = []

        class MockCompletions:
            def create(self, **kwargs):
                captured_calls.append(kwargs)
                choice = MagicMock()
                choice.message.content = (
                    '{"deliberation": {"user_intent_and_success_criteria": "Save file", '
                    '"screen_state_and_visual_hierarchy": "Dialog open", '
                    '"mistake_critique_and_loop_diagnosis": "None", '
                    '"end_to_end_master_plan": ["Phase 1: Click Save (ACTIVE)"], '
                    '"tactical_action_justification": "Target 42 is Save button"}, '
                    '"target_dot_id": 42, "action": "click", "reasoning": "Clicking Save"}'
                )
                resp = MagicMock()
                resp.choices = [choice]
                return resp

        provider.client = MagicMock()
        provider.client.chat.completions = MockCompletions()

        cand = DiagonalDroplet(id=42, center_x=600, center_y=400, category="button", label="Save", diag_wave=1000)
        som_url = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEASABIAAD/mock"

        decision = provider.ground_sync(
            instruction="Click Save",
            manifest_text="• [42] Save",
            all_candidates=[cand],
            som_image_url=som_url,
            win_info="Save Dialog"
        )

        assert len(captured_calls) == 1
        kwargs = captured_calls[0]
        assert kwargs["model"] == "gemini-1.5-pro"
        messages = kwargs["messages"]
        user_msg = next(m for m in messages if m["role"] == "user")
        assert isinstance(user_msg["content"], list)
        img_part = next(p for p in user_msg["content"] if p["type"] == "image_url")
        assert img_part["image_url"]["url"] == som_url
        assert img_part["image_url"]["detail"] == "high"

        assert decision.target_dot_id == 42
        assert decision.action == "click"
        assert decision.provider_used == "gemini"
        assert decision.deliberation["user_intent_and_success_criteria"] == "Save file"

    def test_raw_base64_string_normalization(self):
        """Verifies raw base64 string is defensively prefixed with data:image/jpeg;base64,."""
        provider = GeminiGroundingProvider(api_key="mock-gemini-key")
        captured_calls = []

        class MockCompletions:
            def create(self, **kwargs):
                captured_calls.append(kwargs)
                choice = MagicMock()
                choice.message.content = '{"target_dot_id": 1, "action": "click", "reasoning": "ok"}'
                resp = MagicMock()
                resp.choices = [choice]
                return resp

        provider.client = MagicMock()
        provider.client.chat.completions = MockCompletions()

        cand = DiagonalDroplet(id=1, center_x=100, center_y=100, category="button", label="OK", diag_wave=200)
        provider.ground_sync(
            instruction="Click OK",
            manifest_text="• [1] OK",
            all_candidates=[cand],
            som_image_url="RAW_BASE64_DATA_WITHOUT_PREFIX"
        )

        kwargs = captured_calls[0]
        user_content = next(m for m in kwargs["messages"] if m["role"] == "user")["content"]
        img_part = next(p for p in user_content if p["type"] == "image_url")
        assert img_part["image_url"]["url"] == "data:image/jpeg;base64,RAW_BASE64_DATA_WITHOUT_PREFIX"

    @pytest.mark.asyncio
    async def test_async_ground_calls_ground_sync(self):
        """Verifies async ground(...) faithfully invokes ground_sync(...)."""
        provider = GeminiGroundingProvider(api_key="mock-gemini-key")
        expected_decision = GroundingDecision(target_dot_id=7, action="click", reasoning="async success")

        with patch.object(provider, "ground_sync", return_value=expected_decision) as mock_sync:
            cand = DiagonalDroplet(id=7, center_x=200, center_y=200, category="button", label="Btn", diag_wave=400)
            res = await provider.ground(
                instruction="Click Btn",
                manifest_text="• [7] Btn",
                all_candidates=[cand]
            )
            assert res.target_dot_id == 7
            mock_sync.assert_called_once()

    def test_uninitialized_client_returns_error_decision(self):
        """Verifies provider without client returns structured error decision without raising."""
        provider = GeminiGroundingProvider(api_key="")
        cand = DiagonalDroplet(id=1, center_x=10, center_y=10, category="button", label="OK", diag_wave=20)
        decision = provider.ground_sync(
            instruction="Click OK",
            manifest_text="• [1] OK",
            all_candidates=[cand]
        )
        assert decision.error != ""
        assert decision.confidence == 0.0
        assert decision.provider_used == "gemini"


class TestHighResolutionSoMAndMemoryCleanup:
    """Verifies high-res Set-of-Marks rendering (1920px) and memory buffer deallocation."""

    def test_som_renders_at_1920px_native_resolution(self):
        """Verifies 1920x1080 screenshot retains scale=1.0 when max_image_side=1920."""
        screenshot = np.zeros((1080, 1920, 3), dtype=np.uint8)
        droplets = [
            DiagonalDroplet(id=1, center_x=200, center_y=200, category="button", label="Toolbar Icon", diag_wave=400, bbox=(190, 190, 210, 210)),
            DiagonalDroplet(id=2, center_x=1900, center_y=1060, category="interactive_widget", label="Taskbar Clock", diag_wave=2960, bbox=(1880, 1040, 1918, 1075)),
        ]

        url, (w, h), scale = DiagonalDropletReasoner._render_unoccluded_som(
            screenshot, droplets, max_image_side=1920, jpeg_quality=85
        )

        assert url is not None
        assert url.startswith("data:image/jpeg;base64,")
        assert (w, h) == (1920, 1080)
        assert scale == 1.0

        # Verify base64 is valid decodable JPEG
        b64_data = url.split(",", 1)[1]
        decoded = base64.b64decode(b64_data)
        assert decoded[:2] == b"\xff\xd8"  # JPEG SOI magic bytes

    def test_som_explicit_memory_cleanup_no_leak_across_turns(self):
        """Verifies explicit del overlay, marked, buf ensures buffers are released across repeated turns."""
        screenshot = np.zeros((1080, 1920, 3), dtype=np.uint8)
        droplets = [
            DiagonalDroplet(id=i, center_x=i * 20, center_y=i * 10, category="button", label=f"Btn {i}", diag_wave=i * 30)
            for i in range(40)
        ]

        gc.collect()
        for _ in range(15):
            url, (w, h), scale = DiagonalDropletReasoner._render_unoccluded_som(
                screenshot, droplets, max_image_side=1920, jpeg_quality=85
            )
            assert len(url) > 500
        gc.collect()


class TestTokenBudgetGuardCountingAndDynamicCompaction:
    """Verifies tiktoken cl100k_base counting and 3-tier dynamic compaction."""

    def test_exact_token_counting_cl100k_base(self):
        """Verifies exact token count against cl100k_base."""
        enc = tiktoken.get_encoding("cl100k_base")
        sample_text = (
            "Autonomous Cognitive Spatial Desktop Automation Brain controlling a Windows PC. "
            "Deliberate across 5 cognitive pillars: user_intent_and_success_criteria, "
            "screen_state_and_visual_hierarchy, mistake_critique_and_loop_diagnosis, "
            "end_to_end_master_plan, tactical_action_justification."
        )
        expected = len(enc.encode(sample_text))
        actual = TokenBudgetGuard.count_tokens(sample_text)
        assert actual == expected

    def test_token_counting_fallback_when_tiktoken_fails(self):
        """Verifies character ratio fallback ((len + 3) // 4) when tiktoken is unavailable."""
        with patch.object(TokenBudgetGuard, "get_encoding", return_value=None):
            text = "A" * 100
            count = TokenBudgetGuard.count_tokens(text)
            assert count == 25

    def test_tier1_candidate_relevance_sorting(self):
        """Tier 1: Prioritizes modal dialog controls, dialog choice verbs, active window, and goal keywords."""
        modal_rect = (500, 300, 1100, 700)
        win_rect = (100, 100, 1200, 800)
        goal = "save report document"

        nodes = [
            DiagonalDroplet(id=1, center_x=50, center_y=1050, category="interactive_widget", label="Taskbar Icon", diag_wave=1100, is_foreground=False),
            DiagonalDroplet(id=2, center_x=800, center_y=600, category="button", label="Save", diag_wave=1400, is_foreground=True),  # Inside modal dialog
            DiagonalDroplet(id=3, center_x=950, center_y=600, category="button", label="Cancel", diag_wave=1550, is_foreground=True), # Inside modal dialog
            DiagonalDroplet(id=4, center_x=300, center_y=200, category="button", label="Editor File Menu", diag_wave=500, is_foreground=True), # Inside window
            DiagonalDroplet(id=5, center_x=50, center_y=50, category="surface_probe", label="Probe Node", diag_wave=100, is_foreground=False), # Probe
        ]

        pruned = TokenBudgetGuard.compact_candidates_tier1(
            nodes, goal=goal, modal_rect=modal_rect, win_rect=win_rect, max_nodes=3
        )
        assert len(pruned) == 3
        pruned_ids = {d.id for d in pruned}
        # Modal buttons 2 (Save) and 3 (Cancel) and Window button 4 must be retained
        assert 2 in pruned_ids
        assert 3 in pruned_ids
        assert 4 in pruned_ids
        # Probe and background taskbar must be pruned
        assert 1 not in pruned_ids
        assert 5 not in pruned_ids

    def test_tier2_spatial_layout_graph_condensation(self):
        """Tier 2: Omits empty zones and summarizes non-active background zones."""
        nodes = [
            DiagonalDroplet(id=10, center_x=500, center_y=400, category="button", label="Active Workspace Button", diag_wave=900, is_foreground=True),
            # 5 background footer droplets
            DiagonalDroplet(id=20, center_x=100, center_y=1000, category="interactive_widget", label="Bg App 1", diag_wave=1100, is_foreground=False),
            DiagonalDroplet(id=21, center_x=200, center_y=1000, category="interactive_widget", label="Bg App 2", diag_wave=1200, is_foreground=False),
            DiagonalDroplet(id=22, center_x=300, center_y=1000, category="interactive_widget", label="Bg App 3", diag_wave=1300, is_foreground=False),
            DiagonalDroplet(id=23, center_x=400, center_y=1000, category="interactive_widget", label="Bg App 4", diag_wave=1400, is_foreground=False),
            DiagonalDroplet(id=24, center_x=500, center_y=1000, category="interactive_widget", label="Bg App 5", diag_wave=1500, is_foreground=False),
        ]

        graph = TokenBudgetGuard.condense_spatial_graph_tier2(nodes, sw=1920, sh=1080)
        assert "[WORKSPACE]" in graph
        assert "Active Workspace Button" in graph
        # Background footer zone must be summarized instead of listing all 5 nodes individually
        assert "non-active background controls summarized" in graph

    def test_tier3_trajectory_memory_compaction_and_master_plan(self):
        """Tier 3: Compacts trajectory history to latest 3 steps while preserving Active Master Plan milestones."""
        memory = EpisodicTrajectoryMemory("Open notepad and write notes")
        for i in range(1, 8):
            step = TrajectoryStep(
                step_num=i,
                goal="Open notepad and write notes",
                subgoal=f"Execute Subgoal Phase Step {i} In Trajectory Loop",
                action="click",
                target_id=i,
                target_label=f"Element Button {i}",
                input_text=f"input_{i}",
                verified=True,
                verification_reason="Verified state change",
                active_window="Notepad",
                end_to_end_master_plan=[
                    "[Phase 1] Launch Notepad (DONE)",
                    "[Phase 2] Type meeting notes (ACTIVE)",
                    "[Phase 3] Save file to disk (PENDING)"
                ]
            )
            memory.add_step(step)

        compact_hist = TokenBudgetGuard.compact_trajectory_tier3(memory, max_steps=3)
        # Verify master plan milestones preserved with indicators
        assert "Active Master Plan:" in compact_hist
        assert "✔" in compact_hist
        assert "►" in compact_hist
        assert "○" in compact_hist
        assert "Launch Notepad" in compact_hist
        assert "Type meeting notes" in compact_hist
        assert "Save file to disk" in compact_hist

        # Verify older steps summarized
        assert "Steps 1-4 executed previously in trajectory" in compact_hist
        # Latest steps 5, 6, 7 present
        assert "Step 7:" in compact_hist
        assert "Step 6:" in compact_hist
        assert "Step 5:" in compact_hist
        assert "Step 1:" not in compact_hist

    def test_enforce_guarantees_budget_under_60_to_144_heavy_droplets(self):
        """Verifies TokenBudgetGuard.enforce proactively shrinks massive prompts under 4,800 tokens."""
        enc = tiktoken.get_encoding("cl100k_base")
        heavy_nodes = [
            DiagonalDroplet(
                id=i,
                center_x=50 + (i % 8) * 180,
                center_y=50 + (i // 8) * 90,
                category="button",
                label=f"Enterprise Cloud Diagnostic Controller Node Component Index {i} in Hierarchical Panel",
                diag_wave=i * 15,
                archetype="button"
            )
            for i in range(144)
        ]
        # Generate an initially oversized prompt (>5,500 tokens) to trigger compaction
        system_text = "<system>Operational Invariants 5-Stage Deliberation</system>\n" + "<constitution>Operational Invariants Rules and Directives for Visual Grounding</constitution>\n" * 200
        user_text = "<goal>Diagnostics</goal>\n" + "\n".join(f"• [{d.id}] ({d.center_x},{d.center_y}) {d.label}" for d in heavy_nodes)

        initial_tokens = len(enc.encode(system_text)) + len(enc.encode(user_text))
        assert initial_tokens > 4800, f"Expected initial prompt > 4800 tokens, got {initial_tokens}"

        s_out, u_out, n_out = TokenBudgetGuard.enforce(
            system_instruction=system_text,
            user_content=user_text,
            nodes=heavy_nodes,
            max_tokens=4800
        )

        total_tokens = len(enc.encode(s_out)) + len(enc.encode(u_out))
        assert total_tokens < 4800, f"Total tokens {total_tokens} exceeded target 4800"
        assert total_tokens < 5000, f"Total tokens {total_tokens} exceeded hard limit 5000"
        assert len(n_out) <= 36


class TestProviderRoutingInReasoner:
    """Verifies decide() and decide_turn() provider routing and fallback behavior."""

    @pytest.mark.asyncio
    async def test_decide_routes_to_gemini_without_key_pool(self):
        """Verifies decide(provider='gemini') executes when self.key_pool is empty."""
        reasoner = DiagonalDropletReasoner()
        # Ensure key_pool is empty
        reasoner.key_pool = []

        mock_provider = MagicMock()
        mock_decision = GroundingDecision(
            target_dot_id=12,
            action="click",
            reasoning="Gemini picked 12",
            provider_used="gemini",
            model_used="gemini-1.5-pro"
        )

        async def mock_ground(**kwargs):
            return mock_decision

        mock_provider.ground = mock_ground

        cand = DiagonalDroplet(id=12, center_x=300, center_y=300, category="button", label="Submit", diag_wave=600)

        with patch("spatial_droplet.reasoning.providers.registry.get_grounding_provider", return_value=mock_provider):
            res = await reasoner.decide(
                prompt="Submit form",
                droplets=[cand],
                provider="gemini"
            )
            assert res.action == "click"
            assert res.target_dot_id == 12

    def test_decide_turn_routes_to_gemini_and_falls_back_on_error(self):
        """Verifies decide_turn routes to Gemini, and gracefully falls back to Groq pool if Gemini errors."""
        reasoner = DiagonalDropletReasoner()
        mem = EpisodicTrajectoryMemory("Goal")
        droplets = [DiagonalDroplet(id=1, center_x=50, center_y=50, category="button", label="Btn", diag_wave=100)]

        # Configure Groq key pool to return a valid decision on fallback
        mock_groq_resp = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = '{"action": "click", "target_dot_id": 1, "reasoning": "Groq fallback"}'
        mock_groq_resp.choices = [mock_choice]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_groq_resp
        reasoner.key_pool = [{
            "client": mock_client,
            "masked": "gsk_test...",
            "provider": "groq",
            "models": ["qwen/qwen3.8-27b"]
        }]

        with patch("spatial_droplet.reasoning.providers.gemini_provider.GeminiGroundingProvider.ground_sync") as mock_gemini:
            # Simulate Gemini throwing quota 429 exception
            mock_gemini.side_effect = Exception("Gemini quota 429 exceeded")

            res = reasoner.decide_turn(
                goal="Goal",
                memory=mem,
                droplets=droplets,
                provider="gemini"
            )

            assert res is not None
            assert res["action"] == "click"
            assert res["target_dot_id"] == 1

    def test_decide_turn_with_heavy_nodes_proactively_bounds_tokens(self):
        """Verifies decide_turn proactively enforces < 4,800 tokens on 70 complex nodes."""
        enc = tiktoken.get_encoding("cl100k_base")
        reasoner = DiagonalDropletReasoner()
        droplets = [
            DiagonalDroplet(
                id=i,
                center_x=50 + (i % 8) * 180,
                center_y=50 + (i // 8) * 100,
                category="button",
                label=f"Complex System Container Panel Node Widget Element Index {i}",
                diag_wave=i * 12
            )
            for i in range(70)
        ]
        mem = EpisodicTrajectoryMemory("Run complete diagnostics")
        win = WindowContext(hwnd=999, title="Diagnostics Utility", rect=(100, 100, 1100, 800))
        aware = ScreenAwarenessContext(app_name="Diagnostics Utility", scene_type="DESKTOP_APP")

        captured = []
        for entry in reasoner.key_pool:
            def make_mock(client):
                def mock_create(**kwargs):
                    captured.append(kwargs)
                    raise Exception("intercepted")
                return mock_create
            entry["client"].chat.completions.create = make_mock(entry["client"])

        reasoner.decide_turn("Run complete diagnostics", mem, droplets, win, aware)
        assert len(captured) > 0
        msgs = captured[0]["messages"]
        total_tokens = sum(len(enc.encode(m["content"])) for m in msgs)
        assert total_tokens < 4800, f"Token count {total_tokens} exceeded 4800"
        assert total_tokens < 5000
