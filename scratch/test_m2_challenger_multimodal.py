"""
Milestone 2 Empirical Adversarial Challenge Test Suite:
Multimodal Reasoning & Memory Safety Verifier.

Tests:
1. _render_unoccluded_som:
   - 1920x1080 native frames with 50+ diverse droplets.
   - 4K (3840x2160) frames downsampled to 1920px (scale=0.5, font_scale=0.38).
   - Ultra-wide (3840x1080) frames downsampled to max_image_side=1920.
   - 50 consecutive render calls on 4K frames: verify OpenCV buffer release and memory bounds.
   - Edge cases: None frame, 0x0 frame, empty droplets, negative bboxes.
2. GeminiGroundingProvider:
   - Missing, empty, and whitespace API keys.
   - Malformed API key with 401 AuthenticationError.
   - 429 Quota Exceeded exceptions.
   - Timeout / connection exceptions.
   - Response format fallback retry.
3. DiagonalDropletReasoner Routing & Fallback:
   - decide_turn() with Gemini 429 quota failure falling back to Groq pool.
   - decide_turn() with Gemini timeout falling back to Groq pool.
   - decide_turn() with missing Gemini API key falling back to Groq pool.
   - decide() async with Gemini success routing.
   - decide() async with Gemini failure: verify fallback to Groq pool.
   - decide_turn() with empty key_pool when Gemini fails: verify no crash.
4. Dual Sync / Async Execution & Deadlock Safety:
   - ground_sync in purely synchronous code.
   - ground async in asyncio event loop.
   - Concurrent async ground calls without race conditions.
"""
import pytest
import os
import sys
import gc
import tracemalloc
import asyncio
import base64
import numpy as np
import cv2
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath("."))

from spatial_droplet.reasoning.reasoner import DiagonalDropletReasoner, DecisionResult
from spatial_droplet.reasoning.providers.base import BaseGroundingProvider, GroundingDecision
from spatial_droplet.reasoning.providers.gemini_provider import GeminiGroundingProvider
from spatial_droplet.reasoning.providers.registry import get_grounding_provider
from spatial_droplet.core.models import (
    DiagonalDroplet,
    WindowContext,
    ScreenAwarenessContext,
    TrajectoryStep,
)
from spatial_droplet.core.memory import EpisodicTrajectoryMemory


def generate_synthetic_droplets(count: int, max_x: int = 1920, max_y: int = 1080):
    """Generates `count` diverse droplets with different archetypes and coordinates."""
    archetypes = ["button", "input", "icon", "label", "logo", "container", "surface_probe"]
    droplets = []
    for i in range(count):
        cx = int((i * 37) % (max_x - 40)) + 20
        cy = int((i * 47) % (max_y - 40)) + 20
        arch = archetypes[i % len(archetypes)]
        w = 30 + (i % 5) * 20
        h = 20 + (i % 4) * 10
        bbox = (max(0, cx - w // 2), max(0, cy - h // 2), min(max_x, cx + w // 2), min(max_y, cy + h // 2))
        droplets.append(
            DiagonalDroplet(
                id=i + 1,
                center_x=cx,
                center_y=cy,
                category=arch,
                label=f"Elem_{i+1}_{arch}",
                diag_wave=i * 10,
                archetype=arch,
                bbox=bbox
            )
        )
    return droplets


# ==============================================================================
# 1. _render_unoccluded_som Memory, Resolution & Scaling Tests
# ==============================================================================
class TestRenderUnoccludedSomAdversarial:
    """Stress tests _render_unoccluded_som across native, 4K, ultrawide resolutions and 50-call memory runs."""

    def test_render_1080p_native_resolution_with_50_droplets(self):
        """Verifies 1920x1080 native frame renders without downsampling (scale=1.0) and produces valid data URL."""
        frame = np.full((1080, 1920, 3), 240, dtype=np.uint8)
        droplets = generate_synthetic_droplets(60, 1920, 1080)

        url, (w, h), scale = DiagonalDropletReasoner._render_unoccluded_som(
            frame, droplets, max_image_side=1920, jpeg_quality=85
        )

        assert url is not None
        assert url.startswith("data:image/jpeg;base64,")
        assert (w, h) == (1920, 1080)
        assert scale == 1.0

        b64_data = url.split(",", 1)[1]
        decoded = cv2.imdecode(np.frombuffer(base64.b64decode(b64_data), np.uint8), cv2.IMREAD_COLOR)
        assert decoded is not None
        assert decoded.shape == (1080, 1920, 3)

    def test_render_4k_downsampled_to_1920px(self):
        """Verifies 3840x2160 4K frame is cleanly downsampled to 1920x1080 (scale=0.5)."""
        frame = np.full((2160, 3840, 3), 200, dtype=np.uint8)
        droplets = generate_synthetic_droplets(75, 3840, 2160)

        url, (w, h), scale = DiagonalDropletReasoner._render_unoccluded_som(
            frame, droplets, max_image_side=1920, jpeg_quality=85
        )

        assert url is not None
        assert url.startswith("data:image/jpeg;base64,")
        assert (w, h) == (1920, 1080)
        assert scale == 0.5

        b64_data = url.split(",", 1)[1]
        decoded = cv2.imdecode(np.frombuffer(base64.b64decode(b64_data), np.uint8), cv2.IMREAD_COLOR)
        assert decoded is not None
        assert decoded.shape == (1080, 1920, 3)

    def test_render_ultrawide_downsampled_to_1920px(self):
        """Verifies 3840x1080 ultra-wide frame is downsampled based on longest side (3840 -> 1920, 1080 -> 540)."""
        frame = np.full((1080, 3840, 3), 180, dtype=np.uint8)
        droplets = generate_synthetic_droplets(50, 3840, 1080)

        url, (w, h), scale = DiagonalDropletReasoner._render_unoccluded_som(
            frame, droplets, max_image_side=1920, jpeg_quality=80
        )

        assert url is not None
        assert (w, h) == (1920, 540)
        assert scale == 0.5

    def test_render_50_consecutive_calls_memory_safety(self):
        """Verifies 50 consecutive calls with 4K frames do NOT leak memory and remain bounded."""
        frame = np.random.randint(0, 255, (2160, 3840, 3), dtype=np.uint8)
        droplets = generate_synthetic_droplets(60, 3840, 2160)

        # Warmup call
        warmup_url, _, _ = DiagonalDropletReasoner._render_unoccluded_som(
            frame, droplets, max_image_side=1920, jpeg_quality=85
        )
        del warmup_url

        gc.collect()
        tracemalloc.start()
        snapshot_start = tracemalloc.take_snapshot()

        for _ in range(50):
            url, dims, scale = DiagonalDropletReasoner._render_unoccluded_som(
                frame, droplets, max_image_side=1920, jpeg_quality=85
            )
            assert url is not None
            assert dims == (1920, 1080)
            del url

        gc.collect()
        snapshot_end = tracemalloc.take_snapshot()
        top_stats = snapshot_end.compare_to(snapshot_start, 'lineno')
        tracemalloc.stop()

        # Net memory growth across 50 4K renders must remain < 500 KB (actual observed is < 5 KB).
        total_growth = sum(stat.size_diff for stat in top_stats if stat.size_diff > 0)
        max_allowed_growth_bytes = 500 * 1024  # 500 KB ceiling
        assert total_growth < max_allowed_growth_bytes, (
            f"Potential memory leak detected: net memory growth was {total_growth / 1024:.2f} KB"
        )

    def test_render_edge_cases_and_malformed_inputs(self):
        """Verifies graceful handling of None frame, invalid dimensions, empty droplets, and out-of-bound coords."""
        # None frame
        url, dims, scale = DiagonalDropletReasoner._render_unoccluded_som(None, [])
        assert url is None
        assert dims == (0, 0)

        # 1D or invalid frame
        url, dims, scale = DiagonalDropletReasoner._render_unoccluded_som(np.array([1, 2, 3]), [])
        assert url is None
        assert dims == (0, 0)

        # Frame with empty droplets
        valid_frame = np.full((100, 100, 3), 128, dtype=np.uint8)
        url, dims, scale = DiagonalDropletReasoner._render_unoccluded_som(valid_frame, [])
        assert url is not None
        assert dims == (100, 100)

        # Droplet with negative and out-of-bounds coordinates
        bad_droplet = DiagonalDroplet(
            id=999,
            center_x=-50,
            center_y=2000,
            category="button",
            label="OOB",
            diag_wave=999,
            bbox=(-100, -100, 5000, 5000)
        )
        url, dims, scale = DiagonalDropletReasoner._render_unoccluded_som(valid_frame, [bad_droplet])
        assert url is not None

        # Degenerate 0x0 frame: should return None or (0,0) without throwing unhandled cv2.error
        zero_frame = np.zeros((0, 0, 3), dtype=np.uint8)
        try:
            url, dims, scale = DiagonalDropletReasoner._render_unoccluded_som(zero_frame, [])
            assert url is None or dims == (0, 0)
        except Exception as e:
            pytest.fail(f"EMPIRICAL DEFECT: _render_unoccluded_som raised unhandled exception on 0x0 frame: {type(e).__name__}: {e}")


# ==============================================================================
# 2. GeminiGroundingProvider Error & Quota Handling
# ==============================================================================
class TestGeminiProviderErrorHandling:
    """Stress tests Gemini provider against missing keys, 401s, 429 quota exceptions, and timeouts."""

    def test_missing_or_blank_api_key_returns_graceful_error(self, monkeypatch):
        """Verifies that missing or blank API key returns GroundingDecision with error, not an unhandled exception."""
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        provider = GeminiGroundingProvider(api_key="   ")
        assert not provider.is_available()

        res = provider.ground_sync(
            instruction="Click button",
            manifest_text="[1] button",
            all_candidates=[]
        )
        assert res is not None
        assert res.error is not None
        assert "not initialized" in res.error.lower()
        assert res.confidence == 0.0

    def test_401_authentication_error_handled_gracefully(self):
        """Verifies that a 401 authentication exception is trapped and returned as GroundingDecision(error=...)."""
        provider = GeminiGroundingProvider(api_key="invalid_bad_key_123")
        assert provider.is_available()

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("401 Unauthorized: API key not valid")
        provider.client = mock_client

        res = provider.ground_sync(
            instruction="Click submit",
            manifest_text="[1] button",
            all_candidates=[]
        )
        assert res is not None
        assert res.error is not None
        assert "401" in res.error
        assert res.confidence == 0.0

    def test_429_quota_exceeded_handled_gracefully(self):
        """Verifies that a 429 RateLimit / Quota Exceeded exception is trapped and returned as GroundingDecision(error=...)."""
        provider = GeminiGroundingProvider(api_key="gemini_test_key")
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("429 Resource has been exhausted (quota exceeded)")
        provider.client = mock_client

        res = provider.ground_sync(
            instruction="Search for files",
            manifest_text="[1] input Search",
            all_candidates=[]
        )
        assert res is not None
        assert res.error is not None
        assert "429" in res.error
        assert "quota" in res.error.lower()
        assert res.confidence == 0.0

    def test_timeout_error_handled_gracefully(self):
        """Verifies that an API timeout exception is trapped and returned as GroundingDecision(error=...)."""
        provider = GeminiGroundingProvider(api_key="gemini_test_key")
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("Request timed out after 60.0 seconds")
        provider.client = mock_client

        res = provider.ground_sync(
            instruction="Save file",
            manifest_text="[1] button Save",
            all_candidates=[]
        )
        assert res is not None
        assert res.error is not None
        assert "timed out" in res.error.lower()
        assert res.confidence == 0.0

    def test_response_format_rejection_fallback(self):
        """Verifies that if response_format={'type': 'json_object'} is rejected, it retries without response_format."""
        provider = GeminiGroundingProvider(api_key="gemini_test_key")
        mock_client = MagicMock()

        mock_resp = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = '{"action": "click", "target_dot_id": 5, "reasoning": "Fallback success"}'
        mock_resp.choices = [mock_choice]

        mock_client.chat.completions.create.side_effect = [
            Exception("Unsupported parameter: response_format"),
            mock_resp
        ]
        provider.client = mock_client

        cand = DiagonalDroplet(id=5, center_x=100, center_y=100, category="button", label="OK", diag_wave=50)
        res = provider.ground_sync(
            instruction="Click OK",
            manifest_text="[5] button OK",
            all_candidates=[cand]
        )

        assert res.error is None or res.error == ""
        assert res.target_dot_id == 5
        assert res.action == "click"
        assert mock_client.chat.completions.create.call_count == 2


# ==============================================================================
# 3. DiagonalDropletReasoner Fallback & Quota Resilience
# ==============================================================================
class TestReasonerProviderRoutingAndFallback:
    """Stress tests DiagonalDropletReasoner fallback mechanisms when Gemini fails."""

    def test_decide_turn_gemini_429_falls_back_to_groq_pool(self):
        """Verifies decide_turn catches Gemini 429 quota exception and falls back to Groq pool seamlessly."""
        reasoner = DiagonalDropletReasoner()
        mem = EpisodicTrajectoryMemory("Goal")
        droplet = DiagonalDroplet(id=1, center_x=100, center_y=100, category="button", label="Submit", diag_wave=100)

        mock_groq_resp = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = '{"action": "click", "target_dot_id": 1, "reasoning": "Groq fallback picked 1"}'
        mock_groq_resp.choices = [mock_choice]

        mock_groq_client = MagicMock()
        mock_groq_client.chat.completions.create.return_value = mock_groq_resp

        reasoner.key_pool = [{
            "client": mock_groq_client,
            "masked": "gsk_test123",
            "provider": "groq",
            "models": ["qwen/qwen3.8-27b"]
        }]

        with patch("spatial_droplet.reasoning.providers.gemini_provider.GeminiGroundingProvider.ground_sync") as mock_gemini:
            mock_gemini.side_effect = Exception("429 Resource has been exhausted (quota exceeded)")

            res = reasoner.decide_turn(
                goal="Submit form",
                memory=mem,
                droplets=[droplet],
                provider="gemini"
            )

            assert res is not None
            assert res.get("action") == "click"
            assert res.get("target_dot_id") == 1
            assert mock_groq_client.chat.completions.create.called

    def test_decide_turn_gemini_timeout_falls_back_to_groq_pool(self):
        """Verifies decide_turn catches Gemini timeout and falls back to Groq pool seamlessly."""
        reasoner = DiagonalDropletReasoner()
        mem = EpisodicTrajectoryMemory("Goal")
        droplet = DiagonalDroplet(id=2, center_x=120, center_y=120, category="button", label="Cancel", diag_wave=200)

        mock_groq_resp = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = '{"action": "click", "target_dot_id": 2, "reasoning": "Groq fallback picked 2"}'
        mock_groq_resp.choices = [mock_choice]

        mock_groq_client = MagicMock()
        mock_groq_client.chat.completions.create.return_value = mock_groq_resp

        reasoner.key_pool = [{
            "client": mock_groq_client,
            "masked": "gsk_test456",
            "provider": "groq",
            "models": ["qwen/qwen3.8-27b"]
        }]

        with patch("spatial_droplet.reasoning.providers.gemini_provider.GeminiGroundingProvider.ground_sync") as mock_gemini:
            mock_gemini.side_effect = Exception("Connection timed out after 60s")

            res = reasoner.decide_turn(
                goal="Cancel operation",
                memory=mem,
                droplets=[droplet],
                provider="gemini"
            )

            assert res is not None
            assert res.get("action") == "click"
            assert res.get("target_dot_id") == 2

    def test_decide_turn_unavailable_gemini_falls_back_to_groq_pool(self, monkeypatch):
        """Verifies decide_turn when Gemini has no API key cleanly skips to Groq pool without exception."""
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        reasoner = DiagonalDropletReasoner()
        mem = EpisodicTrajectoryMemory("Goal")
        droplet = DiagonalDroplet(id=3, center_x=150, center_y=150, category="button", label="Next", diag_wave=300)

        mock_groq_resp = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = '{"action": "click", "target_dot_id": 3, "reasoning": "Groq fallback picked 3"}'
        mock_groq_resp.choices = [mock_choice]

        mock_groq_client = MagicMock()
        mock_groq_client.chat.completions.create.return_value = mock_groq_resp

        reasoner.key_pool = [{
            "client": mock_groq_client,
            "masked": "gsk_test789",
            "provider": "groq",
            "models": ["qwen/qwen3.8-27b"]
        }]

        res = reasoner.decide_turn(
            goal="Go to next page",
            memory=mem,
            droplets=[droplet],
            provider="gemini"
        )

        assert res is not None
        assert res.get("action") == "click"
        assert res.get("target_dot_id") == 3

    @pytest.mark.asyncio
    async def test_decide_async_gemini_success_routing(self):
        """Verifies decide() async routes to Gemini and produces DecisionResult."""
        reasoner = DiagonalDropletReasoner()
        droplet = DiagonalDroplet(id=7, center_x=200, center_y=200, category="button", label="Apply", diag_wave=700)

        mock_provider = MagicMock()
        mock_decision = GroundingDecision(
            target_dot_id=7,
            action="click",
            reasoning="Gemini async success",
            provider_used="gemini",
            model_used="gemini-1.5-pro"
        )

        async def mock_ground(**kwargs):
            return mock_decision

        mock_provider.ground = mock_ground

        with patch("spatial_droplet.reasoning.providers.registry.get_grounding_provider", return_value=mock_provider):
            res = await reasoner.decide(
                prompt="Apply settings",
                droplets=[droplet],
                provider="gemini"
            )
            assert res is not None
            assert res.action == "click"
            assert res.target_dot_id == 7

    @pytest.mark.asyncio
    async def test_decide_async_gemini_failure_fallback_to_groq_pool(self):
        """
        Adversarial test: When decide() is called with provider='gemini' and Gemini fails,
        verify that it falls back to the configured Groq pool instead of refusing fallback.
        """
        reasoner = DiagonalDropletReasoner()
        droplet = DiagonalDroplet(id=8, center_x=200, center_y=200, category="button", label="Apply", diag_wave=800)

        # Configure Groq client in reasoner.key_pool
        mock_groq_resp = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = '{"action": "click", "target_dot_id": 8, "reasoning": "Groq fallback picked 8"}'
        mock_groq_resp.choices = [mock_choice]
        mock_groq_client = MagicMock()
        mock_groq_client.chat.completions.create.return_value = mock_groq_resp

        reasoner.key_pool = [{
            "client": mock_groq_client,
            "masked": "gsk_test123",
            "provider": "groq",
            "models": ["qwen/qwen3.8-27b"]
        }]

        mock_provider = MagicMock()
        async def mock_ground_fail(**kwargs):
            return GroundingDecision(error="Gemini 429 quota error", confidence=0.0)

        mock_provider.ground = mock_ground_fail

        with patch("spatial_droplet.reasoning.providers.registry.get_grounding_provider", return_value=mock_provider):
            res = await reasoner.decide(
                prompt="Apply settings",
                droplets=[droplet],
                provider="gemini"
            )
            # The reasoner should fall back to Groq pool
            if res.error and "No configured clients" in str(res.error):
                pytest.fail(
                    f"EMPIRICAL DEFECT: decide() refused fallback to Groq pool when Gemini failed: "
                    f"error='{res.error}', reasoning='{res.reasoning}'"
                )
            assert res.action == "click"
            assert res.target_dot_id == 8

    def test_decide_turn_empty_key_pool_with_gemini_error_does_not_crash(self):
        """
        Adversarial test: If key_pool is empty and Gemini fails, decide_turn must NOT crash with
        ZeroDivisionError, but return a clean error dictionary.
        """
        reasoner = DiagonalDropletReasoner()
        reasoner.key_pool = []  # Empty pool
        mem = EpisodicTrajectoryMemory("Goal")
        droplet = DiagonalDroplet(id=1, center_x=100, center_y=100, category="button", label="Submit", diag_wave=100)

        with patch("spatial_droplet.reasoning.providers.gemini_provider.GeminiGroundingProvider.ground_sync") as mock_gemini:
            mock_gemini.side_effect = Exception("Gemini failed")

            try:
                res = reasoner.decide_turn(
                    goal="Submit form",
                    memory=mem,
                    droplets=[droplet],
                    provider="gemini"
                )
                assert isinstance(res, dict)
            except ZeroDivisionError as zerr:
                pytest.fail(f"EMPIRICAL DEFECT: ZeroDivisionError when key_pool is empty and Gemini fails: {zerr}")


# ==============================================================================
# 4. Dual Sync/Async Execution & Deadlock Safety
# ==============================================================================
class TestDualSyncAsyncExecution:
    """Verifies that ground_sync and ground execute cleanly without deadlocks or thread-blocking issues."""

    def test_ground_sync_in_sync_context(self):
        """Verifies ground_sync executes in a standard synchronous thread."""
        provider = GeminiGroundingProvider(api_key="test_sync_key")
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = '{"action": "click", "target_dot_id": 10, "reasoning": "Sync ok"}'
        mock_resp.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_resp
        provider.client = mock_client

        cand = DiagonalDroplet(id=10, center_x=100, center_y=100, category="button", label="OK", diag_wave=100)
        res = provider.ground_sync("Click OK", "[10] button OK", [cand])
        assert res.target_dot_id == 10
        assert res.action == "click"

    @pytest.mark.asyncio
    async def test_ground_async_in_event_loop(self):
        """Verifies ground() coroutine executes cleanly inside an asyncio event loop."""
        provider = GeminiGroundingProvider(api_key="test_async_key")
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = '{"action": "type_and_enter", "target_dot_id": 20, "reasoning": "Async ok"}'
        mock_resp.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_resp
        provider.client = mock_client

        cand = DiagonalDroplet(id=20, center_x=100, center_y=100, category="input", label="Search", diag_wave=200)
        res = await provider.ground("Type hello", "[20] input Search", [cand])
        assert res.target_dot_id == 20
        assert res.action == "type_and_enter"

    @pytest.mark.asyncio
    async def test_concurrent_async_ground_calls_without_race_conditions(self):
        """Verifies that 10 concurrent async calls to provider.ground() run cleanly without interfering."""
        provider = GeminiGroundingProvider(api_key="test_concurrent_key")
        mock_client = MagicMock()

        def make_resp(dot_id):
            r = MagicMock()
            c = MagicMock()
            c.message.content = f'{{"action": "click", "target_dot_id": {dot_id}, "reasoning": "Decision {dot_id}"}}'
            r.choices = [c]
            return r

        mock_client.chat.completions.create.side_effect = [make_resp(i) for i in range(10)]
        provider.client = mock_client

        async def worker(dot_id):
            cand = DiagonalDroplet(id=dot_id, center_x=100, center_y=100, category="button", label=f"B{dot_id}", diag_wave=dot_id * 10)
            return await provider.ground(f"Click B{dot_id}", f"[{dot_id}] button", [cand])

        tasks = [worker(i) for i in range(10)]
        results = await asyncio.gather(*tasks)

        assert len(results) == 10
        for i, res in enumerate(results):
            assert res.target_dot_id == i
            assert res.action == "click"
