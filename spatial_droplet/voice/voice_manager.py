"""
Spatial Droplet Voice Manager.
Orchestrates 1-click / hotkey toggle audio capture, Auto-VAD end-of-speech detection,
AGC signal conditioning, and Groq Whisper Large V3 Turbo transcription.
Provides thread-safe callbacks for UI integration and global hotkeys (Alt+V, Ctrl+Shift+V, F8).
"""

import re
import threading
import time
from typing import Optional, Callable
import numpy as np
from .audio_recorder import AudioRecorder
from .transcriber import GroqWhisperTranscriber


class VoiceControlManager:
    """
    Manages 1-click / hotkey toggle microphone interaction (with automatic end-of-speech VAD)
    and Groq Whisper Turbo transcription.
    """

    def __init__(self, sample_rate: int = 16000):
        self.recorder = AudioRecorder(sample_rate=sample_rate, auto_vad=True, silence_timeout_sec=1.25)
        self.transcriber = GroqWhisperTranscriber()
        self.is_listening = False
        self.is_transcribing = False
        self._lock = threading.Lock()
        self._session_id = 0
        self._on_recording_state_change: Optional[Callable[[bool], None]] = None

    @property
    def is_available(self) -> bool:
        return self.recorder.is_available() and self.transcriber.is_available

    def toggle(
        self,
        on_status: Optional[Callable[[str, str], None]] = None,
        on_result: Optional[Callable[[str], None]] = None,
        on_state_change: Optional[Callable[[bool], None]] = None
    ) -> bool:
        """
        Toggles voice recording state (1-click / hotkey toggle; no holding required).
        If idle -> starts listening (stops automatically on 1.25s silence after speech, or on second toggle).
        If listening -> stops immediately and transcribes in background thread.
        Returns True if now listening, False if stopped.
        """
        with self._lock:
            if on_state_change:
                self._on_recording_state_change = on_state_change

            if self.is_listening:
                # Manual stop triggered by user click or hotkey
                self.is_listening = False
                active_sid = self._session_id
                threading.Thread(
                    target=self._process_recording,
                    args=(active_sid, on_status, on_result),
                    daemon=True
                ).start()
                return False
            else:
                if not self.recorder.is_available():
                    if on_status:
                        on_status("⚠️ No microphone input device found", "#ff1744")
                    return False

                self._session_id += 1
                active_sid = self._session_id

                def _on_vad_auto_stop():
                    with self._lock:
                        if not self.is_listening or self._session_id != active_sid:
                            return
                        self.is_listening = False
                    if self._on_recording_state_change:
                        try:
                            self._on_recording_state_change(False)
                        except Exception:
                            pass
                    self._process_recording(active_sid, on_status, on_result)

                if not self.recorder.start(on_auto_stop=_on_vad_auto_stop):
                    if on_status:
                        on_status("⚠️ Could not open microphone stream", "#ff1744")
                    return False

                self.is_listening = True
                if on_status:
                    on_status("🎙️ Listening... Speak now (Auto-sends on pause or press Alt+V)", "#00e5ff")

                # Safety max recording duration thread (25 seconds max)
                threading.Thread(
                    target=self._watchdog_timer,
                    args=(active_sid, 25.0, on_status, on_result),
                    daemon=True
                ).start()
                return True

    def _watchdog_timer(self, session_id: int, max_duration: float, on_status, on_result):
        time.sleep(max_duration)
        with self._lock:
            if not self.is_listening or self._session_id != session_id:
                return
            self.is_listening = False
        if self._on_recording_state_change:
            try:
                self._on_recording_state_change(False)
            except Exception:
                pass
        self._process_recording(session_id, on_status, on_result)

    def _process_recording(self, session_id: int = 0, on_status=None, on_result=None):
        audio_data = self.recorder.stop()
        if self._on_recording_state_change:
            try:
                self._on_recording_state_change(False)
            except Exception:
                pass
        self.process_pcm_buffer(audio_data, on_status=on_status, on_result=on_result)

    def process_pcm_buffer(
        self,
        audio_data: Optional[np.ndarray],
        on_status: Optional[Callable[[str, str], None]] = None,
        on_result: Optional[Callable[[str], None]] = None
    ) -> Optional[str]:
        """
        Validates, normalizes, transcribes, cleans, and dispatches a recorded PCM buffer.
        Returns the cleaned instruction string if valid, or None.
        """
        with self._lock:
            self.is_transcribing = True

        if on_status:
            on_status("⚡ Transcribing voice (Whisper Turbo)...", "#ffb300")

        sr = getattr(self.recorder, "sample_rate", 16000)
        if audio_data is None or len(audio_data) < int(sr * 0.25):  # Under 250ms is too short
            with self._lock:
                self.is_transcribing = False
            if on_status:
                on_status("⚠️ Audio too short or empty", "#8b949e")
            return None

        # Check energy after DC offset removal
        try:
            pcm_float = audio_data.astype(np.float64)
            pcm_float = pcm_float - np.mean(pcm_float)
            rms = float(np.sqrt(np.mean(pcm_float ** 2)))
            peak = float(np.max(np.abs(pcm_float)))
            if rms < 25.0 and peak < 140.0:
                with self._lock:
                    self.is_transcribing = False
                if on_status:
                    on_status("⚠️ No speech detected (silence)", "#8b949e")
                return None
        except Exception:
            pass

        # Transcribe with Groq Whisper (AGC-normalized, English)
        text = self.transcriber.transcribe_pcm(audio_data, sample_rate=sr, language="en")

        with self._lock:
            self.is_transcribing = False

        if text and text.strip():
            clean_text = text.strip()
            # Strip extraneous outer quotes, brackets, and trailing sentence punctuation
            clean_text = re.sub(r'^[“"\'`\(\[]+|[”"\'`\)\]]+$', '', clean_text).strip()
            clean_text = re.sub(r'[\s.,!?]+$', '', clean_text).strip()

            # Filter out common Whisper hallucinated filler tokens for silence / background
            HALLUCINATIONS = {
                "thank you", "thank you.", "thank you very much", "thanks for watching",
                "thanks for watching!", "thanks for watching.", "please subscribe",
                "subscribe", "subtitles by", "transcript by", "you", "bye", "bye bye",
                "amara.org", "mbc", "silence", "silence.", "the end", "the end.",
                "like and subscribe", "peace", "translated by", "music", "applause"
            }
            lower = clean_text.lower().rstrip(".,!?")
            if lower in HALLUCINATIONS or re.match(r'^(?:\[.*?\]|\(.*?\))$', clean_text):
                if on_status:
                    on_status("⚠️ No speech detected", "#8b949e")
                return None

            if len(clean_text) < 2:
                if on_status:
                    on_status("⚠️ Audio unclear", "#8b949e")
                return None

            if on_status:
                on_status(f"✔ Voice: \"{clean_text[:32]}\"", "#3fb950")
            if on_result:
                on_result(clean_text)
            return clean_text
        else:
            if on_status:
                on_status("⚠️ Could not transcribe audio", "#ff1744")
            return None
