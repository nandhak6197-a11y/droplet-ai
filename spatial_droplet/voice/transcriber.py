"""
Groq Whisper Speech-to-Text Transcriber with Automatic Gain Control (AGC),
Active Speech Silence Trimming, and Non-Hallucinating Vocabulary Conditioning.
Rotates requests across the 25 Groq API keys using 'whisper-large-v3-turbo'.
"""

import io
import os
import wave
import tempfile
import subprocess
import threading
from typing import Optional, List, Dict, Any
import numpy as np

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

from ..reasoning.groq_rotator import GROQ_API_KEYS


# Concise lexical vocabulary hint (avoids full-sentence continuation hallucinations)
DEFAULT_DESKTOP_PROMPT = (
    "Chrome, Camera, Notepad, VS Code, Calculator, Explorer, Terminal, Settings, "
    "open, launch, switch, click, search, type, close, minimize, maximize, scroll."
)


class GroqWhisperTranscriber:
    """
    Sub-second speech-to-text transcriber using Groq Whisper Large V3 Turbo.
    Features:
      - DC offset removal & active speech envelope trimming
      - Automatic Gain Control (AGC) normalizing quiet microphone speech to -3 dBFS
      - Thread-safe rotation across 25 Groq keys with automatic failover
    """

    def __init__(self, keys: Optional[List[str]] = None, model: str = "whisper-large-v3-turbo"):
        self.keys = keys or GROQ_API_KEYS
        self.model = model
        self.clients: List[Dict[str, Any]] = []
        self.current_idx = 0
        self.lock = threading.Lock()

        if HAS_OPENAI:
            for k in self.keys:
                try:
                    c = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=k, timeout=12.0)
                    masked = f"{k[:8]}...{k[-6:]}"
                    self.clients.append({"key": k, "client": c, "masked": masked})
                except Exception:
                    pass

    @property
    def is_available(self) -> bool:
        return bool(self.clients)

    @staticmethod
    def preprocess_audio_pcm(audio_data: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """
        Applies DC-offset removal, leading/trailing dead-air trimming (with 180ms padding),
        and Automatic Gain Control (AGC peak normalization to -3 dBFS) for crystal-clear Whisper input.
        """
        if audio_data is None or len(audio_data) == 0:
            return np.zeros(0, dtype=np.int16)

        if audio_data.dtype in (np.float32, np.float64):
            pcm = np.clip(audio_data * 32767.0, -32768.0, 32767.0).astype(np.float32)
        else:
            pcm = audio_data.astype(np.float32)

        # 1. Remove DC offset
        pcm = pcm - float(np.mean(pcm))

        # 2. Envelope silence trimming (keep 180ms margin on each side of active speech)
        win_size = max(1, int(sample_rate * 0.02))  # 20ms window
        if len(pcm) > win_size * 4:
            abs_pcm = np.abs(pcm)
            peak_val = float(np.max(abs_pcm))
            if peak_val > 120.0:
                env_thresh = max(80.0, peak_val * 0.08)
                active_indices = np.where(abs_pcm >= env_thresh)[0]
                if len(active_indices) > 0:
                    pad = int(sample_rate * 0.18)
                    start_i = max(0, int(active_indices[0]) - pad)
                    end_i = min(len(pcm), int(active_indices[-1]) + pad)
                    if (end_i - start_i) >= int(sample_rate * 0.25):
                        pcm = pcm[start_i:end_i]

        # 3. Automatic Gain Control (AGC): normalize peak to 23,000 (-3 dBFS), max boost 14x
        peak = float(np.max(np.abs(pcm))) if len(pcm) > 0 else 0.0
        if 100.0 <= peak < 20000.0:
            gain = min(14.0, 23000.0 / peak)
            pcm = np.clip(pcm * gain, -32767.0, 32767.0)

        return pcm.astype(np.int16)

    def transcribe_pcm(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 16000,
        language: Optional[str] = "en",
        prompt: Optional[str] = None
    ) -> Optional[str]:
        """
        Preprocesses PCM samples with AGC, encodes to in-memory WAV, and dispatches to Groq Whisper API.
        """
        if not self.is_available or audio_data is None or len(audio_data) == 0:
            return None

        audio_int16 = self.preprocess_audio_pcm(audio_data, sample_rate=sample_rate)
        if len(audio_int16) == 0:
            return None

        # Build in-memory WAV buffer
        wav_buf = io.BytesIO()
        wav_buf.name = "audio.wav"
        with wave.open(wav_buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(audio_int16.tobytes())
        wav_buf.seek(0)

        num_clients = len(self.clients)
        with self.lock:
            start_idx = self.current_idx
            self.current_idx = (self.current_idx + 1) % num_clients

        last_err = ""
        effective_prompt = prompt if prompt is not None else DEFAULT_DESKTOP_PROMPT
        for offset in range(num_clients):
            active_idx = (start_idx + offset) % num_clients
            entry = self.clients[active_idx]
            client = entry["client"]
            masked = entry["masked"]

            try:
                wav_buf.seek(0)
                kwargs = {
                    "model": self.model,
                    "file": wav_buf,
                    "temperature": 0.0,
                    "response_format": "text",
                    "prompt": effective_prompt
                }
                if language:
                    kwargs["language"] = language

                resp = client.audio.transcriptions.create(**kwargs)
                text = resp if isinstance(resp, str) else getattr(resp, "text", "")
                text = text.strip()
                if text:
                    print(f"[*] Voice transcribed via Groq Key #{active_idx + 1} ({masked}): \"{text}\"")
                    return text
            except Exception as e:
                last_err = str(e)
                continue

        print(f"[!] Groq Whisper transcription failed across all keys: {last_err}")
        return None

    @staticmethod
    def synthesize_speech_pcm(text: str, sample_rate: int = 16000) -> Optional[np.ndarray]:
        """
        Generates a real spoken English 16kHz mono int16 PCM waveform for automated testing
        using Windows native SAPI.SpVoice / System.Speech synthesizer.
        """
        tmp_path = None
        try:
            fd, tmp_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            safe_txt = text.replace("'", " ").replace('"', " ")
            ps_cmd = (
                "Add-Type -AssemblyName System.Speech; "
                "$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                "$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo("
                f"{sample_rate}, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, "
                "[System.Speech.AudioFormat.AudioChannel]::Mono); "
                f"$synth.SetOutputToWaveFile('{tmp_path}', $fmt); "
                f"$synth.Speak('{safe_txt}'); "
                "$synth.Dispose();"
            )
            subprocess.run(
                ["powershell", "-NoProfile", "-Command", ps_cmd],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10.0
            )
            with wave.open(tmp_path, "rb") as wf:
                raw_bytes = wf.readframes(wf.getnframes())
                pcm = np.frombuffer(raw_bytes, dtype=np.int16).copy()
            return pcm
        except Exception as e:
            print(f"[!] Speech synthesis error: {e}")
            return None
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
