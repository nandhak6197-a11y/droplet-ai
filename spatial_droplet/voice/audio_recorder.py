"""
Low-Latency Audio Recorder using sounddevice with Auto-VAD (Voice Activity Detection)
and automatic hardware sample-rate resampling to 16kHz mono PCM.
Supports 1-click / hotkey toggle recording (no holding required) with automatic
end-of-speech detection after 1.25s of trailing silence.
"""

import threading
import time
from typing import Optional, Callable, List
import numpy as np

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    HAS_SOUNDDEVICE = False


class AudioRecorder:
    """
    Non-blocking microphone recorder capturing 16kHz mono int16 PCM.
    Features:
      - Automatic fallback to native hardware sample rate (44.1kHz / 48kHz) with linear resampling to 16kHz
      - Real-time adaptive Voice Activity Detection (Auto-VAD): detects speech onset and automatically
        triggers stop callback after 1.25s of trailing silence so users never need to hold the mic button
      - Thread-safe start, stop, live RMS metering, and buffer extraction
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        auto_vad: bool = True,
        silence_timeout_sec: float = 1.25
    ):
        self.target_sample_rate = sample_rate
        self.sample_rate = sample_rate
        self._hw_sample_rate = sample_rate
        self.channels = channels
        self.auto_vad = auto_vad
        self.silence_timeout_sec = silence_timeout_sec

        self.is_recording = False
        self._stream: Optional["sd.InputStream"] = None
        self._frames: List[np.ndarray] = []
        self._lock = threading.Lock()
        self._start_time = 0.0

        # Live VAD state
        self.current_rms: float = 0.0
        self._noise_floor_rms: float = 120.0
        self._speech_detected: bool = False
        self._speech_frames_count: int = 0
        self._last_speech_time: float = 0.0
        self._auto_stop_triggered: bool = False
        self._on_auto_stop: Optional[Callable[[], None]] = None

    @classmethod
    def is_available(cls) -> bool:
        if not HAS_SOUNDDEVICE:
            return False
        try:
            dev = sd.query_devices(kind="input")
            return bool(dev and dev.get("max_input_channels", 0) > 0)
        except Exception:
            return False

    def start(self, on_auto_stop: Optional[Callable[[], None]] = None) -> bool:
        """Starts microphone recording in a non-blocking stream with optional Auto-VAD callback."""
        if not HAS_SOUNDDEVICE:
            print("[!] sounddevice is not installed or available.")
            return False

        with self._lock:
            if self.is_recording:
                return True
            self._frames = []
            self.is_recording = True
            self._start_time = time.time()
            self._speech_detected = False
            self._speech_frames_count = 0
            self._last_speech_time = time.time()
            self._auto_stop_triggered = False
            self._on_auto_stop = on_auto_stop
            self.current_rms = 0.0

        def _audio_callback(indata, frames, time_info, status):
            now = time.time()
            should_auto_stop = False
            with self._lock:
                if not self.is_recording:
                    return
                chunk = indata.copy()
                if chunk.ndim > 1 and chunk.shape[1] > 1:
                    chunk = np.mean(chunk, axis=1, keepdims=True).astype(np.int16)
                self._frames.append(chunk)

                # Compute live RMS energy for VAD and UI metering
                pcm_f = chunk.astype(np.float32)
                pcm_f = pcm_f - np.mean(pcm_f)
                rms = float(np.sqrt(np.mean(pcm_f ** 2))) if pcm_f.size > 0 else 0.0
                peak = float(np.max(np.abs(pcm_f))) if pcm_f.size > 0 else 0.0
                self.current_rms = rms

                # Adaptive noise floor estimation during initial 250ms
                elapsed = now - self._start_time
                if elapsed < 0.25 and not self._speech_detected:
                    self._noise_floor_rms = max(45.0, min(400.0, 0.7 * self._noise_floor_rms + 0.3 * rms))

                speech_thresh = max(160.0, self._noise_floor_rms * 2.1)
                if rms >= speech_thresh or peak >= max(550.0, speech_thresh * 3.2):
                    self._speech_frames_count += 1
                    self._last_speech_time = now
                    if self._speech_frames_count >= 2:
                        self._speech_detected = True
                else:
                    if self._speech_detected and self.auto_vad and not self._auto_stop_triggered:
                        # Auto-stop after silence_timeout_sec of trailing silence (minimum 0.7s total clip)
                        if elapsed >= 0.70 and (now - self._last_speech_time) >= self.silence_timeout_sec:
                            self._auto_stop_triggered = True
                            should_auto_stop = True

            if should_auto_stop and self._on_auto_stop:
                threading.Thread(target=self._on_auto_stop, daemon=True).start()

        # Try opening at target 16kHz first; fallback to hardware native sample rate if needed
        candidate_rates = [self.target_sample_rate]
        try:
            dev_info = sd.query_devices(kind="input")
            native_sr = int(dev_info.get("default_samplerate", 44100))
            if native_sr not in candidate_rates:
                candidate_rates.append(native_sr)
        except Exception:
            candidate_rates.extend([48000, 44100])

        for sr in candidate_rates:
            try:
                block_sz = int(sr * 0.064)  # ~64ms blocks
                self._stream = sd.InputStream(
                    samplerate=sr,
                    channels=1,
                    dtype="int16",
                    callback=_audio_callback,
                    blocksize=block_sz
                )
                self._stream.start()
                self._hw_sample_rate = sr
                return True
            except Exception:
                continue

        print("[!] Failed to open microphone input stream at any sample rate.")
        with self._lock:
            self.is_recording = False
        return False

    def stop(self) -> Optional[np.ndarray]:
        """Stops recording, resamples to 16kHz if needed, and returns mono int16 PCM array."""
        with self._lock:
            if not self.is_recording:
                return None
            self.is_recording = False
            self.current_rms = 0.0

        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

        with self._lock:
            if not self._frames:
                return None
            full_audio = np.concatenate(self._frames, axis=0).flatten()
            self._frames = []
            hw_sr = self._hw_sample_rate

        # Resample to target_sample_rate (16kHz) if recorded at native 44.1/48kHz
        if hw_sr != self.target_sample_rate and len(full_audio) > 0:
            duration_sec = len(full_audio) / float(hw_sr)
            target_len = max(1, int(round(duration_sec * self.target_sample_rate)))
            x_old = np.linspace(0.0, 1.0, num=len(full_audio), endpoint=False)
            x_new = np.linspace(0.0, 1.0, num=target_len, endpoint=False)
            full_audio = np.interp(x_new, x_old, full_audio.astype(np.float32)).astype(np.int16)

        self.sample_rate = self.target_sample_rate
        return full_audio

    @property
    def duration(self) -> float:
        if not self.is_recording:
            return 0.0
        return max(0.0, time.time() - self._start_time)
