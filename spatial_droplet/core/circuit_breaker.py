"""
Production Subsystem Circuit Breaker & Fault Isolation (Pillar 6).
Prevents cascading hangs by temporarily isolating failing subsystems (UIA, OCR, GDI, API)
with automatic recovery probing and zero-latency fail-soft fallbacks.
"""
import time
import threading
from typing import Dict, Any, Optional, Callable
from enum import Enum


class CircuitState(Enum):
    CLOSED = "CLOSED"      # Normal healthy operation
    OPEN = "OPEN"          # Tripped: bypass subsystem immediately, use fallback
    HALF_OPEN = "HALF_OPEN"# Probing subsystem after cooldown expires


class CircuitBreaker:
    """
    Subsystem-level circuit breaker for resilient, zero-crash desktop automation.
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 3,
        cooldown_sec: float = 15.0
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.cooldown_sec = cooldown_sec
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.success_count = 0
        self._lock = threading.Lock()

    def can_execute(self) -> bool:
        """Returns True if the subsystem is permitted to execute, False if tripped."""
        with self._lock:
            if self.state == CircuitState.CLOSED:
                return True

            now = time.time()
            if self.state == CircuitState.OPEN:
                if (now - self.last_failure_time) >= self.cooldown_sec:
                    self.state = CircuitState.HALF_OPEN
                    return True
                return False

            if self.state == CircuitState.HALF_OPEN:
                return True

            return False

    def record_success(self):
        """Records a successful execution, resetting failure counts."""
        with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
            elif self.state == CircuitState.CLOSED:
                self.failure_count = 0

    def record_failure(self, error: Optional[Exception] = None):
        """Records an execution error or timeout."""
        with self._lock:
            now = time.time()
            self.failure_count += 1
            self.last_failure_time = now

            if self.state in (CircuitState.CLOSED, CircuitState.HALF_OPEN):
                if self.failure_count >= self.failure_threshold or self.state == CircuitState.HALF_OPEN:
                    self.state = CircuitState.OPEN
                    print(f"[CircuitBreaker] [WARNING] Subsystem '{self.name}' tripped -> OPEN. Bypassing for {self.cooldown_sec}s. Error: {error}")

    def execute(self, func: Callable, fallback: Optional[Callable] = None, *args, **kwargs) -> Any:
        """Executes func within circuit breaker protection, or returns fallback result."""
        if not self.can_execute():
            if fallback:
                return fallback(*args, **kwargs)
            return None

        try:
            res = func(*args, **kwargs)
            self.record_success()
            return res
        except Exception as e:
            self.record_failure(e)
            if fallback:
                return fallback(*args, **kwargs)
            return None


class CircuitBreakerRegistry:
    """Global registry of circuit breakers across the spatial droplet framework."""
    _breakers: Dict[str, CircuitBreaker] = {}
    _lock = threading.Lock()

    @classmethod
    def get_breaker(cls, name: str, failure_threshold: int = 3, cooldown_sec: float = 15.0) -> CircuitBreaker:
        with cls._lock:
            if name not in cls._breakers:
                cls._breakers[name] = CircuitBreaker(name, failure_threshold, cooldown_sec)
            return cls._breakers[name]

    @classmethod
    def reset_all(cls):
        with cls._lock:
            for b in cls._breakers.values():
                b.state = CircuitState.CLOSED
                b.failure_count = 0
