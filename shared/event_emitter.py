"""Thread-safe event emitter for decoupled observer/listener communication."""

from __future__ import annotations

import threading
from typing import Any, Callable


class EventEmitter:
    """Simple thread-safe event emitter with on/off/emit."""

    def __init__(self) -> None:
        self._listeners: dict[str, list[Callable]] = {}
        self._lock = threading.Lock()

    def on(self, event_type: str, callback: Callable) -> None:
        """Register a listener for *event_type*."""
        with self._lock:
            self._listeners.setdefault(event_type, []).append(callback)

    def off(self, event_type: str, callback: Callable) -> None:
        """Remove a previously registered listener."""
        with self._lock:
            listeners = self._listeners.get(event_type, [])
            try:
                listeners.remove(callback)
            except ValueError:
                pass

    def emit(self, event_type: str, event: Any = None) -> None:
        """Invoke all listeners registered for *event_type*."""
        with self._lock:
            listeners = list(self._listeners.get(event_type, []))
        for cb in listeners:
            cb(event)
