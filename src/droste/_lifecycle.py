"""Small internal authority for exactly-once synchronous cleanup."""

from __future__ import annotations

from collections.abc import Callable
from threading import Condition, get_ident


class CloseOnce:
    """Run one callback outside the state lock and wait for concurrent closure."""

    __slots__ = ("_callback", "_condition", "_error", "_owner", "_state")

    def __init__(self, callback: Callable[[], None] | None) -> None:
        self._callback = callback
        self._condition = Condition()
        self._error: BaseException | None = None
        self._owner: int | None = None
        self._state = "open"

    def close(self) -> None:
        caller = get_ident()
        with self._condition:
            if self._state == "closed":
                if self._error is not None:
                    raise self._error
                return
            if self._state == "closing":
                if self._owner == caller:
                    return
                self._condition.wait_for(lambda: self._state == "closed")
                if self._error is not None:
                    raise self._error
                return
            self._state = "closing"
            self._owner = caller
        error: BaseException | None = None
        try:
            if self._callback is not None:
                self._callback()
        except BaseException as exc:
            error = exc
        finally:
            with self._condition:
                self._error = error
                self._state = "closed"
                self._owner = None
                self._condition.notify_all()
        if error is not None:
            raise error
