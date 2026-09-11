"""Explicit state transitions; pauses and faults fence every previous request."""

from dataclasses import dataclass
from dataclasses import field
from enum import StrEnum
import uuid


class State(StrEnum):
    DISCONNECTED = "disconnected"
    OBSERVING = "observing"
    READY = "ready"
    ACQUIRING = "acquiring"
    EXECUTING = "executing"
    STOPPING = "stopping"
    PAUSED = "paused"
    FAULT = "fault"


@dataclass
class StateMachine:
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    generation: int = 0
    state: State = State.DISCONNECTED
    fault_reason: str | None = None
    history: list = field(default_factory=list)

    def _move(self, allowed, destination, reason, now):
        if self.state not in allowed:
            raise ValueError(f"Cannot {reason} from {self.state}")
        self.history.append(
            {
                "from": self.state.value,
                "to": destination.value,
                "reason": reason,
                "mono": now,
                "generation": self.generation,
            }
        )
        self.state = destination

    def connect(self, now):
        self._move({State.DISCONNECTED}, State.OBSERVING, "connect", now)

    def ready(self, now):
        self._move({State.OBSERVING}, State.READY, "healthy_inputs", now)

    def acquire(self, now):
        self._move({State.READY}, State.ACQUIRING, "explicit_single_trial", now)

    def begin(self, now):
        self._move({State.ACQUIRING}, State.EXECUTING, "candidate_accepted", now)

    def reject(self, now):
        self._move({State.ACQUIRING}, State.READY, "candidate_rejected_before_output", now)

    def pause(self, now, reason="pause"):
        if self.state in {State.DISCONNECTED, State.FAULT}:
            return
        self.generation += 1
        self._move(set(State) - {State.DISCONNECTED, State.FAULT}, State.STOPPING, reason, now)

    def stopped(self, now):
        self._move({State.STOPPING}, State.PAUSED, "stop_confirmed", now)

    def resume(self, now):
        if self.state != State.PAUSED:
            raise ValueError(f"Cannot explicit_prepare from {self.state}")
        self.generation += 1
        self._move({State.PAUSED}, State.OBSERVING, "explicit_prepare", now)

    def fault(self, reason, now):
        if self.state != State.FAULT:
            self.generation += 1
            self.fault_reason = reason
            self._move(set(State), State.FAULT, reason, now)

    def reset_fault(self, now):
        self._move({State.FAULT}, State.DISCONNECTED, "explicit_fault_reset", now)
        self.generation += 1
        self.run_id = uuid.uuid4().hex
        self.fault_reason = None
