"""System-wide undo stack for human-assisted mode."""

import uuid
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

@dataclass
class UndoableAction:
    action_type: str
    forward_data: Dict[str, Any]
    reverse_data: Dict[str, Any]
    action_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)

class UndoStack:
    def __init__(self, max_depth: int = 10):
        self._undo: List[UndoableAction] = []
        self._redo: List[UndoableAction] = []
        self.max_depth = max_depth

    def push(self, action: UndoableAction) -> None:
        self._undo.append(action)
        if len(self._undo) > self.max_depth:
            self._undo.pop(0)
        self._redo.clear()

    def pop_undo(self) -> Optional[UndoableAction]:
        if not self._undo:
            return None
        action = self._undo.pop()
        self._redo.append(action)
        return action

    def pop_redo(self) -> Optional[UndoableAction]:
        if not self._redo:
            return None
        action = self._redo.pop()
        self._undo.append(action)
        return action

    def get_state(self) -> Dict[str, Any]:
        return {
            "can_undo": len(self._undo) > 0,
            "can_redo": len(self._redo) > 0,
            "last_action_name": self._undo[-1].action_type if self._undo else None,
            "next_redo_name": self._redo[-1].action_type if self._redo else None
        }
