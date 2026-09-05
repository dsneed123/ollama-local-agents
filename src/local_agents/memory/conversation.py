"""In-process conversation memory with a sliding window."""

from __future__ import annotations

from local_agents.memory.base import BaseMemory
from local_agents.models.base import Message


class ConversationMemory(BaseMemory):
    """Keep the most recent ``max_messages`` messages in memory.

    When trimming, tool-result messages that would be orphaned from their
    assistant tool-call are dropped together so the model never sees a tool
    result without the call that produced it.
    """

    def __init__(self, max_messages: int | None = 200) -> None:
        if max_messages is not None and max_messages < 1:
            raise ValueError("max_messages must be >= 1 or None")
        self.max_messages = max_messages
        self._messages: list[Message] = []

    def add(self, message: Message) -> None:
        self._messages.append(message)
        self._trim()

    def get(self, limit: int | None = None) -> list[Message]:
        if limit is None:
            return list(self._messages)
        return list(self._messages[-limit:]) if limit > 0 else []

    def clear(self) -> None:
        self._messages.clear()

    def _trim(self) -> None:
        if self.max_messages is None:
            return
        while len(self._messages) > self.max_messages:
            self._messages.pop(0)
        # Never start with a dangling tool result.
        while self._messages and self._messages[0].role == "tool":
            self._messages.pop(0)
