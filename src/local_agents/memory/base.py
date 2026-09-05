"""Memory interface.

Memory stores the conversation an agent has had so far. Agents depend only on
:class:`BaseMemory`; the default :class:`ConversationMemory` is in-process and
:class:`JSONFileMemory` persists to disk. Both are optional and swappable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

from local_agents.models.base import Message


class BaseMemory(ABC):
    """Ordered message store."""

    @abstractmethod
    def add(self, message: Message) -> None:
        """Append one message."""

    @abstractmethod
    def get(self, limit: int | None = None) -> list[Message]:
        """Return stored messages, oldest first. ``limit`` keeps the most recent N."""

    @abstractmethod
    def clear(self) -> None:
        """Forget everything."""

    def extend(self, messages: Iterable[Message]) -> None:
        for message in messages:
            self.add(message)

    def __len__(self) -> int:
        return len(self.get())

    def __bool__(self) -> bool:  # a memory object is always truthy, even when empty
        return True
