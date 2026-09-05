"""JSON-file backed memory. Optional; nothing in the core requires it."""

from __future__ import annotations

import json
import os
from pathlib import Path

from local_agents.memory.conversation import ConversationMemory
from local_agents.models.base import Message


class JSONFileMemory(ConversationMemory):
    """A :class:`ConversationMemory` mirrored to a JSON file.

    The file is rewritten on every ``add``/``clear`` so a crash never loses
    more than the in-flight message. Loading tolerates a missing or corrupt
    file (it starts empty rather than raising).
    """

    def __init__(self, path: str | os.PathLike[str], max_messages: int | None = 500) -> None:
        super().__init__(max_messages=max_messages)
        self.path = Path(path).expanduser()
        self._load()

    def add(self, message: Message) -> None:
        super().add(message)
        self._save()

    def clear(self) -> None:
        super().clear()
        self._save()

    def _load(self) -> None:
        if not self.path.is_file():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, dict):
                    super().add(Message.from_dict(item))

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = [m.to_dict() for m in self.get()]
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(self.path)
