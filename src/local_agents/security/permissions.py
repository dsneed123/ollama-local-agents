"""Tool permission model.

Each tool declares the :class:`Permission` it needs. A :class:`PermissionSet`
is attached to a tool set (and therefore to an agent) and consulted before any
tool runs. The default set allows reading and network access but not writing,
deleting or executing, so an agent built with no explicit permissions cannot
change anything on disk.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from local_agents.errors import PermissionDeniedError


class Permission(str, Enum):
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    EXECUTE = "execute"
    NETWORK = "network"
    DELEGATE = "delegate"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class PermissionSet:
    """Which permissions an agent's tools may use."""

    read: bool = True
    write: bool = False
    delete: bool = False
    execute: bool = False
    network: bool = True
    delegate: bool = False

    @classmethod
    def read_only(cls) -> PermissionSet:
        return cls(read=True, write=False, delete=False, execute=False, network=False)

    @classmethod
    def all(cls) -> PermissionSet:
        return cls(read=True, write=True, delete=True, execute=True, network=True, delegate=True)

    def allows(self, permission: Permission) -> bool:
        return bool(getattr(self, permission.value))

    def check(self, permission: Permission, *, subject: str = "operation") -> None:
        if not self.allows(permission):
            raise PermissionDeniedError(
                f"{subject} requires the '{permission}' permission, which is disabled",
                hint="Grant it explicitly via PermissionSet or the agent's tool configuration.",
            )

    def granted(self) -> list[str]:
        return [p.value for p in Permission if self.allows(p)]

    def with_(self, **changes: bool) -> PermissionSet:
        data = dict(self.__dict__)
        data.update(changes)
        return PermissionSet(**data)
