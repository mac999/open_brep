"""
Layer 4 - ID Registry.

A centralized symbol table assigning every entity (vertex, edge, face, solid,
loop, curve, surface ...) a unique immutable integer ID. The CLI refers to these
IDs with a leading hash, e.g. ``#100``.

Design goals
    * IDs never get reused once allocated (immutable identity).
    * Explicit IDs (the ``as #<id>`` syntax) are honoured and reserve the value.
    * Lookups are O(1) in both directions.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class IdRegistry:
    """Maps integer IDs to entities and back."""

    def __init__(self, start: int = 100):
        self._next_id = start
        self._by_id: Dict[int, Any] = {}
        # Most recently registered entity per kind ('vertex', 'edge', ...) plus
        # 'any'. Powers the CLI's $vertex / $edge / $last symbolic references.
        self._last: Dict[str, Any] = {}

    def register(self, entity: Any, explicit_id: Optional[int] = None) -> int:
        """
        Register ``entity`` and stamp ``entity.oid`` with its assigned ID.

        If ``explicit_id`` is given it is used as-is (and must be free); otherwise
        the next free auto-increment ID is allocated.
        """
        if explicit_id is not None:
            if explicit_id in self._by_id:
                raise ValueError(f"id #{explicit_id} is already in use")
            oid = explicit_id
            # Keep the auto-counter ahead of any manually chosen id.
            self._next_id = max(self._next_id, oid + 1)
        else:
            oid = self._next_id
            self._next_id += 1

        entity.oid = oid
        self._by_id[oid] = entity
        self._last[type(entity).__name__.lower()] = entity
        self._last["any"] = entity
        return oid

    def last(self, kind: str) -> Optional[Any]:
        """Most recently registered entity of ``kind`` ('vertex', 'edge', 'face',
        'loop', 'solid', 'halfedge', or 'any'); None if none yet."""
        return self._last.get(kind.lower())

    def get(self, oid: int) -> Any:
        """Return the entity for ``oid`` or raise KeyError with a clear message."""
        if oid not in self._by_id:
            raise KeyError(f"no entity registered with id #{oid}")
        return self._by_id[oid]

    def find(self, oid: int) -> Optional[Any]:
        """Like :meth:`get` but returns ``None`` instead of raising."""
        return self._by_id.get(oid)

    def unregister(self, oid: int) -> None:
        """Remove an ID from the table (used by Euler operators that destroy entities)."""
        self._by_id.pop(oid, None)

    def all_of_type(self, cls: type) -> list:
        """Return every registered entity that is an instance of ``cls``."""
        return [e for e in self._by_id.values() if isinstance(e, cls)]

    def __contains__(self, oid: int) -> bool:
        return oid in self._by_id

    def __len__(self) -> int:
        return len(self._by_id)
