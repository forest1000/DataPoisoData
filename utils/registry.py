"""Utility registry for modular components."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Iterator, MutableMapping, TypeVar

T = TypeVar("T")


@dataclass
class Registry(MutableMapping[str, T]):
    """Simple string to object registry with helpful error messages."""

    name: str
    _store: dict[str, T] = field(default_factory=dict)

    def register(self, key: str, value: T) -> None:
        normalized = self._normalize_key(key)
        if normalized in self._store:
            raise KeyError(f"{self.name} '{normalized}' already registered")
        self._store[normalized] = value

    def unregister(self, key: str) -> None:
        normalized = self._normalize_key(key)
        if normalized not in self._store:
            raise KeyError(self._missing_message(normalized))
        del self._store[normalized]

    def get(self, key: str) -> T:  # type: ignore[override]
        normalized = self._normalize_key(key)
        if normalized not in self._store:
            raise KeyError(self._missing_message(normalized))
        return self._store[normalized]

    def __getitem__(self, key: str) -> T:
        return self.get(key)

    def __setitem__(self, key: str, value: T) -> None:
        self.register(key, value)

    def __delitem__(self, key: str) -> None:
        self.unregister(key)

    def __iter__(self) -> Iterator[str]:
        return iter(self._store)

    def __len__(self) -> int:
        return len(self._store)

    def keys(self) -> Iterable[str]:
        return self._store.keys()

    def _normalize_key(self, key: str) -> str:
        return key.strip().lower()

    def _missing_message(self, key: str) -> str:
        options = ", ".join(sorted(self._store)) or "<empty>"
        return f"Unknown {self.name} '{key}'. Available options: {options}."


__all__ = ["Registry"]
