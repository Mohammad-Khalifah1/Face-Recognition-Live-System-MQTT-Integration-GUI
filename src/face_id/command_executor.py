from __future__ import annotations


class CommandExecutor:
    def execute(self, name: str, similarity: float) -> None:
        print(f"[TRIGGER] Recognized: {name} ({similarity:.1f}%)")
