"""Terminal presentation primitives shared by CLI and logging code."""

from __future__ import annotations

import shutil


def terminal_separator(*, width: int | None = None, character: str = "─") -> str:
    """Return a separator sized to the active terminal width."""

    if len(character) != 1:
        raise ValueError("character must contain exactly one character")

    columns = width or shutil.get_terminal_size().columns
    return character * max(columns, 1)
