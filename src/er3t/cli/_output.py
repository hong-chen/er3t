"""Small presentation helpers shared by command-line tools."""

from __future__ import annotations

import sys
from typing import TextIO

from er3t._terminal import terminal_separator


def dialogue_separator(*, width: int | None = None, character: str = "─") -> str:
    """Return a separator sized to the active terminal width.

    This CLI-facing name delegates to the package-wide terminal primitive.
    """

    return terminal_separator(width=width, character=character)


def print_dialogue(message: str, *, stream: TextIO | None = None) -> None:
    """Print a message enclosed by terminal-width dialogue separators."""

    stream = stream or sys.stdout
    separator = dialogue_separator()
    print(separator, file=stream)
    print(message.rstrip(), file=stream)
    print(separator, file=stream)
