"""Citation registry shared by high-level workflows."""


def add_reference(reference: str) -> None:
    """Register a citation once in the package-level reference list."""

    from er3t.common import references

    if reference not in references:
        references.append(reference)


def get_references() -> tuple[str, ...]:
    """Return the current citations as an immutable snapshot."""

    from er3t.common import references

    return tuple(references)


def print_references() -> None:
    """Print all registered citations."""

    for reference in get_references():
        print(reference)
