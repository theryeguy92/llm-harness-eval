"""Shared helper for reading required environment variables."""
import os


def require_key(name: str) -> str:
    """Return a required environment variable or raise with an actionable message.

    Args:
        name: Environment variable name, e.g. "ANTHROPIC_API_KEY".

    Returns:
        The variable's value.

    Raises:
        RuntimeError: If the variable is not set.
    """
    try:
        return os.environ[name]
    except KeyError:
        raise RuntimeError(
            f"Missing required environment variable {name}. "
            "Set it in your shell or in a .env file (see .env.example)."
        ) from None
