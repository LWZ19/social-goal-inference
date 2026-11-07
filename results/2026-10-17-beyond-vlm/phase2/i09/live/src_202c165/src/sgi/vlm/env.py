"""API keys from the git-ignored .env file: the only source of keys. Nothing here prints, logs or touches os.environ."""
from __future__ import annotations

from pathlib import Path

from sgi.config import REPO_ROOT


class MissingKeyError(RuntimeError):
    """A key that the .env file does not hold, or holds empty."""


def load_env(path: Path = REPO_ROOT / ".env") -> dict[str, str]:
    """The NAME=value pairs of a .env file; {} when there is no such file.

    Blank lines, lines starting with # and lines without = are skipped. One pair of matching quotes around a value
    is removed. Nothing else is interpreted: no export prefix, no inline comment, no variable expansion.
    """
    path = Path(path)
    if not path.is_file():
        return {}
    pairs: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        pairs[key.strip()] = value
    return pairs


def require_key(name: str, env_path: Path = REPO_ROOT / ".env") -> str:
    """The value of `name` in the .env file. The error names the file and the variable, never a value."""
    value = load_env(env_path).get(name)
    if not value:
        raise MissingKeyError(
            f"{name} is not set in {env_path}: add the line {name}=<your key> to that file (see .env.example)")
    return value
