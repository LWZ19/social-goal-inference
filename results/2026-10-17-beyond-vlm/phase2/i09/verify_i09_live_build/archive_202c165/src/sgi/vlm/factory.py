"""make_client: the VLMClient for one entry of the model table. And what a command lists of the table and of a
provider: entry_lines and list_models."""
from __future__ import annotations

from sgi.config import REPO_ROOT, Config, ConfigError, ModelEntry
from sgi.vlm.base import Backend, Pacer, VLMClient
from sgi.vlm.cache import Cache
from sgi.vlm.env import load_env, require_key
from sgi.vlm.mock import MockBackend, centre_responder

_PROVIDER_KEYS = {"gemini": "GEMINI_API_KEY", "openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}
_PACERS: dict[tuple[str, str | None], Pacer] = {}       # by provider and address, for as long as the process lives


def make_client(entry_name: str, cfg: Config, backend: Backend | None = None) -> VLMClient:
    """The client for `cfg.models[entry_name]`. A back end passed in is used as it is and no key is read.

    An entry with a request limit gets the pacer its provider's other clients have."""
    entry = _entry(entry_name, cfg)
    if backend is None:
        backend = _backend(entry)
    cache = Cache(REPO_ROOT / cfg.run.cache_dir) if cfg.run.use_cache else None     # an absolute directory is kept
    return VLMClient(entry, backend, cache, pacer=_pacer(entry))


def list_models(entry_name: str, cfg: Config, backend: Backend | None = None) -> list[str]:
    """The model ids the provider of `cfg.models[entry_name]` reports for its key, sorted: a real request.

    An OpenAI-compatible service's ids are lines: each is followed by what else the service says of the model.
    A back end passed in is used as it is and no key is read. A back end that cannot list, the mock, is a
    ConfigError."""
    entry = _entry(entry_name, cfg)
    if backend is None:
        backend = _backend(entry)
    if not hasattr(backend, "list_models"):
        raise ConfigError(f"model entry '{entry_name}': the {backend.provider} back end has no model listing")
    return backend.list_models()


def entry_lines(cfg: Config) -> list[str]:
    """One line per entry of the model table, in its order: name, provider, model id, reasoning effort, point
    format, and whether the .env file holds the entry's key. A key's variable is named; its value never is."""
    held = load_env()
    rows = []
    for name, entry in cfg.models.items():
        variable = _key_variable(entry)
        key = "no key needed" if variable is None else (
            f"{'key set' if held.get(variable) else 'key missing'} ({variable})")
        rows.append((name, entry.provider, entry.model, f"effort {entry.reasoning_effort}", entry.point_format, key))
    widths = [max(len(cell) for cell in column) for column in zip(*rows)]
    return ["  ".join(cell.ljust(width) for cell, width in zip(row, widths)).rstrip() for row in rows]


def _entry(entry_name: str, cfg: Config) -> ModelEntry:
    if entry_name not in cfg.models:
        raise ConfigError(f"model entry '{entry_name}' is not a key of models")
    return cfg.models[entry_name]


def _pacer(entry: ModelEntry) -> Pacer | None:
    """The pacer of the entry's provider and address, one for every client made of them in this process; None
    for an entry without a request limit. Where entries of one address name different limits, the lowest holds."""
    limit = entry.max_requests_per_minute
    if limit is None:
        return None
    key = (entry.provider, entry.base_url)
    if key not in _PACERS:
        _PACERS[key] = Pacer(limit)
    pacer = _PACERS[key]
    pacer.max_per_minute = min(pacer.max_per_minute, limit)
    return pacer


def _key_variable(entry: ModelEntry) -> str | None:
    """The .env variable that holds the key of the entry's provider; None for the mock, which needs none."""
    return entry.api_key_env if entry.provider == "openai_compatible" else _PROVIDER_KEYS.get(entry.provider)


def _backend(entry: ModelEntry) -> Backend:
    if entry.provider == "mock":
        return MockBackend(centre_responder)
    key = require_key(_key_variable(entry))
    if entry.provider == "gemini":
        from sgi.vlm.gemini import GeminiBackend   # in its branch, so a mock run never loads an SDK
        return GeminiBackend(key)
    if entry.provider == "openai_compatible":      # its address comes from the entry too
        from sgi.vlm.compatible import OpenAICompatibleBackend
        return OpenAICompatibleBackend(key, entry.base_url)
    if entry.provider == "anthropic":
        from sgi.vlm.anthropic import AnthropicBackend
        return AnthropicBackend(key)
    from sgi.vlm.openai import OpenAIBackend
    return OpenAIBackend(key)
