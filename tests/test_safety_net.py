import socket

import pytest

from sgi.vlm import factory
from sgi.vlm.env import MissingKeyError, load_env, require_key


def test_the_real_env_file_is_never_read_in_tests():
    assert load_env() == {}
    for name in ("GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        with pytest.raises(MissingKeyError):
            require_key(name)
        with pytest.raises(MissingKeyError):
            factory.require_key(name)                # the name make_client calls


def test_a_network_connection_is_refused():
    with socket.socket() as sock, pytest.raises(RuntimeError, match="offline"):
        sock.connect(("example.com", 443))
