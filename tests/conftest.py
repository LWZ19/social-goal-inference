"""Safety net for the whole suite: no test reads the real .env file or opens a network connection.

The real .env holds API keys. A test that needs a key passes its own file or replaces require_key.
"""
import socket

import pytest

from sgi.vlm import env as _env


@pytest.fixture(scope="session")
def _absent_env(tmp_path_factory):
    return tmp_path_factory.mktemp("no_real_env") / ".env"      # never created, so load_env gives {}


@pytest.fixture(autouse=True)
def _no_real_key_and_no_network(monkeypatch, _absent_env):
    monkeypatch.setattr(_env.load_env, "__defaults__", (_absent_env,))
    monkeypatch.setattr(_env.require_key, "__defaults__", (_absent_env,))

    def refuse(self, *args, **kwargs):
        raise RuntimeError("a test tried to open a network connection; tests must stay offline")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
