import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow (LLM calls, may hit rate limits)")
