"""Tests run on the Warp CPU backend so they are identical on any machine (and in CI)."""

import pytest

from ailab.core import compute


@pytest.fixture(scope="session", autouse=True)
def warp_cpu():
    compute.init("cpu")
    return "cpu"


def pytest_addoption(parser):
    parser.addoption("--gpu", action="store_true", help="also run CUDA determinism checks")
