from __future__ import annotations

import importlib.metadata

import ragged as m


def test_version():
    assert importlib.metadata.version("ragged") == m.__version__
