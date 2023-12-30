# BSD 3-Clause License; see https://github.com/scikit-hep/ragged/blob/main/LICENSE

from __future__ import annotations

from typing import Any


def cupy() -> Any:
    try:
        import cupy as cp  # pylint: disable=C0415

        return cp
    except ModuleNotFoundError as err:
        error_message = """to use the "cuda" backend, you must install cupy:

    pip install cupy

or

    conda install -c conda-forge cupy
"""
        raise ModuleNotFoundError(error_message) from err
