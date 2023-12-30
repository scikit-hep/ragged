# BSD 3-Clause License; see https://github.com/scikit-hep/ragged/blob/main/LICENSE

"""
https://data-apis.org/array-api/latest/API_specification/array_object.html
"""

from __future__ import annotations

import pytest

import ragged


def test_existence():
    assert ragged.array is not None


def test_namespace():
    assert ragged.array(123).__array_namespace__() is ragged
    assert (
        ragged.array(123).__array_namespace__(api_version=ragged.__array_api_version__)
        is ragged
    )
    with pytest.raises(NotImplementedError):
        ragged.array(123).__array_namespace__(api_version="does not exist")


def test_bool():
    assert bool(ragged.array(True)) is True
    assert bool(ragged.array(False)) is False


def test_complex():
    assert isinstance(complex(ragged.array(1.1 + 0.1j)), complex)
    assert complex(ragged.array(1.1 + 0.1j)) == 1.1 + 0.1j


def test_float():
    assert isinstance(float(ragged.array(1.1)), float)
    assert float(ragged.array(1.1)) == 1.1


def test_index():
    assert isinstance(ragged.array(10).__index__(), int)
    assert ragged.array(10).__index__() == 10


def test_int():
    assert isinstance(int(ragged.array(10)), int)
    assert int(ragged.array(10)) == 10
