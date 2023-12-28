# BSD 3-Clause License; see https://github.com/scikit-hep/ragged/blob/main/LICENSE

"""
https://data-apis.org/array-api/latest/API_specification/creation_functions.html
"""

from __future__ import annotations

import ragged


def test_existence():
    assert ragged.arange is not None
    assert ragged.asarray is not None
    assert ragged.empty is not None
    assert ragged.empty_like is not None
    assert ragged.eye is not None
    assert ragged.from_dlpack is not None
    assert ragged.full is not None
    assert ragged.full_like is not None
    assert ragged.linspace is not None
    assert ragged.meshgrid is not None
    assert ragged.ones is not None
    assert ragged.ones_like is not None
    assert ragged.tril is not None
    assert ragged.triu is not None
    assert ragged.zeros is not None
    assert ragged.zeros_like is not None
