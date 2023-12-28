# BSD 3-Clause License; see https://github.com/scikit-hep/ragged/blob/main/LICENSE

"""
https://data-apis.org/array-api/latest/API_specification/constants.html
"""

from __future__ import annotations

import math

import ragged


def test_values():
    assert ragged.e == math.e
    assert not math.isfinite(ragged.inf)
    assert ragged.inf > 0
    assert math.isnan(ragged.nan)
    assert ragged.newaxis is None
    assert ragged.pi == math.pi
