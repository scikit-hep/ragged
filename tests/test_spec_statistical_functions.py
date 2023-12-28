# BSD 3-Clause License; see https://github.com/scikit-hep/ragged/blob/main/LICENSE

"""
https://data-apis.org/array-api/latest/API_specification/statistical_functions.html
"""

from __future__ import annotations

import ragged


def test_existence():
    assert ragged.max is not None
    assert ragged.mean is not None
    assert ragged.min is not None
    assert ragged.prod is not None
    assert ragged.std is not None
    assert ragged.sum is not None
    assert ragged.var is not None
