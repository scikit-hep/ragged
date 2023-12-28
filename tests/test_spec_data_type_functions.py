# BSD 3-Clause License; see https://github.com/scikit-hep/ragged/blob/main/LICENSE

"""
https://data-apis.org/array-api/latest/API_specification/data_type_functions.html
"""

from __future__ import annotations

import ragged


def test_existence():
    assert ragged.astype is not None
    assert ragged.can_cast is not None
    assert ragged.finfo is not None
    assert ragged.iinfo is not None
    assert ragged.isdtype is not None
    assert ragged.result_type is not None
