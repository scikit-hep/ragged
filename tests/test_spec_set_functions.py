# BSD 3-Clause License; see https://github.com/scikit-hep/ragged/blob/main/LICENSE

"""
https://data-apis.org/array-api/latest/API_specification/set_functions.html
"""

from __future__ import annotations

import ragged


def test_existence():
    assert ragged.unique_all is not None
    assert ragged.unique_counts is not None
    assert ragged.unique_inverse is not None
    assert ragged.unique_values is not None
