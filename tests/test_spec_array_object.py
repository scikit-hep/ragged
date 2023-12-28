# BSD 3-Clause License; see https://github.com/scikit-hep/ragged/blob/main/LICENSE

"""
https://data-apis.org/array-api/latest/API_specification/array_object.html
"""

from __future__ import annotations

import ragged


def test_existence():
    assert ragged.array is not None
