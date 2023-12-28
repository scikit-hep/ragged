# BSD 3-Clause License; see https://github.com/scikit-hep/ragged/blob/main/LICENSE

"""
https://data-apis.org/array-api/latest/API_specification/utility_functions.html
"""

from __future__ import annotations

import ragged


def test_existence():
    assert ragged.all is not None
    assert ragged.any is not None
