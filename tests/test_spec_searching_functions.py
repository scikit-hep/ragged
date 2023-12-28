# BSD 3-Clause License; see https://github.com/scikit-hep/ragged/blob/main/LICENSE

"""
https://data-apis.org/array-api/latest/API_specification/searching_functions.html
"""

from __future__ import annotations

import ragged


def test_existence():
    assert ragged.argmax is not None
    assert ragged.argmin is not None
    assert ragged.nonzero is not None
    assert ragged.where is not None
