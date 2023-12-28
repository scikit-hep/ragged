# BSD 3-Clause License; see https://github.com/scikit-hep/ragged/blob/main/LICENSE

"""
https://data-apis.org/array-api/latest/API_specification/manipulation_functions.html
"""

from __future__ import annotations

import ragged


def test_existence():
    assert ragged.broadcast_arrays is not None
    assert ragged.broadcast_to is not None
    assert ragged.concat is not None
    assert ragged.expand_dims is not None
    assert ragged.flip is not None
    assert ragged.permute_dims is not None
    assert ragged.reshape is not None
    assert ragged.roll is not None
    assert ragged.squeeze is not None
    assert ragged.stack is not None
