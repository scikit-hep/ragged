# BSD 3-Clause License; see https://github.com/scikit-hep/ragged/blob/main/LICENSE

from __future__ import annotations

import ragged


def test():
    a = ragged.array([[1, 2], [3]])
    assert a is not None
