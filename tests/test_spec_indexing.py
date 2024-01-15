# BSD 3-Clause License; see https://github.com/scikit-hep/ragged/blob/main/LICENSE

"""
https://data-apis.org/array-api/latest/API_specification/indexing.html
"""

from __future__ import annotations

import ragged


def test():
    # slices are extensively tested in Awkward Array, just check 'axis' argument

    a = ragged.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    assert ragged.take(a, ragged.array([5, 3, 3, 9, 0, 1]), axis=0).tolist() == [
        5.5,
        3.3,
        3.3,
        9.9,
        0,
        1.1,
    ]

    b = ragged.array([[0.0, 1.1, 2.2], [3.3, 4.4], [5.5, 6.6, 7.7, 8.8, 9.9]])
    assert ragged.take(b, ragged.array([0, 1, 1, 0]), axis=1).tolist() == [
        [0, 1.1, 1.1, 0],
        [3.3, 4.4, 4.4, 3.3],
        [5.5, 6.6, 6.6, 5.5],
    ]
