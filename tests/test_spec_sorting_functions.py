# BSD 3-Clause License; see https://github.com/scikit-hep/ragged/blob/main/LICENSE

"""
https://data-apis.org/array-api/latest/API_specification/sorting_functions.html
"""

from __future__ import annotations

import pytest

import ragged

devices = ["cpu"]
try:
    import cupy as cp

    # FIXME!
    # devices.append("cuda")
except ModuleNotFoundError:
    cp = None


def test_existence():
    assert ragged.argsort is not None
    assert ragged.sort is not None


@pytest.mark.parametrize("device", devices)
def test_argsort(device):
    x = ragged.array(
        [[1.1, 0, 2.2], [], [3.3, 4.4], [5.5], [9.9, 7.7, 8.8, 6.6]], device=device
    )
    assert ragged.argsort(x, axis=1, stable=True, descending=False).tolist() == [  # type: ignore[comparison-overlap]
        [1, 0, 2],
        [],
        [0, 1],
        [0],
        [3, 1, 2, 0],
    ]
    assert ragged.argsort(x, axis=1, stable=True, descending=True).tolist() == [  # type: ignore[comparison-overlap]
        [2, 0, 1],
        [],
        [1, 0],
        [0],
        [0, 2, 1, 3],
    ]
    assert ragged.argsort(x, axis=0, stable=True, descending=False).tolist() == [  # type: ignore[comparison-overlap]
        [0, 0, 0],
        [],
        [2, 2],
        [3],
        [4, 4, 4, 4],
    ]
    assert ragged.argsort(x, axis=0, stable=True, descending=True).tolist() == [  # type: ignore[comparison-overlap]
        [4, 4, 4],
        [],
        [3, 2],
        [2],
        [0, 0, 0, 4],
    ]


@pytest.mark.parametrize("device", devices)
def test_sort(device):
    x = ragged.array(
        [[1.1, 0, 2.2], [], [3.3, 4.4], [5.5], [9.9, 7.7, 8.8, 6.6]], device=device
    )
    assert ragged.sort(x, axis=1, stable=True, descending=False).tolist() == [  # type: ignore[comparison-overlap]
        [0, 1.1, 2.2],
        [],
        [3.3, 4.4],
        [5.5],
        [6.6, 7.7, 8.8, 9.9],
    ]
    assert ragged.sort(x, axis=1, stable=True, descending=True).tolist() == [  # type: ignore[comparison-overlap]
        [2.2, 1.1, 0],
        [],
        [4.4, 3.3],
        [5.5],
        [9.9, 8.8, 7.7, 6.6],
    ]
    assert ragged.sort(x, axis=0, stable=True, descending=False).tolist() == [  # type: ignore[comparison-overlap]
        [1.1, 0.0, 2.2],
        [],
        [3.3, 4.4],
        [5.5],
        [9.9, 7.7, 8.8, 6.6],
    ]
    assert ragged.sort(x, axis=0, stable=True, descending=True).tolist() == [  # type: ignore[comparison-overlap]
        [9.9, 7.7, 8.8],
        [],
        [5.5, 4.4],
        [3.3],
        [1.1, 0.0, 2.2, 6.6],
    ]
