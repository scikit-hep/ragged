# BSD 3-Clause License; see https://github.com/scikit-hep/ragged/blob/main/LICENSE

"""
https://data-apis.org/array-api/latest/API_specification/utility_functions.html
"""

from __future__ import annotations

import ragged


def test_existence():
    assert ragged.all is not None
    assert ragged.any is not None


def test_all():
    data = ragged.array([[[0, 1, 2], []], [], [[0, 0], [1], [1, 1, 1, 1]]])
    assert ragged.all(data, axis=None).tolist() is False
    assert (
        ragged.all(data, axis=0).tolist()
        == ragged.all(data, axis=-3).tolist()
        == [[False, False, True], [True], [True, True, True, True]]
    )
    assert (
        ragged.all(data, axis=1).tolist()  # type: ignore[comparison-overlap]
        == ragged.all(data, axis=-2).tolist()
        == [[False, True, True], [], [False, False, True, True]]
    )
    assert (
        ragged.all(data, axis=2).tolist()  # type: ignore[comparison-overlap]
        == ragged.all(data, axis=-1).tolist()
        == [[False, True], [], [False, True, True]]
    )
    assert (
        ragged.all(data, axis=(0, 1)).tolist()
        == ragged.all(data, axis=(1, 0)).tolist()
        == [False, False, True, True]
    )
    assert (
        ragged.all(data, axis=(0, 2)).tolist()
        == ragged.all(data, axis=(2, 0)).tolist()
        == [False, True, True]
    )
    assert (
        ragged.all(data, axis=(1, 2)).tolist()
        == ragged.all(data, axis=(2, 1)).tolist()
        == [False, True, False]
    )
    assert (
        ragged.all(data, axis=(0, 1, 2)).tolist()
        is ragged.all(data, axis=(-1, 0, 1)).tolist()
        is False
    )


def test_any():
    data = ragged.array([[[0, 1, 2], []], [], [[0, 0], [1], [1, 1, 1, 1]]])
    assert ragged.any(data, axis=None).tolist() is True
    assert (
        ragged.any(data, axis=0).tolist()
        == ragged.any(data, axis=-3).tolist()
        == [[False, True, True], [True], [True, True, True, True]]
    )
    assert (
        ragged.any(data, axis=1).tolist()  # type: ignore[comparison-overlap]
        == ragged.any(data, axis=-2).tolist()
        == [[False, True, True], [], [True, True, True, True]]
    )
    assert (
        ragged.any(data, axis=2).tolist()  # type: ignore[comparison-overlap]
        == ragged.any(data, axis=-1).tolist()
        == [[True, False], [], [False, True, True]]
    )
    assert (
        ragged.any(data, axis=(0, 1)).tolist()
        == ragged.any(data, axis=(1, 0)).tolist()
        == [True, True, True, True]
    )
    assert (
        ragged.any(data, axis=(0, 2)).tolist()
        == ragged.any(data, axis=(2, 0)).tolist()
        == [True, True, True]
    )
    assert (
        ragged.any(data, axis=(1, 2)).tolist()
        == ragged.any(data, axis=(2, 1)).tolist()
        == [True, False, True]
    )
    assert (
        ragged.any(data, axis=(0, 1, 2)).tolist()
        is ragged.any(data, axis=(-1, 0, 1)).tolist()
        is True
    )
