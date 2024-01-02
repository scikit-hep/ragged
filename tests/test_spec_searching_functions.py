# BSD 3-Clause License; see https://github.com/scikit-hep/ragged/blob/main/LICENSE

"""
https://data-apis.org/array-api/latest/API_specification/searching_functions.html
"""

from __future__ import annotations

import pytest

import ragged


def test_existence():
    assert ragged.argmax is not None
    assert ragged.argmin is not None
    assert ragged.nonzero is not None
    assert ragged.where is not None


def test_argmax():
    data = ragged.array(
        [[[0, 1.1, 2.2], []], [], [[3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]]]
    )
    assert ragged.argmax(data, axis=None).tolist() == 9
    assert (
        ragged.argmax(data, axis=0).tolist()
        == ragged.argmax(data, axis=-3).tolist()
        == [[1, 1, 0], [1], [0, 0, 0, 0]]
    )
    assert (
        ragged.argmax(data, axis=1).tolist()  # type: ignore[comparison-overlap]
        == ragged.argmax(data, axis=-2).tolist()
        == [[0, 0, 0], [], [2, 2, 2, 2]]
    )
    with pytest.raises(ValueError, match=".*axis.*"):
        ragged.argmax(data, axis=2)
    with pytest.raises(ValueError, match=".*axis.*"):
        ragged.argmax(data, axis=-1)


def test_argmin():
    data = ragged.array(
        [[[0, 1.1, 2.2], []], [], [[3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]]]
    )
    assert ragged.argmin(data, axis=None).tolist() == 0
    assert (
        ragged.argmin(data, axis=0).tolist()
        == ragged.argmin(data, axis=-3).tolist()
        == [[0, 0, 0], [1], [0, 0, 0, 0]]
    )
    assert (
        ragged.argmin(data, axis=1).tolist()  # type: ignore[comparison-overlap]
        == ragged.argmin(data, axis=-2).tolist()
        == [[0, 0, 0], [], [0, 0, 2, 2]]
    )
    with pytest.raises(ValueError, match=".*axis.*"):
        ragged.argmin(data, axis=2)
    with pytest.raises(ValueError, match=".*axis.*"):
        ragged.argmin(data, axis=-1)
