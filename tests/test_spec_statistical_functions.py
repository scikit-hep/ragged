# BSD 3-Clause License; see https://github.com/scikit-hep/ragged/blob/main/LICENSE

"""
https://data-apis.org/array-api/latest/API_specification/statistical_functions.html
"""

from __future__ import annotations

import pytest

import ragged


def test_existence():
    assert ragged.max is not None
    assert ragged.mean is not None
    assert ragged.min is not None
    assert ragged.prod is not None
    assert ragged.std is not None
    assert ragged.sum is not None
    assert ragged.var is not None


def test_sum():
    data = ragged.array(
        [[[0, 1.1, 2.2], []], [], [[3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]]]
    )
    assert ragged.sum(data, axis=None).tolist() == pytest.approx(49.5)
    assert (
        ragged.sum(data, axis=0).tolist()  # type: ignore[comparison-overlap]
        == ragged.sum(data, axis=-3).tolist()
        == [
            pytest.approx([3.3, 5.5, 2.2]),
            pytest.approx([5.5]),
            pytest.approx([6.6, 7.7, 8.8, 9.9]),
        ]
    )
    assert (
        ragged.sum(data, axis=1).tolist()  # type: ignore[comparison-overlap]
        == ragged.sum(data, axis=-2).tolist()
        == [
            pytest.approx([0.0, 1.1, 2.2]),
            pytest.approx([]),
            pytest.approx([15.4, 12.1, 8.8, 9.9]),
        ]
    )
    assert (
        ragged.sum(data, axis=2).tolist()  # type: ignore[comparison-overlap]
        == ragged.sum(data, axis=-1).tolist()
        == [
            pytest.approx([3.3, 0.0]),
            pytest.approx([]),
            pytest.approx([7.7, 5.5, 33.0]),
        ]
    )
    assert (
        ragged.sum(data, axis=(0, 1)).tolist()
        == ragged.sum(data, axis=(1, 0)).tolist()
        == pytest.approx([15.4, 13.2, 11.0, 9.9])
    )
    assert (
        ragged.sum(data, axis=(0, 2)).tolist()
        == ragged.sum(data, axis=(2, 0)).tolist()
        == pytest.approx([11.0, 5.5, 33.0])
    )
    assert (
        ragged.sum(data, axis=(1, 2)).tolist()
        == ragged.sum(data, axis=(2, 1)).tolist()
        == pytest.approx([3.3, 0.0, 46.2])
    )
    assert (
        ragged.sum(data, axis=(0, 1, 2)).tolist()
        == ragged.sum(data, axis=(-1, 0, 1)).tolist()
        == pytest.approx(49.5)
    )
