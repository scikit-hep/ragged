from unique_values1d import unique_values1d
import awkward as ak
import ragged

def test_can_take_none():
    assert unique_values1d(None)==None

def test_can_take_list():
    assert unique_values1d([1,2,4,3,4,5,6,20])

def test_can_take_0():
    assert unique_values1d(ragged.array([0]))==ragged.array([0])

def test_can_take_empty_arr():
    assert unique_values1d(ragged.array())

def test_can_take_moredimensions():
    assert unique_values1d(ragged.array([1,2,3,4,[5,6]]))

def test_can_take_1d_array():
    assert unique_values1d(ragged.array([5,6,7,8,8,9,1,2,3,4,10,0,15,2]))==ragged.array([0,1,2,3,4,5,6,7,8,9,10,15])


