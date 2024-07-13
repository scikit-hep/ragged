from unique_values import unique_values
import awkward as ak
import ragged

def test_can_take_none():
    assert unique_values(None)==None

def test_can_take_list():
    assert unique_values([1,2,4,3,4,5,6,20])

def test_can_take_0():
    assert unique_values(ragged.array([0]))==ragged.array([0])

def test_can_take_empty_arr():
    assert unique_values(ragged.array())

def test_can_take_moredimensions():
    assert unique_values(ragged.array([[1,2,3,4],[5,6]]))

def test_can_take_1d_array():
    assert unique_values(ragged.array([5,6,7,8,8,9,1,2,3,4,10,0,15,2]))==ragged.array([0,1,2,3,4,5,6,7,8,9,10,15])

def test_can_take_awkward():
    assert unique_values(ak.Array([1,2,3,4,[5,6,7]]))
