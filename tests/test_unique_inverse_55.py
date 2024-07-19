from unique_inverse import unique_inverse
import awkward as ak
import ragged

def test_can_take_none():
    assert unique_inverse(None) is None

def test_can_take_list():
    assert unique_inverse([1,2,4,3,4,5,6,20]) is None

def test_can_take_empty_arr():
    arr = ragged.array([])
    expected_unique_values = ragged.array([])
    expected_unique_indices = ragged.array([])
    unique_values, unique_indices = unique_inverse(arr)
    assert ak.to_list(unique_values) == ak.to_list(expected_unique_values)
    assert ak.to_list(unique_indices) == ak.to_list(expected_unique_indices)

def test_can_take_simple_array():
    arr = ragged.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])
    expected_unique_values = ragged.array([1, 2, 3, 4])
    expected_unique_indices = ragged.array([0, 1, 3, 6])
    unique_values, unique_indices = unique_inverse(arr)
    assert ak.to_list(unique_values) == ak.to_list(expected_unique_values)
    assert ak.to_list(unique_indices) == ak.to_list(expected_unique_indices)
    
def test_can_take_normal_array():
    arr = ragged.array([[1, 2, 2], [3], [3, 3], [4, 4, 4], [4]])
    expected_unique_values = ragged.array([1, 2, 3, 4])
    expected_unique_indices = ragged.array([0, 1, 3, 6])
    unique_values, unique_indices = unique_inverse(arr)
    assert ak.to_list(unique_values) == ak.to_list(expected_unique_values)
    assert ak.to_list(unique_indices) == ak.to_list(expected_unique_indices)
       
def test_can_take_awkward():
    arr = ragged.array([[[1, 2, 2], 3, 3], [3, 4, 4, 4, 4]])
    assert unique_inverse(arr) is None

def test_can_take_scalar():
    arr = ragged.array([5])
    expected_unique_values = ragged.array([5])
    expected_unique_indices = ragged.array([0])
    unique_values, unique_indices = unique_inverse(arr)
    assert ak.to_list(unique_values) == ak.to_list(expected_unique_values)
    assert ak.to_list(unique_indices) == ak.to_list(expected_unique_indices)