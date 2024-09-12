## Please fill in all the parts labeled as ### YOUR CODE HERE

import numpy as np
import pytest
from utils import *

def test_dot_product():
    vector1 = np.array([1, 2, 3])
    vector2 = np.array([4, 5, 6])
    
    result = dot_product(vector1, vector2)
    
    assert result == 32, f"Expected 32, but got {result}"
    
def test_cosine_similarity():
    vector1 = np.array([1, 2, 3])
    vector2 = np.array([1, 2, 3])
    
    result = cosine_similarity(vector1, vector2)
    
    expected_result = 1.  # (identical vectors)
    
    assert np.isclose(result, expected_result), f"Expected {expected_result}, but got {result}"

def test_nearest_neighbor():
    target = np.array([1, 2, 3])
    vectors = [
        np.array([1, 2, i]) for i in range(10)
    ] + [np.array([4,5,6])]
    
    result = nearest_neighbor(target, vectors)
    
    expected_index = 3
    
    assert result == expected_index, f"Expected index {expected_index}, but got {result}"
