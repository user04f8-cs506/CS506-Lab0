import numpy as np

def dot_product(v1: np.ndarray, v2: np.ndarray):
    '''
    v1 and v2 are vectors of same shape.
    return the scalar dor product of the two vectors.
    # Hint: use `np.dot`.
    '''
    return np.dot(v1, v2)
    
def cosine_similarity(v1: np.ndarray, v2: np.ndarray):
    '''
    v1 and v2 are vectors of same shape.
    Return the cosine similarity between the two vectors.
    
    # Note: The cosine similarity is a commonly used similarity 
    metric between two vectors. It is the cosine of the angle between 
    two vectors, and always between -1 and 1.
    
    # The formula for cosine similarity is: 
    # (v1 dot v2) / (||v1|| * ||v2||)
    
    # ||v1|| is the 2-norm (Euclidean length) of the vector v1.
    
    # Hint: Use `dot_product` and `np.linalg.norm`.
    '''
    return dot_product(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    
def nearest_neighbor(target_vector: np.ndarray, vectors: np.ndarray):
    '''
    target_vector is a vector of shape d.
    vectors is a matrix of shape N x d.
    return the row index of the vector in vectors that is closest to 
    target_vector in terms of cosine similarity.
    
    # Hint: You should use the cosine_similarity function that you already wrote.
    # Hint: For this lab, you can just use a for loop to iterate through vectors.
    '''
    # best_similarity = best_idx = -1
    # for i, vector in enumerate(vectors): # we COULD do this but this is super inefficient!!
    #     ... 
    return np.argmax( # gets the idx of the lowest cosine_similarity
        [cosine_similarity(target_vector, vector) for vector in vectors]
    )
