import numpy as np

def make_onehot(class_id: int, T: int, num_of_class: int) -> np.ndarray:
    return np.tile(np.eye(num_of_class)[class_id], (T, 1))