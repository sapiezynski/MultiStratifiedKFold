from sklearn.model_selection import StratifiedKFold
import numpy as np


def hash_rows(array):
    """
    given a (N, M) shaped array it returns a (N, ) shaped vector that contains numeric identifiers of unique rows
    useful for stratifying across multiple variables in sklearn StratifiedKFold
    """
    result = np.zeros((array.shape[0], ), dtype=np.int64)
    for idx, row in enumerate(array):
        result[idx] = hash(tuple(row))
    return result

class MultiStratifiedKFold(StratifiedKFold):
    def split(self, X, y, groups=None):
        if type(y) == tuple:
            y = np.stack(y, axis=1)
        y = hash_rows(y)
        return super(MultiStratifiedKFold, self).split(X, y)