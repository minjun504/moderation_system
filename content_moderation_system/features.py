import torch
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def build_vectoriser(ngram_range=(1, 1), max_features=None):
    return TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features
    )

def csr_to_tensor(X):
    coo = X.tocoo()
    coords = torch.from_numpy(np.vstack((coo.row, coo.col)).astype(np.int64))
    vals = torch.from_numpy(coo.data.astype(np.float32))
    shape = torch.Size(coo.shape)
    return torch.sparse_coo_tensor(coords, vals, shape)

