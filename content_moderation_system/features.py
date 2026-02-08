import torch
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def csr_to_tensor(X):
    coo = X.tocoo()
    coords = torch.from_numpy(np.vstack((coo.row, coo.col)).astype(np.int64))
    vals = torch.from_numpy(coo.data.astype(np.float32))
    shape = torch.Size(coo.shape)
    return torch.sparse_coo_tensor(coords, vals, shape)

