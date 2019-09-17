import gc

import numpy as np
import torch


def check_nan(*tensors):
    for t in tensors:
        if t.isnan().sum() > 0:
            print('Found NaN:', t.shape)
            return True
    return False


def check_nan_np(*tensors):
    for t in tensors:
        if np.isnan(t).sum() > 0:
            print('Found NaN:', t.shape)
            return True
    return False


def convert_to_dense(mu, nu, spmatrix, symmetrize=False):
    '''
    Convert mu, nu indices and matrix values into dense matrix.

    :param mu:
    :param nu:
    :param spmatrix:
    :return:
    '''
    if isinstance(spmatrix, torch.Tensor):
        idx = torch.cat([mu[:, 0:1], nu[:, 0:1]], dim=1)
        imax = torch.max(mu[:, 0:1]) + 1
        dense = torch.sparse.FloatTensor(idx.t(), spmatrix,
                                         torch.Size([imax, imax])).to_dense()
        if symmetrize:
            dense = 0.5 * dense + 0.5 * dense.t()
    else:  # numpy
        imax = np.max(mu[:, 0:1]) + 1

        dense = np.zeros((imax, imax), dtype=np.float32)
        dense[mu[:, 0], nu[:, 0]] = spmatrix
        if symmetrize:
            dense = 0.5 * dense + 0.5 * dense.T

    return dense


def tensor_meta_data(tensor):
    element_count = 1;
    for dim in tensor.size():
        element_count = element_count * dim
    size_in_bytes = element_count * tensor.element_size()
    dtype = str(tensor.dtype).replace("torch.", "")
    size = str(tensor.size()).replace("torch.Size(", "").replace(")", "")
    return f"{size_in_bytes / 1000000:5.1f}MB" + \
           f" {dtype}{size} {type(tensor).__name__} {tensor.device}"


def print_tensors():
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            print(tensor_meta_data(obj))