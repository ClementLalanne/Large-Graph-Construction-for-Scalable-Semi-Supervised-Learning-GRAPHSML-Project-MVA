import numpy as np 
import anchors

def closest_indices(data, anchors, K):
    ret = np.zeros((data.shape[0], K), dtype = int)
    for i in range(data.shape[0]):
        tmp = np.linalg.norm(data[i][:, np.newaxis]-anchors, axis=1)
        ret[i, :] = np.argsort(tmp)[-K:]
    return ret

def kernel_association(data, anchors, K, kernel):
    ret = np.zeros((data.shape[0], K), dtype = float)
    ind = closest_indices(data, anchors, K)
    for i in range(data.shape[0]):
        for k in range(K):
            ret[i, ind[k]] = kernel(data[i], anchors[ind[k]])
        ret[i] = ret[i] / np.sum(ret[i])
    return ret 

def matrix_association(data, anchors, K, matrix):
    ret = np.zeros((data.shape[0], K), dtype = float)
    ind = closest_indices(data, anchors, K)
    for i in range(data.shape[0]):
        for k in range(K):
            ret[i, ind[k]] = matrix[i, ind[k]]
    return ret 

def LEA(data, anchors, K):
    ret = np.zeros((data.shape[0], K), dtype = float)
    return ret #Not implemented yet