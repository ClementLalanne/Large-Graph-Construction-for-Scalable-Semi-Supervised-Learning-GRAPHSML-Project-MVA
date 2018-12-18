import numpy as np 
import anchors

def closest_indices(data, anchors, s):
    ret = np.zeros((data.shape[0], s), dtype = int)
    for i in range(data.shape[0]):
        tmp = np.linalg.norm(data[i][:, np.newaxis]-anchors, axis=1)
        ret[i, :] = np.argsort(tmp)[-s:]
    return ret

def kernel_association(data, anchors, s, kernel):
    K = anchors.shape[0]
    ret = np.zeros((data.shape[0], K), dtype = float)
    ind = closest_indices(data, anchors, s)
    for i in range(data.shape[0]):
        for k in range(s):
            ret[i, ind[k]] = kernel(data[i], anchors[ind[k]])
        ret[i] = ret[i] / np.sum(ret[i])
    return ret 

def matrix_association(data, anchors, s, matrix):
    K = anchors.shape[0]
    ret = np.zeros((data.shape[0], K), dtype = float)
    ind = closest_indices(data, anchors, s)
    for i in range(data.shape[0]):
        for k in range(s):
            ret[i, ind[k]] = matrix[i, ind[k]]
    return ret 

def simplex_projection(z):
    pass

def LEA(data, anchors, s):
    K = anchors.shape[0]
    ret = np.zeros((data.shape[0], K), dtype = float)
    return ret #Not implemented yet