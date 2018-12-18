import numpy as np 
from scipy import sparse
from scipy.sparse.linalg import inv
import anchors 
import point_to_anchor
import data_to_anchor_association

def number_classes(Y):
    return np.argsort(Y)[-1] 

def hide_labels(Y, n):
    """Y : classes (1D array), n : number of labels to keep per class"""
    ret = np.zeros(Y.shape[0])
    nclasses = number_classes(Y)
    for i in range(1, nclasses+1):
        ind = np.where(Y==i)
        ind_to_keep = np.random.permutation(ind)[:n]
        ret[ind_to_keep] = Y[ind_to_keep]
    return ret 

def class_to_one_hot(c, n):
    ret = np.zeros(n)
    ret[c-1] = 1. 
    return ret 

def one_hot_to_class(oh):
    c = np.argsort(oh)[-1] +1
    return c

class SSLSolver():
    def __init__(self):
        self.data_to_anchors = data_to_anchor_association.AnchorsAssociation()

    def fit(self, data, Y, m, s, mode='kmeans', association='LEA', kernel=None, tuning_param=1.):

        nclasses = number_classes(Y)
        labeled_indices = np.where(Y>0)

        anchors, Z = self.data_to_anchors.draw(data, m, s, mode, association, kernel)

        Z_l = Z[labeled_indices, :]
        Y_one_hot = np.zeros(labeled_indices.shape[0], nclasses)
        for i in range(labeled_indices.shape[0]):
            Y_one_hot[i] = class_to_one_hot(labeled_indices[i], nclasses)

        Z_sparse = sparse.csr_matrix(Z)
        Z_l_sparse = sparse.csr_matrix(Z_l)
        Y_one_hot_sparse = sparse.csr_matrix(Y_one_hot)

        delta = sparse.diags(Z_sparse.sum(axis=0), 0)
        tmp = Z_sparse.T.dot(Z_sparse)
        reduced_lap_sparse = tmp - tmp.dot(inv(delta)).dot(tmp)

        A_star_sparse = inv(Z_l_sparse.T.dot(Z_l_sparse) + tuning_param * reduced_lap_sparse).dot(Z_l_sparse.T).dot(Y_one_hot_sparse)

        A_star = A_star_sparse.to_array()

        F = Z.dot(A_star)

        lambd = np.sum(F, axis=0)
        Y_one_hot_post = F / lambd[np.newaxis, :]

        Y_post = np.zeros(Y.shape[0], dtype=int)
        for i in range(Y.shape[0]):
            Y_post[i] = one_hot_to_class(Y_one_hot_post[i])

        return Y_post, anchors, Z