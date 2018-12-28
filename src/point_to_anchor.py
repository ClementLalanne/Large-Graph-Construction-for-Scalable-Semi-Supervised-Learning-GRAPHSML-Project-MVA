import numpy as np 
import anchors

epsilon = 1e-4

def closest_indices(data, anchors, s):
    ret = np.zeros((data.shape[0], s), dtype = int)
    for i in range(data.shape[0]):
        tmp = np.linalg.norm(np.tile(data[i],[anchors.shape[0],1])-anchors, axis=1)
        ret[i, :] = np.argsort(tmp)[:s]
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
    v = np.sort(z)[::-1]
    tmps = np.cumsum(v)
    crit = v - (np.ones(z.shape[0]) / (np.arange(z.shape[0]) + 1)) * (tmps - np.ones(z.shape[0]))
    indxs = np.where(crit>0)[0]
    if indxs.shape[0]:
        rho = np.max(indxs)
    else:
        rho = 0
    theta = 1/(rho+1) * (tmps[rho] - 1)
    return ((z-np.full(z.shape[0], theta)) * ((z-np.full(z.shape[0], theta))>0))


def LEA(data, anchors, s):
    K = anchors.shape[0]
    ret = np.zeros((data.shape[0], K), dtype = float)
    c_ind = closest_indices(data, anchors, s)
    for i in range(data.shape[0]):
        U = anchors[c_ind[i], :].T 
        def g(z):
            return np.power(np.linalg.norm(data[i] - U.dot(z)),2)/2
        def grad_g(z):
            return U.T.dot(U.dot(z) - data[i])
        def g_tilde(beta, v, z):
            return g(v) + grad_g(v).T.dot(z-v) + beta*np.power(np.linalg.norm(z-v), 2)/2
        z_old, z_new = np.ones(s)/s, np.ones(s)/s
        delta_old = 0.
        delta_new = 1.
        beta_global = 1.
        t=0
        while True:
            t +=1
            alpha = (delta_old-1)/delta_new
            v = z_new + alpha * (z_new-z_old)
            j=0
            while True:
                if j>40:
                    assert False
                beta = np.power(2, j) * beta_global
                z = simplex_projection(v - 1/beta * grad_g(v))
                if g(z) <= g_tilde(beta, v, z) + epsilon:
                    beta_global = beta 
                    z_old = z_new
                    z_new = z
                    break 
                j+=1
            delta_old = delta_new
            delta_new = (1+np.sqrt(1+4*delta_old**2))/2
            if np.linalg.norm(z_old-z_new)<epsilon:
                break
        for j in range(s):
            ret[i, c_ind[i][j]] = z_new[j]
    return ret 
