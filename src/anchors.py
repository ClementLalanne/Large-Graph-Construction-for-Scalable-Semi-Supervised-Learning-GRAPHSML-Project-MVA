import numpy as np 
from scipy.cluster.vq import vq, kmeans, whiten
import skfuzzy as fuzz

class RandomAnchorsDrawer():
    def __init__(self):
        pass

    def draw(self, data, K):
        mean = np.mean(data, axis=0)
        cov = np.cov(data, rowvar=False)
        ret = np.random.multivariate_normal(mean, cov, size=K)
        return ret 

class KMeansAnchorsDrawer():
    def __init__(self):
        pass

    def draw(self, data, K):
        codebook, _ = kmeans(data, K)
        return codebook

class FuzzyCMeansAnchorsDrawer():
    def __init__(self):
        pass 

    def draw(self, data, K):
        c, tab, _, _, _, _, _ = fuzz.cmeans(data, K, 0.9, 0.1, 20, init=None, seed=None)
        return (c, tab)

class AnchorsDrawer():
    def __init__(self):
        self.random = RandomAnchorsDrawer()
        self.kmeans = KMeansAnchorsDrawer()
        self.fuzzy = FuzzyCMeansAnchorsDrawer()

    def draw(self, data, K, mode='kmeans'):
        if mode=='random':
            return self.random.draw(data, K)
        elif mode=='kmeans':
            return self.kmeans.draw(data, K)
        elif mode=='fuzzy':
            return self.fuzzy.draw(data, K)
        else:
            assert False

