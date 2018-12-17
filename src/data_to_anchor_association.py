import numpy as np 
import anchors 
import point_to_anchor
from scipy import sparse

class AnchorsAssociation():
    def __init__(self):
        self.drawer = anchors.AnchorsDrawer()

    def draw(self, data, K, mode='kmeans', association='LEA', kernel=None):
        if mode=='random':
            anchors = self.drawer.draw(data, K, mode='random')
            if association=='kernel based':
                Z = point_to_anchor.kernel_association(data, anchors, K, kernel)
            elif association=='LEA':
                Z = point_to_anchor.LEA(data, anchors, K)
            else:
                assert False
        elif mode=='kmeans':
            anchors =  self.drawer.draw(data, K, mode='kmeans')
            if association=='kernel based':
                Z = point_to_anchor.kernel_association(data, anchors, K, kernel)
            elif association=='LEA':
                Z = point_to_anchor.LEA(data, anchors, K)
            else:
                assert False
        elif mode=='fuzzy':
            anchors, matrix = self.drawer.draw(data, K, mode='fuzzy')
            if association=='kernel based':
                Z = point_to_anchor.kernel_association(data, anchors, K, kernel)
            elif association=='LEA':
                Z = point_to_anchor.LEA(data, anchors, K)
            elif association=='fuzzy':
                Z = point_to_anchor.matrix_association(data, anchors, K, matrix)
            else:
                assert False
        else:
            assert False
        return anchors, sparse.csr_matrix(Z)