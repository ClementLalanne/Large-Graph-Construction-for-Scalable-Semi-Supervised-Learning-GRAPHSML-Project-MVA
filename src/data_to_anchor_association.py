import numpy as np 
import anchors 
import point_to_anchor
from scipy import sparse

class AnchorsAssociation():
    def __init__(self):
        self.drawer = anchors.AnchorsDrawer()

    def draw(self, data, m, s, mode='kmeans', association='LEA', kernel=None):
        if mode=='random':
            anchors = self.drawer.draw(data, m, mode='random')
            if association=='kernel based':
                Z = point_to_anchor.kernel_association(data, anchors, s, kernel)
            elif association=='LEA':
                Z = point_to_anchor.LEA(data, anchors, s)
            else:
                assert False
        elif mode=='kmeans':
            anchors =  self.drawer.draw(data, m, mode='kmeans')
            if association=='kernel based':
                Z = point_to_anchor.kernel_association(data, anchors, s, kernel)
            elif association=='LEA':
                Z = point_to_anchor.LEA(data, anchors, s)
            else:
                assert False
        elif mode=='fuzzy':
            anchors, matrix = self.drawer.draw(data, m, mode='fuzzy')
            if association=='kernel based':
                Z = point_to_anchor.kernel_association(data, anchors, s, kernel)
            elif association=='LEA':
                Z = point_to_anchor.LEA(data, anchors, s)
            elif association=='fuzzy':
                Z = point_to_anchor.matrix_association(data, anchors, s, matrix)
            else:
                assert False
        else:
            assert False
        return anchors, Z