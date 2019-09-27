from collections import namedtuple
Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')
import numpy as np

class Calculation():
    def __init__(self):
        pass

    def getIOU(self, bb_hat, bb_gt):
        N = bb_hat.shape[0]
        iou = np.zeros((N))
        for i in range(0, N):
            ra = Rectangle(bb_hat[i, 0], bb_hat[i, 1], bb_hat[i, 0] + bb_hat[i, 2], bb_hat[i, 1] + bb_hat[i, 3])
            rb = Rectangle(bb_gt[i, 0], bb_gt[i, 1], bb_gt[i, 0] + bb_gt[i, 2], bb_gt[i, 1] + bb_gt[i, 3])
            intersection = self.getIntersection(ra, rb)
            union = self.getArea(ra) + self.getArea(rb) - intersection
            iou[i] = intersection / union
        return iou

    def getIntersection(self, a, b):  # returns None if rectangles don't intersect
        dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
        dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
        if (dx >= 0) and (dy >= 0):
            return dx * dy

    def getArea(self, r):
        x = (r.xmax - r.xmin)
        y = (r.ymax - r.ymin)
        return x*y