from collections import deque
from queue import PriorityQueue
import random
import numpy as np


class VPTree(object):
    """
    An efficient data structure to perform nearest-neighbor
    search. 
    """

    def __init__(self, points, dist_fn):
        self.left = None
        self.right = None
        self.mu = None
        self.dist_fn = dist_fn

        # choose a better vantage point selection process
        self.vp = points.pop(random.randrange(len(points)))

        if len(points) < 1:
            return

        # choose division boundary at median of distances
        distances = [self.dist_fn(self.vp, p) for p in points]
        self.mu = np.median(distances)

        left_points = []  # all points inside mu radius
        right_points = []  # all points outside mu radius
        for i, p in enumerate(points):
            d = distances[i]
            if d >= self.mu:
                right_points.append(p)
            else:
                left_points.append(p)

        if len(left_points) > 0:
            self.left = VPTree(points=left_points, dist_fn=self.dist_fn)

        if len(right_points) > 0:
            self.right = VPTree(points=right_points, dist_fn=self.dist_fn)

    def is_leaf(self):
        return (self.left is None) and (self.right is None)

    ### Operations
    @staticmethod
    def get_nearest_neighbors(tree, q, k=1):
        """
        find k nearest neighbor(s) of q

        :param tree:  vp-tree
        :param q: a query point
        :param k: number of nearest neighbors

        """

        # buffer for nearest neightbors
        neighbors = PriorityQueue()

        # list of nodes ot visit
        visit_stack = deque([tree])

        # distance of n-nearest neighbors so far
        tau = np.inf
        ret = []

        while len(visit_stack) > 0:
            node = visit_stack.popleft()
            if node is None:
                continue

            d = tree.dist_fn(q, node.vp)
            if d < tau:
                neighbors.put((d, np.random.uniform(), node.vp))
                ret.append((d, node.vp))
                tau, _, _ = neighbors.get()

            if node.is_leaf():
                continue

            if d < node.mu:
                if d < node.mu + tau:
                    visit_stack.append(node.left)
                if d >= node.mu - tau:
                    visit_stack.append(node.right)
            else:
                if d >= node.mu - tau:
                    visit_stack.append(node.right)
                if d < node.mu + tau:
                    visit_stack.append(node.left)
        return ret


    @staticmethod
    def get_all_in_range(tree, q, tau):
        """
        find all points within a given radius of point q

        :param tree: vp-tree
        :param q: a query point
        :param tau: the maximum distance from point q
        """

        # buffer for nearest neightbors
        neighbors = []

        # list of nodes ot visit
        visit_stack = deque([tree])

        while len(visit_stack) > 0:
            node = visit_stack.popleft()
            if node is None:
                continue

            d = tree.dist_fn(q, node.vp)

            if d < tau:
                neighbors.append((d, node.vp))

            if node.is_leaf():
                continue

            if d < node.mu:
                if d < node.mu + tau:
                    visit_stack.append(node.left)
                if d >= node.mu - tau:
                    visit_stack.append(node.right)
            else:
                if d >= node.mu - tau:
                    visit_stack.append(node.right)
                if d < node.mu + tau:
                    visit_stack.append(node.left)
        return neighbors


