from __future__ import division

import numpy as np
import operator
import itertools
from heapq import heappush, heappop
import random

__all__ = ['CoverTree', 'distance_matrix']

#data:输入的数据
#LeafSize:LeaveNode所能包含的最大数据
#n:数据个数
#pt_shape:数据的维度
#root:根节点，InnerNode类型


class CoverTree(object):
    class _lazy_child_dist(dict):
        def __init__(self, base, *a, **kw):
            dict.__init__(self, *a, **kw)
            self.b = base

        def __missing__(self, i):
            self[i] = value = self.b ** i
            return value

    class _lazy_heir_dist(dict):
        def __init__(self, base, *a, **kw):
            dict.__init__(self, *a, **kw)
            self.b = base

        def __missing__(self, i):
            self[i] = value = self.b ** (i + 1) / (self.b - 1)
            return value

    def __init__(self, data, distance, leafsize=10, base=2):
        self.data = np.asarray(data)
        self.n = self.data.shape[0]
        self.pt_shape = self.data.shape[1:]
        self.distance = distance
        self.leafsize = leafsize
        if self.leafsize < 1:
            raise ValueError("leafsize must be at least 1")

        self._child_d = CoverTree._lazy_child_dist(base)
        self._heir_d = CoverTree._lazy_heir_dist(base)

        self.tree = self._build()

    class _Node(object):
        pass

    #children:子节点：InnerNode类型
    #ctr_idx:当前结点对应的数据的下标
    #level:当前节点所在的层
    #num_children:该节点的后继结点的数量
    #radius:当前层的半径
    class _InnerNode(_Node):
        def __init__(self, ctr_idx, level, radius, children):
            self.ctr_idx = ctr_idx
            self.level = level
            self.radius = radius
            self.children = children
            self.num_children = sum(c.num_children for c in children)

        def __repr__(self):
            return ("<_InnerNode: ctr_idx=%d, level=%d (radius=%f), "
                    "len(children)=%d, num_children=%d>" %
                    (self.ctr_idx, self.level,
                     self.radius, len(self.children), self.num_children))

    #ctr_idx:当前结点对应的数据的下标
    # num_children:该节点对应的所有数据的个数
    # radius:叶子结点中的数据到其父节点的最远距离
    class _LeafNode(_Node):
        def __init__(self, idx, ctr_idx, radius):
            self.idx = idx
            self.ctr_idx = ctr_idx
            self.radius = radius
            self.num_children = len(idx)

        def __repr__(self):
            return ('_LeafNode(idx=%s, ctr_idx=%d, radius=%f)' %
                    (repr(self.idx), self.ctr_idx, self.radius))

    def _build(self):
        child_d = self._child_d
        heir_d = self._heir_d

        def split_with_dist(dmax, Dmax, pts_p_ds):
            near_p_ds = []
            far_p_ds = []

            new_pts_len = 0
            for i in xrange(len(pts_p_ds)):
                idx, dp = pts_p_ds[i]
                if dp <= dmax:
                    near_p_ds.append((idx, dp))
                elif dp <= Dmax:
                    far_p_ds.append((idx, dp))
                else:
                    pts_p_ds[new_pts_len] = pts_p_ds[i]
                    new_pts_len += 1
            pts_p_ds[:] = pts_p_ds[:new_pts_len]

            return near_p_ds, far_p_ds

        def split_without_dist(q_idx, dmax, Dmax, pts_p_ds):  # **
            near_q_ds = []
            far_q_ds = []

            new_pts_len = 0
            for i in xrange(len(pts_p_ds)):
                idx, dp = pts_p_ds[i]
                dq = self.distance(self.data[q_idx], self.data[idx])
                if dq <= dmax:
                    near_q_ds.append((idx, dq))
                elif dq <= Dmax:
                    far_q_ds.append((idx, dq))
                else:
                    pts_p_ds[new_pts_len] = pts_p_ds[i]
                    new_pts_len += 1
            pts_p_ds[:] = pts_p_ds[:new_pts_len]

            return near_q_ds, far_q_ds

        def construct(p_idx, near_p_ds, far_p_ds, i):
            if len(near_p_ds) + len(far_p_ds) <= self.leafsize:
                idx = [ii for (ii, d) in itertools.chain(near_p_ds,
                                                         far_p_ds)]
                radius = max(d for (ii, d) in itertools.chain(near_p_ds,
                                                              far_p_ds,
                                                              [(0.0, None)]))
                node = CoverTree._LeafNode(idx, p_idx, radius)
                return node, []
            else:
                nearer_p_ds, so_so_near_p_ds = split_with_dist(
                    child_d[i - 1], child_d[i], near_p_ds)
                p_im1, near_p_ds = construct(p_idx, nearer_p_ds,
                                             so_so_near_p_ds, i - 1)

                if not near_p_ds:
                    return p_im1, far_p_ds
                else:
                    children = [p_im1]
                    while near_p_ds:
                        q_idx, _ = random.choice(near_p_ds)

                        near_q_ds, far_q_ds = split_without_dist(
                            q_idx, child_d[i - 1], child_d[i], near_p_ds)
                        near_q_ds2, far_q_ds2 = split_without_dist(
                            q_idx, child_d[i - 1], child_d[i], far_p_ds)
                        near_q_ds += near_q_ds2
                        far_q_ds += far_q_ds2

                        q_im1, unused_q_ds = construct(
                            q_idx, near_q_ds, far_q_ds, i - 1)

                        children.append(q_im1)

                        new_near_p_ds, new_far_p_ds = split_without_dist(
                            p_idx, child_d[i], child_d[i + 1], unused_q_ds)
                        near_p_ds += new_near_p_ds
                        far_p_ds += new_far_p_ds

                    p_i = CoverTree._InnerNode(p_idx, i, heir_d[i], children)
                    return p_i, far_p_ds

        if self.n == 0:
            self.root = CoverTree._LeafNode(idx=[], ctr_idx=-1, radius=0)
        else:
            p_idx = random.randrange(self.n)
            near_p_ds = [(j, self.distance(self.data[p_idx], self.data[j]))
                         for j in np.arange(self.n)]
            far_p_ds = []
            try:
                maxdist = 2 * max(near_p_ds, key=operator.itemgetter(1))[1]
            except ValueError:
                maxdist = 1

            maxlevel = 0
            while maxdist > child_d[maxlevel]:
                maxlevel += 1

            #根节点的生成,根节点是一个InnerNode
            self.root, unused_p_ds = construct(p_idx, near_p_ds,
                                               far_p_ds, maxlevel)


        def enum_leaves(node):
            if isinstance(node, CoverTree._LeafNode):
                return node.idx
            else:
                return list(itertools.chain.from_iterable(
                    enum_leaves(child)
                    for child in node.children))

        assert sorted(enum_leaves(self.root)) == range(self.data.shape[0])

        return True



    def _query(self, p, k=1, eps=0, distance_upper_bound=np.inf):
        if not self.root:
            return []

        dist_to_ctr = self.distance(p, self.data[self.root.ctr_idx])
        min_distance = max(0.0, dist_to_ctr - self.root.radius)


        q = [(min_distance,
              dist_to_ctr,
              self.root)]

        neighbors = []

        if eps == 0:
            epsfac = 1
        else:
            epsfac = 1 / (1 + eps)

        while q:
            min_distance, dist_to_ctr, node = heappop(q)
            if isinstance(node, CoverTree._LeafNode):
                # brute-force
                for i in node.idx:
                    if i == node.ctr_idx:
                        d = dist_to_ctr
                    else:
                        d = self.distance(p, self.data[i])
                    if d <= distance_upper_bound:
                        if len(neighbors) == k:
                            heappop(neighbors)
                        heappush(neighbors, (-d, i))
                        if len(neighbors) == k:
                            distance_upper_bound = -neighbors[0][0]
            else:

                if min_distance > distance_upper_bound * epsfac:

                    break

                for child in node.children:
                    if child.ctr_idx == node.ctr_idx:
                        d = dist_to_ctr
                    else:
                        d = self.distance(p, self.data[child.ctr_idx])
                    min_distance = max(0.0, d - child.radius)

                    if min_distance <= distance_upper_bound * epsfac:
                        heappush(q, (min_distance, d, child))

        return sorted([(-d, i) for (d, i) in neighbors])

    def query(self, x, k=1, eps=0, distance_upper_bound=np.inf):
        x = np.asarray(x)
        if self.pt_shape:
            if np.shape(x)[-len(self.pt_shape):] != self.pt_shape:
                raise ValueError("x must consist of vectors of shape %s "
                                 "but has shape %s"
                                 % (self.pt_shape, np.shape(x)))
            retshape = np.shape(x)[:-len(self.pt_shape)]
        else:
            retshape = np.shape(x)

        if retshape:
            if k is None:
                dd = np.empty(retshape, dtype=np.object)
                ii = np.empty(retshape, dtype=np.object)
            elif k > 1:
                dd = np.empty(retshape + (k,), dtype=np.float)
                dd.fill(np.inf)
                ii = np.empty(retshape + (k,), dtype=np.int)
                ii.fill(self.n)
            elif k == 1:
                dd = np.empty(retshape, dtype=np.float)
                dd.fill(np.inf)
                ii = np.empty(retshape, dtype=np.int)
                ii.fill(self.n)
            else:
                raise ValueError("Requested %s nearest neighbors; "
                                 "acceptable numbers are integers greater "
                                 "than or equal to one, or None")
            for c in np.ndindex(retshape):
                hits = self._query(
                    x[c], k=k, eps=eps,
                    distance_upper_bound=distance_upper_bound)
                if k is None:
                    dd[c] = [d for (d, i) in hits]
                    ii[c] = [i for (d, i) in hits]
                elif k > 1:
                    for j in range(len(hits)):
                        dd[c + (j,)], ii[c + (j,)] = hits[j]
                elif k == 1:
                    if len(hits) > 0:
                        dd[c], ii[c] = hits[0]
                    else:
                        dd[c] = np.inf
                        ii[c] = self.n
            return dd, ii
        else:
            hits = self._query(x, k=k, eps=eps,
                               distance_upper_bound=distance_upper_bound)
            if k is None:
                return [d for (d, i) in hits], [i for (d, i) in hits]
            elif k == 1:
                if len(hits) > 0:
                    return hits[0]
                else:
                    return np.inf, self.n
            elif k > 1:
                dd = np.empty(k, dtype=np.float)
                dd.fill(np.inf)
                ii = np.empty(k, dtype=np.int)
                ii.fill(self.n)
                for j in range(len(hits)):
                    dd[j], ii[j] = hits[j]
                return dd, ii
            else:
                raise ValueError("Requested %s nearest neighbors; "
                                 "acceptable numbers are integers greater "
                                 "than or equal to one, or None")


        x = np.asarray(x)
        if self.pt_shape and x.shape[-len(self.pt_shape):] != self.pt_shape:
            raise ValueError("Searching for a point of shape %s in a " \
                             "CoverTree with points of shape %s" %
                             (x.shape[-len(self.pt_shape):],
                              self.pt_shape))

        if len(x.shape) == 1:
            return self._query_ball_point(x, r, eps)
        else:
            if self.pt_shape:
                retshape = x.shape[:-len(self.pt_shape)]
            else:
                retshape = x.shape
            result = np.empty(retshape, dtype=np.object)
            for c in np.ndindex(retshape):
                result[c] = self._query_ball_point(x[c], r, eps=eps)
            return result
