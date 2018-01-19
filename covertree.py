#-*- encoding: utf-8 -*-

from __future__ import division
import CoverTreePic
import time
import numpy as np
import operator
import itertools
from heapq import heappush, heappop
import random

__all__ = ['CoverTree']

#data:输入的数据
#LeafSize:LeaveNode所能包含的最大数据
#n:数据个数
#dataDimension:数据的维度
#root:根节点，InnerNode类型



class CoverTree(object):
    class lazyChildDist(dict):
        def __init__(self, base, *a, **kw):
            dict.__init__(self, *a, **kw)
            self.b = base

        def __missing__(self, i):
            self[i] = value = self.b ** i
            return value

    class lazyHaeresDist(dict):
        def __init__(self, base, *a, **kw):
            dict.__init__(self, *a, **kw)
            self.b = base

        def __missing__(self, i):
            self[i] = value = self.b ** (i + 1) / (self.b - 1)
            return value

    #distance:两个点之前的距离
    def __init__(self, data, distance, leafsize=10, base=2):
        self.data = np.asarray(data)
        self.n = self.data.shape[0]
        self.dataDimension = self.data.shape[1:]
        self.distance = distance
        self.leafsize = leafsize

        self._childDist = CoverTree.lazyChildDist(base)
        self._haeresDist = CoverTree.lazyHaeresDist(base)

        self.tree = self._build()

    class _Node(object):
        pass

    #children:子节点：InnerNode类型
    #currentIndex:当前结点对应的数据的下标
    #level:当前节点所在的层
    #childrenNum:该节点的后继结点的数量
    #radius:当前层的半径
    class _InnerNode(_Node):
        def __init__(self, currentIndex, level, radius, children):
            self.currentIndex = currentIndex
            self.level = level
            self.radius = radius
            self.children = children
            self.childrenNum = sum(c.childrenNum for c in children)

        def __repr__(self):
            return ("<_InnerNode: currentIndex=%d, level=%d (radius=%f), "
                    "len(children)=%d, childrenNum=%d>" %
                    (self.currentIndex, self.level,
                     self.radius, len(self.children), self.childrenNum))

    #currentIndex:当前结点对应的数据的下标
    # childrenNum:该节点对应的所有数据的个数
    # radius:叶子结点中的数据到其父节点的最远距离
    class _LeafNode(_Node):
        def __init__(self, idx, currentIndex, radius):
            self.idx = idx
            self.currentIndex = currentIndex
            self.radius = radius
            self.childrenNum = len(idx)

        def __repr__(self):
            return ('_LeafNode(idx=%s, currentIndex=%d, radius=%f)' %
                    (repr(self.idx), self.currentIndex, self.radius))

    def _build(self):
        childDist = self._childDist
        haeresDist = self._haeresDist


        #pointDis:所有点到目标点的距离&&各自的下标的二维元组
        #nearPoints:满足到目标点的距离在0-dmax之间的点的二维元组(包含点的下标及到目标点的距离)
        #farPoints:满足到目标点的距离在dmax-Dmax之间的点的二维元组(包含点的下标及到目标点的距离）
        #dmax&&Dmax:筛选点的阀值
        def splitWithDist(dmax, Dmax, pointDis):
            nearPoints = []
            farPoints = []

            pointCount = 0
            for i in xrange(len(pointDis)):
                idx, dp = pointDis[i]
                if dp <= dmax:
                    nearPoints.append((idx, dp))
                elif dp <= Dmax:
                    farPoints.append((idx, dp))
                else:
                    pointDis[pointCount] = pointDis[i]
                    pointCount += 1
            pointDis[:] = pointDis[:pointCount]

            return nearPoints, farPoints

        def splitWithoutDist(qIndex, dmax, Dmax, pointDis):
            nearQPoints = []
            farQPoints = []

            pointCount = 0
            for i in xrange(len(pointDis)):
                idx, dp = pointDis[i]
                dq = self.distance(self.data[qIndex], self.data[idx])
                if dq <= dmax:
                    nearQPoints.append((idx, dq))
                elif dq <= Dmax:
                    farQPoints.append((idx, dq))
                else:
                    pointDis[pointCount] = pointDis[i]
                    pointCount += 1
            pointDis[:] = pointDis[:pointCount]

            return nearQPoints, farQPoints

        def construct(pIndex, nearPoints, farPoints, i):
            if len(nearPoints) + len(farPoints) <= self.leafsize:
                idx = [ii for (ii, d) in itertools.chain(nearPoints,
                                                         farPoints)]
                radius = max(d for (ii, d) in itertools.chain(nearPoints,
                                                              farPoints,
                                                              [(0.0, None)]))
                node = CoverTree._LeafNode(idx, pIndex, radius)
                return node, []
            else:
                nearerPoints, commonlyNearPoints = splitWithDist(
                    childDist[i - 1], childDist[i], nearPoints)
                pChild, nearPoints = construct(pIndex, nearerPoints,
                                              commonlyNearPoints, i - 1)

                if not nearPoints:
                    return pChild, farPoints
                else:
                    children = [pChild]
                    while nearPoints:
                        qIndex, _ = random.choice(nearPoints)

                        nearQPoints, farQPoints = splitWithoutDist(
                            qIndex, childDist[i - 1], childDist[i], nearPoints)
                        newNearQPoints, newFarQPoints = splitWithoutDist(
                            qIndex, childDist[i - 1], childDist[i], farPoints)
                        nearQPoints += newNearQPoints
                        farQPoints += newFarQPoints

                        qChild, unused_q_ds = construct(
                            qIndex, nearQPoints, farQPoints, i - 1)

                        children.append(qChild)

                        newNearPoints, newFarPoints = splitWithoutDist(
                            pIndex, childDist[i], childDist[i + 1], unused_q_ds)
                        nearPoints += newNearPoints
                        farPoints += newFarPoints

                    p_i = CoverTree._InnerNode(pIndex, i, haeresDist[i], children)
                    # if not children:
                    return p_i, farPoints

        if self.n == 0:
            self.root = CoverTree._LeafNode(idx=[], currentIndex=-1, radius=0)
        else:
            pIndex = random.randrange(self.n)
            nearPoints = [(j, self.distance(self.data[pIndex], self.data[j]))
                         for j in np.arange(self.n)]
            farPoints = []
            try:
                maxdist = 2 * max(nearPoints, key=operator.itemgetter(1))[1]
            except ValueError:
                maxdist = 1

            maxlevel = 0
            while maxdist > childDist[maxlevel]:
                maxlevel += 1

            #根节点的生成,根节点是一个InnerNode
            self.root, unusedPDis = construct(pIndex, nearPoints,
                                               farPoints, maxlevel)
            # CoverTreePic.UpdateTreeStruct(self.root, self.data)


        def enumLeaves(node):
            if isinstance(node, CoverTree._LeafNode):
                return node.idx
            else:
                return list(itertools.chain.from_iterable(
                    enumLeaves(child)
                    for child in node.children))

        assert sorted(enumLeaves(self.root)) == range(self.data.shape[0])

        return True



    def _query(self, p, k=1, eps=0, distanceUpperBound=np.inf):
        if not self.root:
            return []

        pToCurrentDis = self.distance(p, self.data[self.root.currentIndex])
        minDis = max(0.0, pToCurrentDis - self.root.radius)


        q = [(minDis,
              pToCurrentDis,
              self.root)]

        neighbors = []

        if eps == 0:
            epsfac = 1
        else:
            epsfac = 1 / (1 + eps)

        while q:
            minDis, pToCurrentDis, node = heappop(q)
            if isinstance(node, CoverTree._LeafNode):
                # brute-force
                for i in node.idx:
                    if i == node.currentIndex:
                        d = pToCurrentDis
                    else:
                        d = self.distance(p, self.data[i])
                    if d <= distanceUpperBound:
                        if len(neighbors) == k:
                            heappop(neighbors)
                        heappush(neighbors, (-d, i))
                        if len(neighbors) == k:
                            distanceUpperBound = -neighbors[0][0]
            else:
                if minDis > distanceUpperBound * epsfac:

                    break

                for child in node.children:
                    if child.currentIndex == node.currentIndex:
                        d = pToCurrentDis
                    else:
                        d = self.distance(p, self.data[child.currentIndex])
                    minDis = max(0.0, d - child.radius)

                    if minDis <= distanceUpperBound * epsfac:
                        heappush(q, (minDis, d, child))

        return sorted([(-d, i) for (d, i) in neighbors])

    def query(self, x, k=1, eps=0, distanceUpperBound=np.inf):
        x = np.asarray(x)
        if self.dataDimension:
            retshape = np.shape(x)[:-len(self.dataDimension)]
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

            for c in np.ndindex(retshape):
                hits = self._query(
                    x[c], k=k, eps=eps,
                    distanceUpperBound=distanceUpperBound)
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
                               distanceUpperBound=distanceUpperBound)
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
