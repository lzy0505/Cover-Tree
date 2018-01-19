# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 18:03:13 2018

@author: Yang Adam
"""

from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import covertree

Points=[]
LevelWide = []
fig = plt.figure()
ax = Axes3D(fig)
#通过Cover树的结构画树的结构图

def UpdateTreeStruct(root,data):
    level = root.level
    print level
    #initalize(level)
    Points.append(data[root.currentIndex])

    LevelWide.append(root)
    newLevel=[]
    ind=0
    for i in range(5,4-root.level,-1):#对于树的每一层
        for point in Points:
            ax.scatter(point[0], point[1], i, s=30, c='r', marker='o')
            #print point

        for counter in xrange(len(LevelWide)):#已在队列里的每一节点的儿子进入队列
            LevelPoint=LevelWide[counter]
            if i!=5-root.level:
                draw_line(np.append(data[LevelPoint.currentIndex], i), np.append(data[LevelPoint.currentIndex], i - 1))
            if not isinstance(LevelPoint,covertree.CoverTree._LeafNode):
                t=LevelPoint.children
                for child in t:
                    if child not in LevelWide:
                        newLevel.append(child)
                        Points.append(data[child.currentIndex])
                        draw_line(np.append(data[LevelPoint.currentIndex], i), np.append(data[child.currentIndex], i - 1))
                       # ax.scatter(data[LevelPoint.currentIndex][0], data[LevelPoint.currentIndex][1], i, s=30, c='r', marker='o')
                        #ax.scatter(data[child.currentIndex][0], data[child.currentIndex][1], i, s=30, c='r',marker='o')
                    if isinstance(child,covertree.CoverTree._LeafNode):
                        if isinstance(child.currentIndex,np.int64):
                            for index in child.currentIndex.flat:
                                    Points.append(data[index])
                        else:
                                Points.append(data[index])
                for p in newLevel:
                    LevelWide.append(p)
                newLevel=[]

    plt.show()


#两点连线段
def draw_line(point1,point2):
    ax.plot([point1[0],point2[0]], [point1[1],point2[1]],[point1[2],point2[2]])

#计算两点之间的距离
def Distance(point1,point2):
    return np.sqrt(np.sum(np.square(point1-point2)))

#可视化的初始化：坐标系初始化，层数初始化
def initalize(nLevel):
    ax.set_xlabel('X')
    ax.set_xlim(0, 6)
    ax.set_ylabel('Y')
    ax.set_ylim(0, 6)
    ax.set_zlabel('Z')
    ax.set_zlim(0, 6)
    a = np.arange(0, 4, 0.5)
    b = np.arange(0, 4, 0.5)
    X, Y = np.meshgrid(a, b)
    # 具体函数方法可用 help(function) 查看，如：help(ax.plot_surface)
    Z = X - X
    # for i in range(nLevel):
    #     ax.plot_wireframe(X, Y, Z, rstride=2, cstride=2)
    #     Z=Z-1


#
# point=(max_1-min_1)*np.random.rand(5,2)
# z = np.zeros(5)
# o = np.ones(5)
# s = o+o
# points_0=np.c_[point,z]
# points_1=np.c_[point,o]
# points_2=np.c_[point,s]
# #基于ax变量绘制三维图
# #xs表示x方向的变量
# #ys表示y方向的变量
# #zs表示z方向的变量，这三个方向上的变量都可以用list的形式表示
# #m表示点的形式，o是圆形的点，^是三角形（marker)
# #c表示颜色（color for short）
# ax.scatter(points_0[:,0],points_0[:,1],points_0[:,2],s=30, c = 'r', marker = 'o') #
# ax.scatter(points_1[0:3,0],points_1[0:3,1],points_1[0:3,2],s=30, c = 'y', marker = 'o') #
# ax.scatter(points_2[2,0],points_2[2,1],points_2[2,2],s=30, c = 'b', marker = 'o') #
# #ax.scatter(1, 2, 1,s=100, c = 'y', marker = 'o') #点为红色三角形
#
# draw_line(point1=points_0[3],point2=points_1[2])
# draw_line(point1=points_0[4],point2=points_1[1])
# draw_line(point1=points_0[0],point2=points_1[0])
# draw_line(point1=points_0[1],point2=points_1[1])
# draw_line(point1=points_0[2],point2=points_1[2])
# draw_line(point1=points_1[0],point2=points_2[2])
# draw_line(point1=points_1[1],point2=points_2[2])
# draw_line(point1=points_1[2],point2=points_2[2])
