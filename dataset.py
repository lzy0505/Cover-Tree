#-*- encoding: utf-8 -*-

import sqlite3
import numpy as np
import math
from covertree import CoverTree
from scipy.spatial.distance import euclidean
import sys
import datetime
import random
sys.setrecursionlimit(100000)

#preTimes:每次预测所选取的点的个数
#dataset:所选择的数据
dataset='whiteWine'
#dataset = 'abalone'
#dataset = 'iris'
epsilon = 3.0
preTimes=2450
rightTimes=0
ctTime=0
knnTime=0


def run():
    global knnTime,ctTime,ct
    #create_table()
    connect_to_sql()
    #readData()
    getData()
    constructTree()
    #print ct.query([3,3], 10)
    for index in xrange(preTimes):
        runKNN()
    precision()
    print "构造树所花费时间:",ctTime/1000.0,'ms'
    print "K-NN算法所花费时间",knnTime/1000.0,'ms'


def readData():
    global cur, conn
    # for line in open("abalone.data"):
    #     l = line.split(',')
    #     if l[0] is 'M':
    #         l[0] = 0.0
    #     elif l[0] is 'F':
    #         l[0] = 1.0
    #     else:
    #         l[0] = 0.5
    #     l[8] = l[8].rstrip('\n')
    #     if l[8] not in classlist:
    #         classlist.append(l[8])
    for line in open("whiteWine.csv"):
        l = line.split(';')
        l[11] = l[11].rstrip('\n')
        try:
            cur.execute('INSERT INTO whiteWine values(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)'%(l[0],l[1],l[2],l[3],l[4],l[5],l[6],l[7],l[8],l[9],l[10],l[11]))
        except Exception as e:
            print e
            print l
    conn.commit()


def create_table():
    # connect to database
    DB = sqlite3.connect('dataset.db')
    # cursor of database
    cDB = DB.cursor()

    # delete all table in database
    #cDB.execute("DROP TABLE iris")
    print "-INIT- Table data has been deleted."

    # create table
    # cDB.execute("CREATE TABLE abalone (\n"
    # + "Sex REAL  NOT NULL,\n"
    # + "Length REAL NOT NULL,\n"
    # + "Diameter REAL NOT NULL,\n"
    # + "Height REAL NOT NULL,\n"
    # + "Whole_weight REAL NOT NULL,\n"
    # + "Shucked_weight REAL NOT NULL,\n"
    # + "Viscera_weight REAL NOT NULL,\n"
    # + "Shell_weight REAL NOT NULL,\n"
    # + "Rings INTEGER NOT NULL"
    # + ")")

    cDB.execute("CREATE TABLE whiteWine (\n"
    + "fixedAcidity REAL  NOT NULL,\n"
    + "volatileAcidity REAL NOT NULL,\n"
    + "citricAcid REAL NOT NULL,\n"
    + "residualSugar REAL NOT NULL,\n"
    + "chlorides REAL NOT NULL,\n"
    + "freeSulfurDioxide REAL NOT NULL,\n"
    + "totalSulfurDioxide REAL NOT NULL,\n"
    + "density REAL NOT NULL,\n"
    + "pH REAL NOT NULL,\n"
    + "sulphates REAL NOT NULL,\n"
    + "alcohol REAL NOT NULL,\n"
    + "quality INTEGER NOT NULL"
    + ")")
    print "-INIT- Table has been created."


def connect_to_sql():
    global cur, conn
    try:
        conn = sqlite3.connect('dataset.db')
        cur = conn.cursor()
        print '-PREDICTION- Connect to database successfully.'
    except Exception as e:
        print '-- An {} exception occured.'.format(e)


def getData():
    global data, label, cur, classlist
    classlist = []
    cur.execute("SELECT * FROM %s" % dataset)
    rst = cur.fetchone()
    numftr = len(rst)-1
    data = np.zeros((0, numftr), dtype=np.double)
    ftr = np.zeros((1, numftr), dtype=np.double)
    label = []
    cur.execute("SELECT * FROM %s" % dataset)
    rst = cur.fetchall()
    for instance in rst:
        for i in xrange(len(instance)-1):
            ftr[0, i] = instance[i]
        data = np.concatenate((data, ftr))
        label.append(instance[len(instance)-1])
        if instance[len(instance)-1] not in classlist:
            classlist.append(instance[len(instance)-1])
    print data

#label:数据类型
def getLabel(indexlist):
    global classlist, label
    labelcounter = []
    for i in classlist:
        labelcounter.append(0)
    for index in indexlist:
        labelcounter[classlist.index(label[index])] += 1
    max = 0
    maxindex = -1
    for i in xrange(len(labelcounter)):
        if max < labelcounter[i]:
            max = labelcounter[i]
            maxindex = i
    return classlist[maxindex]


def getLabelWithDP(indexlist):
    global classlist, label
    labelcounter = []
    for i in classlist:
        labelcounter.append(0)
    for index in xrange(indexlist.ndim):
        labelcounter[classlist.index(label[indexlist[index]])] += 1
    explist = []
    expsum = 0.0
    for q in labelcounter:
        t = math.exp(epsilon*q/2)
        expsum += t
        explist.append(t)
    max = 0
    maxindex = -1
    for i in xrange(len(explist)):
        if max < explist[i]:
            max = explist[i]
            maxindex = i
    return classlist[maxindex]

#随机选点
def random_point():
    global prePoint
    prePoint=random.randint(0, len(label)-1)
    return  data[prePoint]

def constructTree():
    global ct,ctTime,data
    # d=[]
    # for i in xrange(500):
    #     d.append([random.randint(0, 13),random.randint(0, 13)])
    ctStartTime=datetime.datetime.now()
    ct = CoverTree(data, euclidean, leafsize=20)
    ctEndTime=datetime.datetime.now()
    ctTime=ctTime+(ctEndTime-ctStartTime).microseconds

#f:得到的点到需要被预测的点的距离的集合
#l:得到点的下标的集合
def runKNN():
    global classList,pointIndex,rightTimes,knnTime,ct

    knnStartTime=datetime.datetime.now()
    (f, l) = ct.query(random_point(), 40)
    knnEndTime = datetime.datetime.now()
    knnTime = knnTime + (knnEndTime - knnStartTime).microseconds

    pointIndex=l[0]
    #删除被预测点自身
    f=np.delete(f,0,0)
    l=np.delete(l,0,0)

    if label[pointIndex]==getLabel(l):
        rightTimes+=1
    #print getLabel(l)
    #print getLabelWithDP(l)
    getLabelWithDP(l)
    #getLabel(l)

def precision():
    print "rightTimes:",rightTimes
    print "allTimes:",preTimes
    print "The accuracy of this algorithm is:",(float(rightTimes)/preTimes)*100,"%"

if __name__ == '__main__':
    run()
    # ct = CoverTree([[0,1],[1,1],[2,1],[0,0],[3,3]], euclidean, leafsize=10)