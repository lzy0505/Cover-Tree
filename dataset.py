import sqlite3
import numpy as np


# dataset = 'iris'
dataset = 'abalone'

def run():
    # create_table()
    connect_to_sql()
    # readData()
    getData()
    print getLabel([13,15,16,40,70,60,30,50,120,1300,1500,1501,1900,1234,1235,1565,144,144,144,144,144,144,144])


def readData():
    global cur, conn
    classlist = []
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
    for line in open("iris.data"):
        l = line.split(',')
        l[4] = l[4].rstrip('\n')
        cur.execute('INSERT INTO iris values(?,?,?,?,?)',(l[0],l[1],l[2],l[3],l[4]))
    conn.commit()


def create_table():
    # connect to database
    DB = sqlite3.connect('dataset.db')
    # cursor of database
    cDB = DB.cursor()

    # delete all table in database
    cDB.execute("DROP TABLE iris")
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

    cDB.execute("CREATE TABLE iris (\n"
    + "sepal_length REAL  NOT NULL,\n"
    + "sepal_width REAL NOT NULL,\n"
    + "petal_length REAL NOT NULL,\n"
    + "petal_width REAL NOT NULL,\n"
    + "class TEXT NOT NULL"
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
    rst = cur.fetchall()
    for instance in rst:
        for i in xrange(len(instance)-1):
            ftr[0, i] = instance[i]
        data = np.concatenate((data, ftr))
        label.append(instance[len(instance)-1])
        if instance[len(instance)-1] not in classlist:
            classlist.append(instance[len(instance)-1])
    print data


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


if __name__ == '__main__':
    run()


