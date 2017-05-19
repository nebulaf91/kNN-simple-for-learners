#-*- coding: utf-8 -*-

# implements the knn algorithm
# author : zhikang       date : 2017.5.18
# usage: change the working directory to where knn.py and data files lies in.
#        type in terminal  "python knn.py"  and hit "enter".

import numpy as np

def load_data(fileName):
    X = []
    Y = []
    inputFile = open(fileName);
    for line in inputFile:
        data = map(float, line.split());            #注意！！！这个读入的是字符，要转成number!!
        X.append(data[0 : -1])
        Y.append(data[-1])
    dataDim = len(line.split()) - 1
    dataNum = len(Y)
    X = np.matrix(X)
    Y = np.matrix(Y)
    return X, Y, dataDim, dataNum

def distance(pt1, pt2):         #pt1 代表 point1，是np.matrix格式
    vec = pt1 - pt2
    vec = (vec.tolist())[0]     #转换成列表形式
    distance = sum(map(lambda x:x**2, vec))
    return distance

def cal_err_rate(test_Y, result, testNum):
    result = np.matrix(result)
    temp = ((test_Y - result).tolist())[0]          #标签只为正负1，作差然后相减，取abs，看有多少个不同的
    correctNum = sum(map(abs, temp)) / 2
    return correctNum / testNum
    
if __name__=='__main__':
    k = 3;      #设置kNN中的k。设置成一个奇数可以避免处理平票的情况
    errRate = 0.0;
    train_X, train_Y, trainDim,trainNum = load_data("train_data.dat")
    test_X, test_Y , testDim, testNum = load_data("test_data.dat")
    distMat = np.matrix(np.zeros((testNum, trainNum)))          #distMat(i,j)表示测试集中第i组数据与训练集中第j组数据的距离
    for ii in range(0, testNum):
        for jj in range(0, trainNum):
            distMat[ii, jj] = distance(train_X[jj], test_X[ii])
    # print(distMat)
    #进行kNN投票
    result = []
    for dist in distMat:
        vote = 0
        resultIndex = 0
        index = []
        for kk in range(0, k):
            index.append(dist.argmin());
            dist[0, index[kk]] = float('inf');
            vote = vote + train_Y[0,index[kk]]          #要记住，这是矩阵。要选定某一个元素，要给出(x,y)两个位置参数
        result.append(np.sign(vote))
    # print(result)
    errRate = cal_err_rate(test_Y, result, testNum)
    print("the error rate is: ")
    print(errRate)



        




