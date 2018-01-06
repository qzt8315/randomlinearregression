import random
import numpy as np
import math


class randomLinear:
    def __init__(self, trainData, labels, testData, testLabels, round, n):
        self.trainData = trainData/255.  #训练数据
        self.labels  = labels   #训练样本的标签
        self.testData = testData/255.    #测试数据集
        self.testLabels = testLabels    #  测试集的样本
        self.n = n  #每类采样的数量
        self.round = round  # 设置随机抽取的次数
        self.trainCount = trainData.shape[0]    #训练数据数量
        self.testCount = testData.shape[0]  #测试数据的数量
        self.predictLabels = [] #存储预测出的结果
        self.__dataSort()   #进行训练数据的处理

    def __dataSort(self):
        # 将训练数据集进行按标签归类
        self.dict = {} #训练数据集归类
        for i in range(self.trainCount):
            if self.labels[i] in self.dict.keys():
                self.dict[self.labels[i]] = np.vstack((self.dict[self.labels[i]], self.trainData[i]))
            else:
                self.dict[self.labels[i]] = self.trainData[i]
        # print(self.dict)
        # 模拟源图像
        self.original = {}
        for key in self.dict.keys():
            self.original[key] = np.matrix(np.sum(self.dict[key],axis=0)/self.dict[key].shape[0])
        # print(self.original)
        self.__working()

    def __working(self):
        # 进行学习
        ret = []
        for i in range(self.testCount):  # 遍历测试集
            count = {}
            for j in range(self.round): # 进行round轮
                minlabel = {}
                min = None
                minnorm = None
                for key in self.dict.keys():
                    # 抽取样本
                    # print(self.dict[key].shape)
                    indexset = []
                    c = 0
                    while c <= self.n:
                        t = random.randint(0, self.dict[key].shape[0] - 1)
                        if t not in indexset:
                            indexset.append(t)
                        c += 1
                    trainset = np.matrix([self.dict[key][i, :] for i in indexset])
                    # print(trainset)
                    # a = trainset * trainset.T
                    # print((trainset * trainset.T).I)
                    w = (trainset*trainset.T).I*trainset*np.matrix(self.dict[key][0, :]).T
                    norm = math.pow(np.linalg.norm((self.testData[i].T - trainset.T*w)/self.n, ord=2), 2)
                    if min is None or minnorm > norm:
                        min = key
                        minnorm = norm
                if min in minlabel.keys():
                    minlabel[min] += 1
                else:
                    minlabel[min] = 1
            retMax = None
            retmaxValue = None
            for key in count.keys():
                if retMax is None or retmaxValue < count[key]:
                    retMax = key
                    retmaxValue = count[key]
            ret.append(retMax)
        a = 1



    def printWork(self):
        # 获取学习的结果
        return