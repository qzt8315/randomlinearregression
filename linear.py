import random
import numpy as np
import math


class randomLinear:
    def __init__(self, trainData, labels, testData, testLabels, round, n):
        self.trainData = trainData/256.  #训练数据
        self.labels  = labels   #训练样本的标签
        self.testData = testData/256.    #测试数据集
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

    def getW(self, original, dataset):
        return (dataset*dataset.T).I*dataset*original.T

    def start(self):
        # 开始进行线性回归，实现简易的线性回归选择每类样本的前6个，第一个作为原图像，后面5个作为训练集
        # 计算每类标签的w
        w ={}
        o = {}
        t = {}
        for key in self.dict.keys():
            o[key] = np.matrix(self.dict[key][0, :])
            t[key] = np.matrix(self.dict[key][1:6, :])
            w[key] = self.getW(o[key], t[key])
        # 做测试集预测
        for i in range(self.testCount):
            label = None
            minNorm = None
            # 计算范数
            for key in self.dict.keys():
                norm = np.linalg.norm(self.testData[i] - t[key].T * w[key])
                if label is None or minNorm > norm:
                    label = key
                    minNorm = norm
            