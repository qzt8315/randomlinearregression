'''''
    使用python解析二进制文件
'''
import numpy as np
import struct
from knn import *
import time
from linear import randomLinear
from svm import svm_baseline
from bp import useBP

# K临近的参数
k = 4

def loadImageSet(filename):
    binfile = open(filename, 'rb')  # 读取二进制文件
    buffers = binfile.read()

    head = struct.unpack_from('>IIII', buffers, 0)  # 取前4个整数，返回一个元组

    offset = struct.calcsize('>IIII')  # 定位到data开始的位置
    imgNum = head[1]
    width = head[2]
    height = head[3]

    bits = imgNum * width * height  # data一共有60000*28*28个像素值
    bitsString = '>' + str(bits) + 'B'  # fmt格式：'>47040000B'

    imgs = struct.unpack_from(bitsString, buffers, offset)  # 取data数据，返回一个元组

    binfile.close()
    imgs = np.reshape(imgs, [imgNum, width * height])  # reshape为[60000,784]型数组

    return imgs, head


def loadLabelSet(filename):
    binfile = open(filename, 'rb')  # 读二进制文件
    buffers = binfile.read()

    head = struct.unpack_from('>II', buffers, 0)  # 取label文件前2个整形数

    labelNum = head[1]
    offset = struct.calcsize('>II')  # 定位到label数据开始的位置

    numString = '>' + str(labelNum) + "B"  # fmt格式：'>60000B'
    labels = struct.unpack_from(numString, buffers, offset)  # 取label数据

    binfile.close()
    labels = np.reshape(labels, [labelNum])  # 转型为列表(一维数组)

    return labels, head

def to2(dataset):
    (m, n) = dataset.shape
    for i in range(m):
        for j in range(n):
            if dataset[i,j] >= 127:
                dataset[i, j] =1
            else:
                dataset[i, j] = 0
    return dataset


if __name__ == "__main__":
    trainfile1 = 'train-images.idx3-ubyte'
    trainfile2 = 'train-labels.idx1-ubyte'
    testfile1 = 't10k-images.idx3-ubyte'
    testfile2 = 't10k-labels.idx1-ubyte'

    print('read train dataset from file:'+trainfile1)
    trainData, train_head = loadImageSet(trainfile1)
    trainData = trainData*1.0
    trainLabels, trainL_head = loadLabelSet(trainfile2)
    train = trainData[:50000, :] # to2(trainData[:1000,:])
    # print(trainLabels)
    lanelstrain = trainLabels[:50000]

    print('read test dataset from file:' + testfile1)
    testData, test_head= loadImageSet(testfile1)
    testData = testData*1.0
    testLabels, testL_head = loadLabelSet(testfile2)

    # print('start kNN compute......')
    # missCount = 0
    # print('k=', k)
    # print(time.strftime('%Y-%m-%d %H:%M:%S'))
    # start = time.time()
    # for i in range(testL_head[1]):
    #     out = kNNClassify(testData[i], trainData, trainLabels, k)
    #     if out != testLabels[i]:
    #         missCount += 1
    # print(time.strftime('%Y-%m-%d %H:%M:%S'))
    # end = time.time()
    # print('test dataset total:', test_head[1],
    #       ',miss count:' ,missCount, ',correct ratio:',
    #       1 - missCount/test_head[1], ',miss ratio:',
    #       missCount/test_head[1], ',spend:', end - start)

    #svm
    # 使用svm进行学习
    # print('svm start......')
    # print(train.shape)
    # print(lanelstrain.shape)
    # print(testData.shape)
    # svm_baseline(trainData, trainLabels,testData, testLabels)

    # useBP(trainData,trainLabels,testData, testLabels)