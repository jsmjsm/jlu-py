'''
Description  : jlu-py hw1
Author       : jsmjsm
Github       : https://github.com/jsmjsm
Date         : 2021-04-11 11:05:44
LastEditors  : jsmjsm
LastEditTime : 2021-04-17 21:25:10
'''
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


def main():
    # import data
    rawdata = pd.read_csv("rawdata.csv")
    rawdata = rawdata.values.tolist()
    label = pd.read_csv("label.csv", header=None)
    label = label.values.tolist()
    npdata = np.array(rawdata)

    # delete the first lin, attribute name
    npdata = np.delete(npdata, 0, axis=1)
    nplabel = np.array(label).flatten()  # collapse into one dimension 适应后面的函数
    # Sample: 1102 ，Attributes: 234

    # Replace the name
    nplabel[nplabel == "BC_CML"] = 1
    nplabel[nplabel == "CP_CML"] = 2
    nplabel[nplabel == "k562"] = 3
    nplabel[nplabel == "normal"] = 4
    nplabel[nplabel == "pre_BC"] = 5

    npdata = npdata.astype('float64')
    nplabel = nplabel.astype('int32')

    X_train, X_test, y_train, y_test = train_test_split(
        npdata, nplabel, random_state=0)

    numOfTrain = 826
    numOfTest = 276
    numOfAttribute = 234
    numOfNearest = 3  # 参数

    '''
    @distance: 某一点到训练集内点的距离
    @mins: 下表，距离最短n点
    @mins_label: 类别
    '''
    distance = np.zeros((numOfTrain,))
    mins = np.zeros((numOfNearest,))
    mins_label = np.zeros((numOfNearest,), dtype='int32')

    numOfCorrect = 0

    # Test
    for j in range(0, numOfTest):
        # Find the min
        for i in range(0, numOfTrain):
            distance[i] = (sum((X_train[i] - X_test[j]) ** 2)) ** (1/2)
        mins = distance.argsort()[:numOfNearest]

        # Classify
        for i in range(0, numOfNearest):
            mins_label[i] = y_train[mins[i]]
            result = np.argmax(np.bincount(mins_label))

        if(result == y_test[j]):  # True
            numOfCorrect += 1

    print("模型准确率：", numOfCorrect/numOfTest)


# run
main()
