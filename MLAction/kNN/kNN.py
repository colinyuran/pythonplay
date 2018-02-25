import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import operator
from os import listdir

def createDataSet():
    group = np.array([[1.0,1.1], [1.0,1.0], [0,0], [0,0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    
    diffMat = np.tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}

    for i in range(k):
        voteILabel = labels[sortedDistIndicies[i]]
        classCount[voteILabel]  = classCount.get(voteILabel, 0) + 1
        
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def file2matrix(fileName):
    fr = open(fileName)
    arrayAllLines = fr.readlines()
    numberOfLines = len(arrayAllLines)
    returnMat = np.zeros((numberOfLines, 3))
    classLabelVector = []

    index = 0
    for line in arrayAllLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, : ] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    
    return returnMat, classLabelVector


def autoNorm(dataSet):
    minVals = dataSet.min(axis=0)
    maxVals = dataSet.max(axis=0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    normDataSet = normDataSet / np.tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


def datingClassTest():
    hoRatio = 0.1
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normDataSet, ranges, minVals = autoNorm(datingDataMat)
    m = normDataSet.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    
    for i in range(numTestVecs):
        classifierResult = classify0(normDataSet[i, :], normDataSet[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print('the classifier came back with:{0}, the real answer is:{1}'.format(classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]):
            errorCount += 1.0
    
    print('the total error rate is:{0}'.format((errorCount / float(numTestVecs))))
    return None


def classifyPerson(p=None, f=None, i=None):
    resultList = ['not at all', 'in small dose', 'in large doses']
    if (p == None) : 
        percentTats = float(input('Percentage of time spent playing video games?'))
    else:
        percentTats = p
    
    if (f == None):
        ffMiles = float(input('Frequent flier miles earned per year?'))
    else:
        ffMiles = f
    
    if (i == None):        
        iceCream = float(input('Liters of ice cream consumed per year?'))
    else:
        iceCream = i

    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normDataSet, ranges, minVals = autoNorm(datingDataMat)

    inArr = np.array([ffMiles, percentTats, iceCream])
    inArr = (inArr - minVals) / ranges

    classifierResult = classify0(inArr, normDataSet, datingLabels, 3)
    print('You will propbably like this person: {0}'.format(resultList[classifierResult - 1]))
    return None


def img2vector(fileName):
    returnVect = np.zeros((1,1024))
    fr = open(fileName)

    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i + j] = int(lineStr[j])

    return returnVect


def handWritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('trainingDigits/%s'%fileNameStr)
    
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print('The classifier came back with:{0}, the real answer is:{1}'.format(classifierResult, classNumStr))

        if (classifierResult != classNumStr):
            errorCount += 1.0
        
    print('the total number of errors is:{0}'.format(errorCount))
    print('the total error rate is:{0}'.format(errorCount / float(mTest)))
    return None


if (__name__ == '__main__'):
    # group, labels = createDataSet()
    # result = classify0([0,0], group, labels, 3)
    # print(result)

    # datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    # normDataSet, ranges, minVals = autoNorm(datingDataMat)
    # print(normDataSet)
    # fig = plt.figure()
    # ax = fig.add_subplot(1,1,1)
    # ax.scatter(normDataSet[:, 0], normDataSet[:, 1], 15.0 * np.array(datingLabels), 15.0 * np.array(datingLabels))
    # plt.show()
    # fig.savefig('analysis.png')

    # datingClassTest()

    # classifyPerson(0.2, 1000, 10)

    # testVector = img2vector('testDigits/0_13.txt')
    # print(testVector)

    handWritingClassTest()