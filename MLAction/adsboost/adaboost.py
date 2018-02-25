import numpy as np

def loadSimpData():
    datMat = np.matrix([ [1.0, 2.1], 
                         [2.0, 1.1],
                         [1.3, 1.0],
                         [1.0, 1.0],
                         [2.0, 1.0]])
    
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat, classLabels


def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    retArray = np.ones((np.shape(dataMatrix)[0], 1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray


def buildStump(dataArr, classLabels, D):
    dataMatrix = np.mat(dataArr)
    labelMat = np.mat(classLabels).transpose()
    m, n = np.shape(dataMatrix)

    numSteps = 10.0
    bestStump = {}
    bestClasEst = np.mat(np.zeros((m, 1)))

    minError = np.Infinity
    for i in range(n):
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()
        stepSize = (rangeMax - rangeMin) / numSteps
        for j in range(-1, int(numSteps) + 1):
            for inequal in ['lt', 'gt']:
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)
                errArr = np.mat(np.ones((m, 1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T * errArr
                #print('split: dim {0}, thresh {1}, thresh inequal: {2}, the weighted error is {3}'.format(i, threshVal, inequal, weightedError))

                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClasEst


def adaBoostTrainDS(dataArr, classLabels, numIt = 40):
    weakClassArr = []
    m = np.shape(dataArr)[0]
    D = np.mat(np.ones((m, 1)) / m)
    aggClassEst = np.mat(np.zeros((m, 1)))

    for i in range(numIt):
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        #print('D: {0}'.format(D.T))
        alpha = float(0.5 * np.log((1.0 - error) / max(error, 1e-16)))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        #print('classEst: {0}'.format(classEst.T))

        expon = np.multiply(-1 * alpha * np.mat(classLabels).T, classEst)
        D = np.multiply(D, np.exp(expon))
        D = D / D.sum()

        aggClassEst += alpha * classEst
        #print('aggClassEst:{0}'.format(aggClassEst.T))
        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T, np.ones((m, 1)))
        errorRate = aggErrors.sum() / m
        #print('total error:{0}'.format(errorRate))

        if errorRate == 0.0:
            break
    
    return weakClassArr, aggClassEst


def adaClassify(datToClass, classifierArr):
    dataMatrix = np.mat(datToClass)
    m = np.shape(dataMatrix)[0]
    aggClassEst = np.mat(np.zeros((m, 1)))

    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'], classifierArr[i]['thresh'], classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha'] * classEst
        #print(aggClassEst)
    return np.sign(aggClassEst)


def test():
    datMat, classLabels = loadSimpData()
    classifierArray = adaBoostTrainDS(datMat, classLabels, 30)[0]
    result = adaClassify([[5,5], [0,0]], classifierArray)
    print(result)


def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t'))
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine  = line.strip().split('\t')
        for i in range(numFeat - 1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat

def test2(iterNumber = 10):
    datArr, labelArr = loadDataSet('horseColicTraining2.txt')
    classifierArray = adaBoostTrainDS(datArr, labelArr, iterNumber)[0]
    # print(len(classifierArray))
    # for i in range(len(classifierArray)):
    #     print(classifierArray[i])

    testArr, testLabelArr = loadDataSet('horseColicTest2.txt')
    prediction10 = adaClassify(testArr, classifierArray)
    errArr = np.mat(np.ones((np.shape(testArr)[0], 1)))
    errorCount = errArr[prediction10 != np.mat(testLabelArr).T].sum()
    errorRate = float(errorCount) / np.shape(testArr)[0]

    print('iter is: {0}, classifierArray length is: {1}, error rate is: {2}'.format(iterNumber, len(classifierArray), errorRate))


def plotROC(predStrengths, classLabels):
    import matplotlib.pyplot as plt

    cur = (1.0, 1.0)
    ySum = 0
    numPosClas = np.sum(np.array(classLabels) == 1.0)
    yStep = 1 / float(numPosClas)
    xStep = 1 / float(len(classLabels) - numPosClas)
    sortedIndicies = predStrengths.argsort()

    fig = plt.figure()
    fig.clf()
    ax  = plt.subplot(111)

    for index in sortedIndicies.tolist()[0]:
        print('Classified= {0}, Labeled= {1}, IsEqual= {2}'.format(predStrengths[0, index], classLabels[index], np.sign(predStrengths[0, index]) == np.sign(classLabels[index])))
        if classLabels[index] == 1.0:
            delX = 0
            delY = yStep
        else:
            delX = xStep
            delY = 0
            ySum += cur[1]
        ax.plot([cur[0], cur[0] - delX], [cur[1], cur[1] - delY], c='b')
        cur = (cur[0] - delX, cur[1] - delY)
    
    ax.plot([0,1], [0,1], 'b--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve for AdaBoost Horse Colic Detection System')
    ax.axis([0, 1.1, 0, 1.1])
    plt.show()
    print('the Area Under the Curve is:{0}'.format(ySum * xStep))


def testPlotROC():
    datArr, labelArr = loadDataSet('horseColicTraining2.txt')
    classifierArray, aggClassEst = adaBoostTrainDS(datArr, labelArr, 50)
    plotROC(aggClassEst.T, labelArr)


if __name__ == '__main__':
    # datMat, classLabels = loadSimpData()
    # #print(datMat)
    # #print(classLabels)
    # # D = np.mat(np.ones((5, 1)) / 5)
    # # bestStump, minError, bestClasEst = buildStump(datMat, classLabels, D)

    # classifierArray = adaBoostTrainDS(datMat, classLabels, 9)
    # print(classifierArray)

    #test()
    # test2(1)
    # test2(10)
    # test2(50)
    # test2(100)
    # test2(500)
    # test2(1000)
    # test2(10000)

    testPlotROC()