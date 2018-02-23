import numpy as np
import matplotlib.pyplot as plt
from time import sleep
import json
from urllib.request import urlopen

def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t')) - 1
    dataMat = []
    labelMat = []
    fr = open(fileName)

    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')

        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    
    return dataMat, labelMat


def standRegres(xArr, yArr):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T

    xTx = xMat.T * xMat
    if np.linalg.det(xTx) == 0.0:
        print('This matrix is singular, cannot do inverse')
        return None
    ws = xTx.I * (xMat.T * yMat)
    return ws


def lwlr(testPoint, xArr, yArr, k=1.0):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T

    m = np.shape(xMat)[0]
    weights = np.mat(np.eye(m))

    for j in range(m):
        diffMat = testPoint - xMat[j,:]
        weights[j,j] = np.exp(diffMat * diffMat.T / (-2.0 * (k ** 2)))
    
    xTx = xMat.T * (weights * xMat)
    if np.linalg.det(xTx) == 0.0:
        print('This matrix is singular, cannot do inverse')
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws


def lwlrTest(testArr, xArr, yArr, k=1.0):
    m = np.shape(testArr)[0]
    yHat = np.zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat


def rssError(yArr, yHatArr):
    return ((yArr - yHatArr)**2).sum()


def ridgeRegres(xMat, yMat, lam = 0.2):
    xTx = xMat.T * xMat
    denom = xTx + np.eye(np.shape(xMat)[1]) * lam

    if np.linalg.det(denom) == 0.0:
        print('This matrix is singular, cannot do inverse')
        return
    ws = denom.I * (xMat.T * yMat)
    return ws


def ridgeTest(xArr, yArr):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T

    yMean = np.mean(yMat, axis=0)
    yMat = yMat - yMean

    xMeans = np.mean(xMat, axis=0)
    xVar = np.var(xMat, axis=0)

    xMat = (xMat - xMeans) / xVar
    numTestPts  = 30

    wMat = np.zeros((numTestPts, np.shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat, yMat, np.exp(i-10))
        wMat[i,:] = ws.T
    return wMat


def regularize(xMat):#regularize by columns
    inMat = xMat.copy()
    inMeans = np.mean(inMat,0)   #calc mean then subtract it off
    inVar = np.var(inMat,0)      #calc variance of Xi then divide by it
    inMat = (inMat - inMeans)/inVar
    return inMat


def stageWise(xArr, yArr, eps = 0.01, numIt= 100):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T

    yMean = np.mean(yMat, 0)
    yMat = yMat - yMean

    xMat = regularize(xMat)
    m, n = np.shape(xMat)
    returnMat = np.zeros((numIt, n))
    ws = np.zeros((n, 1))
    wsTest = ws.copy()
    wsMax = ws.copy()

    for i in range(numIt):
        print(ws.T)
        lowestError = np.Infinity

        for j in range(n):
            for sign in [-1, 1]:
                wsTest = ws.copy()
                wsTest[j] += eps * sign
                yTest = xMat * wsTest
                rssE = rssError(yMat.A, yTest.A)
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i, :] = ws.T
    return returnMat


def searchForSet(retX, retY, setNum, yr, numPce, origPrc):
    sleep(10)
    myAPIstr = 'get from code.google.com'
    searchURL = 'https://www.googleapis.com/shopping/search/v1/public/products?key={0}&country=US&q=lego+{1}&alt=json'.format(myAPIstr, setNum)
    pg = urlopen(searchURL)
    retDict = json.loads(pg.read())

    for i in range(len(retDict['items'])):
        try:
            currItem = retDict['items'][i]
            if currItem['product']['condition'] == 'new':
                newFlag = 1
            else:
                newFlag = 0

            listOfInv = currItem['product']['inventories']
            for item in listOfInv:
                sellingPrice = item['price']
                if sellingPrice > origPrc * 0.5:
                    print('{0}\t{1}\t{2}\t{3}\t{4}'.format(yr, numPce, newFlag, origPrc, sellingPrice))
                    retX.append([yr, numPce, newFlag, origPrc])
                    retY.append(sellingPrice)
        except:
            print('problem with item {0}'.format(i))


def setDataCollect(retX, retY):
    searchForSet(retX, retY, 8288, 2006, 800, 49.99)
    searchForSet(retX, retY, 10030, 2002, 3096, 269.99)
    searchForSet(retX, retY, 10179, 2007, 5195, 499.99)
    searchForSet(retX, retY, 10181, 2007, 3428, 199.99)
    searchForSet(retX, retY, 10189, 2008, 5922, 299.99)
    searchForSet(retX, retY, 10196, 2009, 3263, 249.99)




def test():
    xArr, yArr = loadDataSet('ex0.txt')
    # print(xArr[0:2])
    # print(yArr[0:2])
    ws = standRegres(xArr, yArr)
    print(ws)

    xMat = np.mat(xArr)
    yMat = np.mat(yArr)
    yHat = xMat * ws
    relevant = np.corrcoef(yHat.T, yMat)
    print(relevant)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xMat[:, 1].flatten().A[0], yMat.T[:, 0].flatten().A[0])

    # sort x by order to draw a good line
    xCopy = xMat.copy()
    xCopy.sort(0)
    yHat = xCopy * ws
    ax.plot(xCopy[:, 1], yHat)
    plt.show()


def test2(k=1.0):
    xArr, yArr = loadDataSet('ex0.txt')
    yHat = lwlrTest(xArr, xArr, yArr, k)

    xMat = np.mat(xArr)
    srtInd = xMat[:,1].argsort(0)
    xSort = xMat[srtInd][:,0,:]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xSort[:, 1], yHat[srtInd])
    ax.scatter(xMat[:,1].flatten().A[0], np.mat(yArr).T.flatten().A[0], s=2, c='red')
    plt.show()


def test3():
    abX, abY = loadDataSet('abalone.txt')
    yHat01 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 0.1)
    yHat1 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 1.0)
    yHat10 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 10)

    rssError01 = rssError(abY[0:99], yHat01.T)
    print(rssError01)
    rssError1 = rssError(abY[0:99], yHat1.T)
    print(rssError1)
    rssError10 = rssError(abY[0:99], yHat10.T)
    print(rssError10)

    yHat01 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 0.1)
    rssError01 = rssError(abY[100:199], yHat01.T)
    print(rssError01)

    yHat1 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 1.0)
    rssError1 = rssError(abY[100:199], yHat1.T)
    print(rssError1)

    yHat10 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 10)
    rssError10 = rssError(abY[100:199], yHat10.T)
    print(rssError10)

    ws = standRegres(abX[0:99], abY[0:99])
    yHat = np.mat(abX[100:199]) * ws
    rssErrorStand = rssError(abY[100:199], yHat.T.A)
    print(rssErrorStand)


def test4():
    abX, abY = loadDataSet('abalone.txt')
    ridgeWeights = ridgeTest(abX, abY)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ridgeWeights)
    plt.show()


def test5():
    xArr, yArr = loadDataSet('abalone.txt')
    stageW = stageWise(xArr, yArr, 0.01, 200)
    print(stageW)


def test6():
    lgX = []
    lgY = []
    setDataCollect(lgX, lgY)
    print(lgX)
    print(lgY)


if __name__ == '__main__':
    #test()
    # test2(k=1.0)
    # test2(k=0.01)
    # test2(k=0.003)
    #test3()
    #test4()
    #test5()
    test6()
    
