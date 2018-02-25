import numpy as np

def loadDataSet(fileName):
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat


def selectJrand(i, m):
    j = i
    while (j == i):
        j = int(np.random.uniform(0, m))

    return j


def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    b = 0
    m , n = np.shape(dataMatrix)
    alphas = np.mat(np.zeros((m, 1)))
    iter = 0
    while (iter < maxIter):
        alphaPairsChanged = 0

        for i in range(m):
            fXi = float(np.multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[i, :].T)) + b
            Ei = fXi - float(labelMat[i])

            if ((labelMat[i] * Ei < toler) and (alphas[i] < C)) or ((labelMat[i] * Ei > toler) and (alphas[i] > 0)):
                j = selectJrand(i, m)
                fXj = float(np.multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[j, :].T)) + b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()

                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                
                if L == H:
                    print('iter: {0}, i:{1}, alphaPairsChanged: {2}, L == H'.format(iter, i, alphaPairsChanged))
                    continue
                
                eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, : ].T - dataMatrix[i, :] * dataMatrix[i, :].T - dataMatrix[j, :] * dataMatrix[j, :].T
                if eta >= 0:
                    print('iter: {0}, i:{1}, alphaPairsChanged: {2}, eta >= 0'.format(iter, i, alphaPairsChanged))
                    continue
                
                alphas[j] -= labelMat[j] * (Ei - Ej) / eta
                alphas[j] = clipAlpha(alphas[j], H, L)
                if (abs(alphas[j] - alphaJold) < 0.00001) :
                    print('iter: {0}, i:{1}, alphaPairsChanged: {2}, j not moving enough'.format(iter, i, alphaPairsChanged))
                    continue

                alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])

                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[i, :].T - labelMat[j] * \
                    (alphas[j] - alphaJold) * dataMatrix[i, :] * dataMatrix[j, :].T

                b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, : ] * dataMatrix[j, :].T - labelMat[j] * \
                    (alphas[j] - alphaJold) * dataMatrix[j, :] * dataMatrix[j, :].T

                if (0 < alphas[i]) and (C > alphas[i]): 
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else :
                    b = (b1 + b2) / 2.0

                alphaPairsChanged += 1
                print('iter: {0}, i:{1}, alphaPairsChanged: {2}, alpha changed'.format(iter, i, alphaPairsChanged))
            else:
                print('iter: {0}, i:{1}, alphaPairsChanged: {2}, not adjust'.format(iter, i, alphaPairsChanged))
            
        if (alphaPairsChanged == 0):
            iter += 1
        else :
            iter = 0
        print('iteration number:{0}'.format(iter))
        
    return b, alphas

                
def test():
    dataArr, labelArr = loadDataSet('testSet.txt')
    # print(dataArr)
    # print(labelArr)

    b, alphas = smoSimple(dataArr, labelArr, 0.6, 0.001, 40)
    print(b)
    print(alphas[alphas > 0])
    print(np.shape(alphas[alphas > 0]))
    for i in range(100):
        if alphas[i] > 0.0:
            print(dataArr[i], labelArr[i])


class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler, kTup, useKernal=False):
        self.X  = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = np.shape(dataMatIn)[0]
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        self.eCache = np.mat(np.zeros((self.m, 2)))
        self.useKernal = useKernal

        if self.useKernal:
            self.K = np.mat(np.zeros((self.m, self.m)))
            for i in range(self.m):
                self.K[:, i] = kernelTrans(self.X, self.X[i, :], kTup)


    def calcEk(self, k):
        if self.useKernal:
            fXk = float(np.multiply(self.alphas, self.labelMat).T * self.K[:, k]) + self.b
        else:
            fXk = float(np.multiply(self.alphas, self.labelMat).T * (self.X * self.X[k, :].T)) + self.b
        
        Ek = fXk - float(self.labelMat[k])
        return Ek


    def selectJ(self, i, Ei):
        maxK = -1
        maxDeltaE = 0
        Ej = 0
        self.eCache[i] = [1, Ei]
        validEcacheList = np.nonzero(self.eCache[:, 0].A)[0]

        if (len(validEcacheList)) > 1:
            for k in validEcacheList:
                if k == i:
                    continue
                Ek = self.calcEk(k)
                deltaE = abs(Ei -Ek)

                if (deltaE > maxDeltaE):
                    maxK = k
                    maxDeltaE = deltaE
                    Ej = Ek
            return maxK, Ej
        else:
            j = selectJrand(i, self.m)
            Ej = self.calcEk(j)
            return j, Ej
    
    def updateEk(self, k):
        Ek = self.calcEk(k)
        self.eCache[k] = [1, Ek]
        return None

    
    def innerL(self, i):
        Ei = self.calcEk(i)

        if ((self.labelMat[i] * Ei < -self.tol) and (self.alphas[i] < self.C)) or ((self.labelMat[i] * Ei > self.tol) and (self.alphas[i] > 0)):
            j, Ej = self.selectJ(i, Ei)
            alphaIold = self.alphas[i].copy()
            alphaJold = self.alphas[j].copy()

            if (self.labelMat[i] != self.labelMat[j]):
                L = max(0, self.alphas[j] - self.alphas[i])
                H = min(self.C, self.C + self.alphas[j] - self.alphas[i])
            else:
                L = max(0, self.alphas[j] + self.alphas[i] - self.C)
                H = min(self.C, self.alphas[j] + self.alphas[i])
            
            if L == H:
                return 0, 'L == H'
            
            if self.useKernal:
                eta = 2.0 * self.K[i, j] - self.K[i,i] - self.K[j,j]
            else:
                eta = 2.0 * self.X[i, :] * self.X[j, :].T - self.X[i, :] * self.X[i, :].T - self.X[j, :] * self.X[j, :].T
            
            if eta >= 0:
                return 0, 'eta >= 0'

            self.alphas[j] -= self.labelMat[j] * (Ei - Ej) / eta
            self.alphas[j] = clipAlpha(self.alphas[j], H, L)

            self.updateEk(j)
            if (abs(self.alphas[j] - alphaJold) < 0.00001):
                return 0, 'j not moving enough'
            
            self.alphas[i] += self.labelMat[j] * self.labelMat[i] * (alphaJold - self.alphas[j])
            self.updateEk(i)

            if self.useKernal:
                b1 = self.b - Ei - self.labelMat[i] * (self.alphas[i] - alphaIold) * self.K[i, i] - self.labelMat[j] * (self.alphas[j] - alphaJold) * self.K[i, j]
                b2 = self.b - Ej - self.labelMat[i] * (self.alphas[i] - alphaIold) * self.K[i, j] - self.labelMat[j] * (self.alphas[j] - alphaJold) * self.K[j, j]
            else:
                b1 = self.b - Ei - self.labelMat[i] * (self.alphas[i] - alphaIold) * self.X[i, :] * self.X[i, :].T - self.labelMat[j] * \
                        (self.alphas[j] - alphaJold) * self.X[i, :] * self.X[j, :].T

                b2 = self.b - Ej - self.labelMat[i] * (self.alphas[i] - alphaIold) * self.X[i, : ] * self.X[j, :].T - self.labelMat[j] * \
                        (self.alphas[j] - alphaJold) * self.X[j, :] * self.X[j, :].T
                
            if (0 < self.alphas[i]) and (self.C > self.alphas[i]):
                self.b = b1
            elif (0 < self.alphas[j]) and (self.C > self.alphas[j]):
                self.b = b2
            else:
                self.b = (b1 + b2) / 2.0

            return 1, 'update alphas'
        else:
            return 0, 'not adjust'

        
def smoP(dataMatIn, calssLabels, C, toler, maxIter, kTup = ('lin', 0)):
    oS = optStruct(np.mat(dataMatIn), np.mat(calssLabels).transpose(), C, toler, kTup, True)
    
    iter = 0
    entireSet = True
    alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:
            for i in range(oS.m):
                tempAlphaPairsChanged, message = oS.innerL(i)
                alphaPairsChanged += tempAlphaPairsChanged
                #print('fullSet, iter:{0}, i:{1}, alphaPairChanged:{2}, {3}'.format(iter, i , alphaPairsChanged, message))
            iter += 1
        else:
            nonBoundIs = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                tempAlphaPairsChanged, message = oS.innerL(i)
                alphaPairsChanged += tempAlphaPairsChanged
                #print('non-bound, iter:{0}, i:{1}, alphaPairChanged:{2}, {3}'.format(iter, i, alphaPairsChanged, message))
            iter += 1

        if entireSet:
            entireSet = False
        elif (alphaPairsChanged == 0):
            entireSet = True
        
        #print('iteration number:{0}, entireSet:{1}'.format(iter, entireSet))
    
    return oS.b, oS.alphas


def calcWs(alphas, dataArr, classLabels):
    X = np.mat(dataArr)
    labelMat = np.mat(classLabels).transpose()

    m, n = np.shape(X)
    w = np.zeros((n, 1))
    for i in range(m):
        w += np.multiply(alphas[i] * labelMat[i], X[i, :].T)
    return w


def calcWsAndB():
    dataArr, labelArr = loadDataSet('testSet.txt')
    b, alphas = smoP(dataArr, labelArr, 0.6, 0.001, 40)

    print(b)
    print(alphas[alphas > 0])
    print(np.shape(alphas[alphas > 0]))
    for i in range(100):
        if alphas[i] > 0.0:
            print(dataArr[i], labelArr[i])

    ws = calcWs(alphas, dataArr, labelArr)
    print(ws)

    return ws, b


def test2():
    dataArr, labelArr = loadDataSet('testSet.txt')
    ws, b = calcWsAndB()

    dataMat = np.mat(dataArr)
    ws = np.mat(ws)

    m = dataMat.shape[0]
    errorCount = 0
    for i in range(m):
        rawResult = dataMat[i] * ws + b
        if rawResult >= 0:
            result = 1.0
        else:
            result = -1.0
        print('classified result is:{0} and real result is: {1}, rawResult is: {2}'.format(result, labelArr[i], rawResult))

        if result != labelArr[i]:
            errorCount += 1
    
    print('error rate is {0}'.format(float(errorCount) / m))
        

def kernelTrans(X, A, kTup):
    m, n = np.shape(X)
    K = np.mat(np.zeros((m, 1)))
    if kTup[0] == 'lin':
        K = X * A.T
    elif kTup[0] == 'rbf':
        for j in range(m):
            deltaRow = X[j, :] - A
            K[j] = deltaRow * deltaRow.T
        K = np.exp(K / (-1 * kTup[1] ** 2))
    else:
        raise NameError('Houston We Have a Problem -- That Kernel is not recognized')
    return K


def testRbf(k1 = 1.3):
    dataArr, labelArr = loadDataSet('testSetRBF.txt')
    b, alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, ('rbf', k1))
    
    datMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).transpose()

    svInd = np.nonzero(alphas.A > 0)[0]

    sVs = datMat[svInd]
    labelSV = labelMat[svInd]
    print('there are {0} Support Vectors'.format(np.shape(sVs)[0]))
    m, n = np.shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i,:], ('rbf', k1))
        predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
        
        if np.sign(predict) != np.sign(labelArr[i]):
            errorCount += 1
    
    print('the training error rate is:{0}'.format(float(errorCount) / m))


    dataArr, labelArr = loadDataSet('testSetRBF2.txt')
    errorCount = 0
    datMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).transpose()

    m,n = np.shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], ('rbf', k1))
        predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b

        if np.sign(predict) != np.sign(labelArr[i]):
            errorCount += 1
    print('the test error rate is:{0}'.format(float(errorCount) / m))


def img2vector(fileName):
    returnVect = np.zeros((1,1024))
    fr = open(fileName)

    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i + j] = int(lineStr[j])

    return returnVect


def loadImages(dirName):
    from os import listdir
    hwLabels = []
    trainingFileList = listdir(dirName)
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])

        if classNumStr == 9:
            hwLabels.append(-1)
        else:
            hwLabels.append(1)
        
        trainingMat[i, :] = img2vector('{0}/{1}'.format(dirName, fileNameStr))
    return trainingMat, hwLabels


def testDigits(kTup = ('rbf', 10)):
    dataArr, labelArr = loadImages('trainingDigits')
    b, alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, kTup)

    datMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).transpose()

    svInd = np.nonzero(alphas.A > 0)[0]
    sVs = datMat[svInd]
    labelSV = labelMat[svInd]
    print('there are {0} Support Vectors when change rate is:{1}'.format(np.shape(sVs)[0], kTup[1]))

    m, n = np.shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], kTup)
        predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b

        if np.sign(predict) != np.sign(labelArr[i]):
            errorCount += 1
    print('the training error rate is:{0}'.format(float(errorCount) / m))

    dataArr, labelArr = loadImages('testDigits')
    errorCount = 0
    datMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).transpose()

    m, n = np.shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], kTup)
        predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b

        if np.sign(predict) != np.sign(labelArr[i]):
            errorCount += 1
    
    print('the test error rate is:{0}'.format(float(errorCount) / m))




if __name__ == '__main__':
    #test()
    #test2()
    #testRbf(0.5)
    testDigits(('rbf', 0.1))
    testDigits(('rbf', 5))
    testDigits(('rbf', 10))
    testDigits(('rbf', 50))
    testDigits(('rbf', 100))
