import numpy as np

def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float, curLine))
        dataMat.append(fltLine)
    return dataMat


def distEclud(vecA, vecB):
    return np.sqrt(np.sum(np.power(vecA - vecB, 2)))


def randCent(dataSet, k):
    n = np.shape(dataSet)[1]
    centroids = np.mat(np.zeros((k, n)))

    for j in range(n):
        minJ = np.min(dataSet[:, j])
        rangeJ = float(np.max(dataSet[:,j]) - minJ)
        centroids[:, j] = minJ + rangeJ * np.random.rand(k, 1)
    
    return centroids


def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = np.shape(dataSet)[0]
    clusterAssment = np.mat(np.zeros((m, 2)))
    centrodis = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = np.Infinity
            minIndex = -1

            for j in range(k):
                distJI = distMeas(centrodis[j,:], dataSet[i, :])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
                clusterAssment[i, :] = minIndex, minDist**2

        print(centrodis)
        for cent in range(k):
            ptsInClust = dataSet[np.nonzero(clusterAssment[:, 0].A == cent)[0]]
            centrodis[cent, :] = np.mean(ptsInClust, axis=0)

    return centrodis, clusterAssment


def biKmeans(dataSet, k, distMeas=distEclud):
    m = np.shape(dataSet)[0]
    clusterAssment = np.mat(np.zeros((m,2)))
    centroid0 = np.mean(dataSet, axis=0).tolist()[0]
    centList = [centroid0]
    for j in range(m):
        clusterAssment[j , 1] = distMeas(np.mat(centroid0), dataSet[j, :]) ** 2
        
    while (len(centList) < k):
        lowestSSE = np.Infinity
        for i in range(len(centList)):
            ptsIncurrCluster = dataSet[np.nonzero(clusterAssment[:,0].A == i)[0], : ]
            centroidMat , splitClustAss = kMeans(ptsIncurrCluster, 2, distMeas)
            sseSplit = np.sum(splitClustAss[:, 1])
            sseNotSplit = np.sum(clusterAssment[np.nonzero(clusterAssment[:, 0].A != i)[0], 1])
            print('sseSplit, and notSplit:'.format(sseSplit, sseNotSplit))

            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit

        bestClustAss[np.nonzero(bestClustAss[:,0].A == 1)[0], 0] = len(centList)
        bestClustAss[np.nonzero(bestClustAss[:,0].A == 0)[0], 0] = bestCentToSplit
        print('the bestCentToSplit is:{0}'.format(bestCentToSplit))
        print('the len of bestClustAss is:{0}'.format(len(bestClustAss)))

        centList[bestCentToSplit] = bestNewCents[0,:]
        centList.append(bestNewCents[1,:])

        clusterAssment[np.nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0], :] = bestClustAss

        return np.mat(centList), clusterAssment



if __name__ == '__main__':
    datMat = np.mat(loadDataSet('testSet.txt'))    
    # temp = randCent(datMat, 2)

    # print(np.min(datMat[:, 0]))
    # print(np.max(datMat[:, 0]))
    # print(np.min(datMat[:, 1]))
    # print(np.max(datMat[:, 1]))
    # print(temp)

    # print(distEclud(datMat[0], datMat[1]))
    myCentroids, clustAssing = kMeans(datMat, 4)



