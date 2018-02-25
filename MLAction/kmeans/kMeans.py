import numpy as np
import matplotlib
import matplotlib.pyplot as plt


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

        # print(centrodis)
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
            print('sseSplit, and notSplit:{0} and {1}'.format(sseSplit, sseNotSplit))

            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit

        bestClustAss[np.nonzero(bestClustAss[:,0].A == 1)[0], 0] = len(centList)
        bestClustAss[np.nonzero(bestClustAss[:,0].A == 0)[0], 0] = bestCentToSplit
        print('the bestCentToSplit is:{0}'.format(bestCentToSplit))
        print('the len of bestClustAss is:{0}'.format(len(bestClustAss)))

        centList[bestCentToSplit] = bestNewCents[0,:].A[0]
        centList.append(bestNewCents[1,:].A[0])

        clusterAssment[np.nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0], :] = bestClustAss

    return np.mat(centList), clusterAssment


def distSLC(vecA, vecB):
    a = np.sin(vecA[0,1] * np.pi / 180) * np.sin(vecB[0,1] * np.pi / 180)
    b = np.cos(vecA[0,1] * np.pi / 180) * np.cos(vecB[0,1] * np.pi / 180) * np.cos(np.pi * (vecB[0,0] - vecA[0,0]) / 180)
    return np.arccos(a + b) * 6371.0


def clusterClubs(numClust = 5):
    datList = []
    for line in open('places.txt').readlines():
        lineArr = line.split('\t')
        datList.append([float(lineArr[4]), float(lineArr[3])])
    
    datMat = np.mat(datList)
    myCentroids, clustAssing = biKmeans(datMat, numClust, distMeas=distSLC)
    
    fig = plt.figure()
    rect = [0.1, 0.1, 0.8, 0.8]
    scatterMarkers = ['s', 'o', '^', '8', 'p', 'd', 'v', 'h', '>', '<']
    axprops = dict(xticks=[], yticks=[])
    ax0 = fig.add_axes(rect, label='ax0', **axprops)
    imgP = plt.imread('Portland.png')
    ax0.imshow(imgP)
    ax1 = fig.add_axes(rect, label='ax1', frameon = False)
    for i in range(numClust):
        ptsIncurrCluster = datMat[np.nonzero(clustAssing[:, 0].A == i)[0], :]
        marketStyle = scatterMarkers[ i % len(scatterMarkers)]
        ax1.scatter(myCentroids[:,0].flatten().A[0], myCentroids[:, 1].flatten().A[0], marker='+', s=300)
    plt.show()




if __name__ == '__main__':
    # datMat = np.mat(loadDataSet('testSet.txt'))    
    # temp = randCent(datMat, 2)

    # print(np.min(datMat[:, 0]))
    # print(np.max(datMat[:, 0]))
    # print(np.min(datMat[:, 1]))
    # print(np.max(datMat[:, 1]))
    # print(temp)

    # print(distEclud(datMat[0], datMat[1]))
    # myCentroids, clustAssing = kMeans(datMat, 4)
    # datMat3 = np.mat(loadDataSet('testSet2.txt'))
    # myCentroids, clusAssing = biKmeans(datMat3, 3)
    # print(myCentroids)

    clusterClubs(5)



