import numpy as np

def loadExData():
    return [ [1,1,1,0,0],
             [2,2,2,0,0],
             [1,1,1,0,0],
             [5,5,5,0,0],
             [1,1,0,2,2],
             [0,0,0,3,3],
             [0,0,0,1,1]]


def loadExData2():
    return[[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
           [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
           [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
           [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
           [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
           [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
           [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
           [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
           [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]


def euclidSim(inA, inB):
    return 1.0 / (1.0 + np.linalg.norm(inA - inB))


def pearsSim(inA, inB):
    if len(inA) < 3:
        return 1.0
    return 0.5 + 0.5 * np.corrcoef(inA, inB, rowvar=0)[0][1]


def cosSim(inA, inB):
    num = float(inA.T * inB)
    denom = np.linalg.norm(inA) * np.linalg.norm(inB)
    return 0.5 + 0.5 * (num / denom)


def standEst(dataMat, user, simMeas, item):
    n = np.shape(dataMat)[1]
    simTotal = 0.0
    ratSimTotal = 0.0

    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0:
            continue
        
        overLap = np.nonzero(np.logical_and(dataMat[:, item].A > 0, dataMat[:, j].A > 0))[0]
        if len(overLap) == 0:
            similarity = 0
        else:
            similarity = simMeas(dataMat[overLap, item], dataMat[overLap, j])
        print('the {0} and {1} similarity is:{2}'.format(item, j, similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal / simTotal


def recommend(dataMat, user, N = 3, simMeas=cosSim, estMethod=standEst):
    unratedItems = np.nonzero(dataMat[user, :].A == 0)[1]

    if len(unratedItems) == 0:
        return 'you rated everything'
    itemScores = []
    for item in unratedItems:
        estimatedScore = estMethod(dataMat, user, simMeas, item)
        itemScores.append((item, estimatedScore))
    
    return sorted(itemScores, key=lambda jj: jj[1], reverse=True)[:N]


def svdEst(dataMat, user, simMeas, item):
    n = np.shape(dataMat)[1]
    simTotal = 0.0
    ratSimTotal = 0.0
    U, Sigma, VT = np.linalg.svd(dataMat)

    Sig4 = np.mat(np.eye(4) * Sigma[:4])
    xformedItems = dataMat.T * U[:,:4] * Sig4.I
    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0  or j == item:
            continue
        similarity = simMeas(xformedItems[item,:].T, xformedItems[j,:].T)
        print('the {0} and {1} similarity is:{2}'.format(item, j, similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal / simTotal



def test():
    data = [ [4,4,0,2,2],
             [4,0,0,3,3],
             [4,0,0,1,1],
             [1,1,1,2,0],
             [2,2,2,0,0],
             [1,1,1,0,0],
             [5,5,5,0,0]]
    myMat = np.mat(data)
    print(myMat)

    re = recommend(myMat, 2)
    print(re)

def test2():
    # data = np.mat(loadExData2())
    # U, Sigma, VT = np.linalg.svd(data)
    # print(Sigma)

    myMat = np.mat(loadExData2())
    re = recommend(myMat, 1, estMethod=svdEst)
    print(re)
    # Sig3 = np.mat([[Sigma[0], 0, 0], [0, Sigma[1], 0], [0, 0, Sigma[2]]])
    # simu = U[:,:3] * Sig3 * VT[:3, :]
    # print(simu)

if __name__ == '__main__':


    # myMat = np.mat(data)
    # print(euclidSim(myMat[:, 0], myMat[:, 4]))
    # print(euclidSim(myMat[:, 0], myMat[:, 0]))

    # print(cosSim(myMat[:, 0], myMat[:, 4]))
    # print(cosSim(myMat[:, 0], myMat[:, 0]))

    # print(pearsSim(myMat[:, 0], myMat[:, 4]))
    # print(pearsSim(myMat[:, 0], myMat[:, 0]))

    #test()

    test2()