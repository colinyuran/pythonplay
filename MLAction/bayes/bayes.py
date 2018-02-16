import numpy as np


def loadDataSet():
    postingList = [ ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                    ['maybe', 'not', 'take','him', 'to', 'dog','park','stupid'],
                    ['my','dalmation','is','so','cute','I','love','him'],
                    ['stop','posting','stupid','worthless','garbage'],
                    ['mr','licks', 'ate','my','steak','how','to','stop','him'],
                    ['quit','buying','worthless','dog','food','stupid'] ]

    classVec = [0,1,0,1,0,1]
    return postingList, classVec


def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    vocabList = list(vocabSet)
    vocabList.sort()
    return vocabList


def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print('the word:{0} is not in my Vocabulary!'.format(word))
    return returnVec


def trainNBO(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / float(numTrainDocs)

    p0Num = np.zeros(numWords)
    p1Num = np.zeros(numWords)
    p0Denom = 0.0
    p1Denom = 0.0

    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    
    p1Vect = p1Num / p1Denom
    p0Vect = p0Num / p0Denom
    return p0Vect, p1Vect, pAbusive


if __name__ == '__main__':
    listPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listPosts)
    print(myVocabList)
    setVec = setOfWords2Vec(myVocabList, listPosts[0])
    print(setVec)
