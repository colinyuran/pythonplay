import numpy as np
import sys


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


def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
        # else:
        #     print('the word:{0} is not in my Vocabulary!'.format(word))
    return returnVec


def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / float(numTrainDocs)

    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0

    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    
    p1Vect = np.log(p1Num / p1Denom)
    p0Vect = np.log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)

    if p1 > p0:
        return 1
    else:
        return 0


def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postInDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postInDoc))

    p0V, p1V, pAb = trainNB0(trainMat, listClasses)
    # print(p0V)
    # print(p1V)
    # print(pAb)

    testEntry = ['love', 'my','dalmation']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    result = classifyNB(thisDoc, p0V, p1V, pAb)
    print('{0} classified as: {1}'.format(testEntry, result))

    testEntry = ['stupid', 'garbage']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    result = classifyNB(thisDoc, p0V, p1V, pAb)
    print('{0} classified as: {1}'.format(testEntry, result))


def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


def spamTest():
    docList = []
    classList = []
    fullText = []

    for i in range(1,26):
        try :
            wordList = textParse(open('email/spam/{0}.txt'.format(i)).read())
            docList.append(wordList)
            fullText.extend(wordList)
            classList.append(1)

            c = open('email/ham/{0}.txt'.format(i)).read()
            wordList = textParse(c)
            docList.append(wordList)
            fullText.extend(wordList)
            classList.append(0)
        except : 
            print(i)
            print("Unexpected error:", sys.exc_info()[0])

    vocabList = createVocabList(docList)
    trainingSet = list(range(50))
    testSet = []

    for i in range(10):
        randIndex = int(np.random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        trainingSet.remove(trainingSet[randIndex])

    trainMat = []
    trainClasses = []
    
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])

    p0V, p1V, pSpam = trainNB0(np.array(trainMat), np.array(trainClasses))
    
    errorCount = 0

    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        if classifyNB(np.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            #print(docList[docIndex])
            errorCount += 1
    
    print('The error rate is: {0}'.format(float(errorCount) / len(testSet)))


def calcMostFreq(vocabList, fullText):
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token)
    sortedFreq = sorted(freqDict.items(), key=operator.itemgetter(1), reverse=True)
    return sortedFreq[:30]


def localWords(feed1, feed0):
    docList = []
    classList = []
    fullText = []

    minLen = min(len(feed1['entries']), len(feed0['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)

        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)

    vocabList = createVocabList(docList)
    top30Words = calcMostFreq(vocabList, fullText)

    for pairW in top30Words:
        if pairW[0] in vocabList:
            vocabList.remove(pairW[0])
    trainingSet = list(range(2 * minLen))
    testSet = []

    for i in range(20):
        randIndex  = int(np.random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        trainingSet.remove(trainingSet[randIndex])

    trainMat = []
    trainClasses = []

    for docIndex in trainingSet:
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])

    p0V, p1V, pSpam = trainNB0(np.array(trainMat), np.array(trainClasses))

    errorCount = 0
    for docIndex in testSet:
        wordVec = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(np.array(wordVec), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    
    print('the error rate is: {0}'.format(float(errorCount) / len(testSet)))
    return vocabList, p0V, p1V


def testAd():
    import feedparser
    try:
        ny = feedparser.parse('https://newyork.craigslist.org/stp/index.rss')
        sf = feedparser.parse('https://sfbay.craigslist.org/stp/index.rss')

    except:
        print("Unexpected error:", sys.exc_info()[0])
        return None

    vocabList, pSF, pNY = localWords(ny, sf)


def getTopWords():
    import feedparser
    
    ny = feedparser.parse('https://newyork.craigslist.org/stp/index.rss')
    sf = feedparser.parse('https://sfbay.craigslist.org/stp/index.rss')
    vocabList, p0V, p1V = localWords(ny, sf)

    topNY = []
    topSF = []

    for i in range(len(p0V)):
        if p0V[i] > -6.0:
            topSF.append((vocabList[i], p0V[i]))
        if p1V[i] > -6.0:
            topNY.append((vocabList[i], p1V[i]))
    
    sortedSF = sorted(topSF, key = lambda pair: pair[1], reverse=True)
    print('SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**')
    for item in sortedSF:
        print(item[0])

    sortedNY = sorted(topNY, key = lambda pair: pair[1], reverse=True)
    print('NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**')
    for item in sortedNY:
        print(item[0])


if __name__ == '__main__':
    # listPosts, listClasses = loadDataSet()
    # myVocabList = createVocabList(listPosts)
    # print(myVocabList)
    # setVec = setOfWords2Vec(myVocabList, listPosts[0])
    # print(setVec)

    #testingNB()
    #spamTest()
    #testAd()
    getTopWords()