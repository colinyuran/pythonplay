from math import log
import operator
import treePlotter
import pickle

def createDataSet():
    dataSet = [[1,1,'yes'], [1,1,'yes'], [1,0,'no'], [0,1,'no'], [0,1,'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


def calcShannonEnt (dataSet):
    numEntries = len(dataSet)
    labelCounts = {}

    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob, 2)
    
    return shannonEnt


def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if (featVec[axis] == value):
            reducedFeatVec = featVec[ : axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1

    baseEntropy = calcShannonEnt(dataSet)

    bestInfoGrain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [ example[i] for example in dataSet ]
        uniqueVals = set(featList)

        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)

        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGrain):
            bestInfoGrain = infoGain
            bestFeature = i
    
    return bestFeature


def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1

    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), revserse=True)
    return sortedClassCount[0][0]            


def createTree(dataSet, labels):
    classList = [ example[-1] for example in dataSet]

    if classList.count(classList[0]) == len(classList):
        return classList[0]
    
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)

    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]

    myTree = { bestFeatLabel: {}}
    del(labels[bestFeat])

    featValues = [ example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        subDataSet = splitDataSet(dataSet, bestFeat, value)
        myTree[bestFeatLabel][value] = createTree(subDataSet, subLabels)

    return myTree


def  classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)

    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    
    return classLabel

def storeTree(inputTree, fileName):
    fw = open(fileName, 'wb')
    pickle.dump(inputTree, fw)
    fw.close()

def grabTree(fileName):
    fr = open(fileName,'rb')
    return pickle.load(fr)



if (__name__ == '__main__'):
    # myDat, labels = createDataSet()
    # myDat[0][-1] = 'maybe'
    # print(myDat)
    # print(labels)
    # shannonEnt = calcShannonEnt(myDat)
    # print(shannonEnt)

    # myDat, labels = createDataSet()
    # print(myDat)        
    # split0_0 = splitDataSet(myDat, 0, 0)
    # split0_1 = splitDataSet(myDat, 0, 1)
    # print(split0_0)
    # print(split0_1)

    # myDat, labels = createDataSet()
    # print(myDat)        
    # bestFeature = chooseBestFeatureToSplit(myDat)
    # print(bestFeature)

    # myDat, labels = createDataSet()
    # print(myDat)
    # myTree = createTree(myDat, labels)
    # print(myTree)

    # myDat, labels = createDataSet()
    # print(myDat)
    # print(labels)
    # myTree = treePlotter.retrieveTree(0)
    # print(myTree)
    # r = classify(myTree, labels, [1,0])
    # print(r)
    # r = classify(myTree, labels, [1,1])
    # print(r)

    fr = open('lenses.txt')
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    
    lensesTree = createTree(lenses, lensesLabels)
    print(lensesTree)
    treePlotter.createPlot(lensesTree)
    storeTree(lensesTree, 'treeModel')
    loadedTree = grabTree('treeModel')
    treePlotter.createPlot(loadedTree)