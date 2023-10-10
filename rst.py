import numpy as np
import random


# A function that take number of objects n, number of attributes a, and number of classes d and generate a random dataset
def rstDataGenerator(n, a, d):
    # give each attribute a random value between 1 and 4 (including 1 but not 4)
    dataSet = np.random.randint(1, 4, size=(n, a))
    # assign each object a random class of d classes
    classes = np.random.randint(1, d+1, size=(n, 1))
    dataSet = np.append(dataSet, classes, axis=1)
    # give each object an id
    id = np.arange(1, n+1).reshape(n, 1)
    dataSet = np.append(id, dataSet, axis=1)
    # add column names
    columnNames = []
    columnNames.append('ID')
    for i in range(a):
        columnNames.append('A'+str(i+1))

    columnNames.append('Class')
    dataSet = np.append([columnNames], dataSet, axis=0)

    # print(dataSet)
    return dataSet


# format of dataset
# 1st row: 0th index is ID, 1st to last-1 index is attributes, last index is class
# 2nd row onwards: 0th index is ID/object name, 1st to last-1 index is attribute values of object ID, last index is class value of object ID


# a function that takes a dataset and a number of attributes and returns the resultant indiscernibility relation set
def indiscernibility(dataSet, attributeSet):
    # extract attributes from dataSet
    if attributeSet == []:
        return []

    allAttributes = list(dataSet[0][1:-1])
    givenAttributesIndex = []
    givenAttributesLength = len(attributeSet)
    for i in range(givenAttributesLength):
        givenAttributesIndex.append(allAttributes.index(attributeSet[i])+1)
    # print(givenAttributesIndex)

    data = dataSet[1:]
    indA = []

    for i in range(len(data)):
        for j in range(len(data)):
            if i == j:
                indA.append([data[i][0], data[j][0]])
            else:
                flag = True
                for k in range(givenAttributesLength):
                    if data[i][givenAttributesIndex[k]] != data[j][givenAttributesIndex[k]]:
                        flag = False
                        break
                if flag:
                    indA.append([data[i][0], data[j][0]])

    # print(indA, end='\n\n')
    return indA


# A function that returns the equivalence classes of a given equivalence relation set
def eqvClasses(eqvRelSet):

    eqvClasses = []
    for i in range(len(eqvRelSet)):
        if eqvClasses == []:
            eqvClasses.append([eqvRelSet[i][0], eqvRelSet[i][1]])
        else:
            flag = True
            for j in range(len(eqvClasses)):
                if eqvRelSet[i][0] in eqvClasses[j]:
                    eqvClasses[j].append(eqvRelSet[i][1])
                    flag = False
                    break
                elif eqvRelSet[i][1] in eqvClasses[j]:
                    eqvClasses[j].append(eqvRelSet[i][0])
                    flag = False
                    break
            if flag:
                eqvClasses.append([eqvRelSet[i][0], eqvRelSet[i][1]])

    for i in range(len(eqvClasses)):
        eqvClasses[i] = list(
            map(str, sorted(list(map(int, set(eqvClasses[i]))))))

    # print(eqvClasses, end='\n\n')
    return eqvClasses


# a function that takes a set of equivalence classes and target set and
# returns the lower approximation of Rough Set of that target set
def lowerApproximation(eqvClasses, targetSet):
    lowerApprox = []
    for i in range(len(eqvClasses)):
        if set(eqvClasses[i]).issubset(set(targetSet)):
            for j in range(len(eqvClasses[i])):
                if eqvClasses[i][j] not in lowerApprox:
                    lowerApprox.append(eqvClasses[i][j])
    lowerApprox = list(map(str, sorted(list(map(int, set(lowerApprox))))))
    # print(lowerApprox, end='\n\n')
    return lowerApprox


def upperApproximation(eqvClasses, targetSet):
    upperApprox = []
    for i in range(len(eqvClasses)):
        if set(eqvClasses[i]).isdisjoint(set(targetSet)):
            continue
        else:
            for j in range(len(eqvClasses[i])):
                if eqvClasses[i][j] not in upperApprox:
                    upperApprox.append(eqvClasses[i][j])
    upperApprox = list(map(str, sorted(list(map(int, set(upperApprox))))))
    # print(upperApprox, end='\n\n')
    return upperApprox


def rst(dataSet, noOfAttributes, targetSet):
    eqvClassesSet = eqvClasses(indiscernibility(dataSet, noOfAttributes))
    print("------------------------------------------------Equivalent Classes------------------------------------------------\n",
          eqvClassesSet, end='\n\n')
    lowerApproximationSet = lowerApproximation(eqvClassesSet, targetSet)
    upperApproximationSet = upperApproximation(eqvClassesSet, targetSet)
    return lowerApproximationSet, upperApproximationSet


def ranks(sample):
    # Return the ranks of each element in an integer sample.
    indices = sorted(range(len(sample)), key=lambda i: sample[i])
    return sorted(indices, key=lambda i: indices[i])


def sample_with_minimum_distance(n, k, d):
    # Sample of k elements from range(n), with a minimum distance d.
    sample = random.sample(range(1, n-(k-1)*(d-1)), k)
    return [s + (d-1)*r for s, r in zip(sample, ranks(sample))]


n = 10
a = 5
dataSet = rstDataGenerator(n, a, 2)

attributeSetIndex = sorted(
    sample_with_minimum_distance(a+1, random.randint(1, a), 1))

attributeSet = []
for i in range(len(attributeSetIndex)):
    attributeSet.append(dataSet[0][attributeSetIndex[i]])


targetSet = list(map(str, sorted(map(int, (np.random.choice(
    dataSet[1:, 0], random.randint(1, n), replace=False))))))

print("------------------------------------------------Dataset------------------------------------------------\n", dataSet)
print("------------------------------------------------Attribute Set------------------------------------------------\n", attributeSet)
print("------------------------------------------------Target Set------------------------------------------------\n", targetSet)
print("------------------------------------------------RST------------------------------------------------\n",
      rst(dataSet, attributeSet, targetSet))
