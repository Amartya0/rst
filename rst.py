import numpy as np
import random


# A function that take number of objects n, number of attributes a, and number of classes d and generate a random dataset
def rstDataGenerator(n, a, d):
    # give each attribute a random value between 1 and 4 (including 1 but not 3)
    dataSet = np.random.randint(1, 3, size=(n, 1))
    for _ in range(1, a):
        dataSet = np.append(dataSet, np.random.randint(
            1, a+2, size=(n, 1)), axis=1)
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


def rst(eqvClassesSet, targetSet):
    lowerApproximationSet = lowerApproximation(eqvClassesSet, targetSet)
    upperApproximationSet = upperApproximation(eqvClassesSet, targetSet)
    return lowerApproximationSet, upperApproximationSet


# a function that calculates the dependency of attributeSet2(q) on attributeSet1(p) given input dataset equivalent classes of attributeSet1 and attributeSet2
def attributeDependency(dataSet, equivalenceClassesSet1, equivalenceClassesSet2):
    # get the total number of objects in the dataset
    totalObjects = len(dataSet)-1

    # calculate the sum of lower application of each equivalent class of the indecernibility relation of the attribute set 2 for the attribute set 1
    sumLowerApprox = 0
    for i in equivalenceClassesSet2:
        sumLowerApprox += len(lowerApproximation(equivalenceClassesSet1, i))

    # return the dependency of attributeSet2(q) on attributeSet1(p) round of to 6 decimal places
    return round(sumLowerApprox/totalObjects, 6)

#


def discernibilityMatrix(dataset):
    # get all the attributes
    attributes = dataset[0][1:-1]
    # randomly select one object
    object = random.choice(dataset[1:])
    # delete all the objects with same class as the selected object
    dataset = np.delete(dataset, np.where(
        dataset[:, -1] == object[-1]), axis=0)
    # initialize a frequency dictionary of each attribute as key and 0 as value
    attributeFrequency = {}
    for _ in attributes:
        attributeFrequency[_] = 0

    core = []
    # discernMatrix = []
    # for each object in the dataset, if the value of an attribute is different from the selected object, increment the frequency of that attribute
    for i in dataset[1:]:
        count = 0
        coreIndex = 0
        # temp = []
        for j in range(len(attributes)):

            if i[j+1] != object[j+1]:
                attributeFrequency[attributes[j]] += 1
                count += 1
                coreIndex = j
                # temp.append(attributes[j])
        # discernMatrix.append(temp)
        if count == 1:
            if attributes[coreIndex] not in core:
                core.append(attributes[coreIndex])

    # delete the attributes from attributeFrequency that are already in the reduct
    for i in core:
        if i in attributeFrequency:
            del attributeFrequency[i]

    # sort the attributeFrequency dictionary by values in descending order
    attributeFrequency = {k: v for k, v in sorted(
        attributeFrequency.items(), key=lambda item: item[1], reverse=True)}

    return list(sorted(core)), list(attributeFrequency.keys())

    # print(type(core), type(attributeFrequency.keys()))


def reduct(dataSet):
    attributes = list(dataSet[0][1:-1])
    reductSet, attributeFrequency = discernibilityMatrix(dataSet)

    temp = (attributeDependency(dataSet, eqvClasses(indiscernibility(
        dataSet, reductSet)), eqvClasses(indiscernibility(dataSet, attributes))))
    if int(temp) == 1:
        return reductSet
    else:
        for i in attributeFrequency:
            reductSet.append(i)
            temp = (attributeDependency(dataSet, eqvClasses(indiscernibility(
                dataSet, reductSet)), eqvClasses(indiscernibility(dataSet, attributes))))
            if int(temp) == 1:
                return sorted(list(reductSet))

    return sorted(list(reductSet))

# a function that takes in a dataset and returns the horizontally reduced dataset


def horizontalReduct(dataSet):
    reductSet = reduct(dataSet)
    # delete the attributes from the dataset that are not in the reduct
    deleteIndex = []
    for i in range(len(dataSet[0])-2):
        if dataSet[0][i+1] not in reductSet:
            deleteIndex.append(i+1)

    # print(reductSet, deleteIndex)
    dataSet = np.delete(dataSet, deleteIndex, axis=1)
    return dataSet


# a function that takes in a dataset and returns the vertically reduced dataset
def verticalReduct(dataset):
    # delete any two objects with same attributes and class values
    for i in range(len(dataset)-1):
        for j in range(i+1, len(dataset)-1):
            if np.array_equal(dataset[i][1:], dataset[j][1:]):
                dataset = np.delete(dataset, j, axis=0)

    return dataset


def ranks(sample):
    # Return the ranks of each element in an integer sample.
    indices = sorted(range(len(sample)), key=lambda i: sample[i])
    return sorted(indices, key=lambda i: indices[i])


def sample_with_minimum_distance(n, k, d):
    # Sample of k elements from range(n), with a minimum distance d.
    sample = random.sample(range(1, n-(k-1)*(d-1)), k)
    return [s + (d-1)*r for s, r in zip(sample, ranks(sample))]


# n = 10
# a = 4
# dataSet = rstDataGenerator(n, a, 2)

# attributeSetIndex = sorted(
#     sample_with_minimum_distance(a, random.randint(1, a-1), 1))
# attributeSet = []
# for i in range(len(attributeSetIndex)):
#     attributeSet.append(dataSet[0][attributeSetIndex[i]])

# print("------------------------------------------------Attribute Set------------------------------------------------\n", attributeSet)

# eqvClassesSet = eqvClasses(indiscernibility(dataSet, attributeSet))
# print("------------------------------------------------Equivalent Classes------------------------------------------------\n", eqvClassesSet)

# for i in range(1):

#     targetSet = list(map(str, sorted(map(int, (np.random.choice(
#         dataSet[1:, 0], random.randint(1, n), replace=False))))))
#     print("------------------------------------------------", i +
#             1, "------------------------------------------------")
#     print("------------------------------------------------Target Set------------------------------------------------\n", targetSet)
#     print("------------------------------------------------RST------------------------------------------------\n",
#             rst(eqvClassesSet, targetSet))

# print("------------------------------------------------End------------------------------------------------")
