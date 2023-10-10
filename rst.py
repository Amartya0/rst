import numpy as np


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

    return dataSet

# a function that takes a dataset and a set of attributes and returns the resultant indiscernibility relation set
# format of dataset
# 1st row: 0th index is ID, 1st to last-1 index is attributes, last index is class
# 2nd row onwards: 0th index is ID/object name, 1st to last-1 index is attribute values of object ID, last index is class value of object ID


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

    return indA


dataSet = rstDataGenerator(10, 3, 2)
# print(dataSet)
print(indiscernibility(dataSet, ['A1', 'A2']))
