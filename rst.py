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

    return dataSet
