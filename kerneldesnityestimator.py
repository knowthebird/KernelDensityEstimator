import csv
import numpy as np
import math
import matplotlib.pyplot as plt  # used for plotting data points

TrainingFilename = "train.csv"
TestingFilename = "test.csv"
#TrainingFilename = "zipcode_train.csv"
#TestingFilename = "zipcode_test.csv"

h = 0.5

# The following is used to plot the class boundaries
# .....................................................................................
# def plot_by_class_better(dataset, classes, title):
#     dataset[:, 2] = classes
#     x0 = [row[0] for row in dataset if 0 in row]
#     y0 = [row[1] for row in dataset if 0 in row]
#     x1 = [row[0] for row in dataset if 1 in row]
#     y1 = [row[1] for row in dataset if 1 in row]
#     plt.scatter(x0, y0, label='0', color="red", marker="o", s=30)
#     plt.scatter(x1, y1, label='1', color="blue", marker="o", s=30)
#     plt.xlabel('x - axis')
#     plt.ylabel('y - axis')
#     plt.title(title)
#     plt.show()
#     return
#
# def createDataPoint():
#     x = -3
#     y = -8
#     data = []
#     res = 10
#     for i in range(9*res):
#         for j in range(20*res):
#             data.append([(x*res+i+0.1)/res, (y*res+j+0.1)/res, 0])
#     data = np.asarray(data)
#     return data
# .....................................................................................


def load_csv(filename):
    lines = csv.reader(open(filename, "r", encoding='utf-8-sig'))
    dataset = list(lines)
    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]
    dataset = np.asarray(dataset, dtype=np.float32)
    return dataset


# Seperate data by class
def class_sorted_data(dataset):
    classes = np.unique(dataset[:, np.size(dataset, 1) - 1])
    sortedclassdata = []
    for i in range(len(classes)):
        item = classes[i]
        itemindex = np.where(dataset[:, np.size(dataset, 1) - 1] == item)   # index  of rows with label class[i]
        singleclassdataset = dataset[itemindex, 0:np.size(dataset, 1) - 1]  # array  of data for class[i]
        sortedclassdata.append(np.matrix(singleclassdataset))               # matrix of data for class[i]
    return sortedclassdata, classes


# note, this implementation returns value @ class index, not class index
def kerndensitymodel(dataset, classes, h, x):
    probs = np.zeros(shape=(len(dataset),1))

    numClasses = len(dataset)
    D = dataset[0][0].shape[1]

    for i in range(numClasses):
        prob = 0
        N = len(dataset[i])
        for j in range(N):
            exponent = -1 * ((np.linalg.norm(x-dataset[i][j])**2) / ((2 * h) ** 2))
            base = 1 / ((2 * math.pi * (h ** 2)) ** (D / 2))
            prob = prob + base * math.exp(exponent)
        prob = prob / N
        probs[i] = prob

    classPrediction = classes[np.argmax(probs)]
    return classPrediction

def trainAccuracy(dataset, h):
    numCorrect = 0
    for i in range(len(dataset)):
        TrueValue = dataset[i][-1]
        x = np.delete(dataset, -1, axis=1)[i]
        trainer = np.delete(dataset, i, axis=0)
        sortclassdata, classes = class_sorted_data(trainer)
        if kerndensitymodel(sortclassdata, classes, h, x) == TrueValue:
            numCorrect = numCorrect + 1
    accuracy = 100 * (numCorrect/ len(dataset))
    return accuracy

def testAccuracy(traindataset, testdataset, h):
    sortclassdata, classes = class_sorted_data(traindataset)
    numCorrect = 0
    for i in range(len(testdataset)):
        TrueValue = testdataset[i][-1]
        x = np.delete(testdataset, -1, axis=1)[i]
        if kerndensitymodel(sortclassdata, classes, h, x) == TrueValue:
            numCorrect = numCorrect + 1
    accuracy = 100 * (numCorrect/ len(testdataset))
    return accuracy

training_data = load_csv(TrainingFilename)
testing_data = load_csv(TestingFilename)

accuracy = trainAccuracy(training_data, h)
print(f"Training Accuracy {accuracy}% with h = {h}")

accuracy = testAccuracy(training_data, testing_data, h)
print(f"Testing Accuracy {accuracy}% with h = {h}")



# The following is used to plot the class boundaries
# .....................................................................................
# sortclassdata, classes = class_sorted_data(training_data)
# h = 0.5
# testingData = createDataPoint()
# print(len(testingData))
# clas = []
# for i in range(len(testingData)):
#     x = np.delete(testingData, -1, axis=1)[i]
#     clas = np.append(clas, kerndensitymodel(sortclassdata, classes, h, x))
#
# print(clas)
# plot_by_class_better(testingData , clas, 'Kernel Density Estimator')
# .....................................................................................