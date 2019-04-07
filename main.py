from matplotlib import pyplot

import numpy


LEARNING_RATE = 0.01
CRITIC_NUM = 0.00001
ITERATION = 10000


def main():
    fileName = "dataset1.txt"
    file = open(fileName, "r")
    fileContent = file.read()

    """ I changed letters to 0 and 1 to load text file as a numpy array."""
    fileContent = fileContent.replace("R", "0")
    fileContent = fileContent.replace("M", "1")

    file = open("temp.txt", "w")
    file.write(fileContent)
    print(fileName + " has been opened.\n\nTraining started!")
    trainModel()

    print("\n\n")

    fileName = "dataset2.txt"
    file = open(fileName, "r")
    fileContent = file.read()

    """ I changed letters to 0 and 1 to load text file as a numpy array."""
    fileContent = fileContent.replace("b", "0")
    fileContent = fileContent.replace("g", "1")

    file = open("temp.txt", "w")
    file.write(fileContent)
    print(fileName + " has been opened.\n\nTraining is started!")
    trainModel()


def trainModel():
    data = numpy.loadtxt("temp.txt", delimiter=",")
    totalCount = len(data)
    print("Number of data: " + str(totalCount))

    trainCount = int(totalCount*0.8)
    print("Number of train data: " + str(trainCount))
    print("Number of test data: " + str(totalCount-trainCount))

    """Numpy array is shuffled."""
    numpy.random.seed(30)
    numpy.random.shuffle(data)

    columnCount = len(data[0])
    inputs = data[:, 0:columnCount-1]
    results = data[:, columnCount-1:]

    trainData = inputs[0:trainCount, :]
    trainResult = results[0:trainCount, :]

    testData = inputs[trainCount:, :]
    testResult = results[trainCount:, :]

    betaArray = numpy.zeros(shape=(columnCount, 1))
    betaArray, plotData = mini_batch_gradient_descent(trainData, trainResult, betaArray, testData, testResult)

    """Final prediction is calculated."""
    testData = numpy.c_[numpy.ones(len(testData)), testData]
    final = numpy.dot(testData, betaArray)

    """I assigned indexes, which is lower than 0.5, to zero. In the same way, i assigned indexes, which is greater than 0.5, to one"""
    final[final > 0.5] = 1
    final[final < 0.5] = 0
    faultCount = 0
    for i in range(len(final)):
        if final[i] != testResult[i]:
            faultCount = faultCount + 1

    rate = "%.2f" % ((len(final)-faultCount) * 100 / len(final))
    print("Training is finished!\n")
    print("Test accuracy: %" + str(rate))

    x, y = numpy.asarray(plotData).T
    pyplot.scatter(x, y)
    pyplot.show()


def mini_batch_gradient_descent(data, result, beta, testData, testResult, learning_rate=LEARNING_RATE, iterations=ITERATION, batch_size=16):
    global CRITIC_NUM

    size = len(result)

    """It is used to hold prediction results which is produced in every iteration"""
    testAccuracies = []

    beforeOptimizationFunctionResult = 1
    for iteration in range(iterations):
        optimizationFunctionResult = 0.0

        for i in range(0, size, batch_size):
            sub_data = data[i:i + batch_size]
            sub_result = result[i:i + batch_size]

            sub_data = numpy.c_[numpy.ones(len(sub_data)), sub_data]

            prediction = numpy.dot(sub_data, beta)

            beta = beta - (1 / size) * learning_rate * (sub_data.T.dot((prediction - sub_result)))

            optimizationFunctionResult += optimizationFunction(sub_result, prediction)

        tempPrediction = calculatePrediction(beta, testData, testResult)
        testAccuracies.append([iteration, tempPrediction])

        """At below, differences of optimizationFunctionResults are calculated to decide when to stop iteration."""
        difference = abs(optimizationFunctionResult - beforeOptimizationFunctionResult)
        if difference < CRITIC_NUM:
            break

        beforeOptimizationFunctionResult = optimizationFunctionResult

    return beta, testAccuracies


"""Prediction is calculated at below. This method is called in the iteration."""
def calculatePrediction(beta, testData, testResult):
        tempData = numpy.c_[numpy.ones(len(testData)), testData]
        prediction = numpy.dot(tempData, beta)

        prediction[prediction > 0.5] = 1
        prediction[prediction < 0.5] = 0

        faultCount = numpy.count_nonzero(prediction == testResult)*100/len(testResult)
        return faultCount


"""Log-likelihood function."""
def optimizationFunction(y, prediction):
    return (-y * numpy.log(1 / (1 + numpy.exp(-prediction))) - (1 - y) * numpy.log(1 - 1 / (1 + numpy.exp(-prediction)))).mean()


if __name__ == "__main__":
    main()
