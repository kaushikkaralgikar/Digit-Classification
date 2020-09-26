import random
import sys 
import csv
import numpy as np

if(len(sys.argv)==4):
    trainImageFile = sys.argv[1]
    trainLabelFile = sys.argv[2]
    testImageFile = sys.argv[3]
else:
    trainImageFile = 'train_image.csv'
    trainLabelFile = 'train_label.csv'
    testImageFile = 'test_image.csv'

def splitIntoTenNeurons(i):
    x = np.zeros((10, 1))
    x[i] = 1.0
    return x

#'C:\\Users\\kaush\\hello\\.vscode\\train_label.csv'
outputFile = 'test_predictions.csv'
trainLables = []
with open(trainLabelFile, 'r') as train_label_file:
    train_label_file_reader = csv.reader(train_label_file)
    for row in train_label_file_reader:
        trainLables.append(splitIntoTenNeurons(np.int32(row)))
train_label_file.close()

#'C:\\Users\\kaush\\hello\\.vscode\\train_image.csv'
trainImages = []
with open(trainImageFile, 'r') as train_image_file:
    train_image_file_reader = csv.reader(train_image_file)
    for row in train_image_file_reader:
        trainImages.append(np.reshape(np.float32(row)/255.0, (784, 1)))
train_image_file.close()

#'C:\\Users\\kaush\\hello\\.vscode\\test_image.csv'
testImages = []
with open(testImageFile, 'r') as test_image_file:
    test_image_file_reader = csv.reader(test_image_file)
    for row in test_image_file_reader:
        testImages.append(np.reshape(np.float32(row)/255.0, (784, 1)))
test_image_file.close()

trainingData = list(zip(trainImages, trainLables))

class NeuralNetwork(object):

    def __init__(self, arrayOfLayers):
        self.numberOfLayers = len(arrayOfLayers)
        self.arrayOfLayers = arrayOfLayers
        self.weights = [np.random.randn(y, x) for x, y in zip(arrayOfLayers[:-1], arrayOfLayers[1:])]
        self.biases = [np.random.randn(y, 1) for y in arrayOfLayers[1:]]
        self.testResults = []
        
    def train(self, trainingData, noOfEpochs, batchSize, errorConstant):
        random.shuffle(trainingData)
        for i in range(noOfEpochs):
            batches = [trainingData[j:j+batchSize] for j in range(0, len(trainingData), batchSize)]
            for batch in batches:
                self.trainBatch(batch, errorConstant)

    def test(self, testImages):
        count = 0
        testResults = []
        for x in testImages:
            for b, w in zip(self.biases, self.weights):
                count = count+1
                if count == self.numberOfLayers-1:
                    x = softmax(np.dot(w,x)+b)
                else:    
                    x = sigmoid(np.dot(w, x)+b)
            testResults.append(np.argmax(x))        
        self.testResults = testResults

    def outputTestPredictions(self):
        with open(outputFile, mode= 'w', newline='') as result_file:
            result_writer = csv.writer(result_file)
            result_writer.writerows(map(lambda x:[x],self.testResults))
            result_file.close()      

    def trainBatch(self, batch, errorConstant):
        biasAdjust = [np.zeros(b.shape) for b in self.biases]
        weightAdjust = [np.zeros(w.shape) for w in self.weights]
        for batchTrainingInput, bathTrainingOutput in batch:
            biasDelta, weightsDelta = self.updateWeightsAndBias(batchTrainingInput,bathTrainingOutput )
            biasAdjust = [nb+dnb for nb, dnb in zip(biasAdjust, biasDelta)]
            weightAdjust = [nw+dnw for nw, dnw in zip(weightAdjust, weightsDelta)]
        self.weights = [w-(errorConstant*wa) for w, wa in zip(self.weights, weightAdjust)]
        self.biases = [b-(errorConstant*ba) for b, ba in zip(self.biases, biasAdjust)]

    def updateWeightsAndBias(self, batchTrainingInput, bathTrainingOutput):
        activation = batchTrainingInput
        activations = [batchTrainingInput]
        zValueArray = []
        count = 0;
        for b, w in zip(self.biases, self.weights):
            count = count+1
            z = np.dot(w, activation)+b
            zValueArray.append(z)
            if count == self.numberOfLayers-1:
                activation = softmax(z)
            else:    
                activation = sigmoid(z)
            activations.append(activation)

        # backward pass
        biasDelta = [np.zeros(b.shape) for b in self.biases]
        weightsDelta = [np.zeros(w.shape) for w in self.weights]

        delta = activations[-1]-bathTrainingOutput 
        biasDelta[-1] = delta
        weightsDelta[-1] = np.dot(delta, activations[-2].transpose())

        for i in range(2, self.numberOfLayers):
            delta = np.dot(self.weights[-i+1].transpose(), delta) * sigmoidDerivative(zValueArray[-i])
            biasDelta[-i] = delta
            weightsDelta[-i] = np.dot(delta, activations[-i-1].transpose())
        return (biasDelta, weightsDelta)

def sigmoid(i):
    return 1.0/(1.0+np.exp(-i))

def sigmoidDerivative(i):
    return sigmoid(i)*(1-sigmoid(i))

def softmax(i):
    return np.exp(i)/np.sum(np.exp(i), axis = 0)        

net = NeuralNetwork([784, 30, 10])
net.train(trainingData,30,10,0.3)
net.test(testImages)
net.outputTestPredictions()