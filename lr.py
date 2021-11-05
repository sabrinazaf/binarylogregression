import numpy as np
import sys
import matplotlib.pyplot as plt

class lr:
	def __init__(self, trainIn, validIn, testIn, dictionaryInput, trainOut, testOut, metrics, epochs):
		self.trainIn = trainIn
		self.validIn = validIn
		self.testIn = testIn
		self.dictionaryIn = dictionaryInput
		self.trainOut = trainOut
		self.testOut = testOut
		self.metrics = metrics
		self.epochs = epochs

	def dotProduct(self, theta, featureVector): # Calculates sparse dot product
		dot = 0
		for index in featureVector:
			weight = theta[int(index)]
			dot += weight
		return dot

	def loss(self, parameters, features, labels):
		loss = 0
		for feature, label in zip(features, labels):
			dot = self.dotProduct(parameters, feature)
			loss += (-label * dot + np.log(1 + np.exp(dot)))
		return loss

	def totalFeats(self, inputData):
		input = open(inputData, "r")
		input = input.readlines()
		length = 0
		for entry in input:
			length += 1
		return length

	def readData(self, inputData):
		input = open(inputData, "r")
		input = input.readlines()
		length = self.totalFeats(self.dictionaryIn)
		labels = []
		features = []
		for line in input:
			splitLine = line.split('\t')
			label = int(splitLine[0])
			labels.append(label)
			feature = []
			for index in splitLine[1:]:
				item = int(index.split(':')[0])
				feature.append(item)
			feature.append(length)
			features.append(feature)
		return labels, features

	def optimize(self, labels, features, epochs, learningRate):
		length = self.totalFeats(self.dictionaryIn)
		parameters = np.zeros(length + 1)
		numberSamps = len(features)
		for epoch in range(epochs):
			for label, feature in zip(labels, features):
				indicator = np.zeros(length + 1)
				indicator[feature] = 1.0
				dot = self.dotProduct(parameters, feature)
				parameters += learningRate*(indicator/numberSamps)*(label - (np.exp(dot) / (1 + np.exp(dot))))
		return parameters

	def predict(self, parameters, features, labels, outputFile):
		output = open(outputFile, "w")
		MC = 0
		total = 0
		predictions = []
		for ii in range(len(features)):
			feature = features[ii]
			label = labels[ii]
			dot = self.dotProduct(parameters, feature)
			prob = np.exp(dot) / (1 + np.exp(dot))
			predict = 0
			if prob >= .5: 
				predict = 1
			line = str(predict) + "\n"
			output.write(line)
			predictions.append(predict)
		output.close()
		return predictions

	def plots(self, trainLabs, trainFeats, validLabs, validFeats, learningRate):
		length = self.totalFeats(self.dictionaryIn)
		parameters = np.zeros(length + 1)
		trainLoss = []
		validLoss = []
		length = self.totalFeats(self.dictionaryIn)
		for epoch in range(self.epochs):
			for label, feature in zip(trainLabs, trainFeats):
				indicator = np.zeros(length + 1)
				indicator[feature] = 1.0
				dot = self.dotProduct(parameters, feature)
				parameters += learningRate*(indicator/length)*(label - (np.exp(dot) / (1 + np.exp(dot))))
			trainLoss.append(self.loss(parameters, trainFeats, trainLabs)/length)
			validLoss.append(self.loss(parameters, validFeats, validLabs)/length)
		x = np.linspace(0, self.epochs - 1, self.epochs)
		plt.xlabel("Epochs")
		plt.ylabel("Negative Log-Likelihood")
		plt.plot(x, trainLoss, "g", label = "Training Loss")
		plt.plot(x, validLoss, "b", label = "Validation Loss")
		plt.legend(loc = "upper right")
		plt.show()

	def getError(self, labels, predictions):
		misclassification = 0
		total = 0
		for label, prediction in zip(labels, predictions):
			total += 1
			if label != prediction:
				misclassification += 1
		return misclassification/total
		

	def writeMetrics(self, trainMCR, testMCR, outputFile):
		trainStr = "error(train): " + '{:1.6f}'.format(trainMCR) + "\n"
		testStr = "error(test): " + '{:1.6f}'.format(testMCR)
		output = open(outputFile, "w")
		output.write(trainStr)
		output.write(testStr)
		output.close()

	def run(self):
		trainLabs, trainFeats = self.readData(self.trainIn)
		validLabs, validFeats = self.readData(self.validIn)
		testLabs, testFeats = self.readData(self.testIn)
		length = self.totalFeats(self.dictionaryIn)
		parameters = self.optimize(trainLabs, trainFeats, self.epochs, .1)
		trainPreds = self.predict(parameters, trainFeats, trainLabs, self.trainOut)
		trainMCR = self.getError(trainLabs, trainPreds)
		testPreds = self.predict(parameters, testFeats, testLabs, self.testOut)
		testMCR = self.getError(testLabs, testPreds)
		print(trainMCR);print(testMCR)
		self.writeMetrics(trainMCR, testMCR, self.metrics)

		self.plots(trainLabs, trainFeats, validLabs, validFeats, .1)

if __name__ == '__main__': 
    trainIn = sys.argv[1]
    validIn = sys.argv[2] 
    testIn = sys.argv[3]
    dictionaryInput = sys.argv[4]
    trainOut = sys.argv[5]
    testOut = sys.argv[6]
    metrics = sys.argv[7]
    epochs = int(sys.argv[8])
    
    model = lr(trainIn, validIn, testIn, dictionaryInput, trainOut, testOut, metrics, epochs)
    model.run()

