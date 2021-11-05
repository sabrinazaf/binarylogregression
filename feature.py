import numpy as np
import sys

class model1:
	def __init__(self, trainInput, validInput, testInput, dictionaryInput, trainOutput, validOutput, testOutput):
		self.trainIn = trainInput
		self.validIn = validInput
		self.testIn = testInput
		self.dictionaryIn = dictionaryInput
		self.trainOut = trainOutput
		self.validOut = validOutput
		self.testOut = testOutput
		self.wordDict = None

	def read(self, input):
		input = open(input, "r")
		input = input.readlines()
		labels = []
		wordBags = []
		for line in input:
			labels.append(line.split('\t')[0])
			wordBags.append(line.split("\t")[1].split(" "))
		return labels, wordBags

	def buildDict(self, input):
		if self.wordDict != None: return self.wordDict
		input = open(input, "r")
		input = input.readlines()
		result = dict()
		for entry in input:
			word, index = entry.split(" ")
			result[word] = index[:-1]
		self.wordDict = result
		return self.wordDict

	def getFeatures(self, wordDict, wordBags):
		features = []
		for wordBag in wordBags:
			lineFeatures = []
			for word in wordBag:
				if word in wordDict.keys() and wordDict[word] not in lineFeatures:
					index = wordDict[word]
					lineFeatures.append(index)
			features.append(lineFeatures)
		return features

	def output(self, input, output):
		wordDict = self.buildDict(self.dictionaryIn)
		labels, wordBags = self.read(input)
		features = self.getFeatures(wordDict, wordBags)
		output = open(output, "w")
		for label, feature in zip(labels, features):
			current = label
			for element in feature:
				current += "\t" + element + ":1"
			current += "\n"
			output.write(current)

	def run(self):
		self.output(self.trainIn, self.trainOut)
		self.output(self.validIn, self.validOut)
		self.output(self.testIn, self.testOut)

class model2:
	def __init__(self, trainInput, validInput, testInput, dictionaryInput, trainOutput, validOutput, testOutput):
		self.trainIn = trainInput
		self.validIn = validInput
		self.testIn = testInput
		self.dictionaryIn = dictionaryInput
		self.trainOut = trainOutput
		self.validOut = validOutput
		self.testOut = testOutput
		self.t = 4
		self.wordDict = None

	def read(self, input):
		input = open(input, "r")
		input = input.readlines()
		labels = []
		wordBags = []
		for line in input:
			labels.append(line.split('\t')[0])
			wordBags.append(line.split("\t")[1].split(" "))
		return labels, wordBags

	def buildDict(self, input):
		if self.wordDict != None: return self.wordDict
		input = open(input, "r")
		input = input.readlines()
		result = dict()
		for entry in input:
			word, index = entry.split(" ")
			result[word] = index[:-1]
		self.wordDict = result
		return self.wordDict

	def getFeatures(self, wordDict, wordBags):
		features = []
		for wordBag in wordBags:
			lineFeatures = []
			featCount = dict()
			for word in wordBag:
				if word in wordDict.keys():
					if word not in featCount:
						featCount[word] = 1
					elif word in featCount:
						featCount[word] += 1
			for word in featCount:
				if featCount[word] < self.t:
					index = wordDict[word]
					lineFeatures.append(index)
			features.append(lineFeatures)
		return features

	def output(self, input, output):
		wordDict = self.buildDict(self.dictionaryIn)
		labels, wordBags = self.read(input)
		features = self.getFeatures(wordDict, wordBags)
		output = open(output, "w")
		for label, feature in zip(labels, features):
			current = label
			for element in feature:
				current += "\t" + element + ":1"
			current += "\n"
			output.write(current)

	def run(self):
		self.output(self.trainIn, self.trainOut)
		self.output(self.validIn, self.validOut)
		self.output(self.testIn, self.testOut)



if __name__ == '__main__': 
    trainInput = sys.argv[1]
    validInput = sys.argv[2]
    testInput = sys.argv[3]
    dictionaryInput = sys.argv[4]
    trainOutput = sys.argv[5]
    validOutput = sys.argv[6]
    testOutput = sys.argv[7]
    feature_flag = sys.argv[8]
    
    if feature_flag == "1":
    	model = model1(trainInput, validInput, testInput, dictionaryInput, trainOutput, validOutput, testOutput)
    else:
    	model = model2(trainInput, validInput, testInput, dictionaryInput, trainOutput, validOutput, testOutput)
    model.run()