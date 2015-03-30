from Classifier import Classifier
import cv2
import sys, traceback

class SVMClassifier(Classifier):

	def __init__(self):
		self.classifier = cv2.SVM()
		self.setErrorCount(0)
		self.setEvaluationsCount(0)

	def save(self, outputFile):
		try:
			print("Writing Classifier to file %s" % outputFile)
			self.classifier.save(outputFile)

		except Exception, Argument:
			print "Exception happened: ", Argument 
			traceback.print_stack()

	def load (self,inputFile):
		try:
			self.classifier.load(inputFile)

		except Exception, Argument:
			print "Exception happened: ", Argument
			traceback.print_stack()


	def train(self):

	        try:
	            return self.classifier.train(self.trainingData, self.trainingLabels)
	        except Exception, Argument:
	            print "Exception happened: ", Argument
	            traceback.print_stack()

	def predict(self, testData):
		try:
			response = self.classifier.predict(testData)
			return response
		except Exception, Argument:
			print "Exception happened: ", Argument

	def calculateScore(self, testData,returnDFVal=True):
		try:
			response = self.classifier.predict(testData,returnDFVal)
			return response
		except Exception, Argument:
			print "Exception happened: ", Argument