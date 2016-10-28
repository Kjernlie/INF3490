"""
Feel free to use numpy for matrix multiplication and
	other neat features.
	You can write some helper functions to
	place some calculations outside the other functions
	if you like to.
	This pre-code is a nice starting point, but you can
	change it to fit your needs.
"""

import numpy as np

class mlp:
	def __init__(self, sizes):
		self.beta = 1
		self.eta = 0.1
		self.momentum = 0.0

		self.num_layers = len(sizes)
		self.sizes = sizes
		self.biases = [np.random.randn(y,1) for y in sizes[1:]]
		self.weights = [np.random.randn(y,x) for x,y in
				zip(sizes[:-1], sizes[1:])]

	def earlystopping(self, inputs, targets, valid, validtargets):
		self.train(inputs, targets)

	def train(self, inputs, targets, iterations=1):
		#data = zip(inputs, targets)
		#n = len(inputs)
		n = 1
		for i in xrange(iterations):
			for j in xrange(n):
				outputs = self.forward(inputs[j,:])	

	def forward(self, x):

		# Invert the input function, to make dot product possible. Fix this...
		activation = np.zeros((len(x),1))
		for i in xrange(len(x)):
			activation[i,0] = x[i]
		activations = [activation] # list for storing activations, layer-by-layer
		zs = [] # list for storing z values, layer-by-layer
		for b, w in zip(self.biases, self.weights):
			z = self.beta*np.dot(w, activation) + b
			zs.append(z)
			activation = sigmoid(z)
			activations.append(activation)
	
	
		return activations


	def confusion(self, inputs, targets):
		print('To be implemented')



def sigmoid(z):
	"""The sigmoid function"""
	return 1.0/(1.0+np.exp(-z))

def sigmoid_derivative(z):
	"""The derivative of the sigmoid function"""
	return sigmoid(z)*(1-sigmoid(z))
