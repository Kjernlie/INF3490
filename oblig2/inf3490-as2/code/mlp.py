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

		print np.shape(targets[1,:]) 

		nabla_b = [np.zeros(b.shape) for b in self.biases]
		nabla_w = [np.zeros(w.shape) for w in self.weights]		
		delta_nabla_b = [np.zeros(b.shape) for b in self.biases]
		delta_nabla_w = [np.zeros(w.shape) for w in self.weights]
		n = 1
		#print inputs
		#test = zip(inputs, targets)
		#print type(inputs)
		#for x, y in test:
		#	print "here", x
		for i in xrange(iterations):
			data = zip(inputs, targets)
			np.random.shuffle(data)
			#for x, y in data:
			#input_vec, target_vec = map(list, zip(*data))
			#print type(input_vec)
			#for j in xrange(n):
			for x, y in data:
				targets_array = np.zeros((len(y),1))
				targets_array[:,0] += y
				outputs = self.forward(x)	
				
				delta = (targets_array-outputs[-1])*\
					outputs[-1]*(1-outputs[-1])
				delta_nabla_b[-1] = delta
				delta_nabla_w[-1] = np.dot(delta, outputs[-2].transpose())
				
				delta = outputs[-2]*(1-outputs[-2])*\
					np.dot(self.weights[-1].transpose(),delta)
				delta_nabla_b[-2] = delta
				delta_nabla_w[-2] = np.dot(delta, outputs[0].transpose())				
				nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
				nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
				
				print self.weights[1][0], self.biases[0][0]
				self.weights = [w-(self.eta/n)*nw  for w,nw in
						zip(self.weights, nabla_w)]
				self.biases = [b-(self.eta/n)*nb for b,nb in
						zip(self.biases, nabla_b)]					
				print self.weights[1][0], self.biases[0][0]
	def forward(self, x):

		# Invert the input function, to make dot product possible. Fix this...
		activation = np.zeros((len(x),1))
		activation[:,0] += x
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
