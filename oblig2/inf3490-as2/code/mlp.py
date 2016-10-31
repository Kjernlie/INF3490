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
from pandas_ml import ConfusionMatrix

class mlp:
	def __init__(self, sizes):
		"""Here, we initialize beta, eta and the biases and
		weights."""
		self.beta = 1
		self.eta = 0.1
		self.momentum = 0.0

		self.num_layers = len(sizes)
		self.sizes = sizes
		self.biases = [np.random.randn(y,1) for y in sizes[1:]]
		self.weights = [np.random.randn(y,x) for x,y in
				zip(sizes[:-1], sizes[1:])]


	def earlystopping(self, inputs, targets, valid, validtargets, iterations):
		"""Here, we start the algorithm, and compute the squeared error
		for every 20th iteration, using the validation set. If the error
		starts increasing we stop the algoritm. """
		n = len(valid)
		squared_err = np.zeros((len(validtargets[0,:]),1))
		total_sum = []
		count = 0
		total_sum.append(200) # Inital value
		for i in xrange(iterations):
			self.train(inputs, targets) # train the network
			if i % 20 == 0:
				squared_err_arr = np.zeros(n) 
				for j in range(n):
					valid_targets = np.zeros((len(validtargets[0,:]),1))
					valid_targets[:,0] += validtargets[j]
					
					# find outputs using validation set
					output = self.forward(valid[j,:])
					# compute the squared error
					squared_err = (output[-1] - valid_targets)**2
					squared_err_arr[j] = sum(squared_err)
				count += 1	
				total_sum.append(sum(squared_err_arr))
				
				# break out of loop it error starts increasing 
				if total_sum[count] > total_sum[count-1]:
					break
							


	def train(self, inputs, targets):
		"""Here, we do the backward phase 
		of the algorithm, and updates the 
		weights and biases"""
		data = zip(inputs, targets)
		np.random.shuffle(data)
		for x, y in data:
			targets_arr = np.zeros((len(y),1))
			targets_arr[:,0] += y
			outputs = self.forward(x)
			
			# Generate deltas
			delta_out = (targets_arr-outputs[-1])*\
				outputs[-1]*(1-outputs[-1])
			delta_hidden = outputs[-2]*(1-outputs[-2])*\
				np.dot(self.weights[-1].transpose(),delta_out)
			
			# Update weights and biases
			self.weights[-1] += self.eta*np.dot(delta_out, outputs[-2].transpose())
			self.weights[-2] += self.eta*np.dot(delta_hidden, outputs[0].transpose())
			self.biases[-1] += self.eta*delta_out
			self.biases[-2] += self.eta*delta_hidden							

	def forward(self, x):
		"""Here, we do the forward phase of the algorithm, and
		find the activations in the hidden and output layers"""
		activation = np.zeros((len(x),1))
		activation[:,0] += x
		activations = [activation] # list for storing activations, layer-by-layer
		for b, w in zip(self.biases, self.weights):
			z = self.beta*np.dot(w, activation) + b
			activation = sigmoid(z)
			activations.append(activation)
	
	
		return activations

	

	def confusion(self, inputs, targets):
		"""Here, we create the confusion matrix, and print 
		the statistics of the result"""
		n = len(inputs)
		output_list = []
		target_list = []
		for i in xrange(n):
			output = self.forward(inputs[i])
			output_list.append(np.argmax(output[-1])+1)
			target_list.append(np.argmax(targets[i])+1)
		confusion_matrix = ConfusionMatrix(target_list,output_list)
		confusion_matrix.print_stats()
		confusion_matrix.plot(backend='seaborn')



def sigmoid(z):
	"""The sigmoid function"""
	return 1.0/(1.0+np.exp(-z))

