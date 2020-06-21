import numpy as np
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()
#np.random.seed(0)

X =		[ [1, 2, 3, 2.5],
		  [2.0, 5.0, -1.0, 2.0],
		  [-1.5, 2.7, 3.3, -0.8] ]


X, y = spiral_data(100, 3)

'''inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]
output = []

for i in inputs:
	if i > 0:
		output.append(i)
	elif i <=0:
		output.append(0)

print(output)'''







class Layer_Dense:
	def __init__(self, n_inputs, n_neurons):
		self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
		self.biases = np.zeros((1, n_neurons))
	def forward(self, inputs):
		self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
	def forward(self, inputs):
		self.input = np.maximum(0, inputs)

layer1 = Layer_Dense(2, 5)
activation1 = Activation_ReLU()

layer1.forward(X)
#print(layer1.output)
activation1.forward(layer1.output)
print(activation1.output)










'''weights = [[0.2, 0.8, -0.5, 1.0],
		   [0.5, -0.91, 0.26, -0.5],
		   [-0.26, -0.27, 0.17, 0.87]]
			

biases = [2, 3, 0.5]



weights2 = [[0.1, -1.14, 0.5],
		   [-0.5, 0.12, -0.33],
		   [-0.44, 0.73, -0.13]]
			

biases2 = [-1, 2, -0.5]



layer1_outputs = np.dot(inputs, np.array(weights).T) + biases

layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2

print(layer2_outputs)


















#some_value = -0.5
#weight = 0.7
#biass = 0.7


#print(some_value*weight)
#print(some_value+biass)

#bias1 = 2
#bias2 = 3
#bias3 = 0.5

layer_outputs = []
for neuron_weights, neuron_bias in zip(weights, bias):
	neuron_output = 0
	for n_input, weight in zip(inputs, neuron_weights):
		neuron_output+=n_input*weight
	neuron_output += neuron_bias
	layer_outputs.append(neuron_output)
print(layer_outputs)
#output = [inputs[0]*weights1[0] + inputs[1]*weights1[1] + inputs[2]*weights1[2] + inputs[3]*weights1[3] + bias1,
		  #inputs[0]*weights2[0] + inputs[1]*weights2[1] + inputs[2]*weights2[2] + inputs[3]*weights2[3] + bias2,
		 # inputs[0]*weights3[0] + inputs[1]*weights3[1] + inputs[2]*weights3[2] + inputs[3]*weights3[3] + bias3]
#print(output)'''