import numpy as np
# Artificial Neural Network
class ANN(object):

	def __init__(self, layer_size_list):
		self.input_size = layer_size_list[0]
		self.hidden_output_layer = []
		self.cost_result = []
		self.accuracy_result = []
		for layer_index in range(1, len(layer_size_list)):
			self.hidden_output_layer.append( NNlayer( layer_size_list[layer_index - 1], layer_size_list[layer_index], self.sigmoid, self.de_sigmoid ) )

	def propagate_result(self, network_input, save_result = False):
		previous_output = [network_input]
		for layer in self.hidden_output_layer:
			previous_output = layer.CalculateOutput(previous_output,save_data = save_result)
		return previous_output

	def mini_batch_training(self, training_data, batch_size, learning_rate = 0.3,iterations = 10):  # batch_size should be in integer 
		total_num_training_data = len(training_data)
		total_epoch = total_num_training_data // batch_size
		np.random.shuffle(training_data)
		for i in range(total_epoch):
			index = batch_size * i
			success_rate = self.batch_SGD(training_data[index:index+batch_size],learning_rate,maxiter = iterations)
			if i%10 == 0:
				print("epoch:", i+1, "out of", total_epoch,"| Accuracy:", success_rate)

	def batch_SGD(self,training_data,learning_rate,maxiter = 10,gradient_threshold = 1.0E-8):
		batch_size = len(training_data)
		iterations = 0
		correct = 0.0
		total = 0.0
		costTot = 0.0
		while(iterations<=maxiter):
			for data in training_data:
				total += 1.0
				network_result = self.propagate_result(data[0],save_result = True)
				if np.argmax(network_result) == np.argmax(data[1]):
					correct += 1.0
				d_cost = network_result - [data[1]]
				costTot+=0.5 * np.sum( (d_cost)**2 )
				self.backpropagate_result(d_cost)	
			gradient = self.update_layers(batch_size, learning_rate)
			if gradient <= gradient_threshold:
				break
			iterations += 1
		self.cost_result.append( costTot/total )
		self.accuracy_result.append( correct/total )
		return correct/total

	def update_layers(self, batch_size, learning_rate):
		gradient_magnitude = 0
		for layer in self.hidden_output_layer:
			gradient_magnitude += layer.update_constants(learning_rate, batch_size)
		return gradient_magnitude

	def backpropagate_result(self, d_cost):
		final_derivative = d_cost
		for layer in reversed(self.hidden_output_layer):
			final_derivative = layer.backpropagate_layer(final_derivative)

	def sigmoid(self, x): return 1/(1+np.exp(-x))
	def de_sigmoid(self, x): return self.sigmoid(x) * ( 1 - self.sigmoid(x) )

class NNlayer(object):
	def __init__(self, previous_nodes, current_nodes, activating_function, derivative_function):
		self.weightArr = np.random.random((previous_nodes,current_nodes))*2-1
		self.biasArr = np.random.random((1,current_nodes))*2-1
		self.activating_function = activating_function # must be iterative
		self.derivative_function = derivative_function # must be iterative
		self.bias_G_sum = np.copy(self.biasArr) * 0
		self.weight_G_sum = np.copy(self.weightArr) * 0

	def CalculateOutput(self, previous_layer_output, save_data = False):
		pre_activating = np.dot(previous_layer_output, self.weightArr) + self.biasArr
		if save_data: 
			self.derivative_activation = self.derivative_function( pre_activating )
			self.previous_layer_output = np.array(previous_layer_output)
		return self.activating_function( pre_activating )

	def backpropagate_layer(self, next_layer):
		bias_G = next_layer * self.derivative_activation
		weight_G = np.dot( self.previous_layer_output.T, bias_G )
		weight_G = self.previous_layer_output.T.dot( bias_G )
		self.bias_G_sum += bias_G
		self.weight_G_sum += weight_G

		return np.dot( bias_G, self.weightArr.T )

	def update_constants(self, learning_rate, batch_size):
		self.weightArr -= learning_rate * self.weight_G_sum / batch_size
		self.biasArr -= learning_rate * self.bias_G_sum / batch_size
		gradient_magnitude = np.linalg.norm(self.bias_G_sum / batch_size) + np.linalg.norm(self.weight_G_sum / batch_size)
		self.bias_G_sum *= 0
		self.weight_G_sum *= 0
		return gradient_magnitude
