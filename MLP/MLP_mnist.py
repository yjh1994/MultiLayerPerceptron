import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import sklearn


data = []
with open("/Users/yuki/mnist_train.csv") as csvfile:
	readCSV = csv.reader(csvfile,delimiter=',')

	for row in readCSV:
		data.append(row)
#print(data[2,:])

new_data = np.array(data)
target = new_data[:,0].astype(int)
sample = new_data[:,1:]

normalize_sample = preprocessing.normalize(sample)
#print(normalize_sample[:,1:])
#print("sample",sample[1,:])
#print("target",target[1])


#data = list(reader)
#result = numpy.array(data).astype("float")
#print(result)


#train the MLP model
class MultiLayerPerceptron(object):
	
	#3 layers perceptron

	def __init__(self, features, hidden, output, rate, epoch, alpha, batch_size):
		self.n_features = features
		self.n_hidden = hidden
		self.n_output = output
		self.rate = rate
		self.iteration = 0
		self.epoch = epoch
		self.alpha = alpha
		self.weights = []
		self.delta_w1 = []
		self.delta_w2 = []
		self.error_list = []
		self.accuracy_list = []
		self.sample_num = 0
		self.batch_size = batch_size
		self.split_sample_list = [] #mini batch sample (x)
		self.split_target_list = []	#mini batch target (y)

	def fit(self, x, y):

		self.initialize_weights()
		self.sample_num = len(y)
		batch_num = int(self.sample_num/self.batch_size)
		epoch_count = 0
		self.iteration = self.epoch * batch_num

		for i in range(self.iteration):
			batch_count = i % batch_num
			#batch_num = int(self.sample_num/self.batch_size)
			#new_x = self.split_sample_list[i % batch_num]
			#new_y =	self.split_target_list[i % batch_num]
			new_x = x[batch_count*self.batch_size:(batch_count+1)*self.batch_size - 1,:]
			new_y = y[batch_count*self.batch_size:(batch_count+1)*self.batch_size - 1]
			encoding_y = self.one_hot_encoding(new_y)
			
			#print(i)
			mse = self.backpropagation(new_x, encoding_y)
			#accuracy = self.accuracy(new_x, new_y)
			#self.accuracy_list.append(accuracy)
			# if i % 100 == 0:
			# 	print(i)
			# 	print("mse:%f"% mse)
			# 	print("accuracy:%f"% accuracy)


			if i % batch_num == 0:
				print(epoch_count)
				accuracy = self.accuracy(x, y)
				self.accuracy_list.append(accuracy)
				print("mse:%f"% mse)
				print("accuracy:%f"% accuracy)
				epoch_count += 1



	def predict(self, x):
		
		i_value, h_input,h_output,o_input,o_output = self.feedforward(self.weights, x)
		return np.argmax(o_output)

	def predict2(self, x):

		i_value, h_input,h_output,o_input,o_output = self.feedforward(self.weights, x)
		return o_output

	def add_bias_unit(self, X, axis='column'):
		if axis == 'column':
			X_new = np.ones((X.shape[0], X.shape[1]+1))
			X_new[:, 1:] = X
		elif axis == 'row':
			X_new = np.ones((X.shape[0]+1, X.shape[1]))
			X_new[1:, :] = X
		else:
			raise AttributeError('`how` must be `column` or `row`')
		return X_new

	def activation(self, input):
		return 1.0/(1.0 + np.exp(-input))

	def feedforward(self, w, x):
		
		input_layer_value = self.add_bias_unit(x, axis='column')
		hidden_layer_input = np.dot(w[0].T,(input_layer_value).T)
		hidden_layer_output = self.add_bias_unit(self.activation(hidden_layer_input), axis='row')
		output_layer_input = w[1].T.dot(hidden_layer_output)
		output_layer_output = self.activation(output_layer_input)
		
		#print("hidden_layer_input",hidden_layer_input)
		#print("hidden_layer_output",hidden_layer_output)
		#print("output_layer_input",output_layer_input)
		#print("output_layer_output",output_layer_output)
		
		#self.unit_out.append(hidden_layer_output)
		#self.unit_out.append(output_layer_output)
		return input_layer_value, hidden_layer_input, hidden_layer_output, output_layer_input, output_layer_output
		
	def one_hot_encoding(self, label):

		label_binarizer = preprocessing.LabelBinarizer()
		label_binarizer.fit(range(max(label)+1))
		encode = label_binarizer.transform(label)
		
		return encode

	def initialize_weights(self):
		"""Initialize weights with small random numbers."""
		w1 = np.random.uniform(-1.0, 1.0,size=self.n_hidden*(self.n_features + 1))
		w1 = w1.reshape(self.n_features + 1, self.n_hidden) #5x3
		w2 = np.random.uniform(-1.0, 1.0,size=self.n_output*(self.n_hidden + 1))
		w2 = w2.reshape(self.n_hidden + 1, self.n_output) #3x3
		 
		self.delta_w1 = np.zeros((self.n_features + 1, self.n_hidden))
		self.delta_w2 = np.zeros((self.n_hidden + 1, self.n_output))
		
		self.weights.append(w1)
		self.weights.append(w2)
		
		return w1, w2

	def backpropagation(self, x, y):
		
		#step 1 : feedforward and get every units' output
		i_value, h_input,h_output,o_input,o_output = self.feedforward(self.weights,x)
		#print("h_input",h_input)

		#step 2:calculate square error
		mse = self.error_count(o_output,y)
		self.error_list.append(mse)
		#--------------------------------------------------------
		
		#step  3: calculate error
		#output error = o(1-o)(t-o)

		output_error = o_output - y.T
		output_sigmoid_deri = o_output * (1 - o_output)
		delda_output_layer = np.dot(h_output,(output_error * output_sigmoid_deri).T)
		#print(delda_output_layer)
		hidden_error = np.dot((output_error * output_sigmoid_deri).T,self.weights[1].T) #1x4
		hidden_sigmoid_deri = h_output[1:,:] * (1 - h_output[1:,:]) 
		delda_hidden_layer = np.dot(i_value.T,(hidden_error[:,1:] * hidden_sigmoid_deri.T))

		#print(delda_output_layer)
		#print(delda_hidden_layer)
		#step 4 : update weights
		self.delta_w1 = self.rate * delda_hidden_layer + self.alpha * self.delta_w1
		self.delta_w2 = self.rate * delda_output_layer + self.alpha * self.delta_w2
		self.weights[0] = self.weights[0] - self.delta_w1
		self.weights[1] = self.weights[1] - self.delta_w2

		return mse

	def error_count(self, output, y):

		error_sum_list = np.power((output - y.T),2)
		error_sum = (1/(self.sample_num + 1))* np.sum(error_sum_list)
		#print("error_sum: %f"%error_sum)
		return error_sum

	def show_error_figure(self):

		#print(self.error_list)
		plt.plot(self.error_list)
		plt.ylabel('MSE')
		plt.xlabel('iteration')
		plt.show()

	def show_accuracy_figure(self):
		#print(self.error_list)
		plt.plot(self.accuracy_list)
		plt.ylabel('accuracy')
		plt.xlabel('epoch')
		plt.show()

	def accuracy(self, x, y):

		#Type casting 
		x_list = list(x)
		y_list = list(y)

		correct = 0
		for num in range(len(x_list)):
			if np.equal(y_list[num] , self.predict(np.array([x_list[num]]))):
				correct += 1
		#print("accuracy: %f"% (correct/len(x_list)))
		return correct/len(x_list)


	def get_weights(self):

		print(self.weights[0])
		print(self.weights[1])
		return self.weights

	def get_mse(self):

		print("final mse: %f"%self.error_list[-1])
		return self.error_list[-1]

	def get_accuracy(self):

		print("final accuracy: %f"%self.accuracy_list[-1])
		return self.accuracy_list[-1]

#MultiLayerPerceptron(features,hidden layer num,output layer num,rate,epoch,momentun,batch_size)
mlp = MultiLayerPerceptron(784,100,10,0.01,200,0.9,200)
mlp.fit(normalize_sample,target)
mlp.get_weights()
mlp.show_error_figure()
mlp.show_accuracy_figure()


