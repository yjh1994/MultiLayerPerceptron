
# coding: utf-8

# In[3]:


import numpy as np
import sklearn
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import sklearn.preprocessing
import csv
import matplotlib.pyplot as plt


# In[4]:


iris_data = load_iris().data
iris_target = load_iris().target



# In[49]:


#train the MLP model
class MultiLayerPerceptron(object):
    
    #3 layers perceptron
    
    def __init__(self, features, hidden, output, rate, iteration, alpha):
        self.n_features = features
        self.n_hidden = hidden
        self.n_output = output
        self.rate = rate
        self.iteration = iteration
        self.alpha = alpha
        self.weights = []
        self.unit_out = []
        self.delta_w1 = []
        self.delta_w2 = []
        self.error_list = []
        self.accuracy_list = []
        self.sample_num = 0
    
    def fit(self, x, y):
        
        self.initialize_weights()
        self.sample_num = len(y)
        encoding_y = self.one_hot_encoding(y)
        for i in range(self.iteration):
            #print(i)
            self.backpropagation(x, encoding_y)
            self.accuracy_list.append(self.accuracy(x, y))
            if i % 2000 == 0:
                print(i)
                print("mse:%f"% self.backpropagation(x, encoding_y))
                print("accuracy:%f"% self.accuracy(x, y))

            if  self.accuracy(x, y) == 1:
                print(i)
                print("mse:%f"% self.backpropagation(x, encoding_y))
                print("accuracy:%f"% self.accuracy(x, y))
                break
            
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
        
        self.unit_out.append(hidden_layer_output)
        self.unit_out.append(output_layer_output)
        return input_layer_value, hidden_layer_input, hidden_layer_output, output_layer_input, output_layer_output
    
    def one_hot_encoding(self, label):

        label_binarizer = sklearn.preprocessing.LabelBinarizer()
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
        
        #step 2calculate square error
        self.error_list.append(self.error_count(o_output,y))
        #--------------------------------------------------------
        mse = self.error_count(o_output,y)
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
        error_sum_list = np.power(output - y.T,2)
        error_sum = (1/(self.sample_num + 1))* np.sum(error_sum_list)
        #print("error_sum: %f"%error_sum)
        return error_sum
    
    def show_error_figure(self):
        #print(self.error_list)
        plt.plot(self.error_list)
        plt.ylabel('MSE')
        plt.xlabel('epoch')
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

# In[50]:


#test
test_data = np.asarray([( 4.4,  3.2,  1.3,  0.2)])
test_target = np.asarray([0])

#features, hidden, output, rate, iteration, alpha

mlp = MultiLayerPerceptron(4,3,3,0.01,10000,0.9)
#w1,w2 = mlp.initialize_weights()
#mlp.feedforward(mlp.weights,test_data)

#mlp.backpropagation(iris_data,iris_target)
#mlp.predict(np.array([(6.5,  3. ,  5.2,  2. )]))

mlp.fit(iris_data,iris_target)
mlp.predict(np.array([( 4.4,  3.2,  1.3,  0.2)]))
mlp.get_weights()
mlp.get_accuracy()
mlp.get_mse()
# In[51]:


mlp.predict(np.array([(5.9,  3,  5.1,  1.8)]))


# In[52]:


mlp.show_error_figure()


# In[53]:


mlp.show_accuracy_figure()



