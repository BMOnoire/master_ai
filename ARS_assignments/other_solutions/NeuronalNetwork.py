
import numpy as np
import time
import matplotlib.pyplot as plt


class Neural_Network(object):

  def __init__(self, inputLayer, hiddenLayer, outputLayer):
    self.inputSize = inputLayer
    self.hiddenSize = hiddenLayer
    self.outputSize = outputLayer
    self.weights1 = np.random.randn(self.inputSize+1, self.hiddenSize)  #Random weights at the beginning (One more for the bias)
    self.weights2 = np.random.randn(self.hiddenSize+1, self.outputSize) #Random weights at the beginning (One more for the bias)
    self.bias1 = 1
    self.bias2 = 1
    self.extraoutput_backprop = 1

  #Forward propagation
  def forwardprop(self, x):
    x = np.concatenate((np.array([[self.bias1]]*len(x)), x), axis=1) #Adding the first bias (input)
    self.hiddenLayer = self.sigmoid(np.dot(x, self.weights1))        #HiddenLayerActivation => Input x Weights1
    self.hiddenLayer = np.concatenate((np.array([[self.bias2]]*len(self.hiddenLayer)), self.hiddenLayer), axis=1) #Adding the second bias (Hidden layer)
    output = self.sigmoid(np.dot(self.hiddenLayer, self.weights2))   #Output => HiddenLayerActivation x Weights2
    return output

  #Backward propagation
  def backprop(self, x, y, output, l_rate):
    output_error = y - output #Error in output
    output_delta = output_error*self.sigmoid(output, True)   #Derivative of sigmoid over output error
    hiddenLayer_error = np.dot(output_delta, self.weights2.T)#Error in hidden layer
    hiddenLayer_delta = hiddenLayer_error*self.sigmoid(self.hiddenLayer, True)      #Derivative of sigmoid over hidden layer error
    x = np.concatenate((np.array([[self.extraoutput_backprop]]*len(x)), x), axis=1) #Adding a extra term in the input (1) for backpropagating the bias
    hiddenLayer_delta = np.array(hiddenLayer_delta)[:,1:]    #Removing the bias in the hidden layer for backprop
    var_weights1 = np.dot(x.T, hiddenLayer_delta)            #Variation of the first weights
    var_weights2 = np.dot(self.hiddenLayer.T, output_delta)  #Variation of the second weights
    self.weights1 += l_rate*var_weights1
    self.weights2 += l_rate*var_weights2

  #Activation function - second parameter for derivate
  def sigmoid(self, x, derivate = False):
    if derivate == True:
        return x*(1-x)
    return 1/(1+np.exp(-x))

  #Training weights
  def train(self, x, y, n_epochs, l_rate=1):
    errors = []
    init_time=time.time()
    print("Training with",n_epochs, "epochs...")
    for i in range(0, n_epochs):
      output = self.forwardprop(x)
      self.backprop(x, y, output, l_rate)
      errors.append([np.square(np.subtract(x,output)).mean(),i])
    print("Training completed in",str(round((time.time()-init_time),3)),"seconds")
    errors = np.array(errors)
    plt.plot(errors[:,1],errors[:,0])
    plt.xlabel('Mean square error')
    plt.ylabel('Number epochs')
    plt.show()

  #Returns a prediction given an output
  def predict(self, x, showActivationAndWeights=False):    
    print ("\n>>>Input: \n", str(x))
    x = np.append(self.bias1, x) #Adding the first bias
    self.hiddenLayer = self.sigmoid(np.dot(x, self.weights1))      #Hidden layer activation => Input x Weights1
    self.hiddenLayer = np.append(self.bias2, self.hiddenLayer)     #Adding the second bias
    output = self.sigmoid(np.dot(self.hiddenLayer, self.weights2)) #Output => HiddenLayerActivation x Weights2
    if showActivationAndWeights: 
      print("\n\n------------------------------------------------------------------")
      print("\n**Weights 1: \n",self.weights1)
      print("\n**Weights 2: \n",self.weights2)
      print("\n**Activation Hidden Layer: \n",self.hiddenLayer)
      x = np.array(x)[1:] #Removing the bias
      print("\n**Mean square error: ", np.square(np.subtract(x,output)).mean())
      print("\n------------------------------------------------------------------\n")
    print (">>>Output: \n", str(output))

    return output

  #Saves the model in a txt file 
  def saveModel(self, name):
  	np.savetxt(("models\\"+name+"-w1.txt"), self.weights1, fmt="%s")
  	np.savetxt(("models\\"+name+"-w2.txt"), self.weights2, fmt="%s")

  #Load a model from a txt file
  def loadModel(self, name):
  	try:
	  	w1 = np.loadtxt(("models\\"+name+"-w1.txt"))
	  	w2 = np.loadtxt(("models\\"+name+"-w2.txt"))
  	except:
  		print("The indicated model doesn't exist!")
  		time.sleep(2)
  		exit(1)
  	self.weights1 = w1
  	self.weights2 = w2


# # # # # # # # # # #
#                   #
#  Testing the NN   #
#                   #
# # # # # # # # # # #

#Creation dataset
dataset = np.array([[1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0],[0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1]])
#dataset = np.eye(8)
labels = np.array([[1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0],[0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1]])

#Split dataset
#train_data = np.split(dataset, [7])[0] #The first seven
#train_labels = np.split(labels, [7])[0] #The first seven
train_labels = labels #All
train_data = dataset #All
eval_data = np.split(dataset, [7])[1]  #The last one

#Implementation
NN = Neural_Network(inputLayer=8, hiddenLayer=3, outputLayer=8)

#Training
NN.train(train_data, train_labels, 1000, 0.01)

#Saving/Loading model
#NN.saveModel("model2")
#NN.loadModel("model1")


#Predictions - Testing
output = NN.predict(eval_data, True)
print("Greater: ",np.where(output==np.sort(output)[-1])[0][0])

#Analysis intern mechanism
# output = NN.predict([0,0,0,0,0,0,0,0], True)
# output = NN.predict([1,1,1,1,1,1,1,1], True)
# output = NN.predict([0,1,0,1,0,1,0,1], True)
# output = NN.predict([1,0,1,0,1,0,1,0], True)
# output = NN.predict([0,0,0,0,1,1,1,1], True)
# output = NN.predict([1,1,1,1,0,0,0,0], True)