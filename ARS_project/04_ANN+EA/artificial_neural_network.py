# GLAUCO
import numpy as np
from scipy.special import expit

X, Y = 0, 1

class ArtificialNeuralNetwork:

    def __init__(self, n_inputs, n_hidden, n_outputs):
        self.inputSize = n_inputs
        self.hiddenSize = n_hidden
        self.outputSize = n_outputs
        self.weights_0L = np.ones((n_inputs, n_hidden))
        self.weights_1L = np.ones((n_hidden, n_outputs))
        self.bias_0L = np.ones((1, n_hidden))
        self.bias_1L = np.ones((1, n_outputs))

    def initialize_random_weights(self):
        W0 = np.random.randn(self.inputSize, self.hiddenSize) #np.random.rand(self.inputSize, self.hiddenSize) #
        W1 = np.random.randn(self.hiddenSize, self.outputSize)
        return [W0, W1]

    def _random_weights(self, n, m):
        return np.random.randn(n, m)

    def _sigmoid(self, x):
        return expit(x)  # 1.0 / (1.0 + np.exp(-x))

    def forward_propagation(self, inputs):
        hidden_layer = np.dot(inputs, self.weights_0L) + self.bias_0L
        hidden_layer = self._sigmoid(hidden_layer[0])
        outputs = np.dot(hidden_layer, self.weights_1L) + self.bias_1L
        outputs = self._sigmoid(outputs[0])
        return outputs

    def mapping_output(self, output, output_range):  #  [0,1] -> [A,B]
        new_output = []
        for i in range(len(output)):
            new_value= np.interp(output[i], [0, 1], [output_range[i][0], output_range[i][1]])
            new_output.append(new_value)
        return new_output

    def mapping_output_velocity(self, output, limit):  #  [0,1] -> [-A,A]
        new_output = []
        for i in range(len(output)):
            new_value = np.interp(output[i], [0, 1], [-limit, limit])
            new_output.append(new_value)
        return new_output
