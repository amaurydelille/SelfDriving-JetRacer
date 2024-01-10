import random
import math

INPUTS = 720
HIDDEN = 20
OUTPUTS = 4 
LEARNING_RATE = 0.2
EPOCHS = 200
SAMPLES = 300

"""
The issue here is to know what our neural network should return :
- discrete values => distinct actions (turn left, right, keep forward, stop).
- continuous value => steering angle (127.0, 23.0, etc...).
""" 

class NeuralNetwork():
    def __init__(self, inputs = INPUTS, hidden = HIDDEN, outputs = OUTPUTS, samples = SAMPLES ) -> None:
        """
        Constructor of the NeuralNetwork class, siple attributes and properties.
        """
        self.inputs = inputs
        self.hidden = hidden
        self.outputs = outputs
        self.weights_ih = [[0 for _ in range(self.inputs)] for _ in range(self.hidden)]
        self.weights_ho = [[0 for _ in range(self.hidden)] for _ in range(self.outputs)]
        self.bias_ih = [0] * self.hidden
        self.bias_ho = [0] * self.outputs 
        self.dW2 = [[0 for _ in range(self.hidden)] for _ in range(self.outputs)]
        self.dW1 = [[0 for _ in range(self.inputs)] for _ in range(self.hidden)]
        self.dB1 = [0] * self.hidden
        self.dB2 = [0] * self.outputs
        self.dZ1 = [0] * self.hidden
        self.dZ2 = [0] * self.outputs 
        self.learning_rate = LEARNING_RATE
        self.epochs = EPOCHS
        self.samples = samples

    def xavier_weights(self, shape) -> float:
        """
        Xavier weights function is a method which assigns precise floating value to weights and biases,
        instead of assigning random values. 
        """
        fan_in, fan_out = shape[0], shape[1]
        limit = math.sqrt(6 / (fan_in + fan_out))
        return [[random.uniform(-limit, limit) for _ in range(shape[1])] for _ in range(shape[0])]

    def reLU(self, x):
        return max(x, 0)
    
    def reLU_derivative(self, x):
        if x > 0:
            return 1
        else:
            return 0
    
    def initialize_network(self) -> None:
        self.weights_ih = self.xavier_weights((self.hidden, self.inputs))
        self.weights_ho = self.xavier_weights((self.outputs, self.hidden))
        self.bias_ho = random.randint(-2, 2)
        self.bias_ih = random.randint(-2, 2)

    def forward_propagation(self) -> None:
        for i in range(self.hidden):
            for j in range(self.inputs):
                self.hidden[i] = self.weights_ih[i][j] * self.inputs[j]
            self.hidden[i] = self.reLU(self.hidden[i] + self.bias_ih[i])

        for i in range(self.outputs):
            for j in range(self.hidden):
                self.outputs[i] = self.weights_ho[i][j] * self.hidden[j]
            self.outputs[i] = self.reLU(self.outputs[i] + self.bias_ho[i])

    def backward_propagation(self, target: float, y_pred: list) -> None:
        one_hot = [0] * self.outputs
        one_hot[y_pred[target]] = 1
        
        for i in range(self.outputs):
            self.dZ2[i] = self.outputs[i] - one_hot[i]

        for i in range(self.outputs):
            for j in range(self.hidden):
                self.dW2[i][j] = self.dZ2 * self.hidden[j] * 1/self.samples
            self.dB2[i] = self.dZ2[i] * 1/self.samples

        for i in range(self.hidden):
            for j in range(self.outputs):
                self.dZ1[i] += self.weights_ho[j][i] * self.dZ2[j]
            self.dZ1[i] *= self.reLU_derivative(self.hidden[i])

        for i in range(self.hidden):
            for j in range(self.inputs):
                self.dW1[i][j] = self.dZ1[i] * self.inputs[j]
            self.dB1[i] = self.dZ1[i]

    def update_parameters(self):
        for i in range(self.hidden):
            for j in range(self.inputs):
                self.weights_ih[i][j] -= self.learning_rate * self.dW1[i][j]
            self.bias_ih[i] -= self.learning_rate * self.dB2[i]

        for i in range(self.outputs):
            for j in range(self.hidden):
                self.weights_ho[i][j] -= self.learning_rate * self.dW2[i][j]
            self.bias_ho[i] -= self.learning_rate * self.dB2[i]

    def gradient_descent(self, y, trainset):
        for epoch in range(self.epochs):
            for sample in range(self.samples):
                self.inupts = trainset[sample]

                self.forward_propagation()
                self.backward_propagation(y, sample)
                self.update_parameters()
            if epoch % 10 == 0:
                print(epoch)