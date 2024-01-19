import random
import math

INPUTS = 720
HIDDEN = 20
OUTPUTS = 1
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
        Constructor of the NeuralNetwork class, simple attributes and properties.
        """
        self.inputs = [0] * inputs
        self.hidden = [0] * hidden
        self.outputs = [0] * outputs
        self.weights_ih = [[0 for _ in range(inputs)] for _ in range(hidden)]
        self.weights_ho = [[0 for _ in range(hidden)] for _ in range(outputs)]
        self.bias_ih = [0] * hidden
        self.bias_ho = [0] * outputs 
        self.dW2 = [[0 for _ in range(hidden)] for _ in range(outputs)]
        self.dW1 = [[0 for _ in range(inputs)] for _ in range(hidden)]
        self.dB1 = [0] * hidden
        self.dB2 = [0] * outputs
        self.dZ1 = [0] * hidden
        self.dZ2 = [0] * outputs 
        self.learning_rate = LEARNING_RATE
        self.epochs = EPOCHS
        self.samples = samples
        self.target = []
        self.accuracy = 0

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
    
    def softmax(self):
        max_val = max(self.outputs)
        s = 0.0

        for i in range(len(self.outputs)):
            self.outputs[i] = math.exp(self.outputs[i] - max_val)
            s += self.outputs[i]

        for i in range(len(self.outputs)):
            self.outputs[i] /= s 
    
    def initialize_network(self) -> None:
        self.weights_ih = self.xavier_weights((self.hidden, self.inputs))
        self.weights_ho = self.xavier_weights((self.outputs, self.hidden))
        self.bias_ho = random.randint(-2, 2)
        self.bias_ih = random.randint(-2, 2)

    def forward_propagation(self) -> None:
        for i in range(len(self.hidden)):
            self.hidden[i] = 0
            for j in range(len(self.inputs)):
                self.hidden[i] += self.weights_ih[i][j] * self.inputs[j]
            self.hidden[i] = self.reLU(self.hidden[i] + self.bias_ih[i])

        for i in range(len(self.outputs)):
            self.outputs[i] = 0
            for j in range(len(self.hidden)):
                self.outputs[i] += self.weights_ho[i][j] * self.hidden[j]
            self.outputs[i] = self.outputs[i] + self.bias_ho[i]

    def backward_propagation(self, target, sample) -> None:
        for i in range(len(self.outputs)):
            for j in range(len(self.hidden)):
                self.dW2[i][j] = self.dZ2[i] * self.hidden[j] * 1/self.samples
            self.dB2[i] = self.dZ2[i] * 1/self.samples

        for i in range(len(self.hidden)):
            for j in range(len(self.outputs)):
                self.dZ1[i] += self.weights_ho[j][i] * self.dZ2[j]
            self.dZ1[i] *= self.reLU_derivative(self.hidden[i])

        for i in range(len(self.hidden)):
            for j in range(len(self.inputs)):
                self.dW1[i][j] = self.dZ1[i] * self.inputs[j]
            self.dB1[i] = self.dZ1[i]

    def update_parameters(self):
        for i in range(len(self.hidden)):
            for j in range(len(self.inputs)):
                self.weights_ih[i][j] -= self.learning_rate * self.dW1[i][j]
            self.bias_ih[i] -= self.learning_rate * self.dB1[i]

        for i in range(len(self.outputs)):
            for j in range(len(self.hidden)):
                self.weights_ho[i][j] -= self.learning_rate * self.dW2[i][j]
            self.bias_ho[i] -= self.learning_rate * self.dB2[i]

    def get_prediction(self):
        return self.outputs.index(max(self.outputs))

    def gradient_descent(self, target, trainset):
        for epoch in range(self.epochs):
            self.accuracy = 0
            for sample in range(self.samples):
                self.inupts = trainset[sample]

                self.forward_propagation()

                loss = sum((target[sample] - self.outputs[0]) ** 2) / 2.0
                total_loss += loss

                if self.get_prediction() == self.target[sample]:
                    self.accuracy += 1

                if epoch % 10 == 0 and sample == self.samples - 1:
                    print(f"Epoch {epoch}, Loss: {total_loss / self.samples}, Accuracy: {self.accuracy / self.samples}")

                self.backward_propagation(target, sample)
                self.update_parameters()