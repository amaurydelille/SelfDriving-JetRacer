import random
INPUTS = 360
HIDDEN = 20
OUTPUTS = 4 
LEARNING_RATE = 0.2
EPOCHS = 200

"""
The issue here is to know what our neural network should return :
- discrete values => distinct actions (turn left, right, keep forward, stop).
- continuous value => steering angle (127.0, 23.0, etc...).
""" 

class NeuralNetwork():
    def __init__(self, inputs = INPUTS, hidden = HIDDEN, outputs = OUTPUTS ) -> None:
        """
        Constructor of the NeuralNetwork class, siple attributes and properties.
        """
        self.inputs = inputs
        self.hidden = hidden
        self.outputs = outputs
        self.weights_ih = [[0 for _ in range(self.inputs)] for _ in range(self.hidden)]
        self.weights_ho = [[0 for _ in range(self.hidden)] for _ in range(self.outputs)]
        self.bias_ih = [0 for _ in range(self.hidden)]
        self.bias_ho = [0 for _ in range(self.outputs)]
        self.learning_rate = LEARNING_RATE
        self.epochs = EPOCHS

    def xavier_weights() -> float:
        """
        Xavier weights function is a method which assigns precise floating value to weights and biases,
        instead of assigning random values. 
        """