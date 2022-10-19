__author__ = '{Nianheng Wu}'

import numpy as np


class BaseLayer:
    """
    An abstract class for all kinds of layers
    """

    def __init__(self):
        pass

    def forward(self, input):
        return input

    def backward(self, input, gradients):
        # d(loss) / d(x) = gradients * ( d(layer) / d(x) )
        d_layer_d_x = np.eye(input.shape[1])
        d_loss_d_x = np.dot(gradients, d_layer_d_x)
        return d_loss_d_x

    def update(self, weights_grads, bias_grads):
        """
        Weights updating
        :param weights_grads: the gradients for the weights
        :param bias_grads: the gradients for the bias
        :return:
        """
        pass


class Dense(BaseLayer):
    def __init__(self, shape, lr=0.01):
        """
        A fully connected dense layer
        :param shape: the shape of this layer (feature_size, parameter_size)
        :param lr: learning rate
        """
        # Xavier initialization on ReLu using nets
        self.weights = np.random.normal(loc=0, scale=1, size=shape) * np.sqrt(2 / shape[1])
        self.bias = np.zeros(shape[1])
        self.lr = lr

    def forward(self, input):
        return np.dot(input, self.weights) + self.bias

    def backward(self, input, gradients):
        # d f / d x = (d f / d layer) * (d layer / d x)
        d_dense_d_input = np.dot(gradients, self.weights.T)
        weights_grads = np.dot(input.T, gradients)
        bias_grads = gradients.mean(axis=0) * input.shape[0]
        self.update(weights_grads, bias_grads)
        return d_dense_d_input

    def update(self, weights_grads, bias_grads):
        self.weights = self.weights - self.lr * weights_grads
        self.bias = self.bias - self.lr * bias_grads


class ReLU(BaseLayer):
    """
    ReLU layer
    """

    def __init__(self):
        pass

    def forward(self, input):
        logits = np.maximum(0, input)
        return logits

    def backward(self, input, gradients):
        return input * (input > 0)


class MultiLayerPerceptron:
    """
    A multilayer perceptron model. One could add dense layers and ReLU layers to it.
    """

    def __init__(self):
        self.layers = list()

    def add_layer(self, shape):
        """
        Add one dense layer
        :param shape: the shape of the dense layer
        :return: None
        """
        self.layers.append(Dense(shape))

    def add_activation_func(self):
        """
        Add one ReLU layer
        :return: None
        """
        self.layers.append(ReLU())

    def forward(self, input):
        intermediate_outputs = list()
        this_input = input
        if len(self.layers) > 0:
            for layer in self.layers:
                intermediate_outputs.append(layer.forward(this_input))
                this_input = intermediate_outputs[-1]
        else:
            raise ValueError("Must have at least one layer!")
        return intermediate_outputs

    def loss_n_grads(self, init_grads, y):
        """
        Calculating the loss and loss gradients.
        :param init_grads: the initial gradient passed from the last layer
        :param y: the golden labels
        :return: loss, loss_grads
        """
        # loss = -[golden] + log (SUM([predict n]))
        x_id = np.arange(len(init_grads))
        golden_grads = init_grads[x_id, y]

        golden_labels = np.zeros(init_grads.shape, dtype=init_grads.dtype)
        golden_labels[x_id, y] = 1
        loss = - golden_grads + np.log(np.sum(np.exp(init_grads), axis=-1))

        # keepdims = True very important! Takes forever to debug this.
        softmax = np.exp(init_grads) / (np.exp(init_grads).sum(keepdims=True, axis=-1))
        loss_grads = (softmax - golden_labels) / init_grads.shape[0]

        return loss, loss_grads

    def train(self, x, y):
        intermediate_outputs = self.forward(x)
        initial_grads = intermediate_outputs[-1]
        loss, loss_grads = self.loss_n_grads(initial_grads, y)

        intermediate_inputs = [x] + intermediate_outputs
        for layer_id in reversed(range(len(self.layers))):
            loss_grads = self.layers[layer_id].backward(intermediate_inputs[layer_id], loss_grads)

    def predict(self, x):
        return self.forward(x)[-1].argmax(axis=-1)

    def validate(self, x, y):
        return np.mean(self.forward(x)[-1].argmax(axis=-1) == y)
