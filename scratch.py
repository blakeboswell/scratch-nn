import numpy as np

np.random.seed(1)


def init_layer(input_n, neuron_n):
    ''' initialize layer with n neurons
        expecting n input features
    '''
    w = np.random.normal(0, 1, [input_n, neuron_n]) 
    b = np.random.rand([input_n, neuron_n])
    return {'weight': w, 'bias': b}


def init_network(input_n, hidden_n, output_n):
    ''' build network structure
        arguments:
            input_n: number of inputs
            hidden_n: number of neurons in hidden layer
            output_n: number of outputs
    '''
    hidden = init_layer(input_n, hidden_n)
    output = init_layer(input_n, hidden_n)
    return [hidden, output]


def activate(weights, bias, inputs):
    '''
    '''
    return np.dot(inputs, weights) + bias


def transfer(activation):
    '''
    '''
    return 1.0 / (1.0 + np.exp(-activation))


def transfer_derivative(output):
    '''
    '''
    return output * (1 - output)


def forward_propagate(network, row):
    '''
    '''
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            act = activate(neuron['weights'],  neuron['bias'], inputs)
            neuron['output'] = transfer(act)
            new_inputs.append(neuron['output'])
        inputs = np.array(new_inputs)
    return inputs


class Network(object):

    def __init__(self, sizes):
        '''
        '''
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        '''
        '''
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def sgd(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        '''
        '''
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            np.random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size]
                            for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print('epoch {}: {} / {}'.format(
                    j, self.evaluate(test_data), n_test))
            else:
                print('epoch {} complete'.format(j))

    def update_mini_batch(self, mini_batch, eta):
        '''
        '''
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        '''
        '''
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        delta = self.cost_derivative(activations[-1], y) * \
                sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return nabla_b, nabla_w

    def evaluate(self, test_data):
        '''
        '''
        test_results = [(np.argmax(self.feedforward(x)),
                         np.argmax(y))
                        for x, y in test_data]
        return sum(int(x==y) for x, y in test_results)
    
    def cost_derivative(self, output_activations, y):
        '''
        '''
        return output_activations - y


def sigmoid(z):
    return 1./(1. + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))


