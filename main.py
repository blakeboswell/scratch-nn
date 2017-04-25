import mnist_loader as loader
import network as net
import numpy as np

if __name__ == '__main__':

#    train, valid, test = loader.load_data()
#    net = Network([784, 30, 10])
#    net.sgd(train, 30, 10, 3.0, test_data=test)

    network = [[{'weights': np.array([0.13436424411240122, 0.8474337369372327]),
                 'bias': np.array([0.763774618976614])}],
               [{'weights': np.array([0.2550690257394217]),
                 'bias': np.array([0.49543508709194095])},
                {'weights': np.array([0.4494910647887381]),
                 'bias': np.array([0.651592972722763])}]]

    row = np.array([1., 0.])
    output = net.forward_propagate(network, row)
    print(output)
    print(type(output))
    print(output.shape)
