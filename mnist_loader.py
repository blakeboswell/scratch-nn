import numpy as np
import _pickle
import gzip


def load_data():
     
    with gzip.open('mnist.pkl.gz', 'rb') as f:
        train, valid, test = _pickle.load(f, encoding='bytes')
    
    def onehot(j):
        e = np.zeros((10, 1))
        e[j] = 1.0
        return e

    def transform(data):
        x = [np.reshape(i, (784, 1)) for i in data[0]]
        y = [onehot(y) for  y in data[1]]
        return list(zip(x, y))

    return transform(train), transform(valid), transform(test)

