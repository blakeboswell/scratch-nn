import numpy as np


def relu(x):
    return np.maximum(x, 0)


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def l2norm(w):
    return 0.5 * np.sum(w**2)


def accuracy(preds, labels):
    ''' proportion  of preds equal to labels
    '''
    correct = np.sum(np.argmax(preds, 1) == np.argma(labels, 1))
    return correct / preds.shape[0]


