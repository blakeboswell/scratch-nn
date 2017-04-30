from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


def relu(x):
    return np.maximum(x, 0)


def softmax(x):
    logit = np.exp(x)
    return logit / np.sum(logit, axis=1, keepdims=True)


def cross_entropy_loss(softmax_prob, y_onehot):
    indices = np.argmax(y_onehot, axis=1).astype(int)
    pred = softmax_prob[np.arange(len(softmax_prob)), indices]
    log_pred = np.log(pred)
    return -1 * np.sum(log_pred) / len(log_pred)


def regularization(reg_lambda, w1, w2):
    w1_loss = 0.5 * reg_lambda * np.sum(w1 * w1)
    w2_loss = 0.5 * reg_lambda * np.sum(w2 * w2)
    return w1_loss + w2_loss


def accuracy(preds, labels):
    correct = np.sum(np.argmax(preds, 1) == np.argmax(labels, 1))
    return 100.0 * correct / preds.shape[0]  


np.random.seed(12)
num_observations = 5000

x1 = np.random.multivariate_normal([0, 0], [[2, .75],[.75, 2]], num_observations)
x2 = np.random.multivariate_normal([1, 4], [[1, .75],[.75, 1]], num_observations)
x3 = np.random.multivariate_normal([2, 8], [[0, .75],[.75, 0]], num_observations)

X = np.vstack((x1, x2, x3)).astype(np.float32)
y = np.hstack((np.zeros(num_observations),
	       np.ones(num_observations),
               np.ones(num_observations) + 1))


y_onehot = np.zeros((y.shape[0], 3)).astype(int)
y_onehot[np.arange(len(y)), y.astype(int)] = 1

train_X, test_X, train_y, test_y  = train_test_split(
        X, y_onehot, test_size=0.1, random_state=12)

hidden_nodes = 5
num_features = train_X.shape[1]
num_labels = train_y.shape[1]
learning_rate = 0.01
reg_lambda = 0.01

w1 = np.random.normal(0, 1, (num_features, hidden_nodes))
b1 = np.zeros((1, hidden_nodes))

w2 = np.random.normal(0, 1, (hidden_nodes, num_labels))
b2 = np.zeros((1, num_labels))

for step in range(5001):

    input_layer = np.dot(train_X, w1) + b1
    hidden_layer = relu(input_layer)
    output_layer = np.dot(hidden_layer, w2) + b2
    output_prob = softmax(output_layer)

    loss = cross_entropy_loss(output_prob, train_y)
    loss += regularization(reg_lambda, w1, w2)

    output_error_signal = (output_prob - train_y) / output_prob.shape[0]

    error_signal_hidden = np.dot(output_error_signal, w2.T)
    error_signal_hidden[hidden_layer <= 0] = 0

    gradient_layer2_weights = np.dot(hidden_layer.T, output_error_signal)
    gradient_layer2_bias = np.sum(output_error_signal, axis=0, keepdims=True)

    gradient_layer1_weights = np.dot(train_X.T, error_signal_hidden)
    gradient_layer1_bias = np.sum(error_signal_hidden, axis=0, keepdims=True)
    
    gradient_layer2_weights += reg_lambda * w2
    gradient_layer1_weights += reg_lambda * w1

    w1 -= learning_rate * gradient_layer1_weights
    b1 -= learning_rate * gradient_layer1_bias
    w2 -= learning_rate * gradient_layer2_weights
    b2 -= learning_rate * gradient_layer2_bias

    if step % 500 == 0:
        print('loss at step {}: {}'.format(step, loss))
 

input_layer = np.dot(test_X, w1)
hidden_layer = relu(input_layer + b1)
scores = np.dot(hidden_layer, w2) + b2
probs = softmax(scores)

print('test accuracy: {}%'.format(accuracy(probs, test_y)))
