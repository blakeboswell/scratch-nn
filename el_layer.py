import numpy as np

from utils import load_data


def cache_feed(fun):
    """ return fun(args), args
    """
    def with_cache(*args):
        if len(args) < 2:
            (arg,) = args
            return fun(arg), arg
        return fun(*args), args
    return with_cache


@cache_feed
def sigmoid(Z):
    """
    """
    return (1 + np.exp(-Z))**-1


def sigmoid_prime(Z):
    """
    """
    return sigmoid(Z) * (1 - sigmoid(Z))


@cache_feed
def relu(Z):
    """
    """
    return np.max(Z, 0)


def relu_prime(Z):
    """
    """
    return 1 if Z > 0 else 0


def cost(AL, Y):
    """
    """
    m = Y.shape[0]
    log_prob = np.multiply(Y, np.log(AL)) + np.multiply((1 - Y), AL)
    return -1/m*np.sum(log_prob)


def cost_prime(AL, Y):
    """
    """
    return -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))


def initialize_params(layer_dims):
    """
    """

    def init_weight(dim_l, dim_l_prev):
        return np.random.randn(dim_l, dim_l_prev) * 0.01

    def init_bias(dim_l):
        return np.zeros((dim_l, 1))

    params = {}
    for i in range(1, len(layer_dims)):
        params[f'W{i}'] = init_weight(layer_dims[i], layer_dims[i-1])
        params[f'b{i}'] = init_bias(layer_dims[i])

    return params


def linear_forward(A, W, b):
    """
    """
    return np.dot(W, A) + b, (A, W, b)


def linear_activation_forward(A_prev, W, b, activation):
    """
    """
    Z, linear_cache = linear_forward(A_prev, W, b)
    A, activation_cache = activation(Z)
    return A, (linear_cache, activation_cache)


def el_model_forward(X, params):
    """
    """
    caches = []
    A = X
    L = len(params) // 2

    for i in range(1, L):
        A_prev = A
        W, b = params[f'W{i}'], params[f'b{i}']
        A, cache = linear_activation_forward(A_prev, W, b, relu)
        caches.append(cache)

    W, b = params[f'W{L}'], params[f'b{L}']
    AL, cache = linear_activation_forward(A, W, b, sigmoid)
    caches.append(cache)

    return AL, caches


def linear_backward(dZ, cache):
    """
    """
    A_prev, W, b = cache
    m = A_prev.shape[0]

    dW = 1/m*np.dot(dZ, A_prev.T)
    db = 1/m*np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db


def linear_backward_activation(dA, cache, activation):
    """
    """
    linear_cache, activation_cache = cache
    dZ = activation(dA, activation_cache)
    return linear_backward(dZ, linear_cache)


def el_model_backward(AL, Y, cache):
    """
    """
    grads = {}
    L = len(caches)
    m = AL.shape[0]
    Y = Y.reshape(AL.shape)

    dAL = cost_prime(AL, Y)
    current_cache = caches[L-1]
    grads[f'dA{L}'], grads[f'dW{L}'], grads[f'db{L}'] = linear_activation_backward(
            dAL, curren_cache, sigmoid_prime
    )

    for i in reversed(range(1, L-1)):
        current_cache = caches[i-1]
        dA, dW, db = linear_activation_backward(
            grads[f'dA{i+2}'], curren_cache, relu_prime
        )
        grads[f'dA{i}'] = dA
        grads[f'dW{i}'] = dW
        grads[f'db{i}'] = db

    return grads


def update_params(params, grads, alpha):
    """
    """
    L = len(params) // 2
    for i in range(L):
        params[f'W{i+1}'] = params[f'W{i+1}'] + alpha*grads['dW{i+1}']
        params[f'b{i+1}'] = params[f'b{i+1}'] + alpha*grads['db{i+1}']

    return params


def el_layer_model(X, Y, layer_dims,
                   alpha=0.0075, num_iterations=2500, print_cost=True):
    """
    """
    np.random.seed(1)
    costs = []

    params = initialize_params(layer_dims)

    for i in range(num_iterations):
        AL, caches = el_model_forward(X, params)
        cost = compute_cost(AL, Y)
        grads = el_model_backwards(AL, Y, caches)
        params = update_params(params, grads, alpha)

        if print_cost and i % 100 == 0:
            print(f'Cost after iteration {i}: {cost}')
            costs.append(cost)

    return params


def main(dir_path):
    """
    """
    X_train, X_test, Y_train, Y_test = load_data(dir_path)
    layer_dims = [X_train.shape[0], 20, 7, 5, 1]
    params = el_layer_model(X_train, X_test, layer_dims)


if __name__ == '__main__':
    data_dir ='data/train'
    main(data_dir)

