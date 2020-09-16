import os
import numpy as np
import random
import argparse
import json

def sample(parameters, char_to_ix, seed):
    None

def init_hidden_state(n_a):
    return np.zeros((n_a, 1))

def init_gradients(params, a):
    gradients = {}
    Waa, Wax, Wya, b, by = params["Waa"], params["Wax"], params["Wya"], params["b"], params["by"]
    gradients["dWaa"] = np.zeros_like(Waa)
    gradients["dWax"] = np.zeros_like(Wax)
    gradients["dWya"] = np.zeros_like(Wya)
    gradients["db"] = np.zeros_like(b)
    gradients["dby"] = np.zeros_like(by)
    gradients["da_next"] = np.zeros_like(a[0])
    return gradients

def init_params(n_a, vocab_size):
    Waa, Wax, Wya = np.random.randn(n_a, n_a), np.random.randn(n_a, vocab_size), np.random.randn(vocab_size, n_a)
    Waa *= 0.01
    Wax *= 0.01
    Wya *= 0.01
    b, by = np.random.randn(n_a, 1), np.random.randn(vocab_size, 1)
    return {"Waa": Waa, "Wax": Wax, "Wya": Wya, "b":b, "by": by}

def char_to_one_hot(x, chars_to_ix):
    oh = np.zeros((len(chars_to_ix), 1))
    if x != None:
        oh[chars_to_ix[x]] = 1
    return np.array(oh)

def optimize():
    None

def clip(gradients, maxValue):
    for gradient in ["dWax", "dWaa", "dWya", "db", "dby"]:
        np.clip(gradients[gradient], -maxValue, maxValue, gradients[gradient])
    return gradients

def get_initial_loss(vocab_size):
    return -np.log(1.0/vocab_size)

def smooth(loss, cur_loss):
    return loss * 0.999 + cur_loss * 0.001

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def rnn_step_forward(input, a_prev, params):
    a_curr = np.tanh(np.dot(params["Waa"], a_prev) + np.dot(params["Wax"], input) + params["b"])
    y_hat = softmax(np.dot(params["Wya"], a_curr) + params["by"])
    return a_curr, y_hat

def rnn_step_backward(dy, gradients, params, x, a, a_prev):
    gradients['dWya'] += np.dot(dy, a.T)
    gradients['dby'] += dy
    da = np.dot(params['Wya'].T, dy) + gradients['da_next']  # backprop into h
    daraw = (1 - a * a) * da  # backprop through tanh nonlinearity
    gradients['db'] += daraw
    gradients['dWax'] += np.dot(daraw, x.T)
    gradients['dWaa'] += np.dot(daraw, a_prev.T)
    gradients['da_next'] = np.dot(params['Waa'].T, daraw)
    return gradients

def rnn_forward(X, Y, parameters, chars, char_to_ix, a_prev):
    a = {}
    x = {}
    y_hat = {}
    loss = 0
    a[-1] = a_prev
    for t, char in enumerate(X):
        x[t] = char_to_one_hot(X[t], char_to_ix)
        a[t], y_hat[t] = rnn_step_forward(x[t], a_prev, parameters)
        a_prev = a[t]
        loss -= np.log(y_hat[t][char_to_ix[Y[t]], 0])
    cache = (y_hat, a, x)
    return loss, cache

def rnn_backward(X, Y, parameters, cache, char_to_ix):
    (y_hat, a, x) = cache
    gradients = init_gradients(parameters,a)
    for t in reversed(range(len(X))):
        dy = np.copy(y_hat[t])
        dy[char_to_ix[Y[t]]] -= 1
        gradients = rnn_step_backward(dy, gradients, parameters, x[t], a[t], a[t-1])
    return gradients, a

def update_params(params, grads, lr):
    params["Waa"] += -lr * grads["dWaa"]
    params["Wax"] += -lr * grads["dWax"]
    params["Wya"] += -lr * grads["dWya"]
    params["b"] += -lr * grads["db"]
    params["by"] += -lr * grads["dby"]
    return params

def rnn(train_data, chars, char_to_ix, n_a, lr, num_iterations):
    vocab_size = len(chars)
    parameters = init_params(n_a, vocab_size)
    loss = get_initial_loss(vocab_size)
    for j in range(0,num_iterations):
        i = j % len(train_data)
        example_chars = [c for c in train_data[i]]
        X = [None] + example_chars
        Y = example_chars + ["<EOM>"]
        a0 = init_hidden_state(n_a)
        curr_loss, cache = rnn_forward(X, Y, parameters, chars, char_to_ix, a0)
        gradients, a = rnn_backward(X, Y, parameters, cache, char_to_ix)
        gradients = clip(gradients, 5)
        parameters = update_params(parameters, gradients, lr)
        loss = smooth(loss, curr_loss)
        if i % 1000 == 0:
            print("ON TRAINING EXAMPLE " + str(i) + " : " + train_data[i][0:50] + "...")
    return parameters

def sample(params, chars, char_to_ix):
    Waa, Wax, Wya, b, by = params["Waa"], params["Wax"], params["Wya"], params["b"], params["by"]
    vocab_size = by.shape[0]
    n_a = Waa.shape[1]
    msg = ""
    new_char = None
    a_prev = np.zeros((n_a, 1))
    iter = 0
    while new_char != "<EOM>" and iter < 300:
        x = char_to_one_hot(new_char, char_to_ix)
        a_prev, y_hat = rnn_step_forward(x, a_prev, params)
        char_int = np.random.choice(range(y_hat.shape[0]), p=y_hat.ravel())
        new_char = chars[char_int]
        msg += new_char
        iter += 1
    return msg


def main(dir):
    train_data = []
    test_data = []
    chars = set()
    num_msgs = 0
    for r, d, f in os.walk(dir):
        for file in f:
            if "csv" not in file:
                continue
            with open(os.path.join(r, file), 'r') as fp:
                for msg in fp:
                    num_msgs+=1
                    msg = msg.lower()
                    if(random.random() < 0.9):
                        train_data.append(msg)
                    else:
                        test_data.append(msg)
                    chars = chars.union(set(msg))
    chars = sorted(list(chars))
    chars.append("<EOM>")
    char_to_ix = {c:i for i,c in enumerate(chars)}
    params = rnn(train_data, chars, char_to_ix, 128, 0.01, 150000)
    params_to_write = {}
    for key in params:
        params_to_write[key] = params[key].tolist()
    with open(os.path.join(r,"parameters.json"), 'w') as fp:
        json.dump(params_to_write, fp)
    for i in range(0,5):
        msg = sample(params, chars, char_to_ix)
        print(msg)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", default="/slackdata/")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args.datadir)

