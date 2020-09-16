import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import argparse

# def model(X, Y, layer_dims, optimizer, learning_rate = 0.0007, mini_batch_size = 64, beta = 0.9,
#           beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8, num_epochs = 10000, print_cost = True):

print("BEGINNING", flush=True)

print_debug = False
def normalize_input(X,Xtest):
    m = X.shape[1]
    mu = np.sum(X, axis = 1, keepdims=True) / m
    X -= mu
    sigsquared = np.sum(X**2, axis=1, keepdims=True) / m
    offset = np.int64(sigsquared == 0)
    X /= (sigsquared + offset)
    Xtest -= mu
    Xtest /= sigsquared
    return X, Xtest

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="/slackdata/features/")
    parser.add_argument("--xtrain", default="/slackdata/features/trainx_0.csv")
    parser.add_argument("--ytrain", default="/slackdata/features/trainy_0.csv")
    parser.add_argument("--xtest", default="/slackdata/features/testx_0.csv")
    parser.add_argument("--ytest", default="/slackdata/features/testy_0.csv")
    args = parser.parse_args()
    return args

def initialize_adam(layer_dims):
    params = []
    for i in range(1, len(layer_dims)):
        layer_params =  {}
        layer_params["VdW"] = np.zeros((layer_dims[i], layer_dims[i-1]))
        layer_params["Vdb"] = np.zeros((layer_dims[i], 1))
        layer_params["SdW"] = np.zeros((layer_dims[i], layer_dims[i-1]))
        layer_params["Sdb"] = np.zeros((layer_dims[i], 1))
        params.append(layer_params)
    return params

def initialize_params(layer_dims):
    params = []
    np.random.seed(5)
    for i in range(1,len(layer_dims)):
        # Xavier initialization?
        layer_params =  {}
        layer_params["W"] = np.random.randn(layer_dims[i], layer_dims[i-1]) * np.sqrt(1 / layer_dims[i-1])
        layer_params["b"] = np.zeros((layer_dims[i], 1))

        assert layer_params['W'].shape[0] == layer_dims[i], layer_dims[i - 1]
        assert layer_params['W'].shape[0] == layer_dims[i], 1
        params.append(layer_params)
    return params

def random_mini_batches(X, Y, mini_batch_size, seed):
    m = X.shape[1]
    permutation = np.random.permutation(m)
    randX = X[:, permutation]
    randY = Y[:, permutation]
    numbatches = math.floor(m/mini_batch_size)
    if m % mini_batch_size != 0:
        numbatches += 1
    minibatches = []
    for i in range(0, numbatches):
        start = i * mini_batch_size
        end = min((i+1) * mini_batch_size, m)
        minibatches.append((randX[:, start:end], randY[:, start:end]))
    return minibatches

def sigmoid(x):
    return 1/(1+np.exp(-x))

def relu(x):
    s = np.maximum(x, 0)
    return s

def softmax(x):
    if print_debug:
        print(np.amin(x))
        print(np.amax(x))
    x = np.exp(x)
    x /= np.sum(x, axis=0, keepdims=True)
    return x

def forward_prop(X, params):
    cache = []
    Z0 = np.dot(params[0]["W"], X) + params[0]["b"]
    Aprev = relu(Z0)
    cache.append({"Z": Z0, "A": Aprev, "W": params[0]["W"], "b": params[0]["b"]})
    for l in range(1,len(params)-1):
        Zl = np.dot(params[l]["W"], Aprev) + params[l]["b"]
        Al = relu(Zl)
        cache.append({"Z": Zl, "A": Al, "W": params[l]["W"], "b": params[l]["b"]})
        Aprev = Al
    l = len(params) - 1
    Zout = np.dot(params[l]["W"], Aprev) + params[l]["b"]
    Aout = softmax(Zout)
    cache.append({"Z": Zout, "A": Aout, "W": params[l]["W"], "b": params[l]["b"]})
    return Aout, cache

def compute_cost(output, Y):
    m = Y.shape[1]
    cost = -1./m * np.sum(np.sum(Y * np.log(output), axis = 0, keepdims=True), axis = 1, keepdims=True).flatten()
    return cost

def back_prop(X, Y, caches):
    L = len(caches)
    grads = [{} for i in range(0, L)]
    m = X.shape[1]
    dZL = caches[L-1]["A"] - Y
    dWL = 1./m * np.dot(dZL, caches[L-2]["A"].T)
    dbL = 1./m * np.sum(dZL, axis = 1, keepdims=True)
    grads[L-1] = {"dZ": dZL, "dW": dWL, "db": dbL}
    for l in range(L-2,-1,-1):
        dAl = np.dot(caches[l+1]["W"].T, caches[l+1]["Z"])
        dZl = np.multiply(dAl, np.int64(caches[l]["A"] > 0))
        if l == 0:
            dWl = 1./m * np.dot(dZl, X.T)
        else:
            dWl = 1./m * np.dot(dZl, caches[l-1]["A"].T)
        dbl = 1./m * np.sum(dZl, axis=1, keepdims=True)
        grads[l] = {"dA": dAl, "dZ": dZl, "dW": dWl, "db": dbl}
    return grads

def update_parameters(params, grads, learning_rate):
    L = len(params)
    for l in range(0,L):
        params[l]["W"] -= learning_rate * grads[l]["dW"]
        params[l]["b"] -= learning_rate * grads[l]["db"]
    return params

def update_parameters_with_adam(params, adam_params, grads, t, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
    L = len(params)
    for l in range(0,L):
        adam_params[l]["VdW"] = beta1 * adam_params[l]["VdW"] + (1-beta1) * grads[l]["dW"]
        adam_params[l]["Vdb"] = beta1 * adam_params[l]["Vdb"] + (1-beta1) * grads[l]["db"]
        vw_corrected = adam_params[l]["VdW"] / (1-beta1 ** t)
        vb_corrected = adam_params[l]["Vdb"] / (1-beta1 ** t)

        adam_params[l]["SdW"] = beta2 * adam_params[l]["SdW"] + (1-beta2) * (grads[l]["dW"] ** 2)
        adam_params[l]["Sb"] = beta2 * adam_params[l]["Sdb"] + (1-beta2) * (grads[l]["db"] ** 2)
        sw_corrected = adam_params[l]["SdW"] / (1-beta2 ** t)
        sb_corrected = adam_params[l]["Sdb"] / (1-beta2 ** t)

        params[l]["W"] -= learning_rate * vw_corrected / np.sqrt(sw_corrected + epsilon)
        params[l]["b"] -= learning_rate * vb_corrected / np.sqrt(sb_corrected + epsilon)
    return params, adam_params

def predict(X, Y, params):
    m = X.shape[1]
    n = len(params)
    preds = np.zeros((1,m))

    output, _ = forward_prop(X,params)

    preds = np.int64(output == np.max(output, axis = 0))
    accuracy = 100/m * np.sum(np.multiply.reduce(preds == np.int64(Y), axis=0))
    print("Accuracy: " + str(accuracy), flush=True)

def plotcosts(costs, learning_rate):
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per 25)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

def model(X, Y, layer_dims, learning_rate = 0.0007, mini_batch_size = 64, num_epochs=1):
    # initialize weights
    # Loop over epochs and mini batches
    #    Forward/Back Prop
    # Optimize
    # Plot cost/metrics
    # Evaluate
    print("HEEERE", flush=True)
    m = X.shape[1]
    params = initialize_params(layer_dims)
    adam_params = initialize_adam(layer_dims)
    seed = 10
    costs = []
    t = 1
    if print_debug:
        print(np.amin(X))
        print(np.amax(X))
    for i in range(0, num_epochs):
        seed += 1
        minibatches = random_mini_batches(X, Y, mini_batch_size, seed)
        cost_total = 0
        batchnumber = 0
        for batch in minibatches:
            batchnumber+=1
            if print_debug:
                print("======================BATCH NUMBER: " + str(batchnumber))
                print(np.amax(params[0]["W"]))
                print(np.amax(params[1]["W"]))
                print(np.amax(params[2]["W"]))
                print(np.amax(params[3]["W"]))
                print(np.amax(params[4]["W"]))
            batchX, batchY = batch
            output, caches = forward_prop(batchX, params)
            grads = back_prop(batchX, batchY, caches)
            # params = update_parameters(params, grads, learning_rate)
            params, adam_params = update_parameters_with_adam(params, adam_params, grads, t, learning_rate)
            t += 1
            cost = compute_cost(output, batchY)
            cost_total += cost
        cost_avg = cost_total / m
        if i % 25 == 0:
            print("Cost after " + str(i) + " iterations: " + str(cost_avg), flush = True)
            costs.append(cost_avg)

    # plotcosts(costs, learning_rate)

    print(costs)
    return params

if __name__=="__main__":
    print("HERE", flush=True)
    # xtrain = "/Users/davidlyle/slack_ml/data/features/trainx.csv"
    # ytrain = "/Users/davidlyle/slack_ml/data/features/trainy.csv"
    # xtest = "/Users/davidlyle/slack_ml/data/features/testx.csv"
    # ytest = "/Users/davidlyle/slack_ml/data/features/testy.csv"
    args = parse_args()
    if False and args.features != "/slackdata/features/":
        args.xtrain = args.features + "trainx.csv"
        args.ytrain = args.features + "trainy.csv"
        args.xtest = args.features + "testx.csv"
        args.ytest = args.features + "testy.csv"
    X = pd.read_csv(args.xtrain, header=0, index_col=0)
    X = X.T.to_numpy()
    Y = pd.read_csv(args.ytrain, header=0, index_col=0)
    Y= Y.T.to_numpy()
    Xtest = pd.read_csv(args.xtest, header=0, index_col=0)
    Xtest = Xtest.T.to_numpy()
    Ytest = pd.read_csv(args.ytest, header=0, index_col=0)
    Ytest = Ytest.T.to_numpy()

    #X,Xtest = normalize_input(X,Xtest)

    layer_dims = [X.shape[0], 1000, 500, 250, 100, 50, Y.shape[0]]
    params = model(X,Y,layer_dims, learning_rate=0.001, num_epochs=200)

    predict(X, Y, params)
    predict(Xtest, Ytest, params)

