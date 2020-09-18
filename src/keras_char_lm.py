from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Input
from keras.optimizers import Adam
from keras.utils.data_utils import get_file
import numpy as np
import random
import math
import sys
import io
import os
import argparse
from functools import reduce
from data.load import load_datasets

def create_model(chars, n_a, maxlen, lr):
    vocab_size = len(chars)
    print(maxlen)
    print(vocab_size)
    model = Sequential([
        LSTM(n_a, input_shape=(maxlen, vocab_size), return_sequences=True),
        Dense(vocab_size, activation="softmax")
    ])
    model.compile(loss='categorical_crossentropy', optimizer="adam")
    print(model.summary())
    return model

def sample(model, maxlen, chars, char_to_ix):
    vocab_size = len(chars)
    x = np.zeros((1,maxlen, vocab_size))
    next_char = ""
    num_chars = 0
    #while next_char != "<EOM>" and num_chars < maxlen:
    preds = model(x, training=False)[0]
    print(preds)
    return

def on_epoch_end(epoch, _):
    print("COMPLETED EPOCH %d" % epoch)

def main(datadir):
    mini_batch_size = 64
    n_a = 32
    num_epochs = 1
    train, test = load_datasets(datadir)
    m = len(train)
    numbatches = math.floor(m / mini_batch_size)
    if m % mini_batch_size != 0:
        numbatches += 1
    chars = set()
    maxlen = max([len(msg) for msg in train]) + 1
    #maxlen = reduce(lambda a,b: len(a) if len(a) > b else b, train) + 1 # Adding one for appended <EOM>
    for msg in train:
        chars = chars.union(set(msg))
    chars = sorted(list(chars))
    chars.append("<EOM>")
    char_to_ix = {c: i for i, c in enumerate(chars)}

    model = create_model(chars, 32, maxlen, 0.01)
    for e in range(0,num_epochs):
        random.shuffle(train)
        for b in range(0,numbatches):
            X = np.zeros((mini_batch_size, maxlen, len(chars)))
            Y = np.zeros((mini_batch_size, maxlen, len(chars)))
            for i, msg in enumerate(train[b*mini_batch_size:min(m, (b+1)*mini_batch_size)]):
                for t,c in enumerate(msg):
                    X[i, t+1, char_to_ix[c]] = 1
                    Y[i, t, char_to_ix[c]] = 1
                Y[i, -1, char_to_ix["<EOM>"]] = 1
            model.train_on_batch(X,Y)
            print("FINISHED BATCH %d on slice [%d:%d]" % (b, b*mini_batch_size,min(m, (b+1)*mini_batch_size)))

if __name__=="__main__":
    main("data/user_msgs/U0AR782AV/")