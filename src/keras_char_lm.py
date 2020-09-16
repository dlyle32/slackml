from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Input
from keras.optimizers import Adam
from keras.utils.data_utils import get_file
import numpy as np
import random
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
    train, test = load_datasets(datadir)
    print(len(train))
    chars = set()
    maxlen = max([len(msg) for msg in train]) + 1
    #maxlen = reduce(lambda a,b: len(a) if len(a) > b else b, train) + 1 # Adding one for appended <EOM>
    for msg in train:
        chars = chars.union(set(msg))
    chars = sorted(list(chars))
    chars.append("<EOM>")
    char_to_ix = {c: i for i, c in enumerate(chars)}
    X = np.zeros((len(train), maxlen, len(chars)))
    Y = np.zeros((len(train), maxlen, len(chars)))
    for i, msg in enumerate(train):
        for t,c in enumerate(msg):
            X[i, t+1, char_to_ix[c]] = 1
            Y[i, t, char_to_ix[c]] = 1
        Y[i, -1, char_to_ix["<EOM>"]] = 1

    print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
    model = create_model(chars, 128, maxlen, 0.01)
    model.fit(X,Y, batch_size=128, epochs=1, callbacks=[print_callback])

if __name__=="__main__":
    main("data/user_msgs/U0AR782AV/")