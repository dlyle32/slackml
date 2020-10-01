import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.models import load_model
import numpy as np
import random
import math
import sys
import io
import os
import argparse
import json
from functools import reduce
from data.load import load_datasets

def create_model(chars, n_a, maxlen, lr):
    vocab_size = len(chars)
    model = Sequential([
        LSTM(n_a, input_shape=(maxlen, vocab_size), return_sequences=True),
        LSTM(n_a, return_sequences=True),
        TimeDistributed(Dense(vocab_size, activation="softmax"))
    ])
    opt = Adam(learning_rate=lr, clipvalue=5)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=["accuracy"])
    print(model.summary())
    return model

def sample(model, chars, char_to_ix, temperature=1.0):
    maxlen=model.layers[0].input_shape[1]
    vocab_size =model.layers[-1].output_shape[-1]
    output = ""
    x = np.zeros((1, maxlen, vocab_size))
    char_index = -1
    i = 0
    while char_index != char_to_ix['\n'] and i < maxlen:
        preds = model.predict(x, verbose=0)[0][i]
        #preds = np.asarray(preds).astype('float64')
        #preds = np.log(preds) / temperature
        #exp_preds = np.exp(preds)
        #preds = exp_preds / np.sum(exp_preds)
        char_index = np.random.choice(range(vocab_size), p = preds.ravel())
        #probas = np.random.multinomial(1, preds, 1)
        #char_index = np.argmax(probas)
        x[0,i,char_index] = 1
        output += chars[char_index]
        i+=1
    return output

def get_ix_from_char(char_to_ix, chars, c):
    if c in chars:
        return char_to_ix[c]
    else:
        return char_to_ix["<UNK>"]

def on_epoch_end(epoch, model, chars, char_to_ix, metrics, model_path):
    #print("COMPLETED EPOCH %d" % epoch)
    #for lbl in metrics.keys():
    #    print(lbl + ": " + str(metrics[lbl]))
    model.save(model_path)
    sample_msg = sample(model, chars, char_to_ix)
    print("\n" + sample_msg)

def oh_to_char(chars, oh):
    char = [chars[i] for i,c in enumerate(oh) if c == 1]
    return "0" if len(char) == 0 else char[0]

def main(datadir, args):
    model_path = "/slackml/mymodel3.keras"
    mini_batch_size = 512
    n_a = 128
    num_epochs = 60
    train, test = load_datasets(datadir)
    m = len(train)
    chars = set()
    maxlen = 25
    #for msg in train:
    #    chars = chars.union(set(msg))
    #chars = sorted(list(chars))
    #chars = ['\t', '\n', ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~', '<UNK>']
    chars = ['\n', ' ', '!', '"', ',', '.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', '?', '@', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '~', '<UNK>']
    char_to_ix = {c: i for i, c in enumerate(chars)}

    if args.loadmodel:
        print("LOADING FROM FILE")
    else:
        print("CREATING MODEL")
    model = create_model(chars, n_a, maxlen, 0.01) if not args.loadmodel else load_model(model_path)
    metrics = []
    data = "".join(train)
    nummsgs = math.floor(len(data) / maxlen)
    if len(data) % maxlen != 0:
        nummsgs += 1
    X = np.zeros((nummsgs, maxlen, len(chars)))
    Y = np.zeros((nummsgs, maxlen, len(chars)))
    msgs = 0
    for i in range(0,len(data), maxlen):
        last_ix = min(i+maxlen, len(data)-1)
        for t,c in enumerate(data[i:last_ix]):
            char_index = get_ix_from_char(char_to_ix, chars, c)
            if t < maxlen-1:
                X[msgs, t + 1, char_index] = 1
            Y[msgs, t, char_index] = 1
        Y[msgs, maxlen-1, get_ix_from_char(char_to_ix, chars, data[last_ix])] = 1
        msgs+=1
    epoch_callback = LambdaCallback(on_epoch_end=lambda epoch, logs: on_epoch_end(epoch, model, chars, char_to_ix, logs, model_path))
    model.fit(X,Y,
              batch_size=mini_batch_size,
              epochs=num_epochs,
              callbacks=[epoch_callback])

def parse_args():
    parser= argparse.ArgumentParser()
    parser.add_argument("--loadmodel", action="store_true", default=False)
    return parser.parse_args()

if __name__=="__main__":
    args = parse_args()
    main("/slackdata/", args)
