import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.optimizers import Adam
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
        LSTM(n_a, input_shape=(maxlen, n_a), return_sequences=True),
        TimeDistributed(Dense(vocab_size, activation="softmax"))
    ])
    opt = Adam(learning_rate=lr)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=["accuracy"])
    print(model.summary())
    return model

def sample(model, chars, temperature=1.0):
    maxlen=model.layers[0].input_shape[1]
    vocab_size =model.layers[-1].output_shape[-1]
    output = ""
    x = np.zeros((1, maxlen, vocab_size))
    char_index = -1
    i = 0
    while char_index != vocab_size-1 and i < maxlen:
        preds = model.predict(x, verbose=0)[0][i]
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        char_index = np.argmax(probas)
        x[0,i,char_index] = 1
        output += chars[char_index]
        i+=1
    return output

def get_ix_from_char(char_to_ix, chars, c):
    if c in chars:
        return char_to_ix[c]
    else:
        return char_to_ix["<UNK>"]

def on_epoch_end(epoch, model, chars, metrics):
    print("COMPLETED EPOCH %d" % epoch)
    for i,lbl in enumerate(model.metrics_names):
        print(lbl + ": " + str(metrics[i]))
    sample_msg = sample(model, chars) 
    print(sample_msg)

def oh_to_char(chars, oh):
    char = [chars[i] for i,c in enumerate(oh) if c == 1]
    return "0" if len(char) == 0 else char[0]

def main(datadir):
    mini_batch_size = 10
    n_a = 512
    num_epochs = 200
    train, test = load_datasets(datadir)
    m = len(train)
    numbatches = math.floor(m / mini_batch_size)
    if m % mini_batch_size != 0:
        numbatches += 1
    chars = set()
    maxlen = 60
    #maxlen = max([len(msg) for msg in train]) + 1
    #maxlen = reduce(lambda a,b: len(a) if len(a) > b else b, train) + 1 # Adding one for appended <EOM>
    #for msg in train:
    #    chars = chars.union(set(msg))
    #chars = sorted(list(chars))
    chars = ['\t', '\n', ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~', '<UNK>', '<EOM>']
    char_to_ix = {c: i for i, c in enumerate(chars)}

    model = create_model(chars, n_a, maxlen, 0.01)
    metrics = []
    for e in range(0,num_epochs):
        random.shuffle(train)
        for b in range(0,numbatches):
            X = np.zeros((mini_batch_size, maxlen, len(chars)))
            Y = np.zeros((mini_batch_size, maxlen, len(chars)))
            for i, msg in enumerate(train[b*mini_batch_size:min(m, (b+1)*mini_batch_size)]):
                for t,c in enumerate(msg[0:maxlen-1]):
                    char_index = get_ix_from_char(char_to_ix, chars,c)
                    X[i, t+1, char_index] = 1
                    Y[i, t, char_index] = 1
                Y[i, min(t+1,maxlen-1), char_to_ix["<EOM>"]] = 1
                print(msg)
                for j in range(maxlen):
                    print("X: " + oh_to_char(chars, X[i,j,:]))
                    print("Y: " + oh_to_char(chars, Y[i,j,:]))
                return
            metrics = model.train_on_batch(X,Y)
            #print("FINISHED BATCH %d on slice [%d:%d]" % (b, b*mini_batch_size,min(m, (b+1)*mini_batch_size)))
        model.save("/slackml/mymodel5.keras")
        on_epoch_end(e, model, chars, metrics)

if __name__=="__main__":
    main("/slackdata/")
