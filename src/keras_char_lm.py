from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Input
from keras.layers import TimeDistributed
from keras.optimizers import Adam
from keras.utils.data_utils import get_file
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
    model.compile(loss='categorical_crossentropy', optimizer="adam")
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

def on_epoch_end(epoch, model, chars):
    print("COMPLETED EPOCH %d" % epoch)
    sample_msg = sample(model, chars) 
    print(sample_msg)

def main(datadir):
    mini_batch_size = 256
    n_a = 64
    num_epochs = 50
    train, test = load_datasets(datadir)
    m = len(train)
    numbatches = math.floor(m / mini_batch_size)
    if m % mini_batch_size != 0:
        numbatches += 1
    chars = set()
    maxlen = 500
    #maxlen = max([len(msg) for msg in train]) + 1
    #maxlen = reduce(lambda a,b: len(a) if len(a) > b else b, train) + 1 # Adding one for appended <EOM>
    for msg in train:
        chars = chars.union(set(msg))
    chars = sorted(list(chars))
    chars.append("<EOM>")
    char_to_ix = {c: i for i, c in enumerate(chars)}

    model = create_model(chars, n_a, maxlen, 0.01)
    for e in range(0,num_epochs):
        random.shuffle(train)
        for b in range(0,numbatches):
            X = np.zeros((mini_batch_size, maxlen, len(chars)))
            Y = np.zeros((mini_batch_size, maxlen, len(chars)))
            for i, msg in enumerate(train[b*mini_batch_size:min(m, (b+1)*mini_batch_size)]):
                for t,c in enumerate(msg[0:maxlen-1]):
                    X[i, t+1, char_to_ix[c]] = 1
                    Y[i, t, char_to_ix[c]] = 1
                Y[i, min(t+1,maxlen-1), char_to_ix["<EOM>"]] = 1
            model.train_on_batch(X,Y)
            #print("FINISHED BATCH %d on slice [%d:%d]" % (b, b*mini_batch_size,min(m, (b+1)*mini_batch_size)))
        model.save("mymodel2.keras")
        on_epoch_end(e, model, chars)

if __name__=="__main__":
    main("/slackdata/")
