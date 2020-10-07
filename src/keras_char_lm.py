import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import LambdaCallback, ModelCheckpoint, CSVLogger
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM, Dropout
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.models import load_model
import numpy as np
import random
import math
import requests
import datetime
import time
import glob
import csv
import sys
import io
import os
import argparse
import logging
import json
from functools import reduce
from data.load import load_datasets

logger = logging.getLogger('keras_char_lm')

def create_model(chars, n_a, maxlen, lr):
    vocab_size = len(chars)
    model = Sequential([
        LSTM(n_a, input_shape=(maxlen, vocab_size), return_sequences=True),
        Dropout(0.3),
        LSTM(n_a),
        Dropout(0.3),
        Dense(vocab_size, activation="softmax")
    ])
    opt = RMSprop(learning_rate=lr, clipvalue=3)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=["accuracy"])
    model.summary(print_fn=logger.info)
    return model

def sample(data, model, chars, char_to_ix, temperature=1.0):
    maxlen=model.layers[0].input_shape[1]
    vocab_size =model.layers[-1].output_shape[-1]
    char_index = -1
    i = random.randint(0, len(data) - maxlen - 1)
    inpt = data[i:i+maxlen]
    output = inpt
    while char_index != char_to_ix['\n']:
        x = np.zeros((1, maxlen, vocab_size))
        x[0] = [char_to_oh(get_ix_from_char(char_to_ix, chars, c), vocab_size) for c in inpt]
        preds = model.predict(x, verbose=0)[0]
        #preds = np.asarray(preds).astype('float64')
        #preds = np.log(preds) / temperature
        #exp_preds = np.exp(preds)
        #preds = exp_preds / np.sum(exp_preds)
        char_index = np.random.choice(range(vocab_size), p = preds.ravel())
        #probas = np.random.multinomial(1, preds, 1)
        #char_index = np.argmax(probas)
        new_char = chars[char_index]
        output += new_char
        inpt = inpt[1:] + new_char
    logger.info("\n" + output[:40] + "->" + output[40:])
    return output

def get_ix_from_char(char_to_ix, chars, c):
    if c in chars:
        return char_to_ix[c]
    else:
        return char_to_ix["*"]

def on_epoch_end(data, epoch, model, chars, char_to_ix, metrics):
    #print("COMPLETED EPOCH %d" % epoch)
    #for lbl in metrics.keys():
    #    print(lbl + ": " + str(metrics[lbl]))
    sample_msg = sample(data, model, chars, char_to_ix)

def on_batch_end(batch, logs, volumedir):
    if batch % 100 == 0:
        fieldnames = logs.keys()
        with open(os.path.join(volumedir, "batch_metrics.csv"), "a+") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames)
            if batch == 0:
                writer.writeheader()
            writer.writerow(logs)

def oh_to_char(chars, oh):
    char = [chars[i] for i,c in enumerate(oh) if c == 1]
    return "0" if len(char) == 0 else char[0]

def char_to_oh(index, vocab_size):
    x = np.zeros((vocab_size))
    x[index] = 1
    return x

def load_checkpoint_model(checkpoint_path, checkpoint_names):
    list_of_checkpoint_files = glob.glob(os.path.join(checkpoint_path, '*'))
    checkpoint_epoch_number = max([int(file.split(".")[1]) for file in list_of_checkpoint_files])
    checkpoint_epoch_path = os.path.join(checkpoint_path,
                                         checkpoint_names.format(epoch=checkpoint_epoch_number))
    resume_model = load_model(checkpoint_epoch_path)
    return resume_model, checkpoint_epoch_number

def get_callbacks(volume_mount_dir, checkpoint_path, checkpoint_names, chars, char_to_ix, data, model):
    today_date = datetime.datetime.today().strftime('%Y-%m-%d')
    if not os.path.isdir(checkpoint_path):
        os.makedirs(checkpoint_path)
    filepath = os.path.join(checkpoint_path, checkpoint_names)
    checkpoint_callback = ModelCheckpoint(filepath=filepath,
                                          save_weights_only=False,
                                          monitor='val_loss')
    # Loss history callback
    epoch_results_callback = CSVLogger(os.path.join(volume_mount_dir, 'training_log_{}.csv'.format(today_date)),
                                       append=True)
    sample_callback = LambdaCallback(on_epoch_end=lambda epoch, logs: on_epoch_end(data,epoch, model, chars, char_to_ix, logs))

    batch_callback = LambdaCallback(on_batch_end=lambda batch, logs: on_batch_end(batch, logs, volume_mount_dir))

    class SpotTermination(keras.callbacks.Callback):
        def on_batch_begin(self, batch, logs={}):
            try:
                status_code = requests.get("http://169.254.169.254/latest/meta-data/spot/instance-action").status_code
                if status_code != 404:
                    time.sleep(150)
            except:
                logger.warning("Request unsuccessful")

    spot_termination_callback = SpotTermination()

    callbacks = [checkpoint_callback, epoch_results_callback, spot_termination_callback, sample_callback, batch_callback]
    return callbacks

def main(args):
    volumedir = args.volumedir
    datadir = os.path.join(volumedir, args.datadir)
    checkpointdir = os.path.join(volumedir, args.checkpointdir)
    checkpointnames = args.checkpointnames
    mini_batch_size = args.minibatchsize
    learning_rate = args.learningrate
    n_a = args.hiddensize
    num_epochs = args.numepochs
    hdlr = logging.FileHandler(os.path.join(volumedir, "training_output_%d.log" % time.time()))
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)

    train, test = load_datasets(datadir)
    m = len(train)
    chars = set()
    step = args.step
    maxlen = args.seqlength
    #for msg in train:
    #    chars = chars.union(set(msg))
    #chars = sorted(list(chars))
    #chars = ['\t', '\n', ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~', '*']
    chars = ['\n', ' ', '!', '"', ',', '.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', '?', '@', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '~', '*']
    char_to_ix = {c: i for i, c in enumerate(chars)}

    if os.path.isdir(checkpointdir) and any(glob.glob(os.path.join(checkpointdir, '*'))):
        model, epoch_number = load_checkpoint_model(checkpointdir, checkpointnames)
    else:
        model = create_model(chars, n_a, maxlen, learning_rate)
        epoch_number = 0
    metrics = []
    data = "".join(train)
    nummsgs = math.floor(len(data) / maxlen)
    if len(data) % maxlen != 0:
        nummsgs += 1
    nummsgs = len(data) - maxlen
    X = np.zeros((nummsgs, maxlen, len(chars)))
    Y = np.zeros((nummsgs, len(chars)))
    msgs = 0
    for i in range(0,nummsgs, step):
        last_ix = min(i+maxlen, len(data)-1)
        for t,c in enumerate(data[i:last_ix]):
            char_index = get_ix_from_char(char_to_ix, chars, c)
            X[msgs, t, char_index] = 1
        Y[msgs, get_ix_from_char(char_to_ix, chars, data[last_ix])] = 1
        msgs+=1
    callbacks = get_callbacks(volumedir, checkpointdir, checkpointnames, chars, char_to_ix, data, model)
    model.fit(X,Y,
              batch_size=mini_batch_size,
              epochs=num_epochs,
              initial_epoch=epoch_number,
              callbacks=callbacks)

def parse_args():
    parser= argparse.ArgumentParser()
    parser.add_argument("--loadmodel", action="store_true", default=False)
    parser.add_argument("--datadir", default="data/")
    parser.add_argument("--volumedir", default="/training/")
    parser.add_argument("--checkpointdir", default="checkpoints/")
    parser.add_argument("--checkpointnames", default="nodle_char_model.{epoch:03d}.h5")
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument("--hiddensize", type=int, default=128)
    parser.add_argument("--minibatchsize", type=int, default=1024)
    parser.add_argument("--numepochs", type=int, default=60)
    parser.add_argument("--seqlength", type=int, default=40)
    parser.add_argument("--learningrate", type=float, default=0.01)
    return parser.parse_args()

if __name__=="__main__":
    args = parse_args()
    main(args)
