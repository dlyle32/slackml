import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import LambdaCallback, ModelCheckpoint, CSVLogger
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.models import load_model
from tensorflow.keras import regularizers
import numpy as np
import nltk
import random
import math
import datetime
import time
import os
import argparse
import logging
from data.load import load_datasets

logger = logging.getLogger('keras_char_lm')

class WordLanguageModelBuilder:
    def __init__(self, args):
        self.learning_rate = args.learningrate
        self.dropout_rate = args.dropoutrate
        self.reg_factor = args.regfactor
        self.n_a = args.hiddensize
        self.step = args.step
        self.maxlen = args.seqlength
        #self.tokenizer = nltk.RegexpTokenizer("\S+|\n+")
        self.tokenizer = nltk.RegexpTokenizer("&gt|\¯\\\_\(\ツ\)\_\/\¯|\<\@\w+\>|\:\w+\:|\/gif|_|\"| |\w+\'\w+|\w+|\n")

    def tokenize(self, data, freq_threshold=5):
        #tokens = " ".join(data).split(" ")
        tokens = self.tokenizer.tokenize(" ".join(data))
        token_counts = {}
        for t in tokens:
            if t not in token_counts.keys():
                token_counts[t] = 1
            else:
                token_counts[t] += 1
        freq_filtered = filter(lambda elem: elem[1] >= freq_threshold, token_counts.items())
        vocab = sorted([elem[0] for elem in list(freq_filtered)])
        #vocab = sorted(list(set(tokens)))
        vocab += ["<UNK>"]
        reverse_token_map = {t: i for i, t in enumerate(vocab)}
        return tokens, vocab, reverse_token_map

    def create_model(self, vocab):
        vocab_size = len(vocab)
        reg = regularizers.l2(self.reg_factor)
        tf.keras.backend.set_floatx('float64')
        x = Input(shape=(self.maxlen, vocab_size), name="input")
        out = LSTM(self.n_a, return_sequences=True, kernel_regularizer=reg, recurrent_regularizer=reg)(x)
        out = Dropout(self.dropout_rate)(out)
        out = LSTM(self.n_a, kernel_regularizer=reg, recurrent_regularizer=reg)(out)
        out = Dropout(self.dropout_rate)(out)
        out = Dense(vocab_size, activation='softmax', kernel_regularizer=reg)(out)
        model = keras.Model(inputs=x, outputs=out)
        opt = RMSprop(learning_rate=self.learning_rate, clipvalue=3)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=["accuracy"])
        model.summary(print_fn=logger.info)
        return model

    def sample(self, model, tokens, vocab, reverse_token_map):
        maxlen = self.maxlen
        vocab_size = len(vocab)
        token_ix = -1
        i = random.randint(0, len(tokens) - maxlen - 1)
        inpt = tokens[i:i + maxlen]
        output = ""
        for t in inpt:
            output += t
        while token_ix != reverse_token_map['\n']:
            x = np.zeros((1, maxlen, vocab_size))
            x[0] = [token_to_oh(get_ix_from_token(reverse_token_map, token), vocab_size) for token in inpt]
            preds = model.predict(x, verbose=0)[0]
            token_ix = np.random.choice(range(vocab_size), p=preds.ravel())
            new_token = vocab[token_ix]
            output += new_token
            inpt = inpt[1:] + [new_token]
        logger.info("\n" + output[:maxlen] + "->" + output[maxlen:])
        return output

    def format_input(self, tokens, vocab, reverse_token_map):
        nummsgs = math.floor(len(tokens) - self.maxlen / self.step) + 1
        X = np.zeros((nummsgs, self.maxlen, len(vocab)))
        Y = np.zeros((nummsgs, len(vocab)))
        j = 0
        for i in range(0, len(tokens) - self.maxlen, self.step):
            last_ix = min(i + self.maxlen, len(tokens)-1)
            X[j, :, :] = [token_to_oh(reverse_token_map[token], len(vocab)) for token in tokens[i:last_ix]]
            Y[j, :] = token_to_oh(reverse_token_map[tokens[last_ix]], len(vocab))
            j += 1
        return X, Y


def get_ix_from_token(reverse_token_map, token):
    if token in reverse_token_map.keys():
        return reverse_token_map[token]
    else:
        return reverse_token_map["<UNK>"]

def oh_to_token(vocab, oh):
    tokens = [vocab[i] for i,c in enumerate(oh) if c == 1]
    return "0" if len(tokens) == 0 else tokens[0]

def token_to_oh(index, vocab_size):
    x = np.zeros((vocab_size))
    x[index] = 1
    return x

def get_callbacks(volumedir, checkpointdir, checkpointnames, timestamp, sample_func):
    today_date = datetime.datetime.today().strftime('%Y-%m-%d')
    if not os.path.isdir(checkpointdir):
        os.makedirs(checkpointdir)
    filepath = os.path.join(checkpointdir, checkpointnames)
    checkpoint_callback = ModelCheckpoint(filepath=filepath,
                                          save_weights_only=False,
                                          monitor='val_loss')
    # Loss history callback
    epoch_results_callback = CSVLogger(
        os.path.join(volumedir, 'training_log_{}_{:d}.csv'.format(today_date, timestamp)),
        append=True)

    sample_callback = LambdaCallback(on_epoch_end=lambda epoch, logs: sample_func())

    callbacks = [checkpoint_callback, epoch_results_callback, sample_callback]
    return callbacks

def main(args):
    datadir = os.path.join(args.volumedir, args.datadir)
    train, test = load_datasets(datadir)
    train = train[:min(len(train), args.datacap)]

    modelBuilder = WordLanguageModelBuilder(args)
    tokens, vocab, reverse_token_map = modelBuilder.tokenize(train, freq_threshold=args.freqthreshold)

    timestamp = int(time.time())
    init_epoch = 0
    if args.loadmodel and os.path.exists(args.loadmodel):
        modelpath = args.loadmodel
        timestamp = int(modelpath.split(".")[1])
        init_epoch = int(modelpath.split(".")[2])
        model = load_model(modelpath), init_epoch
    else:
        model = modelBuilder.create_model(vocab)

    hdlr = logging.FileHandler(os.path.join(args.volumedir, "training_output_%d.log" % timestamp))
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)

    X, Y = modelBuilder.format_input(tokens, vocab, reverse_token_map)

    checkpointdir = os.path.join(args.volumedir, args.checkpointdir)
    checkpointnames = args.checkpointnames % timestamp
    sample_func = lambda : modelBuilder.sample(model, tokens, vocab, reverse_token_map)
    callbacks = get_callbacks(args.volumedir, checkpointdir, checkpointnames, timestamp, sample_func)

    model.fit(X, Y,
               batch_size=args.minibatchsize,
               epochs=args.numepochs,
               initial_epoch=init_epoch,
               validation_split=0.2,
               shuffle=True,
               callbacks=callbacks)


def parse_args():
    parser= argparse.ArgumentParser()
    parser.add_argument("--loadmodel", type=str)
    parser.add_argument("--datadir", default="data/")
    parser.add_argument("--volumedir", default="/training/")
    parser.add_argument("--checkpointdir", default="checkpoints/")
    parser.add_argument("--checkpointnames", default="nodle_char_model.%d.{epoch:03d}.h5")
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument("--hiddensize", type=int, default=128)
    parser.add_argument("--minibatchsize", type=int, default=256)
    parser.add_argument("--numepochs", type=int, default=25)
    parser.add_argument("--seqlength", type=int, default=40)
    parser.add_argument("--learningrate", type=float, default=0.01)
    parser.add_argument("--dropoutrate", type=float, default=0.2)
    parser.add_argument("--regfactor", type=float, default=0.01)
    parser.add_argument("--datacap", type=int, default=10000)
    parser.add_argument("--freqthreshold", type=int, default=5)
    return parser.parse_args()

if __name__=="__main__":
    args = parse_args()
    main(args)
