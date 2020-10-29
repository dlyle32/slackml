import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import LambdaCallback, ModelCheckpoint, CSVLogger
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM, Dropout, Embedding
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.models import load_model
from tensorflow.keras import regularizers
import numpy as np
import nltk
import random
import math
import logging
from models.helpers import get_ix_from_token, token_to_oh, oh_to_token

logger = logging.getLogger('keras_char_lm')

class WordLanguageModelEmbeddingsBuilder:
    def __init__(self, args):
        self.learning_rate = args.learningrate
        self.dropout_rate = args.dropoutrate
        self.reg_factor = args.regfactor
        self.n_a = args.hiddensize
        self.step = args.step
        self.maxlen = args.seqlength
        #self.tokenizer = nltk.RegexpTokenizer("\S+|\n+")
        self.tokenizer = nltk.RegexpTokenizer("\,|\.|&gt|\¯\\\_\(\ツ\)\_\/\¯|\<\@\w+\>|\:\w+\:|\/gif|_|\"| |\w+\'\w+|\w+|\n")

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
        x = Input(shape=(self.maxlen), name="input")
        out = Embedding(vocab_size, 100, input_length=self.maxlen)(x)
        out = LSTM(self.n_a, return_sequences=True, kernel_regularizer=reg, recurrent_regularizer=reg)(out)
        out = Dropout(self.dropout_rate)(out)
        out = LSTM(self.n_a, kernel_regularizer=reg, recurrent_regularizer=reg)(out)
        out = Dropout(self.dropout_rate)(out)
        out = Dense(self.n_a, activation='relu', kernel_regularizer=reg)(out)
        out = Dense(vocab_size, activation='softmax', kernel_regularizer=reg)(out)
        model = keras.Model(inputs=x, outputs=out)
        opt = RMSprop(learning_rate=self.learning_rate, clipvalue=3)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=["accuracy"])
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
        output += "->"
        mintokens = 15
        maxtokens = 100
        i = 0
        while i < maxtokens and (i < mintokens or token_ix != reverse_token_map['\n']):
            x = np.zeros((1, maxlen))
            x[0] = [get_ix_from_token(reverse_token_map, token) for token in inpt]
            preds = model.predict(x, verbose=0)[0]
            token_ix = np.random.choice(range(vocab_size), p=preds.ravel())
            new_token = vocab[token_ix]
            output += new_token
            inpt = inpt[1:] + [new_token]
            i+=1
        logger.info("\n" + output)
        return output

    def get_input_sequences(self, tokens, reverse_token_map):
        nummsgs = math.floor((len(tokens) - self.maxlen) / self.step) + 1
        j = 0
        seqs = []
        for i in range(0, len(tokens) - self.maxlen, self.step):
            last_ix = min(i + self.maxlen, len(tokens)-1)
            Xseq = [get_ix_from_token(reverse_token_map, token) for token in tokens[i:last_ix]]
            Yix = get_ix_from_token(reverse_token_map, tokens[last_ix])
            seqs.append((Xseq, Yix))
            j += 1
        return seqs

    def build_input_vectors(self, seqs, vocab, reverse_token_map):
        X = np.zeros((len(seqs), self.maxlen))
        Y = np.zeros((len(seqs), len(vocab)))
        j = 0
        for Xseq, Yix in seqs:
            X[j, :] = Xseq
            Y[j, :] = token_to_oh(Yix, len(vocab))
            j+=1
        return X, Y


