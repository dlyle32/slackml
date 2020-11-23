import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import LambdaCallback, ModelCheckpoint, CSVLogger
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM, Dropout, Embedding
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.models import load_model
from tensorflow.keras import regularizers
import numpy as np
import nltk
import random
import math
import logging
from models.helpers import get_ix_from_token, token_to_oh, oh_to_token, char_padded

class PerMessageLanguageModelBuilder:
    def __init__(self, args):
        self.learning_rate = args.learningrate
        self.dropout_rate = args.dropoutrate
        self.reg_factor = args.regfactor
        self.n_a = args.hiddensize
        self.step = args.step
        self.seqlen = args.seqlength
        self.minlen = args.minlength
        self.maxlen = args.maxlength
        self.embedding = args.embedding

        #self.tokenizer = nltk.RegexpTokenizer("\S+|\n+")
        # self.tokenizer = nltk.RegexpTokenizer("\<\@\w+\>|\:\w+\:|\/gif|_|\"| |\w+\'\w+|\w+|\n")
        self.tokenizer = nltk.RegexpTokenizer("\,|\.|\¯\\\_\(\ツ\)\_\/\¯|\<\@\w+\>|\:\w+\:|\/gif|_|\"| |\w+\'\w+|\w+|\n")

    def sliding_window_tokenize(self, data, freq_threshold=5):
        #tokens = " ".join(data).split(" ")
        tokens = self.tokenizer.tokenize("START ".join(data))
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

    def tokenize(self, data, freq_threshold=5, sliding_window=True):
        if sliding_window:
            return self.sliding_window_tokenize(data, freq_threshold)
        #tokens = " ".join(data).split(" ")
        # tokens = self.tokenizer.tokenize("<START>".join(data))
        tokens = []
        token_counts = {}
        for line in data:
            linetokens = self.tokenizer.tokenize(line)
            for t in linetokens:
                if t not in token_counts.keys():
                    token_counts[t] = 1
                else:
                    token_counts[t] += 1
            tokens.append(linetokens)

        # filter out messages that are too short or too long
        tokens = list(filter(lambda a: len(a) >= self.minlen and len(a) <= self.maxlen, tokens))

        freq_filtered = filter(lambda elem: elem[1] >= freq_threshold, token_counts.items())
        vocab = sorted([elem[0] for elem in list(freq_filtered)])
        #vocab = sorted(list(set(tokens)))
        vocab += ["<UNK>"]
        vocab += ["START"]
        reverse_token_map = {t: i for i, t in enumerate(vocab)}
        return tokens, vocab, reverse_token_map

    def create_model(self, vocab):
        vocab_size = len(vocab)
        reg = regularizers.l2(self.reg_factor)
        tf.keras.backend.set_floatx('float64')
        if self.embedding:
            x = Input(shape=(self.seqlen), name="input")
            out = Embedding(vocab_size, 256, input_length=self.seqlen)(x)
            out = LSTM(self.n_a, return_sequences=True, kernel_regularizer=reg, recurrent_regularizer=reg)(out)
        else:
            x = Input(shape=(self.seqlen, vocab_size), name="input")
            out = LSTM(self.n_a, return_sequences=True, kernel_regularizer=reg, recurrent_regularizer=reg)(x)
        out = Dropout(self.dropout_rate)(out)
        # out = LSTM(self.n_a, return_sequences=True, kernel_regularizer=reg, recurrent_regularizer=reg)(out)
        # out = Dropout(self.dropout_rate)(out)
        out = Dense(self.n_a, activation='relu', kernel_regularizer=reg)(out)
        out = Dense(vocab_size, activation='softmax', kernel_regularizer=reg)(out)
        model = keras.Model(inputs=x, outputs=out)
        return model

    def sample(self, model, tokens, vocab, reverse_token_map):
        seqlen = self.seqlen
        vocab_size = len(vocab)
        token_ix = -1
        inpt = ["START" for i in range(self.seqlen)]
        output = ""
        mintokens = 15
        maxtokens = 100
        i = 0
        while i < maxtokens and (i < mintokens or token_ix != reverse_token_map['START']):
            if self.embedding:
                x = np.zeros((1, seqlen))
                x[0] = [get_ix_from_token(reverse_token_map, token) for token in inpt]
            else:
                x = np.zeros((1, seqlen, vocab_size))
                x[0] = [token_to_oh(get_ix_from_token(reverse_token_map, token), vocab_size) for token in inpt]
            preds = model.predict(x, verbose=0)[0][min(i,self.seqlen-1)]
            token_ix = np.random.choice(range(vocab_size), p=preds.ravel())
            while token_ix == reverse_token_map["<UNK>"]:
                token_ix = np.random.choice(range(vocab_size), p=preds.ravel())
            new_token = vocab[token_ix]
            output += new_token
            if(i+1 < len(inpt)):
                inpt[i+1] = new_token
            else:
                inpt = inpt[1:] + [new_token]
            i+=1
        return output

    def get_full_input_sequences(self, tokens, reverse_token_map):
        tokens = sorted(tokens, key=lambda a: len(a))
        left = 0
        right = len(tokens)-1
        seqs = []
        Xseq = []
        Yseq = []
        while left < right:
            if len(Yseq) + len(tokens[right][:self.seqlen]) <= self.seqlen:
                newSeq = tokens[right][:self.seqlen]
                Yseq += newSeq
                Xseq += ["START"] + newSeq[:-1]
                right -= 1
            if len(Yseq) + len(tokens[left][:self.seqlen]) <= self.seqlen:
                newSeq = tokens[left][:self.seqlen]
                Yseq += newSeq
                Xseq += ["START"] + newSeq[:-1]
                left += 1
            else:
                paddedX = [get_ix_from_token(reverse_token_map, token) for token in char_padded(Xseq, " ", self.seqlen)]
                paddedY = [get_ix_from_token(reverse_token_map, token) for token in char_padded(Yseq, " ", self.seqlen)]
                seqs.append((paddedX, paddedY))
                Yseq = []
                Xseq = []
        paddedX = [get_ix_from_token(reverse_token_map, token) for token in char_padded(Xseq, " ", self.seqlen)]
        paddedY = [get_ix_from_token(reverse_token_map, token) for token in char_padded(Yseq, " ", self.seqlen)]
        seqs.append((paddedX, paddedY))
        return seqs


    def sliding_window_input_sequences(self, tokens, reverse_token_map):
        nummsgs = math.floor((len(tokens) - self.seqlen) / self.step) + 1
        seqs = []
        x0 = "START"
        for i in range(0, len(tokens) - self.seqlen, self.step):
            x0 = "START" if i == 0 else tokens[i-1]
            last_ix = min(i + self.seqlen, len(tokens) - 1)
            padded_sequence = char_padded(tokens[i:last_ix], " ", self.seqlen)
            Yseq = [get_ix_from_token(reverse_token_map, token) for token in padded_sequence]
            Xseq = [get_ix_from_token(reverse_token_map, x0)] + Yseq[:-1]
            seqs.append((Xseq, Yseq))
        return seqs

    def get_input_sequences(self, tokens, reverse_token_map, full=True, sliding_window=True):
        if full:
            return self.get_full_input_sequences(tokens, reverse_token_map)
        if sliding_window:
            return self.sliding_window_input_sequences(tokens, reverse_token_map)
        nummsgs = math.floor((len(tokens) - self.seqlen) / self.step) + 1
        seqs = []
        for line in tokens:
            padded_sequence = char_padded(line[:self.seqlen], " ", self.seqlen)
            Yseq = [get_ix_from_token(reverse_token_map, token) for token in padded_sequence]
            Xseq = [get_ix_from_token(reverse_token_map, " ")] + Yseq[:-1]
            seqs.append((Xseq, Yseq))
        return seqs

    def build_input_vectors(self, seqs, vocab, reverse_token_map):
        Y = np.zeros((len(seqs), self.seqlen, len(vocab)))
        if self.embedding:
            X = np.zeros((len(seqs), self.seqlen))
        else:
            X = np.zeros((len(seqs), self.seqlen, len(vocab)))
        j = 0
        for Xseq, Yseq in seqs:
            if self.embedding:
                X[j, :] = Xseq
            else:
                X[j, :, :] = [token_to_oh(ix, len(vocab)) for ix in Xseq]
            Y[j, :, :] = [token_to_oh(ix, len(vocab)) for ix in Yseq]
            j+=1
        return X, Y


