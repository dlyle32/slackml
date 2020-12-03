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
import os
import logging
from models.helpers import get_ix_from_token, token_to_oh, oh_to_token, char_padded


def load_vocab(fname):
    with open(fname, "r") as fp:
        vocab = fp.read().split("\t")
    return vocab

def get_lstm_weights(modelpath):
    model = load_model(modelpath)
    return model.get_layer(name="lstm").get_weights()


class ContextualLanguageModelBuilder:
    def __init__(self, args):
        self.learning_rate = args.learningrate
        self.dropout_rate = args.dropoutrate
        self.reg_factor = args.regfactor
        self.n_a = args.hiddensize
        self.step = args.step
        self.seqlen = args.seqlength
        self.context_len = args.conlength
        self.minlen = args.minlength
        self.maxlen = args.maxlength
        self.embedding = args.embedding
        self.vocabfile = args.vocabfile
        self.modelweightspath = args.modelweightspath

        #self.tokenizer = nltk.RegexpTokenizer("\S+|\n+")
        # self.tokenizer = nltk.RegexpTokenizer("\<\@\w+\>|\:\w+\:|\/gif|_|\"| |\w+\'\w+|\w+|\n")
        self.tokenizer = nltk.RegexpTokenizer("\,|\.|\¯\\\_\(\ツ\)\_\/\¯|\<\@\w+\>|\:\w+\:|\/gif|_|\"| |\w+\'\w+|\w+|\n")

    def tokenize(self, data, freq_threshold=5, sliding_window=True):
        vocab = load_vocab(self.vocabfile)
        tokens = []
        token_counts = {}
        for context, target in data:
            context_tokens = self.tokenizer.tokenize(context)
            target_tokens = self.tokenizer.tokenize(target)
            tokens.append((context_tokens, target_tokens))
        reverse_token_map = {t: i for i, t in enumerate(vocab)}
        return tokens, vocab, reverse_token_map

    def create_model(self, vocab):
        vocab_size = len(vocab)
        reg = regularizers.l2(self.reg_factor)
        tf.keras.backend.set_floatx('float64')

        encoder_input = Input(shape=(None, vocab_size), name="encoder_input")
        encoder_lstm = LSTM(self.n_a, return_state=True, kernel_regularizer=reg, recurrent_regularizer=reg)
        encoder_output, state_h, state_c =encoder_lstm(encoder_input)
        encoder_states = [state_h, state_c]

        decoder_lstm_weights = get_lstm_weights(self.modelweightspath)
        decoder_input = Input(shape=(None, vocab_size), name="decoder_input")
        decoder_lstm = LSTM(self.n_a, weights=decoder_lstm_weights, return_sequences=True, return_state=True, kernel_regularizer=reg, recurrent_regularizer=reg)
        decoder_output, _, _ = decoder_lstm(decoder_input, initial_state=encoder_states)
        decoder_output = Dropout(self.dropout_rate)(decoder_output)
        decoder_dense = Dense(vocab_size, activation='softmax', kernel_regularizer=reg)
        decoder_output = decoder_dense(decoder_output)

        model = keras.Model(inputs=[encoder_input, decoder_input], outputs=decoder_output)

        return model

    def build_sample_model(self, model):
        encoder_input = model.input[0]
        encoder_lstm = model.layers[2]
        encoder_output, enc_state_h, enc_state_c = encoder_lstm(encoder_input)
        encoder_states = [enc_state_h, enc_state_c]
        self.encoder_model = keras.Model(encoder_input, encoder_states)

        decoder_input = model.input[1]
        decoder_state_input_h = Input(shape=(self.n_a,), name="decoder_state_input_h")
        decoder_state_input_c = Input(shape=(self.n_a,), name="decoder_state_input_c")
        decoder_states_input = [decoder_state_input_h, decoder_state_input_c]
        decoder_lstm = model.layers[3]
        decoder_output, dec_state_h, dec_state_c = decoder_lstm(decoder_input, initial_state=decoder_states_input)
        decoder_states = [dec_state_h, dec_state_c]
        decoder_dense = model.layers[4]
        decoder_output = decoder_dense(decoder_output)
        self.decoder_model = keras.Model([decoder_input] + decoder_states_input, [decoder_output] + decoder_states)

    def sample(self, model, tokens, vocab, reverse_token_map):
        seqlen = self.seqlen
        vocab_size = len(vocab)
        tf.keras.backend.set_floatx('float64')

        if(not hasattr(self, "encoder_model") or not hasattr(self, "decoder_model")):
            self.build_sample_model(model)

        i = random.randint(0, len(tokens))
        context = tokens[i][0]
        actual_message = tokens[i][1]
        encoder_input = [get_ix_from_token(reverse_token_map, token) for token in context]
        encoder_input = [token_to_oh(ix, len(vocab)) for ix in encoder_input]
        encoder_input = np.array([encoder_input])
        encoder_state = self.encoder_model.predict(encoder_input)

        inpt = ["START" for i in range(self.seqlen)]
        output = ""
        token_ix = -1
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
            preds = self.decoder_model.predict([x] + encoder_state, verbose=0)[0]
            preds = preds[0][min(i,self.seqlen-1)]
            probs = preds.ravel()
            token_ix = np.random.choice(range(vocab_size), p=probs)
            while token_ix == reverse_token_map["<UNK>"] or (token_ix == reverse_token_map[" "] and output[-1] == " "):
                token_ix = np.random.choice(range(vocab_size), p=preds.ravel())
            new_token = vocab[token_ix]
            output += new_token
            if(i+1 < len(inpt)):
                inpt[i+1] = new_token
            else:
                inpt = inpt[1:] + [new_token]
            i+=1

        print(context)
        print(output)
        print(actual_message)
        print(actual_message[-1] == "\n")
        print(len(output))
        return output

    def get_input_sequences(self, tokens, reverse_token_map, full=True, sliding_window=True):
        nummsgs = math.floor((len(tokens) - self.seqlen) / self.step) + 1
        seqs = []
        for context, target in tokens:
            padded_sequence = char_padded(target[:self.seqlen], " ", self.seqlen)
            decoder_output = [get_ix_from_token(reverse_token_map, token) for token in padded_sequence]
            decoder_input = [get_ix_from_token(reverse_token_map, "START")] + decoder_output[:-1]

            encoder_input = char_padded(context[:self.context_len], " ", self.context_len)
            encoder_input = [get_ix_from_token(reverse_token_map, token) for token in encoder_input]
            seqs.append((encoder_input, decoder_input, decoder_output))
        return seqs

    def build_input_vectors(self, seqs, vocab, reverse_token_map):
        decoder_output = np.zeros((len(seqs), self.seqlen, len(vocab)))
        if self.embedding:
            decoder_input = np.zeros((len(seqs), self.seqlen))
        else:
            decoder_input = np.zeros((len(seqs), self.seqlen, len(vocab)))
        encoder_input = np.zeros((len(seqs), self.context_len, len(vocab)))
        j = 0
        for context_input, target_input, target_output in seqs:
            if self.embedding:
                decoder_input[j, :] = target_input
            else:
                decoder_input[j, :, :] = [token_to_oh(ix, len(vocab)) for ix in target_input]
            decoder_output[j, :, :] = [token_to_oh(ix, len(vocab)) for ix in target_output]
            encoder_input[j, :, :] = [token_to_oh(ix, len(vocab)) for ix in context_input]
            j+=1
        return [encoder_input, decoder_input], decoder_output


