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
from tokenizers.sliding_window import SlidingWindowTokenizer
import random
import math
import logging
from models.helpers import get_ix_from_token, token_to_oh, oh_to_token, char_padded, create_oh

def attention_head(q, k, v, dropout, mask=None):
    # Q dot K scaled -> softmax = attention parameters -> ap * V summed = output
    dk = k.shape[-1]
    ndim = len(k.shape)
    perm = list(range(ndim))
    perm[-2] = ndim - 1
    perm[-1] = ndim - 2

    # use perm to transpose final two dimensions of key vector
    attn_factor = tf.matmul(q, tf.transpose(k, perm=perm)) / (dk ** 0.5)
    if mask is not None:
        attn_factor[mask == False] == -1e9
    attn_factor = keras.layers.Softmax()(attn_factor)
    attn_factor = keras.layers.Dropout(dropout)(attn_factor)
    return tf.matmul(attn_factor, v), attn_factor


def multihead_attention(q, k, v, h, n_a, m, reg, dropout, mask=None):
    dim = n_a // h
    Wq = keras.layers.Dense(n_a, kernel_regularizer=reg)
    Wk = keras.layers.Dense(n_a, kernel_regularizer=reg)
    Wv = keras.layers.Dense(n_a, kernel_regularizer=reg)
    Wo = keras.layers.Dense(n_a, kernel_regularizer=reg)
    if mask is not None:
        mask = tf.expand_dims(mask, 1)
        mask = tf.repeat(mask, 4, 1)

    seqlen = q.shape[1]
    shape = [m, seqlen, h, dim]
    Q = Wq(q)
    Q = tf.reshape(Q, shape)
    Q = tf.transpose(Q, perm=[0, 2, 1, 3]) # reshape for heads x seqlen x model_dim
    K = tf.transpose(tf.reshape(Wk(k), shape), perm=[0, 2, 1, 3])
    V = tf.transpose(tf.reshape(Wv(v), shape), perm=[0, 2, 1, 3])

    C, attn_factor = attention_head(Q, K, V, dropout, mask)
    C = tf.reshape(tf.transpose(C, perm=[0, 2, 1, 3]), (m, seqlen, n_a))

    return Wo(C)

def positional_encoding(seqlen, d_emb):
    pos_enc = np.array(
        [
            [pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)]
            if pos != 0
            else np.zeros(d_emb)
            for pos in range(seqlen)
        ]
    )
    pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2])  # dim 2i
    pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2])  # dim 2i+1
    return keras.layers.Dropout(0.10)(pos_enc)

class AttentionModelBuilder:
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
        self.embeddingsize = args.embeddingsize
        self.ffdim = args.ffdim

        self.tokenizer = SlidingWindowTokenizer(self.seqlen, self.step, args.freqthreshold)

    def tokenize(self, data, freq_threshold=None):
        return self.tokenizer.tokenize(data)

    def get_masked_datasets(self, dataset, mask_token_ix, vocab_size, pad_mask=None):
        #     dataset = dataset.to_numpy().reshape((dataset.shape[0],1))

        # 15% BERT masking
        if pad_mask is None:
            pad_mask = np.zeros(dataset.shape)
        inp_mask = np.logical_and(np.random.rand(*dataset.shape) < 0.15, pad_mask == False)

        labels = -1 * np.ones(dataset.shape)
        labels[inp_mask] = 0

        masked_dataset = np.copy(dataset)

        # 90% of mask indices get set to mask token
        inp_mask = np.logical_and(inp_mask, np.random.rand(*dataset.shape) < 0.9)
        masked_dataset[inp_mask] = mask_token_ix

        # 10% of mask indices get set to a random token (and 10% remain unchanged)
        inp_mask = np.logical_and(inp_mask, np.random.rand(*dataset.shape) < 1 / 9)
        masked_dataset[inp_mask] = np.random.randint(0, mask_token_ix, inp_mask.sum())

        # To be used to scale loss function to only count masked tokens
        loss_mask = np.ones(dataset.shape, dtype=int)
        loss_mask[labels == -1] = 0

        # The Y labels are just the original dataset
        y_labels = np.copy(dataset)

        return masked_dataset, y_labels, loss_mask

    def get_input_sequences(self, tokens, reverse_token_map):
        return self.tokenizer.get_input_sequences(tokens,reverse_token_map)

    def build_masked_input_vectors(self, seqs, vocab, reverse_token_map):
        mask_token_ix = reverse_token_map["<MASK>"]
        seqs = np.asarray(seqs)
        masked_ds, masked_y, sample_weights = self.get_masked_datasets(seqs, mask_token_ix, len(vocab))
        return masked_ds, masked_y, sample_weights

    def build_input_vectors(self, seqs, vocab, reverse_token_map):
        X = np.zeros((len(seqs), self.seqlen))
        Y = np.zeros((len(seqs), self.seqlen))

        for i, (Xseq, Yseq) in enumerate(seqs):
            X[i, :] = Xseq
            Y[i, :] = Yseq
        return X,Y, None


    def transformer_encoder(self, x, i, reg, mask=None):
        # Embedding, self-attention, dropout, residual layerNorm, ffn, residual layerNorm
        m = tf.shape(x)[0]

        attn_layer = keras.layers.MultiHeadAttention(4, self.n_a//4)
        attn_out = attn_layer(x,x,x, attention_mask=mask)
        # attn_out = multihead_attention(x, x, x, 4, self.n_a, m, reg, self.dropout_rate, mask=mask)
        #     attn_out = tf.reshape(out, (m,seqlen*n_a))

        x = keras.layers.LayerNormalization(epsilon=1e-6, name="encoder_{}/attn_norm".format(i))(x + attn_out)

        # Feed-forward layer
        ffn = keras.Sequential(
            [
                keras.layers.Dense(self.ffdim, kernel_regularizer=reg, activation="relu"),
                keras.layers.Dense(self.n_a, kernel_regularizer=reg),
            ],
            name="encoder_{}/ffn".format(i),
        )

        ffn_out = ffn(x)
        ffn_out = keras.layers.Dropout(self.dropout_rate, name="encoder_{}/ffn_dropout".format(i))(ffn_out)

        x = keras.layers.LayerNormalization(epsilon=1e-6, name="encoder_{}/ffn_norm".format(i))(x + ffn_out)
        return x

    def subsequent_mask(self, shape):
        "Mask out subsequent positions."
        subsequent_mask = np.triu(np.ones(shape), k=1).astype('uint8')
        return subsequent_mask == 0

    def transformer_decoder(self, encoder_out, targets, reg, mask=None):
        # Embedding, self attention, encoder attention, dropout, residual layerNorm, ffn, dropout, res norm, dense softmax
        m = tf.shape(x)[0]

        attn_layer_1 = keras.layers.MultiHeadAttention(4, self.n_a // 4)
        attn_out_1 = attn_layer_1(targets, targets, targets, attention_mask=mask)
        # attn_out = multihead_attention(x, x, x, 4, self.n_a, m, reg, self.dropout_rate, mask=mask)
        #     attn_out = tf.reshape(out, (m,seqlen*n_a))

        attn_out_1 = keras.layers.LayerNormalization(epsilon=1e-6, name="decoder/attn_norm_1")(targets + attn_out_1)

        attn_layer_2 = keras.layers.MultiHeadAttention(4, self.n_a // 4)
        attn_out_2 = attn_layer_2(encoder_out, attn_out_1, attn_out_1, attention_mask=mask)
        # attn_out = multihead_attention(x, x, x, 4, self.n_a, m, reg, self.dropout_rate, mask=mask)
        #     attn_out = tf.reshape(out, (m,seqlen*n_a))

        attn_out_2 = keras.layers.LayerNormalization(epsilon=1e-6, name="decoder/attn_norm_2")(attn_out_1 + attn_out_2)

        # Feed-forward layer
        ffn = keras.Sequential(
            [
                keras.layers.Dense(self.n_a, kernel_regularizer=reg, activation="relu"),
                keras.layers.Dense(self.n_a, kernel_regularizer=reg),
            ],
            name="decoder/ffn",
        )

        ffn_out = ffn(attn_out_2)
        ffn_out = keras.layers.Dropout(self.dropout_rate, name="decoder/ffn_dropout")(ffn_out)

        x = keras.layers.LayerNormalization(epsilon=1e-6, name="decoder/ffn_norm")(attn_out_2 + ffn_out)
        return x

    def create_model(self, vocab, mask=None):
        vocab_size = len(vocab)
        reg = keras.regularizers.l2(self.reg_factor)

        inpt = keras.layers.Input(shape=(self.seqlen), name="input")
        targets = keras.layers.Input(shape=(self.seqlen), name="targets")
        out = keras.layers.Embedding(vocab_size, self.n_a, input_length=self.seqlen)(inpt)
        target_emb = keras.layers.Embedding(vocab_size, self.n_a, input_length=self.seqlen)(targets)
        pos_enc = positional_encoding(self.seqlen, self.n_a)
        pos_emb = keras.layers.Embedding(
            input_dim=self.seqlen,
            output_dim=self.n_a,
            weights=[pos_enc],
            name="position_embedding",
        )(tf.range(start=0, limit=self.seqlen, delta=1))
        encoder_out = out + pos_emb
        target_emb = target_emb + pos_emb
        m = tf.shape(out)[0]
        mask = self.subsequent_mask(self.seqlen)
        for i in range(6):
            encoder_out = self.transformer_encoder(encoder_out, i, reg, mask)
        # decoder_out = self.transformer_decoder(encoder_out, target_emb, reg, mask)
        out = keras.layers.Dense(self.n_a, activation="relu", kernel_regularizer=reg)(encoder_out)
        out = keras.layers.Dense(vocab_size, activation="softmax", kernel_regularizer=reg)(out)

        # masked_model = MaskedLanguageModel(inputs=inpt, outputs=masked_out)
        # return masked_model

        model = keras.Model(inputs=inpt, outputs=out)
        return model

    def sample(self, model, tokens, vocab, reverse_token_map):
        seqlen = self.seqlen
        vocab_size = len(vocab)
        token_ix = -1
        inpt = ["<START>" for i in range(self.seqlen)]
        output = ""
        mintokens = 15
        maxtokens = 100
        i = 1
        while i < maxtokens and (i < mintokens or token_ix != reverse_token_map['<START>']):
            # x = np.zeros((1, seqlen))
            x = [get_ix_from_token(reverse_token_map, token) for token in inpt]
            x = np.asarray(x)
            x = x.reshape((1,seqlen))
            preds = model.predict(x, verbose=0)[0][min(i, self.seqlen - 1)]
            token_ix = np.random.choice(range(vocab_size), p=preds.ravel())
            while token_ix == reverse_token_map["<UNK>"]:
                token_ix = np.random.choice(range(vocab_size), p=preds.ravel())
            new_token = vocab[token_ix]
            output += new_token
            if (i + 1 < len(inpt)):
                inpt[i + 1] = new_token
            else:
                inpt = inpt[1:] + [new_token]
            i += 1
        return output

    def masked_sample(self, model, tokens, vocab, reverse_token_map):
        seqlen = self.seqlen
        vocab_size = len(vocab)
        token_ix = -1
        inpt = ["<MASK>" for i in range(self.seqlen)]
        inpt[0] = "<START>"
        output = ""
        mintokens = 15
        maxtokens = 100
        i = 1
        while i < maxtokens and (i < mintokens or token_ix != reverse_token_map['<START>']):
            maskix = min(i, self.seqlen - 1)
            # x = np.zeros((1, seqlen))
            x = [get_ix_from_token(reverse_token_map, token) for token in inpt]
            x = np.asarray(x)
            x = x.reshape((1,seqlen))
            preds = model.predict(x, verbose=0)[0][min(i, self.seqlen - 1)]
            token_ix = np.random.choice(range(vocab_size), p=preds.ravel())
            while token_ix == reverse_token_map["<UNK>"]:
                token_ix = np.random.choice(range(vocab_size), p=preds.ravel())
            new_token = vocab[token_ix]
            output += new_token
            inpt[maskix] = new_token
            if maskix == self.seqlen - 1:
                inpt = inpt[1:] + ["<MASK>"]
            i += 1
        return output


loss_fn = keras.losses.SparseCategoricalCrossentropy(
            reduction=tf.keras.losses.Reduction.NONE
        )
loss_tracker = tf.keras.metrics.Mean(name="loss")
accr_tracker = tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")
class MaskedLanguageModel(tf.keras.Model):
    def train_step(self, inputs):
        if len(inputs) == 3:
            features, labels, sample_weight = inputs
        else:
            features, labels = inputs
            sample_weight = None

        with tf.GradientTape() as tape:
            predictions = self(features, training=True)
            loss = loss_fn(labels, predictions, sample_weight=sample_weight)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Compute our own metrics
        loss_tracker.update_state(loss, sample_weight=sample_weight)
        accr_tracker.update_state(labels, predictions, sample_weight=sample_weight)

        # Return a dict mapping metric names to current value
        return {"loss": loss_tracker.result(), "accuracy": accr_tracker.result()}

    def test_step(self, inputs):
        if len(inputs) == 3:
            features, labels, sample_weight = inputs
        else:
            features, labels = inputs
            sample_weight = None

        predictions = self(features, training=False)
        loss = loss_fn(labels, predictions, sample_weight=sample_weight)
        loss_tracker.update_state(loss, sample_weight=sample_weight)
        accr_tracker.update_state(labels, predictions, sample_weight=sample_weight)

        return {"loss": loss_tracker.result(), "accuracy": accr_tracker.result()}


    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [loss_tracker]