import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import LambdaCallback, ModelCheckpoint, CSVLogger
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM, Dropout, Embedding
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.models import load_model
from tensorflow.keras import regularizers
from tensorflow.python.ops import special_math_ops
from tensorflow.python.ops import math_ops
import numpy as np
import nltk
from tokenizers.sliding_window import SlidingWindowTokenizer
import random
import math
import logging
from models.helpers import get_ix_from_token, token_to_oh, oh_to_token, char_padded, create_oh

logger = logging.getLogger('keras_char_lm')

class EinsumOp(keras.layers.Layer):
    def __init__(self, op, **kwargs):
        super(EinsumOp, self).__init__(**kwargs)
        self.op = op

    def call(self, inputs):
        a1 = inputs[0]
        a2 = inputs[1]
        attn_factor = special_math_ops.einsum(self.op, a1, a2)
        return attn_factor

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            "op" : self.op
        })
        return config

def einsum_attn(i,q,k,v, dropout, dk, mask):
    dk = tf.cast(dk, tf.float32)
    ndim = len(k.shape)
    perm = list(range(ndim))
    perm[-2] = ndim - 1
    perm[-1] = ndim - 2
    dot = "aecd,abcd->acbe"
    com = "acbe,aecd->abcd"

    attn_factor = EinsumOp(dot, name="einsum_dot_%d" % i)([k,q])
    attn_factor = attn_factor / tf.math.sqrt(dk)
    if mask is not None:
        adder = (1.0 - math_ops.cast(mask, attn_factor.dtype)) * -1e9
        attn_factor += adder
    attn_factor = keras.layers.Softmax(name="attention_values_%d" % i)(attn_factor)
    attn_factor = keras.layers.Dropout(dropout, name="attention_dropout_%d" % i)(attn_factor)
    C = EinsumOp(com, name="einsum_com_%d" % i)([attn_factor,v])
    return C, attn_factor


def attention_head(i,q, k, v, dropout, dim, mask=None):
    return einsum_attn(i,q,k,v, dropout, dim, mask)
    # Q dot K scaled -> softmax = attention parameters -> ap * V summed = output
    dk = k.shape[-1]
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    ndim = len(k.shape)
    perm = list(range(ndim))
    perm[-2] = ndim - 1
    perm[-1] = ndim - 2

    # use perm to transpose final two dimensions of key vector
    # attn_factor = tf.matmul(q, tf.transpose(k, perm=perm)) / (dk ** 0.5)
    attn_factor = tf.matmul(q,k, transpose_b=True) / tf.math.sqrt(dk)
    if mask is not None:
        # attn_factor[mask == False] = -1e9
        mask = mask == False
        attn_factor = (mask * -1e9)
    attn_factor = keras.layers.Softmax()(attn_factor)
    attn_factor = keras.layers.Dropout(dropout)(attn_factor)
    return tf.matmul(attn_factor, v), attn_factor


def multihead_attention(i, q, k, v, h, n_a, reg, dropout, mask=None):
    dim = n_a // h
    Wq = keras.layers.Dense(n_a, kernel_regularizer=reg, name="dense_q_%d" % i)
    Wk = keras.layers.Dense(n_a, kernel_regularizer=reg, name="dense_k_%d" % i)
    Wv = keras.layers.Dense(n_a, kernel_regularizer=reg, name="dense_v_%d" % i)
    Wo = keras.layers.Dense(n_a, kernel_regularizer=reg, name="dense_o_%d" % i)

    seqlen = q.shape[1]
    shape = [-1, seqlen, h, dim]
    Q = Wq(q)
    Q = tf.reshape(Q, shape)
    #Q = tf.transpose(Q, perm=[0, 1, 2, 3]) # reshape for heads x seqlen x model_dim
    K = tf.reshape(Wk(k), shape)
    V = tf.reshape(Wv(v), shape)

    C, attn_factor = attention_head(i,Q, K, V, dropout, dim, mask)
    C = tf.reshape(tf.transpose(C, perm=[0, 2, 1, 3]), (-1, seqlen, n_a))

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
        self.transformer_layers = args.transformer_layers
        self.attention_heads = args.attention_heads

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

        # attn_layer = keras.layers.MultiHeadAttention(self.attention_heads, self.n_a//self.attention_heads)
        # attn_out = attn_layer(x,x,x, attention_mask=mask)
        attn_out = multihead_attention(i, x, x, x, self.attention_heads, self.n_a, reg, self.dropout_rate, mask=mask)
        #     attn_out = tf.reshape(out, (m,seqlen*n_a))

        x = keras.layers.add([attn_out, x])
        x = keras.layers.LayerNormalization(epsilon=1e-6, name="encoder_{}/attn_norm".format(i))(x)

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

        x = keras.layers.add([ffn_out, x])
        x = keras.layers.LayerNormalization(epsilon=1e-6, name="encoder_{}/ffn_norm".format(i))(x)
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
        out = keras.layers.Embedding(vocab_size, self.n_a, input_length=self.seqlen)(inpt)
        pos_enc = positional_encoding(self.seqlen, self.n_a)
        pos_emb = keras.layers.Embedding(
            input_dim=self.seqlen,
            output_dim=self.n_a,
            weights=[pos_enc],
            name="position_embedding",
        )(tf.range(start=0, limit=self.seqlen, delta=1))
        # encoder_out = out + pos_emb
        encoder_out = tf.math.add(out, pos_emb)
        mask = self.subsequent_mask(self.seqlen)
        for i in range(self.transformer_layers):
            encoder_out = self.transformer_encoder(encoder_out, i, reg, mask)
        # decoder_out = self.transformer_decoder(encoder_out, target_emb, reg, mask)
        out = keras.layers.Dense(self.ffdim, activation="relu", kernel_regularizer=reg, name="penult_dense")(encoder_out)
        out = keras.layers.Dense(vocab_size, activation="softmax", kernel_regularizer=reg, name="final_dense")(out)

        # masked_model = MaskedLanguageModel(inputs=inpt, outputs=masked_out)
        # return masked_model

        model = keras.Model(inputs=inpt, outputs=out)
        return model

    def logtop5(self,p, vocab):
        top5 = tf.math.top_k(p, k=5)
        top5 = [(vocab[tix], top5.values[ix].numpy()) for (ix, tix) in enumerate(top5.indices)]
        logger.info(top5)

    def sample(self, model, tokens, vocab, reverse_token_map, temp=1):
        seqlen = self.seqlen
        vocab_size = len(vocab)
        token_ix = -1
        # start = np.random.randint(0, len(tokens) - self.seqlen)
        # inpt = tokens[start:start+self.seqlen]
        inpt = [" " for i in range(self.seqlen)]
        inpt[0] = "\n"
        x = [get_ix_from_token(reverse_token_map, token) for token in inpt]
        x = np.asarray(x)
        x = x.reshape((1, seqlen))
        preds = model.predict(x, verbose=0)[0]
        for j in range(self.seqlen):
            p = preds[j]
            top5 = tf.math.top_k(p, k=5)
            top5 = [(vocab[tix], top5.values[ix].numpy()) for (ix, tix) in enumerate(top5.indices)]
            logger.info(top5)
        output = ""
        mintokens = 15
        maxtokens = 100
        i = 0
        while i < maxtokens and (i < mintokens or token_ix != reverse_token_map['<START>']):
            # x = np.zeros((1, seqlen))
            logger.info(inpt)
            x = [get_ix_from_token(reverse_token_map, token) for token in inpt]
            x = np.asarray(x)
            x = x.reshape((1,seqlen))
            preds = model.predict(x, verbose=0)[0]
            preds = preds[min(i, self.seqlen - 1)]
            self.logtop5(preds,vocab)
            # topk = tf.math.top_k(preds, k=50)
            # topk_preds = keras.layers.Softmax()(topk.values/temp)
            # token_ix = np.random.choice(topk.indices, p=topk_preds)
            token_ix = np.random.choice(range(vocab_size), p=preds.ravel())
            retries = 0
            while retries < 10 and token_ix == reverse_token_map["<UNK>"]:
                token_ix = np.random.choice(range(vocab_size), p=preds.ravel())
                retries += 1
            new_token = vocab[token_ix]
            logger.info(new_token)
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
