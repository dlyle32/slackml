import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tokenizers.sliding_window import SlidingWindowTokenizer
from models.helpers import get_ix_from_token, token_to_oh, oh_to_token, char_padded


def count_tokens(token_counts, tokens):
    for t in tokens:
        if t not in token_counts:
            token_counts[t] = 1
        else:
            token_counts[t] += 1
    return token_counts

def hoops_in(s):
    return "hoops.csv" in bytes.decode(s)

class SlackTextLineDataset():
    def __init__(self, args, data):
        self.freq_threshold=args.freqthreshold
        self.len_threshold=args.lenthreshold
        self.step = args.step
        self.seqlen = args.seqlength
        self.tokenizer = SlidingWindowTokenizer(args)
        # datadir = os.path.join(args.volumedir, args.datadir)
        # file_pattern = datadir + "*.csv"
        # files = tf.data.Dataset.list_files(file_pattern)
        # self.ds = tf.data.TextLineDataset(files)
        self.preprocess(data)

    def preprocess(self, data):
        token_counts = {}
        self.tokens = []
        for msg in data:
            tokens = self.tokenizer.word_tokenize(msg)
            if len(tokens) < self.len_threshold:
                continue
            token_counts = count_tokens(token_counts, tokens)
            self.tokens.append(tokens)

        freq_filtered = filter(lambda elem: elem[1] >= self.freq_threshold, token_counts.items())
        self.vocab = sorted([elem[0] for elem in list(freq_filtered)])
        self.vocab += ["<UNK>", "<START>", "<PAD>"]
        self.reverse_token_map = {t: i for i, t in enumerate(self.vocab)}

        # self.ds = self.ds.map(lambda x: tf.py_function(func=self.tokenize, inp=[x], Tout=tf.string))
        # token_counts = self.ds.reduce({}, lambda state, x: tf.py_function(count_tokens,[state,x], tf.)
        # self.ds = tf.data.Dataset.from_tensor_slices([vect for seq in seqs for vect in self.tokens_to_sequences(seq)])
        self.ds = None
        self.Xseqs = []
        self.Yseqs = []
        for seq in self.tokens:
            Xs, Ys = self.tokens_to_sequences(seq)
            self.Xseqs.extend(Xs)
            self.Yseqs.extend(Ys)
        self.Xseqs = np.array(self.Xseqs)
        self.Yseqs = np.array(self.Yseqs)
        #     ds = self.tokens_to_sequences(seq)
        #     if self.ds is None:
        #         self.ds = ds
        #     else:
        #         self.ds = self.ds.concatenate(ds)
        # print("SEQUENCES BUILT")
        # return self.ds
        # return selfXseqs, Yseqs

    def tokens_to_sequences(self, tokens):
        if len(tokens) < self.seqlen:
            tokens = char_padded(tokens, "<PAD>", self.seqlen)
        Xseqs = []
        Yseqs = []
        pad_masks = []
        for i in range(0,len(tokens)-self.seqlen+1, self.step):
            x0 = "<START>" if i == 0 else tokens[i - 1]
            Yseq = [get_ix_from_token(self.reverse_token_map, token) for token in tokens[i:i+self.seqlen]]
            Xseq = [get_ix_from_token(self.reverse_token_map, x0)] + Yseq[:-1]
            Yseq = np.array(Yseq)
            Xseq = np.array(Xseq)
            # pad_mask = (Yseq != get_ix_from_token(self.reverse_token_map, "<PAD>")).astype(np.int64)
            # pad_masks.append(pad_mask)
            Yseqs.append(Yseq)
            Xseqs.append(Xseq)
        # Yseqs = tf.data.Dataset.from_tensor_slices(Yseqs)
        # Xseqs = tf.data.Dataset.from_tensor_slices(Xseqs, Yseqs)
        # seqs = tf.data.Dataset.from_tensor_slices((Xseqs,Yseqs))
        return Xseqs, Yseqs
        # return tf.data.Dataset.from_tensor_slices((Xseqs,Yseqs))

    def get_dataset(self):
        return self.Xseqs, self.Yseqs, self.vocab, self.tokens



