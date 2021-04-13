import tensorflow as tf
import numpy as np
import math

class SequenceVectors(tf.keras.utils.Sequence):

    def __init__(self, args, seqs, vocab):
        self.batch_size = args.minibatchsize
        self.seqs = seqs
        self.seqlen = args.seqlength
        self.vocab = vocab
        self.onehot = True

    def __len__(self):
        return math.ceil(len(self.seqs) / self.batch_size)

    def __getitem__(self, index):
        start = index * self.batch_size
        end = min(start+self.batch_size, len(self.seqs))
        return self.get_input_vectors(end-start, self.seqs[start:end])

    def on_epoch_end(self):
        np.random.shuffle(self.seqs)

    def get_input_vectors(self, batch_size, seqs):
        X = np.zeros((batch_size, self.seqlen))
        Y = np.zeros((batch_size, self.seqlen))

        for i, (Xseq, Yseq) in enumerate(seqs):
            X[i, :] = Xseq
            Y[i, :] = Yseq

        if self.onehot:
            X = tf.keras.utils.to_categorical(X, num_classes=len(self.vocab))

        return X,Y

