import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import LambdaCallback, ModelCheckpoint, CSVLogger
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import numpy as np
import random
import math
import datetime
import time
import os
import argparse
import logging
from data.load import load_datasets, load_context_target_pairs

logger = logging.getLogger('keras_char_lm')

def get_callbacks(volumedir, checkpointdir, checkpointnames, timestamp, sample_func):
    today_date = datetime.datetime.today().strftime('%Y-%m-%d')
    if not os.path.isdir(checkpointdir):
        os.makedirs(checkpointdir)
    filepath = os.path.join(checkpointdir, checkpointnames)
    checkpoint_callback = ModelCheckpoint(filepath=filepath,
                                          save_weights_only=False,
                                          monitor='val_loss')
    # Loss history callback
    # WILL NEED TO ADD LOGDIR IF I USE CALLBACKS AGAIN
    epoch_results_callback = CSVLogger(
        os.path.join(volumedir, 'training_log_{}_{:d}.csv'.format(today_date, timestamp)),
        append=True)

    sample_callback = LambdaCallback(on_epoch_end=lambda epoch, logs: sample_func())

    callbacks = [checkpoint_callback, epoch_results_callback, sample_callback]
    return callbacks

def rand_mini_batches(seqs, mini_batch_size):
    m = len(seqs)
    np.random.shuffle(seqs)
    numbatches = math.floor(m / mini_batch_size)
    if m % mini_batch_size != 0:
        numbatches += 1
    minibatches = []
    for i in range(0, numbatches):
        start = i * mini_batch_size
        end = min((i + 1) * mini_batch_size, m)
        minibatches.append(seqs[start:end])
    return minibatches

def validation_split(data, val_split = 0.2):
    train = []
    val = []
    for dat in data:
        if (random.random() < val_split):
            val.append(dat)
        else:
            train.append(dat)
    return train, val

def evaluate_mini_batches(model, modelBuilder, vocab, reverse_token_map, data, mini_batch_size):
    batches = rand_mini_batches(data, mini_batch_size)
    metrics = {}
    for i, batch in enumerate(batches):
        X, Y = modelBuilder.build_input_vectors(batch, vocab, reverse_token_map)
        metrics = model.test_on_batch(X, Y, reset_metrics=i == 0, return_dict=True)
    return metrics

def save_vocab(vocab, checkpointsdir, timestamp):
    fname = os.path.join(checkpointsdir, "vocab.%d.tsv" % timestamp)
    with open(fname, "w") as fp:
        fp.write("\t".join(vocab))

def load_vocab(checkpointdir, timestamp):
    fname = os.path.join(checkpointdir, "vocab.%d.tsv" % timestamp)
    with open(fname, "r") as fp:
        vocab = fp.read().split("\t")
    return vocab

def save_tokens(tokens, checkpointsdir, timestamp):
    fname = os.path.join(checkpointsdir, "tokenized_train_data.%d.tsv" % timestamp)
    with open(fname, "w") as fp:
        fp.write("\t".join(tokens))

def load_tokens(checkpointdir, timestamp):
    fname = os.path.join(checkpointdir, "tokenized_train_data.%d.tsv" % timestamp)
    if(not os.path.exists(fname)):
        return []
    with open(fname, "r") as fp:
        tokens = fp.read().split("\t")
    return tokens

def main(args):
    # load train/test data
    datadir = os.path.join(args.volumedir, args.datadir)
    # train, test = load_datasets(datadir)
    train, test = load_context_target_pairs(datadir)
    # train = sorted(train, key=lambda a: len(a), reverse=True)
    train = train[:min(len(train), args.datacap)]

    # Dynamically load modelBuilder class
    moduleName, klassName = args.modelbuilder.split(".")
    mod = __import__('models.%s' % moduleName, fromlist=[klassName])
    klass = getattr(mod,klassName)
    modelBuilder = klass(args)


    timestamp = int(time.time())
    logdir = os.path.join(args.volumedir, datetime.datetime.today().strftime('%Y%m%d'), args.logdir)
    if not os.path.isdir(logdir):
        os.makedirs(logdir)
    hdlr = logging.FileHandler(os.path.join(logdir, "training_output_%d.log" % timestamp))
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)

    checkpointdir = os.path.join(args.volumedir, datetime.datetime.today().strftime('%Y%m%d'), args.checkpointdir)
    if not os.path.isdir(checkpointdir):
        os.makedirs(checkpointdir)

    # Create or load existing model
    init_epoch = 0
    tokens, vocab, reverse_token_map = modelBuilder.tokenize(train, freq_threshold=args.freqthreshold)
    if args.loadmodel and os.path.exists(args.loadmodel):
        modelpath = args.loadmodel
        timestamp = int(modelpath.split(".")[1])
        init_epoch = int(modelpath.split(".")[2])
        loaddir = "/".join(modelpath.split("/")[:-1])
        model = load_model(modelpath)
        vocab = load_vocab(loaddir, timestamp)
        # tokens = load_tokens(loaddir, timestamp)
        reverse_token_map = {t: i for i, t in enumerate(vocab)}
    else:
        model = modelBuilder.create_model(vocab)
        save_vocab(vocab, checkpointdir, timestamp)
        if args.savetokens:
            save_tokens(tokens, checkpointdir, timestamp)

    optimizer_map = {"adam": Adam, "rmsprop": RMSprop, "sgd": SGD}
    optimizer = optimizer_map[args.optimizer] if args.optimizer in optimizer_map.keys() else RMSprop
    lr_decay = ExponentialDecay(initial_learning_rate=args.learningrate,
                                decay_rate=args.decayrate,
                                decay_steps=args.decaysteps)
    opt = optimizer(learning_rate=lr_decay, clipvalue=3)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=["accuracy"])

    model.summary(print_fn=logger.info)

    checkpointnames = args.checkpointnames % timestamp
    sample_func = lambda : modelBuilder.sample(model, tokens, vocab, reverse_token_map)
    callbacks = get_callbacks(args.volumedir, checkpointdir, checkpointnames, timestamp, sample_func)

    seqs = modelBuilder.get_input_sequences(tokens, reverse_token_map, full=args.fillseqs)

    trainseqs, valseqs = validation_split(seqs, val_split=args.valsplit)

    metrics = {}
    for epoch in range(init_epoch, args.numepochs):
        batches = rand_mini_batches(trainseqs, args.minibatchsize)
        for i, batch in enumerate(batches):
            X, Y = modelBuilder.build_input_vectors(batch, vocab, reverse_token_map)
            metrics = model.train_on_batch(X, Y, reset_metrics = i==0, return_dict=True)
            if i % 100 == 0:
                print("Batch %d of %d in epoch %d: %s" % (i, len(batches), epoch, str(metrics)))
        logger.info("Epoch %d: %s" % (epoch, str(metrics)))
        valmetrics = evaluate_mini_batches(model, modelBuilder, vocab, reverse_token_map, valseqs, args.minibatchsize)
        logger.info("Validation metrics %s" % str(valmetrics))
        if args.runsamples:
            sample_output = sample_func()
            logger.info("\n" + sample_output)
        model.save(os.path.join(checkpointdir, checkpointnames).format(epoch=epoch))

    # model.fit(X, Y,
    #            batch_size=args.minibatchsize,
    #            epochs=args.numepochs,
    #            initial_epoch=init_epoch,
    #            validation_split=0.2,
    #            shuffle=True,
    #            callbacks=callbacks)


def parse_args():
    parser= argparse.ArgumentParser()
    parser.add_argument("--loadmodel", type=str)
    parser.add_argument("--datadir", default="data/")
    parser.add_argument("--logdir", default="logs/")
    parser.add_argument("--volumedir", default="/training/")
    parser.add_argument("--checkpointdir", default="checkpoints/")
    parser.add_argument("--checkpointnames", default="nodle_char_model.%d.{epoch:03d}.h5")
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument("--hiddensize", type=int, default=128)
    parser.add_argument("--minibatchsize", type=int, default=256)
    parser.add_argument("--numepochs", type=int, default=25)
    parser.add_argument("--seqlength", type=int, default=40)
    parser.add_argument("--minlength", type=int, default=15)
    parser.add_argument("--maxlength", type=int, default=200)
    parser.add_argument("--learningrate", type=float, default=0.01)
    parser.add_argument("--dropoutrate", type=float, default=0.2)
    parser.add_argument("--regfactor", type=float, default=0.01)
    parser.add_argument("--datacap", type=int, default=10000)
    parser.add_argument("--freqthreshold", type=int, default=5)
    parser.add_argument("--modelbuilder", type=str)
    parser.add_argument("--valsplit", type=float, default=0.2)
    parser.add_argument("--fillseqs", action="store_true")
    parser.add_argument("--savetokens", action="store_true")
    parser.add_argument("--embedding", action="store_true")
    parser.add_argument("--embeddingsize", type=int, default=100)
    parser.add_argument("--optimizer", default="rmsprop")
    parser.add_argument("--decaysteps", type=int, default=10000)
    parser.add_argument("--decayrate", type=float, default=1.0)
    parser.add_argument("--runsamples", action="store_true")
    parser.add_argument("--vocabfile", type=str)
    parser.add_argument("--modelweightspath", type=str)
    return parser.parse_args()

if __name__=="__main__":
    args = parse_args()
    main(args)
