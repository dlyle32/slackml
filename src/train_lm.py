import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import LambdaCallback, ModelCheckpoint, CSVLogger
from tensorflow.keras.models import load_model
import numpy as np
import random
import math
import datetime
import time
import os
import argparse
import logging
from data.load import load_datasets

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

def main(args):
    # load train/test data
    datadir = os.path.join(args.volumedir, args.datadir)
    train, test = load_datasets(datadir)
    train = train[:min(len(train), args.datacap)]

    # Dynamically load modelBuilder class
    moduleName, klassName = args.modelbuilder.split(".")
    mod = __import__('models.%s' % moduleName, fromlist=[klassName])
    klass = getattr(mod,klassName)
    modelBuilder = klass(args)

    tokens, vocab, reverse_token_map = modelBuilder.tokenize(train, freq_threshold=args.freqthreshold)

    # Create or load existing model
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

    checkpointdir = os.path.join(args.volumedir, args.checkpointdir)
    checkpointnames = args.checkpointnames % timestamp
    sample_func = lambda : modelBuilder.sample(model, tokens, vocab, reverse_token_map)
    callbacks = get_callbacks(args.volumedir, checkpointdir, checkpointnames, timestamp, sample_func)

    seqs = modelBuilder.get_input_sequences(tokens, reverse_token_map)

    trainseqs, valseqs = validation_split(seqs, val_split=args.valsplit)

    metrics = {}
    logger.info(vocab)
    for epoch in range(init_epoch, args.numepochs):
        batches = rand_mini_batches(trainseqs, args.minibatchsize)
        for i, batch in enumerate(batches):
            X, Y = modelBuilder.build_input_vectors(batch, vocab, reverse_token_map)
            metrics = model.train_on_batch(X, Y, reset_metrics = i==0, return_dict=True)
            if i % 100 == 0:
                print("Batch %d of %d in epoch %d: %s" % (i, len(batches), epoch, str(metrics)))
        logger.info("Epoch %d: %s" % (epoch, str(valmetrics)))
        valmetrics = evaluate_mini_batches(model, modelBuilder, vocab, reverse_token_map, valseqs, args.minibatchsize)
        logger.info("Validation metrics %s" % str(valmetrics))
        sample_func()
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
    parser.add_argument("--modelbuilder", type=str)
    parser.add_argument("--valsplit", type=float, default=0.2)
    return parser.parse_args()

if __name__=="__main__":
    args = parse_args()
    main(args)
