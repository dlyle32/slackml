import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import LambdaCallback, ModelCheckpoint, CSVLogger
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import matplotlib.pyplot as plt
import numpy as np
import random
import math
import datetime
import time
import os
import argparse
import logging
from data.load import load_datasets, load_context_target_pairs, imdb_data_load
from models.helpers import get_ix_from_token, token_to_oh, oh_to_token, char_padded, create_oh
from models.kattn_lm import EinsumOp
from callbacks.text_gen import TextGenerator
from tensorflow.keras.utils import plot_model

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
        X, Y, sample_weight = modelBuilder.build_input_vectors(batch, vocab, reverse_token_map)
        metrics = model.test_on_batch(X, Y, sample_weight= sample_weight, reset_metrics=i == 0, return_dict=True)
    for key in metrics.keys():
        if "val_" in key:
           continue
        metrics["val_" + key] = metrics[key]
        del metrics[key]
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

# Learning rate schedule from Attention is all you need
# Taken from https://www.tensorflow.org/tutorials/text/transformer
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super(CustomSchedule, self).__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps

  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

  def get_config(self):
      config = {
          'd_model': self.d_model,
          'warmup_steps': self.warmup_steps,

      }
      return config

def configure_checkpointing(args, timestamp):
    checkpoint_dir = os.path.join(args.volumedir, datetime.datetime.today().strftime('%Y%m%d'), "checkpoints/")
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_name = "nodle_word_lm.%d.{epoch:03d}.h5" % timestamp
    return os.path.join(checkpoint_dir, checkpoint_name)

def plot_history(metrics, lr, logdir, timestamp):
    plt.plot(np.squeeze(metrics["loss"]),"b")
    plt.plot(np.squeeze(metrics["val_loss"]),"r")
    plt.ylabel('loss')
    plt.xlabel('iterations')
    plt.title("Learning rate =" + str(lr))
    lossfname = os.path.join(logdir, "model_loss_%d.png" % timestamp)
    plt.savefig(lossfname)
    if "accuracy" not in metrics.keys():
        return
    plt.plot(np.squeeze(metrics["accuracy"]), "b")
    plt.ylabel('accuracy')
    plt.xlabel('iterations')
    plt.title("Learning rate =" + str(lr))
    accrfname = os.path.join(logdir, "model_accuracy_%d.png" % timestamp)
    plt.savefig(accrfname)

def last_word_prediction_accuracy(batch_size, seqlen):
    def last_word_cat_accuracy(y_true, y_pred):
        # amax = np.argmax(y_pred, axis=-1)
        # acc = np.dot(sample_weights, np.equal(y_true, amax))
        # return acc
        batch_size = tf.shape(y_true).numpy()[0]
        sample_weights = np.zeros((batch_size, seqlen))
        sample_weights[:, seqlen - 1] = 1
        m = tf.keras.metrics.SparseCategoricalAccuracy()
        m.update_state(y_true, y_pred, sample_weight=sample_weights)
        return m.result().numpy()
    return last_word_cat_accuracy

def last_word_prediction_topk_accuracy(batch_size, seqlen, k):
    def last_word_topk_cat_accuracy(y_true, y_pred):
        sample_weights = np.zeros((batch_size, seqlen, 1))
        sample_weights[:, seqlen - 1, :] = 1
        m = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=k)
        m.update_state(y_true, y_pred, sample_weight=sample_weights)
        return m.result().numpy()
    return last_word_topk_cat_accuracy

def main(args):
    # load train/test data
    datadir = os.path.join(args.volumedir, args.datadir)
    # train = imdb_data_load(datadir)
    train, test = load_datasets(datadir)
    # train, test = load_context_target_pairs(datadir, context_len = args.conlength)
    # train = sorted(train, key=lambda a: len(a), reverse=True)
    # train = train[:min(len(train), args.datacap)]

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
    checkpointpath = configure_checkpointing(args, timestamp)
    checkpoint_callback = ModelCheckpoint(filepath=checkpointpath,
                                          save_weights_only=False)

    # Create or load existing model
    init_epoch = 0
    tokens, vocab, reverse_token_map = modelBuilder.tokenize(train, freq_threshold=args.freqthreshold)
    if args.loadmodel and os.path.exists(args.loadmodel):
        modelpath = args.loadmodel
        timestamp = int(modelpath.split(".")[1])
        init_epoch = int(modelpath.split(".")[2])
        loaddir = "/".join(modelpath.split("/")[:-1])
        model = load_model(modelpath, custom_objects={"EinsumOp" : EinsumOp})
        vocab = load_vocab(loaddir, timestamp)
        # tokens = load_tokens(loaddir, timestamp)
        reverse_token_map = {t: i for i, t in enumerate(vocab)}
    else:
        model = modelBuilder.create_model(vocab)
        save_vocab(vocab, checkpointdir, timestamp)
        if args.savetokens:
            save_tokens(tokens, checkpointdir, timestamp)

    plot_model(model, to_file='model_plot_2.png', show_shapes=True, show_layer_names=True)
    optimizer_map = {"adam": Adam, "rmsprop": RMSprop, "sgd": SGD}
    optimizer = optimizer_map[args.optimizer] if args.optimizer in optimizer_map.keys() else RMSprop
    lr_decay = ExponentialDecay(initial_learning_rate=args.learningrate,
                                decay_rate=args.decayrate,
                                decay_steps=args.decaysteps)
    custom_lr = CustomSchedule(args.hiddensize)
    opt = optimizer(learning_rate=lr_decay, clipvalue=3)
    # model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=["accuracy"])
    # attn_4_output = model.get_layer("attention_values_4").output
    # dense_v_out = model.get_layer("dense_v_4").output
    # einsum_com_output = model.get_layer("einsum_com_4").output
    # inpt = model.get_layer("input")
    # attn_factor_model = keras.Model(inputs=inpt.input, outputs=attn_4_output)
    # einsum_com_model = keras.Model(inputs=inpt.input, outputs=einsum_com_output)
    # dense_v_model = keras.Model(inputs=inpt.input, outputs=dense_v_out)
    model.compile(loss=keras.losses.SparseCategoricalCrossentropy(name="loss"),
                  run_eagerly=True,
                  optimizer=opt,
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
                           tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3,name="top_3_accuracy"),
                           tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5,name="top_5_accuracy"),
                           last_word_prediction_accuracy(args.minibatchsize, args.seqlength)])
                           # last_word_prediction_topk_accuracy(args.minibatchsize, args.seqlength, 5)])

    # model.summary(print_fn=logger.info)

    checkpointnames = args.checkpointnames % timestamp
    sample_func = lambda : modelBuilder.sample(model, tokens, vocab, reverse_token_map)
    callbacks = get_callbacks(args.volumedir, checkpointdir, checkpointnames, timestamp, sample_func)
    sample_callback = LambdaCallback(on_epoch_end=lambda epoch, logs: sample_func())
    logger_callback = LambdaCallback(on_epoch_end=lambda epoch, logs: logger.info("Epoch %d: %s" % (epoch, str(logs))))

    trainseqs = modelBuilder.get_input_sequences(tokens, reverse_token_map)

    # trainseqs, valseqs = validation_split(seqs, val_split=args.valsplit)

    X, Y, sample_weights = modelBuilder.build_input_vectors(trainseqs, vocab, reverse_token_map)
    # ds = modelBuilder.build_input_vectors(trainseqs, vocab, reverse_token_map)
    # model.fit(X, Y,
    # print(ds)
    # start_prompt = "this movie is"
    # start_tokens = [reverse_token_map[t] for t in start_prompt.split()]
    # num_tokens_generated = 40
    # text_gen_callback = TextGenerator(num_tokens_generated, args.seqlength, start_tokens, vocab)
    history = model.fit(X,Y,
                        epochs=args.numepochs,
                        initial_epoch=init_epoch,
                        batch_size=args.minibatchsize,
                        validation_split=0.1,
                        shuffle=True,
                        callbacks=[sample_callback, logger_callback, checkpoint_callback])
    logger.info(history.history)
    return
    allmetrics = {}
    for epoch in range(init_epoch, args.numepochs):
        batches = rand_mini_batches(trainseqs, args.minibatchsize)
        for i, batch in enumerate(batches):
            X, Y, sample_weights = modelBuilder.build_input_vectors(batch, vocab, reverse_token_map)
            metrics = model.train_on_batch(X, Y, sample_weight=sample_weights, reset_metrics = i==0, return_dict=True)
            if i % 100 == 0:
                valmetrics = evaluate_mini_batches(model, modelBuilder, vocab, reverse_token_map, valseqs,
                                                   args.minibatchsize)
                metrics.update(valmetrics)
                for key in metrics.keys():
                    if key in allmetrics.keys():
                        allmetrics[key] += [metrics[key]]
                    else:
                        allmetrics[key] = [metrics[key]]
                print("Batch %d of %d in epoch %d: %s" % (i, len(batches), epoch, str(metrics)))
        logger.info("Epoch %d: %s" % (epoch, str(metrics)))
        # logger.info("Validation metrics %s" % str(valmetrics))
        if args.runsamples:
            sample_output = sample_func()
            logger.info("\n" + sample_output)
        model.save(os.path.join(checkpointdir, checkpointnames).format(epoch=epoch))
        plot_history(allmetrics, args.learningrate, logdir, timestamp)
    # for i in range(10):
    #     sample_output = sample_func()
    #     logger.info("\n" + sample_output)


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
    parser.add_argument("--ffdim", type=int, default=256)
    parser.add_argument("--attention_heads", type=int, default=4)
    parser.add_argument("--transformer_layers", type=int, default=5)
    parser.add_argument("--minibatchsize", type=int, default=256)
    parser.add_argument("--numepochs", type=int, default=25)
    parser.add_argument("--seqlength", type=int, default=40)
    parser.add_argument("--conlength", type=int, default=40)
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
