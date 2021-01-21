from tensorflow.keras.models import load_model
import numpy as np
import json
import os
import argparse
import requests
from models.helpers import get_ix_from_token, token_to_oh

def sample(data, model, chars, char_to_ix, temperature=1.0):
    maxlen = model.get_layer(name="input").input_shape[0][1]
    vocab_size =model.layers[-1].output_shape[-1]
    char_index = -1
    inpt = [" " for i in range(0,maxlen)]
    output = inpt
    while char_index != char_to_ix['\n']:
        x = np.zeros((1, maxlen, vocab_size))
        x[0] = [char_to_oh(get_ix_from_char(char_to_ix, chars, c), vocab_size) for c in inpt]
        preds = model.predict(x, verbose=0)[0]
        #preds = np.asarray(preds).astype('float64')
        #preds = np.log(preds) / temperature
        #exp_preds = np.exp(preds)
        #preds = exp_preds / np.sum(exp_preds)
        char_index = np.random.choice(range(vocab_size), p = preds.ravel())
        #probas = np.random.multinomial(1, preds, 1)
        #char_index = np.argmax(probas)
        new_char = chars[char_index]
        output += new_char
        inpt = inpt[1:] + new_char
    logger.info("\n" + output[:40] + "->" + output[40:])
    return output

def oh_to_char(chars, oh):
    char = [chars[i] for i,c in enumerate(oh) if c == 1]
    return "0" if len(char) == 0 else char[0]

def char_to_oh(index, vocab_size):
    x = np.zeros((vocab_size))
    x[index] = 1
    return x

def get_ix_from_char(char_to_ix, chars, c):
    if c in chars:
        return char_to_ix[c]
    else:
        return char_to_ix["*"]

def sample(model, vocab_size, maxlen, temperature=1.0):
    #chars = json.loads('["\\t", "\\n", " ", "!", "\\"", "#", "$", "%", "&", "\'", "(", ")", "*", "+", ",", "-", ".", "/", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ":", ";", "<", "=", ">", "?", "@", "[", "\\\\", "]", "^", "_", "`", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "{", "|", "}", "~", "\\u00a0", "\\u00a3", "\\u00ae", "\\u00af", "\\u00b0", "\\u00e1", "\\u00e9", "\\u00ed", "\\u00ef", "\\u00f1", "\\u00f4", "\\u00f6", "\\u00f8", "\\u00fc", "\\u015f", "\\u01bd", "\\u02b0", "\\u02b3", "\\u0430", "\\u0432", "\\u0433", "\\u0435", "\\u0437", "\\u0438", "\\u043b", "\\u043c", "\\u043d", "\\u043e", "\\u043f", "\\u0440", "\\u0442", "\\u0447", "\\u044b", "\\u04af", "\\u0501", "\\u0b20", "\\u0c66", "\\u0f3c", "\\u0f3d", "\\u1d49", "\\u1d4d", "\\u1d52", "\\u1d57", "\\u1da6", "\\u2005", "\\u200a", "\\u2013", "\\u2014", "\\u2018", "\\u2019", "\\u201c", "\\u201d", "\\u2022", "\\u2026", "\\u2122", "\\u2500", "\\u2502", "\\u2514", "\\u251c", "\\u2588", "\\u25d5", "\\u267c", "\\u2683", "\\u3064", "\\u30c4", "\\uab87", "\\ud83d\\udd47", "\\ud83d\\udec7", "\\ud83e\\udd21", "\\ud83e\\udd47", "\\ud83e\\udd54", "<EOM>"]')
    chars = ['\n', ' ', '!', '"', ',', '.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', '?', '@', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '~', '*']
    char_to_ix = {c: i for i, c in enumerate(chars)}
    inpt = "                                                      you know what the crowd says,"
    inpt = inpt[len(inpt)-maxlen:]
    output = inpt
    numchars = 0
    char_index = -1
    while numchars < maxlen or char_index != char_to_ix['\n']:
        x = np.zeros((1, maxlen, vocab_size))
        x[0] = [char_to_oh(get_ix_from_char(char_to_ix, chars, c), vocab_size) for c in inpt]
        preds = model.predict(x, verbose=0)[0]
        # preds = np.asarray(preds).astype('float64')
        # preds = np.log(preds) / temperature
        # exp_preds = np.exp(preds)
        # preds = exp_preds / np.sum(exp_preds)
        char_index = np.random.choice(range(vocab_size), p=preds.ravel())
        # probas = np.random.multinomial(1, preds, 1)
        # char_index = np.argmax(probas)
        new_char = chars[char_index]
        output += new_char
        inpt = inpt[1:] + new_char
        numchars += 1
    print(output)

    # output = ""
    # x = np.zeros((1, maxlen, vocab_size))
    # char_index = -1
    # i = 0
    # while char_index != 1:
    #     preds = model.predict(x, verbose=0)[0][i]
    #     print("SO FAR: %s" % output)
    #     print("PREDS : " + str(["%s:%0.3f" % (chars[i], p) for i,p in enumerate(preds)]))
    #     char_index = np.random.choice(range(vocab_size), p = preds.ravel())
    #     x[0,i,char_index] = 1
    #     output += chars[char_index]
    #     i+=1
    # print(output)
    # return


def sample_seq2seq(model, seqlen, vocab, reverse_token_map):
    seqlen = seqlen
    vocab_size = len(vocab)
    token_ix = -1
    inpt = ["START" for i in range(seqlen)]
    output = ""
    mintokens = 10
    maxtokens = 1000
    i = 0
    while i < maxtokens and (i < mintokens or token_ix != reverse_token_map['START']):
        x = np.zeros((1, seqlen, vocab_size))
        x[0] = [token_to_oh(get_ix_from_token(reverse_token_map, token), vocab_size) for token in inpt]
        preds = model.predict(x, verbose=0)[0][min(i, seqlen - 1)]
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

def load_vocab(checkpointdir, timestamp):
    fname = os.path.join(checkpointdir, "vocab.%d.tsv" % timestamp)
    with open(fname, "r") as fp:
        vocab = fp.read().split("\t")
    return vocab

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelpath", required=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    slackurl = "slackurl"
    model = load_model(args.modelpath)
    timestamp = int(args.modelpath.split(".")[1])
    loaddir = "/".join(args.modelpath.split("/")[:-1])
    vocab = load_vocab(loaddir, timestamp)
    reverse_token_map = {t: i for i, t in enumerate(vocab)}
    seqlen = model.get_layer(name="input").input_shape[0][1]
    vocab_size = model.get_layer(name="dense").output_shape[-1]
    print(model.summary())
    args.seqlen = seqlen
    for i in range(0,10):
        output = sample_seq2seq(model, seqlen, vocab, reverse_token_map)
        msg = output.replace("START","")
        print(msg)
        payload = {"text": msg }
        resp = requests.post(slackurl, headers = {"Content-type": "application/json"}, data=json.dumps(payload))
