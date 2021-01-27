import os
import time
import hmac
import hashlib

efs_path = "/mnt/sampler"
print("PYTHONPATH:", os.environ.get('PYTHONPATH'))
print("PATH:", os.environ.get('PATH'))
SLACK_SIGNING_SECRET = os.environ.get("SLACK_SIGNING_SECRET")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import numpy as np
from numpy import zeros
from numpy.random import choice
import json
import requests

import matplotlib
import matplotlib.pyplot as plt
from models.kattn_lm import EinsumOp

def verify_request(event):
    body=event["body"]
    headers = event["headers"]
    timestamp = headers["X-Slack-Request-Timestamp"] if "X-Slack-Request-Timestamp" in headers.keys() else None
    slack_signature = headers["X-Slack-Signature"] if "X-Slack-Signature" in headers.keys() else None

    if timestamp == None or slack_signature == None:
        return False

    if abs(time.time() - float(timestamp)) > 60 * 5:
        # request is too old
        return False

    sig_basestring = "v0:" + timestamp + ":" + json.dumps(body).replace(" ", "")
    print(sig_basestring)
    print(slack_signature)
    my_signature = "v0=" + hmac.new(
        key=SLACK_SIGNING_SECRET.encode("utf-8"),
        msg=sig_basestring.encode("utf-8"),
        digestmod=hashlib.sha256
    ).hexdigest()

    print(my_signature)
    return hmac.compare_digest(my_signature, slack_signature)

def lambda_handler(event, context):
    # TODO implement
    print("IN HANDLER")
    print(event)
    print(context)
    if not verify_request(event):
        print("Could not verify request")
        return {
            'statusCode': 400,
            'body' : json.dumps({"error" : "Unverified request"})
        }

    body = event["body"]
    request_type = body["type"]
    if request_type == "url_verification":
        challenge_resp = {"challenge": body["challenge"]}
        print(challenge_resp)
        return {
            'statusCode': 200,
            'body': body["challenge"]
        }

    payload = main()
    return {
        'statusCode': 200,
        'body': json.dumps(payload)
    }
    

def get_ix_from_token(reverse_token_map, token):
    if token in reverse_token_map.keys():
        return reverse_token_map[token]
    else:
        return reverse_token_map["<UNK>"]

def oh_to_token(vocab, oh):
    tokens = [vocab[i] for i,c in enumerate(oh) if c == 1]
    return "0" if len(tokens) == 0 else tokens[0]

def token_to_oh(index, vocab_size):
    x = zeros((vocab_size))
    x[index] = 1
    return x

def char_padded(sequence, pad, maxlen):
    return [pad if i >= len(sequence) else sequence[i] for i in range(maxlen)]

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-90, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def gettop5(p, vocab):
        top5 = tf.math.top_k(p, k=5)
        top5 = [(vocab[tix], top5.values[ix].numpy()) for (ix, tix) in enumerate(top5.indices)]
        return top5

def logtop5(p, vocab):
        print(gettop5(p,vocab))

def sample( model, seqlen, vocab, reverse_token_map, temp=1):
    vocab_size = len(vocab)
    token_ix = -1
    # output = "hello this is nodle and you know what the crowd says i just gotta say this thing that is on my mind grapes ya know"
    # output = "you know what the crowd says "
    # s = ["you"," ","know"," ", "what", " ", "the", " ", "crowd", " ", "says", " "]
    # output = "<START> finally, someone made an image out of the sensible argument for fahrenheit everywhere as human readable temperature"
 
    output ="<START> when i walk crowd i put on sandals mostly because the floor is dirty now with the sand and salt and whatnot in the winter"
    output ="<START> this fast dog gum jump sad distinct surely dank flavor dingus is the a to and of sure for a nice ten second job stuff shit guys"
    s = output.split(" ")
    spaces = [" " for i in range(len(s))]
    s = [val for pair in zip(s,spaces) for val in pair]
    # inpt = ["<START>" for i in range(seqlen)]
    inpt = [s[i] if i < len(s) else "<START>" for i in range(seqlen)]
    output = ""

    attn_4_output = model.get_layer("attention_values_4").output
    attn_out = model.get_layer("dense_v_4").output
    dense_v_out = model.get_layer("dense_v_4").output
    einsum_com_output = model.get_layer("einsum_com_4").output
    input_layer = model.get_layer("input")
    attn_factor_model = keras.Model(inputs=input_layer.input, outputs=attn_4_output)
    einsum_com_model = keras.Model(inputs=input_layer.input, outputs=einsum_com_output)
    dense_v_model = keras.Model(inputs=input_layer.input, outputs=dense_v_out)

    x = [get_ix_from_token(reverse_token_map, token) for token in inpt]
    x = np.asarray(x)
    x = x.reshape((1, seqlen))
    attn_out = attn_factor_model.predict(x, verbose=0)[0]
    preds = model.predict(x, verbose=0)[0]
    top5s = ["\'%s\'" % gettop5(p,vocab)[0][0] for p in preds]

    fig, ax = plt.subplots()
    lbls = ["\'%s\'" % t for t in inpt]
    im, cbar = heatmap(attn_out[0,:,:], lbls, lbls, ax=ax,
                   cmap="YlGn", cbarlabel="attn")
    plt.savefig("attention_heatmap_%d.jpg" % 0)
    for i in range(1,attn_out.shape[0]):
        im = ax.imshow(attn_out[i,:,:],cmap="YlGn")
        
        #texts = annotate_heatmap(im, valfmt="{x:.1f} t")

        #fig.tight_layout()
        #plt.show()
        plt.savefig("attention_heatmap_%d.jpg" % i)
    return ""

    print(attn_out)
    print(attn_out.shape)
    print(inpt)
    print("===============================================================================================")
    for h in range(attn_out.shape[0]):
        for i in range(attn_out.shape[1]):
            print(inpt[i])
            print([(inpt[j],attn_out[h,i,j]) for j in range(attn_out.shape[2]) if attn_out[h,i,j] != 0])
            logtop5(preds[i],vocab)
        print("===============================================================================================")
    return ""

    mintokens = 15
    maxtokens = 100
    # i = len(s)-1
    i = 0
    while i < maxtokens and (i < mintokens or token_ix != reverse_token_map['<START>']):
        x = [get_ix_from_token(reverse_token_map, token) for token in inpt]
        x = np.asarray(x)
        x = x.reshape((1, seqlen))
        preds = model.predict(x, verbose=0)[0][min(i, seqlen - 1)]
        logtop5(preds, vocab)
        # topk = tf.math.top_k(preds, k=10)
        # topk_preds = keras.layers.Softmax()(topk.values/temp)
        # s = np.random.multinomial(1,topk_preds)
        # token_ix = topk.indices[np.flatnonzero(s)[0]]
        # token_ix = np.random.choice(topk.indices, p=topk_preds)
        token_ix = np.random.choice(range(vocab_size), p=preds.ravel())
        retries = 0
        while retries < 10 and token_ix in [reverse_token_map["<UNK>"]]:
            #token_ix = np.random.choice(topk.indices, p=topk_preds)
            #s = np.random.multinomial(1,topk_preds) 
            #token_ix = topk.indices[np.flatnonzero(s)[0]]
            token_ix = np.random.choice(range(vocab_size), p=preds.ravel())
            retries += 1
        new_token = vocab[token_ix]
        print(inpt)
        print(inpt[min(i,seqlen-1)] + " ----> " + new_token)
        output += new_token
        if (i + 1 < len(inpt)):
            inpt[i+1] = new_token
        else:
            inpt = inpt[1:] + [new_token]
        i += 1
    return output


def sample_seq2seq(model, seqlen, vocab, reverse_token_map):
    seqlen = seqlen
    vocab_size = len(vocab)
    print(model.summary())
    print(vocab[:100])
    token_ix = -1
    inpt = ["START" for i in range(seqlen)]
    output = ""
    mintokens = 10
    maxtokens = 1000
    i = 0
    while i < maxtokens and (i < mintokens or token_ix != reverse_token_map['START']):
        x = zeros((1, seqlen, vocab_size))
        x[0] = [token_to_oh(get_ix_from_token(reverse_token_map, token), vocab_size) for token in inpt]
        print("PREDICTING....")
        print(x.shape)
        preds = model.predict(x, verbose=0)[0][min(i, seqlen - 1)]
        print("GOT PREDICTION....")
        token_ix = choice(range(vocab_size), p=preds.ravel())
        while token_ix == reverse_token_map["<UNK>"]:
            token_ix = choice(range(vocab_size), p=preds.ravel())
        new_token = vocab[token_ix]
        output += new_token
        if (i + 1 < len(inpt)):
            inpt[i + 1] = new_token
        else:
            inpt = inpt[1:] + [new_token]
        i += 1
    return output

def load_vocab(fname):
    #fname = os.path.join(efs_path,"vocab.tsv")
    with open(fname, "r") as fp:
        vocab = fp.read().split("\t")
    return vocab

def transformer_encoder(x, i, reg,  n_a, ffdim, seqlen, dropout_rate, mask=None):
    # Embedding, self-attention, dropout, residual layerNorm, ffn, residual layerNorm
    m = tf.shape(x)[0]
    attention_heads = 4

    attn_layer = keras.layers.MultiHeadAttention(attention_heads, n_a//attention_heads)
    attn_out = attn_layer(x,x,x, attention_mask=mask)
    # attn_out = multihead_attention(x, x, x, attention_heads, n_a, m, reg, dropout_rate, mask=mask)
    #     attn_out = tf.reshape(out, (m,seqlen*n_a))

    x = keras.layers.LayerNormalization(epsilon=1e-6, name="encoder_{}/attn_norm".format(i))(x + attn_out)

    # Feed-forward layer
    ffn = keras.Sequential(
        [
            keras.layers.Dense(ffdim, kernel_regularizer=reg, activation="relu"),
            keras.layers.Dense(n_a, kernel_regularizer=reg),
        ],
        name="encoder_{}/ffn".format(i),
    )

    ffn_out = ffn(x)
    ffn_out = keras.layers.Dropout(dropout_rate, name="encoder_{}/ffn_dropout".format(i))(ffn_out)

    x = keras.layers.LayerNormalization(epsilon=1e-6, name="encoder_{}/ffn_norm".format(i))(x + ffn_out)
    return x

def subsequent_mask( shape):
    "Mask out subsequent positions."
    subsequent_mask = np.triu(np.ones(shape), k=1).astype('uint8')
    return subsequent_mask == 0

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

def create_model( vocab, mask=None):
    seqlen = 40
    reg_factor = 0
    n_a = 128
    ffdim = 512
    dropout_rate = 0.1
    transformer_layers=5
    vocab_size = len(vocab)
    reg = keras.regularizers.l2(reg_factor)

    inpt = keras.layers.Input(shape=(seqlen), name="input")
    targets = keras.layers.Input(shape=(seqlen), name="targets")
    out = keras.layers.Embedding(vocab_size, n_a, input_length=seqlen)(inpt)
    target_emb = keras.layers.Embedding(vocab_size, n_a, input_length=seqlen)(targets)
    pos_enc = positional_encoding(seqlen, n_a)
    pos_emb = keras.layers.Embedding(
        input_dim=seqlen,
        output_dim=n_a,
        weights=[pos_enc],
        name="position_embedding",
    )(tf.range(start=0, limit=seqlen, delta=1))
    encoder_out = out + pos_emb
    target_emb = target_emb + pos_emb
    m = tf.shape(out)[0]
    mask = subsequent_mask(seqlen)
    for i in range(transformer_layers):
        encoder_out = transformer_encoder(encoder_out, i, reg, n_a, ffdim, seqlen, dropout_rate, mask=mask)
    # decoder_out = transformer_decoder(encoder_out, target_emb, reg, mask)
    out = keras.layers.Dense(n_a, activation="relu", kernel_regularizer=reg)(encoder_out)
    out = keras.layers.Dense(vocab_size, activation="softmax", kernel_regularizer=reg)(out)

    # masked_model = MaskedLanguageModel(inputs=inpt, outputs=masked_out)
    # return masked_model

    model = keras.Model(inputs=inpt, outputs=out)
    return model

def main():
    slackurl = "https://hooks.slack.com/services/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"
    #modelpath = os.path.join(efs_path, "model.h5")
    modelpath = "model.h5"
    vocab_path = "vocab.tsv"
    print("LOADING VOCAB")
    vocab = load_vocab(vocab_path)
    print("LOADING MODEL")
    print(modelpath)
    print(os.path.exists(modelpath))
    model = load_model(modelpath, custom_objects={"EinsumOp": EinsumOp})
    #model = create_model(vocab)
    #model.load_weights(modelpath)
    # print(model.summary())
    reverse_token_map = {t: i for i, t in enumerate(vocab)}
    # seqlen = model.get_layer(name="input").input_shape[0][1]
    seqlen = 40
    # vocab_size = model.get_layer(name="dense").output_shape[-1]
    output = sample(model, seqlen, vocab, reverse_token_map)
    msg = output.replace("START","")
    payload = {"text": msg}
    print(payload)
    #resp = requests.post(slackurl, headers = {"Content-type": "application/json"}, data=json.dumps(payload))
    #print(resp)
    return payload

if __name__=="__main__":
    main()

