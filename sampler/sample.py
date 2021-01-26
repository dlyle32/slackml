from tensorflow.keras.models import load_model
import numpy as np
import json
import os
import argparse
import requests

def get_ix_from_token(reverse_token_map, token):
    if token in reverse_token_map.keys():
        return reverse_token_map[token]
    else:
        return reverse_token_map["<UNK>"]

def oh_to_token(vocab, oh):
    tokens = [vocab[i] for i,c in enumerate(oh) if c == 1]
    return "0" if len(tokens) == 0 else tokens[0]

def token_to_oh(index, vocab_size):
    x = np.zeros((vocab_size))
    x[index] = 1
    return x

def char_padded(sequence, pad, maxlen):
    return [pad if i >= len(sequence) else sequence[i] for i in range(maxlen)]


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

def load_vocab():
    fname = "vocab.tsv"
    with open(fname, "r") as fp:
        vocab = fp.read().split("\t")
    return vocab

def respond(err, res=None):
    return {
        'statusCode': '400' if err else '200',
        'body': err.message if err else json.dumps(res),
        'headers': {
            'Content-Type': 'application/json',
        },
    }

def lambda_handler(event, context):
    '''Demonstrates a simple HTTP endpoint using API Gateway. You have full
    access to the request and response payload, including headers and
    status code.

    To scan a DynamoDB table, make a GET request with the TableName as a
    query string parameter. To put, update, or delete an item, make a POST,
    PUT, or DELETE request respectively, passing in the payload to the
    DynamoDB API as a JSON body.
    '''
    #print("Received event: " + json.dumps(event, indent=2))

    operations = {
        'DELETE': lambda dynamo, x: dynamo.delete_item(**x),
        'GET': lambda dynamo, x: dynamo.scan(**x),
        'POST': lambda dynamo, x: dynamo.put_item(**x),
        'PUT': lambda dynamo, x: dynamo.update_item(**x),
    }
    sample = main()
    return respond(None, sample)

    operation = event['httpMethod']
    if operation in operations:
        payload = event['queryStringParameters'] if operation == 'GET' else json.loads(event['body'])
        return respond(None, operations[operation](dynamo, payload))
    else:
        return respond(ValueError('Unsupported method "{}"'.format(operation)))


def main():
    slackurl = "https://hooks.slack.com/services/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"
    modelpath = "model.h5"
    model = load_model(modelpath)
    vocab = load_vocab()
    reverse_token_map = {t: i for i, t in enumerate(vocab)}
    seqlen = model.get_layer(name="input").input_shape[0][1]
    vocab_size = model.get_layer(name="dense").output_shape[-1]
    print(model.summary())
    output = sample_seq2seq(model, seqlen, vocab, reverse_token_map)
    msg = output.replace("START","")
    payload = {"text": msg }
    print(payload)
    resp = requests.post(slackurl, headers = {"Content-type": "application/json"}, data=json.dumps(payload))
    return payload

if __name__ == "__main__":
    main()
