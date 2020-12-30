import numpy as np

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

def create_oh(int_array, vocab_size):
    oh_shape = int_array.shape + (vocab_size,)
    rows = int_array.shape[-2]
    cols = int_array.shape[-1]

    rindices = np.arange(rows).reshape(rows,1)
    cindices = np.arange(cols).reshape(1,cols)

    oh = np.zeros(oh_shape)

    oh[rindices, cindices, int_array] = 1
    return oh

def char_padded(sequence, pad, maxlen):
    return [pad if i >= len(sequence) else sequence[i] for i in range(maxlen)]
