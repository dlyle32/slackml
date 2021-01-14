import nltk
from models.helpers import get_ix_from_token, token_to_oh, oh_to_token, char_padded

class SlidingWindowTokenizer:
    def __init__(self, seqlen, step, freq_threshold):
        self.freq_threshold=freq_threshold
        self.seqlen = seqlen
        self.step = step
        self.tokenizer = nltk.RegexpTokenizer("\<START\>|\,|\.|\¯\\\_\(\ツ\)\_\/\¯|\<\@\w+\>|\:\w+\:|\/gif|_|\"| |\w+\'\w+|\w+|\n")

    def tokenize(self, data):
        tokens = self.tokenizer.tokenize("<START> ".join(data))
        token_counts = {}
        for t in tokens:
            if t not in token_counts.keys():
                token_counts[t] = 1
            else:
                token_counts[t] += 1
        freq_filtered = filter(lambda elem: elem[1] >= self.freq_threshold, token_counts.items())
        vocab = sorted([elem[0] for elem in list(freq_filtered)])
        vocab += ["<MASK>","<UNK>","<START>"]
        reverse_token_map = {t: i for i, t in enumerate(vocab)}
        return tokens, vocab, reverse_token_map

    def get_input_sequences(self, tokens, reverse_token_map):
        seqs = []
        for i in range(0, len(tokens) - self.seqlen, self.step):
            x0 = "<START>" if i == 0 else tokens[i - 1]
            last_ix = min(i + self.seqlen, len(tokens) - 1)
            padded_sequence = char_padded(tokens[i:last_ix], " ", self.seqlen)
            Yseq = [get_ix_from_token(reverse_token_map, token) for token in padded_sequence]
            Xseq = [get_ix_from_token(reverse_token_map, x0)] + Yseq[:-1]
            seqs.append((Xseq, Yseq))
        return seqs