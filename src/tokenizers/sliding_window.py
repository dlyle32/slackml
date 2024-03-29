import nltk
from models.helpers import get_ix_from_token, token_to_oh, oh_to_token, char_padded
import re

class SlidingWindowTokenizer:
    def __init__(self, args):
        self.freq_threshold=args.freqthreshold
        self.len_threshold=args.lenthreshold
        # self.freq_threshold=0
        self.seqlen = args.seqlength
        self.step = args.step
        self.tokenizer = nltk.RegexpTokenizer("\<START\>|\,|\.|\¯\\\_\(\ツ\)\_\/\¯|\<\@\w+\>|\:\w+\:|\/gif|_|\"| |\w+\'\w+|\w+|\n")

        # self._re_word_start = r"[^\(\"\`{\[:;&\#\*@\)}\]\-,]"
        self._re_word_start = r"[^\(\"\`{\[;&\#\*\)}\]\-,]"
        """Excludes some characters from starting word tokens"""

        # self._re_non_word_chars = r"(?:[?!)\";}\]\*@\'\({\[])"
        self._re_non_word_chars = r"(?:[?!)\"\.;}\]\*\({\[])"
        """Characters that cannot appear within words"""

        self._re_multi_char_punct = r"(?:\-{2,}|\.{2,}|(?:\.[ \t\r\f\v]){2,}\.)"
        """Hyphen and ellipsis are multi-character punctuation"""

        self._word_tokenize_fmt = r"""(
                %(MultiChar)s
                |
                (?=%(WordStart)s)[^ \t\r\f\v]+?  # Accept word characters until end is found
                (?= # Sequences marking a word's end
                    [ \t\r\f\v]|                                 # White-space
                    $|                                  # End-of-string
                    %(NonWord)s|%(MultiChar)s|          # Punctuation
                    ,(?=$|[ \t\r\f\v]||%(NonWord)s|%(MultiChar)s) # Comma if at end of word
                )
                |
                [^ \t\r\f\v]
            )"""
        """Format of a regular expression to split punctuation from words,
        excluding period."""

        self._re_word_tokenizer = re.compile(
            self._word_tokenize_fmt
            % {
                "NonWord": self._re_non_word_chars,
                "MultiChar": self._re_multi_char_punct,
                "WordStart": self._re_word_start,
            },
            re.UNICODE | re.VERBOSE,
        )

    def _word_tokenizer_re(self):
        """Compiles and returns a regular expression for word tokenization"""
        try:
            return self._re_word_tokenizer
        except AttributeError:
            self._re_word_tokenizer = re.compile(
                self._word_tokenize_fmt
                % {
                    "NonWord": self._re_non_word_chars,
                    "MultiChar": self._re_multi_char_punct,
                    "WordStart": self._re_word_start,
                },
                re.UNICODE | re.VERBOSE,
            )
            return self._re_word_tokenizer

    def word_tokenize(self, s):
        """Tokenize a string to split off punctuation other than periods"""
        return self._word_tokenizer_re().findall(s)

    def tokenize(self, data):
        # tokens = self.tokenizer.tokenize("<START> ".join(data))
        # tokens = self.word_tokenize(" <START> ".join(data))
        tokens = []
        for msg in data:
            msg_tokens = self.word_tokenize(msg)
            if len(msg_tokens) < self.len_threshold:
                continue
            tokens.append("<START>")
            tokens.extend(msg_tokens)

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

    def slice_padded_sequence(self, tokens, start):
        end = min(start+self.seqlen, len(tokens) - 1)
        padded_sequence = char_padded(tokens[start:end], " ", self.seqlen)
        return padded_sequence

    def get_input_sequences(self, tokens, reverse_token_map):
        seqs = []
        for i in range(0, len(tokens) - self.seqlen, self.step):
            # x0 = "<START>" if i == 0 else tokens[i - 1]
            # last_ix = min(i + self.seqlen, len(tokens) - 1)
            # padded_sequence = char_padded(tokens[i:last_ix], " ", self.seqlen)
            X = self.slice_padded_sequence(tokens, i)
            Y = self.slice_padded_sequence(tokens, i+1)
            Xseq = [get_ix_from_token(reverse_token_map, token) for token in X]
            Yseq = [get_ix_from_token(reverse_token_map, token) for token in Y]
            # Yseq = [get_ix_from_token(reverse_token_map, token) for token in padded_sequence]
            # Xseq = [get_ix_from_token(reverse_token_map, x0)] + Yseq[:-1]
            seqs.append((Xseq, Yseq))
        return seqs
