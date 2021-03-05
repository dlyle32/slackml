import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization


def custom_standardization(input_string):
    """ Remove html line-break tags and handle punctuation """
    lowercased = tf.strings.lower(input_string)
    stripped_html = tf.strings.regex_replace(lowercased, "<br />", " ")
    return tf.strings.regex_replace(stripped_html, f"([{string.punctuation}])", r" \1")

def prepare_lm_inputs_labels(text):
    """
    Shift word sequences by 1 position so that the target for position (i) is
    word at position (i+1). The model will use all words up till position (i)
    to predict the next word.
    """
    text = tf.expand_dims(text, -1)
    tokenized_sentences = vectorize_layer(text)
    x = tokenized_sentences[:, :-1]
    y = tokenized_sentences[:, 1:]
    return x, y



class TFVectTokenizer:

    def __init__(self,seqlen, step, freq_threshold):
        self.freq_threshold = freq_threshold
        self.freq_threshold = 0
        self.seqlen = seqlen
        self.step = step
        self.vocab_size = 20000

    def tokenize(self, text_ds):
        # Create a vectorization layer and adapt it to the text
        vectorize_layer = TextVectorization(
            standardize=custom_standardization,
            max_tokens=self.vocab_size - 1,
            output_mode="int",
            output_sequence_length=self.seqlen + 1,
        )
        vectorize_layer.adapt(text_ds)
        vocab = vectorize_layer.get_vocabulary()  # To get words back from token indices
        reverse_token_map = {t: i for i, t in enumerate(vocab)}
        return vocab, text_ds, reverse_token_map

    def get_input_sequences(self, text_ds, reverse_token_map):
        text_ds = text_ds.map(prepare_lm_inputs_labels)
        text_ds = text_ds.prefetch(tf.data.experimental.AUTOTUNE)
        return text_ds
