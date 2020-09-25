from tensorflow.keras.models import load_model
import numpy as np
import json

def sample(model_path, temperature=1.0):
    model = load_model(model_path)
    maxlen=model.layers[0].input_shape[1]
    vocab_size =model.layers[-1].output_shape[-1]
    char_to_ix = {}
    chars = json.loads('["\\t", "\\n", " ", "!", "\\"", "#", "$", "%", "&", "\'", "(", ")", "*", "+", ",", "-", ".", "/", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ":", ";", "<", "=", ">", "?", "@", "[", "\\\\", "]", "^", "_", "`", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "{", "|", "}", "~", "\\u00a0", "\\u00a3", "\\u00ae", "\\u00af", "\\u00b0", "\\u00e1", "\\u00e9", "\\u00ed", "\\u00ef", "\\u00f1", "\\u00f4", "\\u00f6", "\\u00f8", "\\u00fc", "\\u015f", "\\u01bd", "\\u02b0", "\\u02b3", "\\u0430", "\\u0432", "\\u0433", "\\u0435", "\\u0437", "\\u0438", "\\u043b", "\\u043c", "\\u043d", "\\u043e", "\\u043f", "\\u0440", "\\u0442", "\\u0447", "\\u044b", "\\u04af", "\\u0501", "\\u0b20", "\\u0c66", "\\u0f3c", "\\u0f3d", "\\u1d49", "\\u1d4d", "\\u1d52", "\\u1d57", "\\u1da6", "\\u2005", "\\u200a", "\\u2013", "\\u2014", "\\u2018", "\\u2019", "\\u201c", "\\u201d", "\\u2022", "\\u2026", "\\u2122", "\\u2500", "\\u2502", "\\u2514", "\\u251c", "\\u2588", "\\u25d5", "\\u267c", "\\u2683", "\\u3064", "\\u30c4", "\\uab87", "\\ud83d\\udd47", "\\ud83d\\udec7", "\\ud83e\\udd21", "\\ud83e\\udd47", "\\ud83e\\udd54", "<EOM>"]')
    chars = ['\t', '\n', ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~', '<UNK>']
    output = ""
    x = np.zeros((1, maxlen, vocab_size))
    char_index = -1
    i = 0
    while char_index != 1:
        preds = model.predict(x, verbose=0)[0][i]
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        char_index = np.argmax(probas)
        x[0,i,char_index] = 1
        output += chars[char_index]
        i+=1
    print(output)
    return

if __name__ == "__main__":
    for i in range(0,10):
        print("SAMPLING.......")
        sample("/slackml/mymodel3.keras")
