import os
import random

def load_datasets(directory, training_threshold = 0.9):
    train_data = []
    test_data = []
    chars = set()
    for r, d, f in os.walk(directory):
        for file in f:
            if "csv" not in file:
                continue
            with open(os.path.join(r, file), 'r') as fp:
                for msg in fp:
                    msg = msg.lower()
                    if(random.random() < training_threshold):
                        train_data.append(msg)
                    else:
                        test_data.append(msg)
                    chars = chars.union(set(msg))
    return train_data, test_data