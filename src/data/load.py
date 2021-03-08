import os
import random
import json
import tensorflow as tf

def imdb_data_load(directory):
    batch_size=128
    filenames = []
    directories = [
        os.path.join(directory,"train/pos"),
        os.path.join(directory,"train/neg"),
        os.path.join(directory,"test/pos"),
        os.path.join(directory,"test/neg"),
    ]
    for dir in directories:
        for f in os.listdir(dir):
            filenames.append(os.path.join(dir, f))

    print(f"{len(filenames)} files")

    # Create a dataset from text files
    random.shuffle(filenames)
    text_ds = tf.data.TextLineDataset(filenames)
    text_ds = text_ds.shuffle(buffer_size=256)
    text_ds = text_ds.batch(batch_size)
    return text_ds

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

def is_event_valid(event, word_threshold = 2):
    is_user_message = event.get("type") == "message" and event.get("user") and event.get("text");
    words = event.get("text").split(" ") if event.get("text") != None else []
    return is_user_message and len(words) > word_threshold

def get_user_msg_context(data,i, context_len):
    context = []
    while len(context) < context_len and i > 0:
        event = data[i]
        if is_event_valid(event, word_threshold=0):
            context = [event["user"]] + event["text"].split(" ") + context
        i -= 1
    return " ".join(context)

def process_file_context_target(fname, target_user, context_len):
    msgs_with_context = []
    with open(fname, 'r') as fp:
        data = json.load(fp)
        for i, event in enumerate(data):
            if is_event_valid(event, word_threshold=0) and event["user"] == target_user:
                context = get_user_msg_context(data, i-1, context_len)
                target_msg = event["text"] + "\n"
                if len(context) <= 1:
                    continue
                msgs_with_context.append((context, target_msg))
    return msgs_with_context

def load_context_target_pairs(datadir, training_threshold=0.9, context_len = 40):
    context_target_pairs = []
    for r, d, f in os.walk(datadir):
        for file in f:
            fname = os.path.join(r,file)
            pairs = process_file_context_target(fname, "U0AR782AV", context_len)
            context_target_pairs.extend(pairs)
    return context_target_pairs, []

