import os
import random
import json

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

def get_user_msg_context(data,i, context_len = 40):
    context = []
    while len(context) < context_len and i > 0:
        event = data[i]
        if is_event_valid(event, word_threshold=0):
            context = [event["user"]] + event["text"].split(" ") + context
        i -= 1
    return " ".join(context)

def process_file_context_target(fname, target_user):
    msgs_with_context = []
    with open(fname, 'r') as fp:
        data = json.load(fp)
        for i, event in enumerate(data):
            if is_event_valid(event, word_threshold=0) and event["user"] == target_user:
                context = get_user_msg_context(data, i-1)
                target_msg = event["text"]
                msgs_with_context.append((context, target_msg))
    return msgs_with_context

def load_context_target_pairs(datadir, training_threshold=0.9):
    context_target_pairs = []
    for r, d, f in os.walk(datadir):
        for file in f:
            fname = os.path.join(r,file)
            pairs = process_file_context_target(fname, target_user="U0AR782AV")
            context_target_pairs.extend(pairs)
    return context_target_pairs, []

