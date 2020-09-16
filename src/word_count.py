import os
import argparse
import json

def count_words(dir):
    words = {}
    for r, d, f in os.walk(dir):
        for file in f:
            print(file)
            with open(os.path.join(r,file),'r') as fp:
                for msg in fp:
                    for word in msg.split(" "):
                        word = word.strip()
                        word = word.strip("\n")
                        word = word.lower()
                        if word not in words:
                            words[word] = 1
                        else:
                            words[word] += 1
    with open("noodle_words.txt", "w") as fp:
        fp.write(json.dumps(sorted(words.items(), key=lambda x : x[1], reverse=True)))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", default="/slackdata/")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    count_words(args.datadir)