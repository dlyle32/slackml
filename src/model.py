import os
import pandas as pd
import numpy as np

# def model(X, Y, layers_dims, optimizer, learning_rate = 0.0007, mini_batch_size = 64, beta = 0.9,
#           beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8, num_epochs = 10000, print_cost = True):

def main(features_dir):
    files = os.listdir(features_dir)
    trainx = pd.read_csv(features_dir+"trainx_0.csv")
    trainy = pd.read_csv(features_dir+"trainy_0.csv")

if __name__ == "__main__":
    features_dir = "/Users/davidlyle/slack_ml/features/"
    main(features_dir)