import pandas as pd
import numpy as np


def csv_to_numpy(csvpath, csv_type):
    if csv_type is "centroid":
        df = pd.read_csv(csvpath)
        header = df.columns.to_numpy()
        header = np.reshape(header, ((1,) + header.shape))
        new_df = df.to_numpy()
        new_df = np.concatenate((header, new_df), axis=0)
    elif csv_type is "kalman":
        df = csvpath
        df = df.drop("time", axis=1)
        df = df.drop("cluster", axis=1)
        new_df = df.to_numpy()
    elif csv_type is "word":
        df = pd.read_csv(csvpath)
        df = df.drop("word_list", axis=1)
        new_df = df.to_numpy()
    else:
        df = pd.read_csv(csvpath, index_col=[0])
        df = df.drop("time", axis=1)
        new_df = df.to_numpy()
    return new_df
