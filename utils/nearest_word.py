from .csv_to_numpy import csv_to_numpy
import pandas as pd
import numpy as np
import tqdm

from kalman_filter import euclidean_distance


def get_neighbors(centroid, words, num_neighbors):
    distances = list()
    for idx, word in enumerate(words):
        dist = euclidean_distance(centroid, word)
        distances.append((idx, dist))
    distances.sort(key=lambda tup: tup[1])
    distances = distances[:num_neighbors]

    neighbor = [i[0] for i in distances]
    distance = [i[1] for i in distances]

    return neighbor, distance


def find_nearest_word(n_cluster, n_word):
    whole_centroid_list = csv_to_numpy(
        f"../dataset_timeseries/kmean/{n_cluster}/whole/whole_centroid_list.csv",
        "centroid",
    )
    kr_word_v = np.load("../dataset/kr_word_v.npy")
    en_word_v = np.load("../dataset/en_word_v.npy")
    words_v = np.concatenate([kr_word_v, en_word_v])

    words = pd.read_csv("../dataset_timeseries/words.csv")
    words.to_csv(f"../dataset_timeseries/words.csv", index=None)

    total_word_list = []
    for idx, centroid in enumerate(tqdm.tqdm(whole_centroid_list)):
        neighbors, distances = get_neighbors(centroid, words_v, n_word)
        centroid_word_list = []
        for neighbor in neighbors:
            centroid_word_list.append(words["word_list"][neighbor])
        total_word_list.append(centroid_word_list)

    for i in range(len(total_word_list)):
        print(f"centroid {i} : word {total_word_list[i]}")


if __name__ == "__main__":
    find_nearest_word(15, 10)
