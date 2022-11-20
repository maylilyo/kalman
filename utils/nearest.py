from .csv_to_numpy import csv_to_numpy
import pandas as pd
import numpy as np
import tqdm
import pickle

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


def find_nearest_word(folderpath, datapath, n_word):
    if "Ai" in datapath:
        words_v = np.load(f"{datapath}/ai_word_kr.npy")
        words = np.load(f"{datapath}/ai_word_list_kr.npy")
        words = pd.DataFrame(words, columns=["word_list"])
        print(words_v.shape, words.shape)
        return

    else:
        kr_word_v = np.load(f"{datapath}/kr_word_v.npy")
        en_word_v = np.load(f"{datapath}/en_word_v.npy")
        words_v = np.concatenate([kr_word_v, en_word_v])
        words = pd.read_csv(f"../dataset_timeseries/words.csv")

    with open(f"{folderpath}/{words_v.shape[1]}/centroids", "rb") as fp2:
        whole_centroid_list = pickle.load(fp2)

    total_word_list = []
    for idx, centroid in enumerate(tqdm.tqdm(whole_centroid_list)):
        neighbors, _ = get_neighbors(centroid, words_v, n_word)
        print(neighbors)
        centroid_word_list = []
        for neighbor in neighbors:
            centroid_word_list.append(words["word_list"][neighbor])
        total_word_list.append(centroid_word_list)

    for i in range(len(total_word_list)):
        print(f"centroid {i} : word {total_word_list[i]}")


def find_nearest_docx(custompath, datapath, n_docs):
    if "Ai" in datapath:
        doc_v = np.load(f"{datapath}/ai_doc_kr.npy")
        docs = pd.read_csv(f"{datapath}/data.csv")

    else:
        doc_v = np.load(f"{datapath}/kr_doc_v.npy")
        docs = pd.read_csv(f"{datapath}/kr_en_10000_bench.csv")

    with open(f"{custompath}/{doc_v.shape[1]}/centroids", "rb") as fp:
        whole_centroid_list = pickle.load(fp)

    total_docs_list = []
    for idx, centroid in enumerate(tqdm.tqdm(whole_centroid_list)):
        neighbors, distances = get_neighbors(centroid, doc_v, n_docs)
        centroid_word_list = []
        for neighbor in neighbors:
            centroid_word_list.append(docs["kr"][neighbor])
        total_docs_list.append(centroid_word_list)

    for i in range(len(total_docs_list)):
        print(f"centroid {i} : docs {total_docs_list[i]}")


if __name__ == "__main__":
    n_cluster = 70
    n_word = 10
    n_docs = 1
    custompath = f"../label/{n_cluster}"
    datapath = f"../dataset/"

    find_nearest_word(custompath, datapath, n_word)
    find_nearest_docx(custompath, datapath, n_docs)
