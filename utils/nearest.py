from .csv_to_numpy import csv_to_numpy
import pandas as pd
import numpy as np
import tqdm
import pickle

from kalman_filter import euclidean_distance


def get_neighbors(centroid, words, num_neighbors):
    distances = []
    for idx, word in enumerate(words):
        dist = np.linalg.norm(centroid - word) #
        distances.append((idx, dist))
    distances.sort(key=lambda tup: tup[1])
    distances = distances[:num_neighbors]

    neighbor = [i[0] for i in distances]
    distance = [i[1] for i in distances]

    # distance = np.linalg.norm(forecast - test)
    # distance_list.append(distance)
    # print(np.mean(np.array(distance_list)))
    return neighbor, distance


def find_nearest_word(custompath, args):
    if "Ai" in args.datapath:
        words_v = np.load(f"{args.datapath}/ai_word_kr.npy")
        words = np.load(f"{args.datapath}/ai_word_list_kr.npy")
        words = pd.DataFrame(words, columns=["word_list"])

    else:
        kr_word_v = np.load(f"{args.datapath}/kr_word_v.npy")
        en_word_v = np.load(f"{args.datapath}/en_word_v.npy")
        words_v = np.concatenate([kr_word_v, en_word_v])
        words = pd.read_csv(f"../dataset_timeseries/words.csv")

    with open(f"{custompath}/{words_v.shape[1]}/centroids", "rb") as fp2:
        whole_centroid_list = pickle.load(fp2)

    total_word_list = []
    for idx, centroid in enumerate(tqdm.tqdm(whole_centroid_list)):
        neighbors, _ = get_neighbors(centroid, words_v, args.n_word)
        total_word_list.append(words["word_list"].loc[neighbors].values.tolist())

    total_word_list = pd.DataFrame(total_word_list)
    total_word_list.to_csv(f"./results/wordlist.csv", index=False)


def find_nearest_docx(custompath, args):
    if "Ai" in args.datapath:
        doc_v = np.load(f"{args.datapath}/ai_doc_kr.npy")
        docs = pd.read_csv(f"{args.datapath}/data.csv")

    else:
        doc_v = np.load(f"{args.datapath}/kr_doc_v.npy")
        docs = pd.read_csv(f"{args.datapath}/kr_en_10000_bench.csv")

    with open(f"{custompath}/{doc_v.shape[1]}/centroids", "rb") as fp:
        whole_centroid_list = pickle.load(fp)

    total_docs_list = []
    for idx, centroid in enumerate(tqdm.tqdm(whole_centroid_list)):
        neighbors, distances = get_neighbors(centroid, doc_v, args.n_docs)
        centroid_word_list = []
        total_docs_list.append(docs["kr"].loc[neighbors].values.tolist())

    total_docs_list = pd.DataFrame(total_docs_list)
    total_docs_list.to_csv(f"./results/docxlist.csv", index=False)
    # for i in range(len(total_docs_list)):
    #     print(f"centroid {i} : docs {total_docs_list[i]}")
