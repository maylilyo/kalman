from .csv_to_numpy import csv_to_numpy
import pandas as pd
import numpy as np
import tqdm
import pickle

from kalman_filter import euclidean_distance
from utils.tf_idf import tf_idf


def get_neighbors(centroid, words, num_neighbors):
    distances = []
    for idx, word in enumerate(words):
        dist = np.linalg.norm(centroid - word)
        distances.append((idx, dist))
    distances.sort(key=lambda tup: tup[1])
    distances = distances[:num_neighbors]

    neighbor = [i[0] for i in distances]
    distance = [i[1] for i in distances]

    return neighbor, distance


def find_nearest_word(custompath, args, whole_centroid_list):
    if "Ai" in args.datapath:
        words_v = np.load(f"{args.datapath}/ai_word_kr.npy")
        words = np.load(f"{args.datapath}/ai_word_list_kr.npy")
        words = pd.DataFrame(words, columns=["word_list"])

    else:
        kr_word_v = np.load(f"{args.datapath}/kr_word_v.npy")
        en_word_v = np.load(f"{args.datapath}/en_word_v.npy")
        words_v = np.concatenate([kr_word_v, en_word_v])
        words = pd.read_csv(f"../dataset_timeseries/words.csv")

    # with open(f"{custompath}/{words_v.shape[1]}/centroids", "rb") as fp2:
    #     whole_centroid_list = pickle.load(fp2)

    total_word_list = []
    for idx, centroid in enumerate(tqdm.tqdm(whole_centroid_list)):
        neighbors, _ = get_neighbors(centroid, words_v, args.n_word)
        total_word_list.append(words["word_list"].loc[neighbors].values.tolist())

    total_word_list = pd.DataFrame(total_word_list)
    total_word_list.to_csv(f"./results/wordlist.csv", index=False)


def find_nearest_docx(custompath, args, origin_df, whole_centroid_list):
    if "Ai" in args.datapath:
        doc_v = np.load(f"{args.datapath}/ai_doc_kr.npy")
        docs = pd.read_csv(f"{args.datapath}/data.csv")

    else:
        doc_v = np.load(f"{args.datapath}/kr_doc_v.npy")
        docs = pd.read_csv(f"{args.datapath}/kr_en_10000_bench.csv")

    # with open(f"{custompath}/{doc_v.shape[1]}/centroids", "rb") as fp:
    #     whole_centroid_list = pickle.load(fp)
    # print(whole_centroid_list)
    total_docs_list = []
    # TODO
    """
    Cluster로 분류하고 찾은 게 아니라 전체 data 중 가장 가까운 docs를 찾아서 생긴 연관성 문제
    """
    for idx, centroid in enumerate(tqdm.tqdm(whole_centroid_list)):
        doc_cluster = origin_df.loc[origin_df["cluster"] == idx]
        doc_cluster = doc_cluster.drop(["time", "cluster"], axis="columns")
        doc_v = doc_cluster.to_numpy()

        neighbors, distances = get_neighbors(centroid, doc_v, args.n_docs)
        neighbors_index = doc_cluster.iloc[neighbors].index.tolist()
        total_docs_list.append(docs["kr"].loc[neighbors_index].values.tolist())

    total_docs_df = pd.DataFrame(total_docs_list)
    total_docs_df.to_csv(f"./results/docxlist.csv", index=False)

    tf_idf_list = []
    for docs in total_docs_list:
        tf_idf_words = tf_idf(docs, args)
        tf_idf_list.append(tf_idf_words)

    tf_idf_list = pd.DataFrame(tf_idf_list)
    tf_idf_list.to_csv(f"./results/tf_idf.csv", index=False)
