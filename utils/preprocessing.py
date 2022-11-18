import numpy as np
import pandas as pd
import re

DATASET_YEAR_START = 1991


def length_data(concat_df):
    for date in range(DATASET_YEAR_START, 2022):
        date_df = concat_df[(concat_df["time"] == date)]
        print(f"{date} data length : {len(date_df)}")


def data_to_timeseries():
    """
    input : None
    output : None
    시간 정보가 없는 vector csv 파일에 원본의 time information을 추가하는 함수
    """
    time_df = pd.read_csv("../dataset_timeseries/kr_en_v.csv")
    time_df = time_df[["time"]]

    kr_doc_df = np.load("../dataset/kr_doc_v.npy")
    kr_doc_df = pd.DataFrame(kr_doc_df)
    kr_doc_df = pd.concat([time_df, kr_doc_df], axis=1)

    en_doc_df = np.load("../dataset/en_doc_v.npy")
    en_doc_df = pd.DataFrame(en_doc_df)
    en_doc_df = pd.concat([time_df, en_doc_df], axis=1)

    concat_df = pd.concat([kr_doc_df, en_doc_df], axis=0)
    concat_df = concat_df.sort_values(by="time", ascending=True)

    for date in range(2011, 2021):
        date_df = concat_df[(concat_df["time"] == date)]
        date_df.to_csv(f"../dataset_timeseries/{date}.csv")


def concat_ko_en_vec():
    time_df = pd.read_csv("../dataset/kr_en_10000_bench.csv")
    time_df = time_df[["time"]]

    kr_doc_df = np.load("../dataset/kr_doc_v.npy")
    kr_doc_df = pd.DataFrame(kr_doc_df)
    kr_doc_df = pd.concat([time_df, kr_doc_df], axis=1)

    en_doc_df = np.load("../dataset/en_doc_v.npy")
    en_doc_df = pd.DataFrame(en_doc_df)
    en_doc_df = pd.concat([time_df, en_doc_df], axis=1)

    concat_df = pd.concat([kr_doc_df, en_doc_df], axis=0)
    concat_df = concat_df.sort_values(by="time", ascending=True)

    concat_df.to_csv(f"../dataset_timeseries/kr_en_v.csv")


def concat_monolingual_vec(language):
    time_df = pd.read_csv("../dataset/kr_en_10000_bench.csv")
    time_df = time_df[["time"]]

    doc_df = np.load(f"../dataset/{language}_doc_v.npy")
    doc_df = pd.DataFrame(doc_df)
    doc_df = pd.concat([time_df, doc_df], axis=1)

    doc_df.to_csv(f"../custom_dataset/{language}_v.csv")


def concat_ko_en_wordlist():
    kr_word = pd.read_csv("../dataset/kr_word_list.csv")
    en_word = pd.read_csv("../dataset/en_word_list.csv")
    words = pd.concat([kr_word, en_word], axis=0)
    words.to_csv(f"../custom_dataset/concat_words.csv", index=None)


if __name__ == "__main__":
    data_to_timeseries()
