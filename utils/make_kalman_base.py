import operator
import pandas as pd
import numpy as np
import pickle
from .csv_to_numpy import csv_to_numpy
import os.path
from .kmeans import clustering


def make_origin_df(datapath):
    if "Ai" in datapath:
        origin_df = np.load(f"{datapath}/ai_doc_kr.npy")
        shape = origin_df.shape[1]
        origin_df = pd.DataFrame(origin_df)

        time_df = pd.read_csv(f"{datapath}/data.csv")["time"]
        time_df = time_df[: origin_df.shape[0]]  # TMP

        label_centroid_df = origin_df
        origin_df = pd.concat([time_df, origin_df], axis=1)
    else:
        origin_df = pd.read_csv(f"{datapath}/kr_en_v.csv")
        shape = origin_df.shape[1]
        if shape == 770:
            origin_df.drop(["Unnamed: 0"], axis=1, inplace=True)
            label_centroid_df = origin_df.drop(columns=["time"], axis=1)

    return origin_df, label_centroid_df, shape


def make_cluster_information_csv(folderpath, datapath, n_cluster):
    # origin dataset에 KMeans를 거친 label을 추가한다.
    origin_df, label_centroid_df, shape = make_origin_df(datapath)
    # origin_df, label_centroid_df, shape = make_origin_df("../dataset_timeseries")

    if os.path.isfile(f"{folderpath}/{shape}/label") and os.path.isfile(
        f"{folderpath}/{shape}/centroids"
    ):
        pass
    else:
        if not os.path.isdir(f"{folderpath}/{shape}"):
            os.makedirs(f"{folderpath}/{shape}")
        print("Trying to make label of original data.")
        label, centroids, _ = clustering(label_centroid_df, "kmeans", n_cluster)
        with open(f"{folderpath}/{shape}/label", "wb") as fp:
            pickle.dump(label, fp)
        with open(f"{folderpath}/{shape}/centroids", "wb") as fp:
            pickle.dump(centroids, fp)

    with open(f"{folderpath}/{shape}/label", "rb") as fp:
        label = pickle.load(fp)
        print("label is loaded.")
    with open(f"{folderpath}/{shape}/centroids", "rb") as fp2:
        centroids = pickle.load(fp2)
        print("Centroids is loaded.")

    label_df = pd.DataFrame(label, columns=["label"])
    origin_df.insert(1, "cluster", label_df)

    return origin_df, centroids


def find_docx_centroid(docx):
    # 여러 docx의 중심 vector를 찾는 함수
    # print(docx.mean())
    docx_np = csv_to_numpy(docx, "kalman")
    centroid = docx_np.mean(axis=0)
    return centroid


def cluster_to_time_centroid(n_cluster, start, end, origin_df):
    """
    n번째 cluster에 지정한 time dataset이 얼마나 존재하는지 셈한다.
    input = n_cluster
    output = 2차원 list 2개
            centroid_list : [n번째 cluster][m time의 문서 중심]
            docx_count_list : [n번째 cluster][m time의 문서 개수]
    """
    centroid_list = [[] for i in range(n_cluster)]
    docx_count_list = [[] for i in range(n_cluster)]

    for i in range(n_cluster):  # n번째 cluster에 대해
        n_cluster = origin_df[origin_df["cluster"] == i]
        for j in range(start, end + 1):  # m번째 time의 문서 개수 / 중심(mean)
            time_docx = n_cluster[n_cluster["time"] == j]
            # TODO : cluster 개수가 많아지면 cluster가 없을 수도 있음. 그럴 경우엔 어떻게?
            docx_count_list[i].append(len(time_docx))
            centroid_list[i].append(find_docx_centroid(time_docx))
    centroid_list = np.array(centroid_list)
    docx_count_list = np.array(docx_count_list)
    # print(docx_count_list.shape)  # (n_cluster, len_time)
    # print(centroid_list.shape)  # (n_cluster, len_time, len_centroid_vector)
    return centroid_list, docx_count_list


def make_base_scala(docx_count_list):
    """
    cluster의 문서 개수를 기반으로, 차년도 문서가 몇 개 생성될지 예측하는 task
    해당 task에 사용하기 위해 변수 반환
    input : docx_count_list(n_cluster, len_time)
    output : measurement, motion_control
            measurement(list) = cluster 내부의 연도별 문서 개수
                measurement[0](list) = 0번째 cluster의 연도별 문서 개수
            motion_control(list) = 연도별 문서 개수의 변화량(scala)
    """
    motion_controls = [[] for i in range(len(docx_count_list))]
    measurements = docx_count_list

    total_sum = np.sum(measurements)  # 전체 문서의 합계

    for idx1, cluster in enumerate(measurements):
        for idx2 in range(len(cluster) - 1):
            motion_controls[idx1].append(cluster[idx2 + 1] - cluster[idx2])

    motion_controls = np.array(motion_controls)
    measurements = np.array(measurements)

    return motion_controls, measurements


def make_base_vector(centroid_list, whole_centroid_list):
    """
    cluster의 중심 좌표를 기반으로, 차년도 중심 좌표가 어디에 생성될지 예측하는 task
    해당 task에 사용하기 위해 변수 반환
    input :     centroid_list(n_cluster, len_time, vector dim)
                whole_centroid_list(n_cluster, vector dim)
    output :    measurement, motion_control
                motion_control(list) =  전체 데이터의 중심좌표와의 연도별 중심좌표 사이의 변화량(vector)
                                        (n_cluster, len_time, vector dim)
                measurement(list) = cluster 내부의 연도별 중심 좌표 (n_cluster, len_time, vector dim)
    """

    motion_controls = [[] for i in range(len(centroid_list))]
    measurements = centroid_list

    for idx1, cluster in enumerate(measurements):
        for idx2 in range(len(cluster) - 1):
            motion_controls[idx1].append(cluster[idx2 + 1] - cluster[idx2])

    # if you want to compare to whole centroid vec, use this code
    # for idx1, cluster in enumerate(measurements):
    #     whole_centorid = np.expand_dims(whole_centroid_list[idx1], axis=0).astype(
    #         np.float
    #     )
    #     cluster = cluster - whole_centorid
    #     motion_controls[idx1] = cluster

    motion_controls = np.array(motion_controls)
    # print(motion_controls.shape)
    return motion_controls, measurements


def make_base(folderpath, args):

    datapath = args.datapath
    n_cluster = args.n_cluster
    start = args.start_date
    end = args.end_date

    origin_df, whole_centroid_list = make_cluster_information_csv(
        folderpath, datapath, n_cluster
    )
    # print(origin_df.shape, whole_centroid_list.shape)

    centroid_list, docx_count_list = cluster_to_time_centroid(
        n_cluster, start, end, origin_df
    )

    # centroid_list[n][m] : n번 cluster의 m time에서의 중심점
    # docx_count_list[n][m] : n번 cluster의 m time의 문서 개수
    # whole_centroid_list : 전체 dataset에서 n개로 clustering한 좌표. (n_cluster, len_centroid_vector)

    scala_motion_controls, scala_measurements = make_base_scala(docx_count_list)
    vector_motion_controls, vector_measurements = make_base_vector(
        centroid_list, whole_centroid_list
    )
    return (
        scala_motion_controls,
        scala_measurements,
        vector_motion_controls,
        vector_measurements,
    )


if __name__ == "__main__":
    make_base(15, 2011, 2021)
