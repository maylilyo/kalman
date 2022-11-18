import operator
import pandas as pd
import numpy as np
import pickle
from .csv_to_numpy import csv_to_numpy
import os.path
from .kmeans import clustering


def make_cluster_information_csv(folderpath, n_cluster):
    # origin dataset에 KMeans(n_cluster=15)를 거친 label을 추가한다.
    origin_df = pd.read_csv("../dataset_timeseries/kr_en_v.csv")
    if os.path.isfile(f"{folderpath}/label") and os.path.isfile(
        f"{folderpath}/cluster"
    ):
        pass
    else:
        os.makedirs(folderpath)
        print("Trying to make label of original data.")
        label, centroids, _ = clustering(origin_df, "kmeans", n_cluster)
        with open(f"{folderpath}/label", "wb") as fp:
            pickle.dump(label, fp)
        with open(f"{folderpath}/centroids", "wb") as fp:
            pickle.dump(label, fp)

    with open(f"{folderpath}/label", "rb") as fp:
        label = pickle.load(fp)
        print("label is loaded.")
    with open(f"{folderpath}/centroids", "rb") as fp2:
        centroids = pickle.load(fp2)
        print("Centroids is loaded.")

    label_df = pd.DataFrame(label, columns=["label"])
    origin_df.insert(1, "cluster", label_df)
    origin_df = origin_df.drop(["Unnamed: 0"], axis=1)

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
        # print(n_cluster)
        for j in range(start, end + 1):  # m번째 time의 문서 개수 / 중심(mean)
            time_docx = n_cluster[n_cluster["time"] == j]
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

    # 클러스터 내부 문서 / 모든 클러스터(전체) 문서 비율
    total_sum = np.sum(measurements)  # 전체 문서의 합계
    print(f"total sum is {total_sum}")
    measurements = np.divide(measurements, total_sum) * 100

    # 클러스터 내부 문서 / 클러스터별 전체 문서 비율
    # cluster_sum = measurements.sum(axis=1)  # 각 cluste 문서의 합계(2011~2021)

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
    print(motion_controls.shape)
    return motion_controls, measurements


def make_base(n_cluster, start, end):
    folderpath = f"../label/{n_cluster}"
    origin_df, whole_centroid_list = make_cluster_information_csv(folderpath, n_cluster)
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
