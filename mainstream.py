import pandas as pd
import numpy as np

from kalman_filter import kalman_filter_scala, kalman_filter_vector, average
from utils.arguments import get_args
from utils.make_kalman_base import make_base, make_cluster_information_csv
from utils.nearest import find_nearest_word, find_nearest_docx
from utils.visualization import centroid_visualization_2D, scala_visualization_2D


def main():
    args = get_args()
    custompath = f"../label/{args.n_cluster}"  # 연산을 위해 새롭게 생성된/로드할 데이터의 위치
    # data preprocessing
    (
        scala_motion_controls,
        scala_measurements,
        vector_motion_controls,
        vector_measurements,
    ) = make_base(custompath, args)

    # print(scala_measurements)
    tmp = pd.DataFrame(
        scala_measurements,
        columns=[str(i) for i in range(args.start_date, args.end_date + 1)],
    )
    tmp.to_csv("./tmptmptmp.csv", index=False)
    origin_df, whole_centroid_list = make_cluster_information_csv(
        custompath, args.datapath, args.n_cluster
    )
    print(origin_df)
    print("----------")

    # scala_df = pd.DataFrame(scala_measurements[:, -1])
    # scala_df.to_csv("scala_goldlabel.csv", index=False)

    # scala kalman filter (문서 개수) => 추후 시계열(VAR)로 동작 예정
    predict = kalman_filter_scala(scala_motion_controls, scala_measurements)
    average(scala_measurements, args.average_years, "scala")
    # for i in range(args.n_cluster):
    #     scala_visualization_2D(custompath, predict, scala_measurements, args, i)

    # vector kalman filter (좌표)
    predict = kalman_filter_vector(vector_motion_controls, vector_measurements)
    average(vector_measurements, args.average_years, "vector")
    # for i in range(args.n_cluster):
    #     centroid_visualization_2D(custompath, predict, vector_measurements, args, i)
    # predict 하나로 합쳐서 잘못 찍고 있음. 다시 찍어보기 => 실제로 있는 이슈 맞음

    # centroid로 nearest words, nearest docx 탐색   # TODO : (이거 2중for문 말고 numpy로 한번에 연산 필요)
    find_nearest_word(custompath, args, whole_centroid_list)

    # nearest docx와 nearest words 사이의 좌표 비교  # TODO : (이것도)
    find_nearest_docx(custompath, args, origin_df, whole_centroid_list)


if __name__ == "__main__":
    main()
