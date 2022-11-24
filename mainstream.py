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

    origin_df, whole_centroid_list = make_cluster_information_csv(
        custompath, args.datapath, args.n_cluster
    )

    # scala_df = pd.DataFrame(scala_measurements[:, -1])
    # scala_df.to_csv("scala_goldlabel.csv", index=False)

    # scala kalman filter (문서 개수) => 추후 시계열(VAR)로 동작 예정
    predict = kalman_filter_scala(scala_motion_controls, scala_measurements)
    average(scala_measurements, args.average_years, "scala")
    scala_visualization_2D(custompath, predict, scala_measurements, args, 1)

    # vector kalman filter (좌표)
    predict = kalman_filter_vector(vector_motion_controls, vector_measurements)
    average(vector_measurements, args.average_years, "vector")
    centroid_visualization_2D(custompath, predict, vector_measurements, args, 11)

    return
    # centroid로 nearest words, nearest docx 탐색   # TODO : (이거 2중for문 말고 numpy로 한번에 연산 필요)
    find_nearest_word(custompath, args)

    # nearest docx와 nearest words 사이의 좌표 비교  # TODO : (이것도)
    find_nearest_docx(custompath, args)


if __name__ == "__main__":
    main()
