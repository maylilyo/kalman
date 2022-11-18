from utils.make_kalman_base import make_base
from kalman_filter import kalman_filter_scala, kalman_filter_vector, average
from utils.nearest_word import find_nearest_word
import pandas as pd


def main():
    START_DATE = 2011
    END_DATE = 2021
    n_cluster = 16
    n_word = 10  # number of catch nearest word
    methods = "kmeans"
    average_years = 5  # 최근 n년 연도의 평균과 비교할 때 사용하는 n의 수

    # origin_df = pd.read_csv("../dataset_timeseries/kr_en_v.csv") # 전체 데이터셋
    # data preprocessing
    (
        scala_motion_controls,
        scala_measurements,
        vector_motion_controls,
        vector_measurements,
    ) = make_base(n_cluster, START_DATE, END_DATE)

    # scala kalman filter (문서 개수) => 추후 시계열로 동작 예정
    kalman_filter_scala(scala_motion_controls, scala_measurements)
    average(scala_measurements, average_years, "scala")

    # vector kalman filter (좌표)
    kalman_filter_vector(vector_motion_controls, vector_measurements)
    average(vector_measurements, average_years, "vector")

    # centroid로 nearest words, nearest docx 탐색 (이거 2중for문 말고 numpy로 한번에 연산 필요)
    # find_nearest_word(n_cluster, n_word)

    # nearest docx와 nearest words 사이의 좌표 비교


if __name__ == "__main__":
    main()
