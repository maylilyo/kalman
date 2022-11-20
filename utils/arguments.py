import argparse


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--start_date", type=int, default=2011, help="time series 시작 연도"
    )
    parser.add_argument(
        "--end_date",
        type=int,
        default=2021,
        help="end_date-1 까지의 time series를 통해 forecasting하는 연도",
    )
    parser.add_argument(
        "--n_cluster",
        type=int,
        default=5,
        help="number of cluster",
    )
    parser.add_argument(
        "--n_word",
        type=int,
        default=10,
        help="number of catch nearest words",
    )
    parser.add_argument(
        "--n_docs",
        type=int,
        default=1,
        help="number of catch nearest docs",
    )
    parser.add_argument(
        "--methods", type=str, default="kmeans", help="clustering method"
    )
    parser.add_argument(
        "--average_years", type=int, default=5, help="최근 n년 연도의 평균과 비교할 때 사용하는 n의 수"
    )
    parser.add_argument("--datapath", type=str, default="../Aihub_data", help="데이터 경로")

    # custompath = f"../label/{n_cluster}"  # 연산을 위해 새롭게 생성된/로드할 데이터의 위치

    args = parser.parse_args()
    return args
