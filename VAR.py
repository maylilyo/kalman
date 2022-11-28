import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller

from utils.arguments import get_args
from utils.make_kalman_base import make_base


def differencing(mydata):
    mydata_diff = mydata.diff().dropna()
    return mydata_diff


def stationary_timeseries(mydata):
    for i in range(mydata.shape[1]):
        adfuller_test = adfuller(mydata[f"{i}"], autolag="AIC")
        statistic = adfuller_test[0]
        p_value = adfuller_test[1]
        print(f"p-value of {i} cluster = {p_value}")


def make_data(args, mode, custompath):
    if mode == "scala":
        (motion_controls, measurements, _, _) = make_base(custompath, args)
        # print(measurements)  # 문서 개수(int)
        # print(motion_controls)  # 다음 연도로의 움직임
        measurements_df = pd.DataFrame(measurements).transpose()
        m_columns = [f"{i}" for i in range(measurements_df.shape[1])]
        measurements_df.columns = m_columns
        return measurements_df

    elif mode == "vector":
        (_, _, motion_controls, measurements) = make_base(custompath, args)
        """
        TMP
        """
        datalist = []
        for i in range(args.n_cluster):
            data = pd.DataFrame(measurements[i])
            datalist.append(data)
        return datalist


def issue():
    # n_fit을 찾는 과정에서 aic를 구했을 때 왜 오류가 생기는지?
    # for p in range(1, 2):
    #     results = model.fit(p)
    #     if results.aic is None:
    #         results_aic.append(0)
    #     else:
    #         results_aic.append(results.aic)
    # # n_fit = results_aic.index(min(results_aic)) + 1
    return 3


def forecasting_var(model, train, test):
    n_fit = issue()
    for p in range(1, 10):
        results = model.fit(n_fit)
        laaged_values = train.values[-n_fit:]
        forecast = pd.DataFrame(
            results.forecast(y=laaged_values, steps=1), index=test.index
        )
    return forecast


def print_average(forecast, test):
    forecast = forecast.values.tolist()[0]
    test = test.values.tolist()[0]
    sub_list = []
    for ai, bi in zip(forecast, test):
        sub_list.append(abs(ai - bi))
    result = np.mean(np.array(sub_list))
    print(f"VAR Average(|Predoct-Actual|) = {result}")
    pass


def var_scala():
    test_length = 1
    args = get_args()
    custompath = f"../label/{args.n_cluster}"  # 연산을 위해 새롭게 생성된/로드할 데이터의 위치
    mydata = make_data(args, "scala", custompath)
    print(mydata)
    # stationary_timeseries(mydata)  # 차분 이전
    # mydata = differencing(mydata)  # 차분

    train = mydata.iloc[:-test_length, :]
    test = mydata.iloc[-test_length:, :]

    # print(train)
    forecasting_model = VAR(train)
    forecast = forecasting_var(forecasting_model, train, test)

    print_average(forecast, test)


def var_vector():
    test_length = 1
    args = get_args()
    custompath = f"../label/{args.n_cluster}"  # 연산을 위해 새롭게 생성된/로드할 데이터의 위치
    mydata = make_data(args, "vector", custompath)

    distance_list = []
    for data in mydata:
        train = data.iloc[:-test_length, :]
        test = data.iloc[-test_length:, :]

        forecasting_model = VAR(train)
        forecast = forecasting_var(forecasting_model, train, test)

        distance = np.linalg.norm(forecast - test)
        distance_list.append(distance)
    print(np.mean(np.array(distance_list)))


if __name__ == "__main__":
    # var_scala()
    var_vector()
