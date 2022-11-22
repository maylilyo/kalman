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


def forecasting_var(model, train, test):
    results_aic = []
    for p in range(1, 6):
        results = model.fit(p)
        results_aic.append(results.aic)
    # n_fit = results_aic.index(min(results_aic)) + 1
    n_fit = 5
    results = model.fit(n_fit)
    laaged_values = train.values[-n_fit:]
    forecast = pd.DataFrame(
        results.forecast(y=laaged_values, steps=1), index=test.index
    )
    return forecast


def var_scala():
    test_length = 1
    args = get_args()
    custompath = f"../label/{args.n_cluster}"  # 연산을 위해 새롭게 생성된/로드할 데이터의 위치
    mydata = make_data(args, "scala", custompath)
    # stationary_timeseries(mydata)  # 차분 이전
    # mydata = differencing(mydata)  # 차분

    train = mydata.iloc[:-test_length, :]
    test = mydata.iloc[-test_length:, :]

    # print(train)
    # print(test)
    # print(np.var(train))
    forecasting_model = VAR(train)
    forecast = forecasting_var(forecasting_model, train, test)
    print(forecast)


if __name__ == "__main__":
    var_scala()
