import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.cluster as cluster
import pandas as pd
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from .make_kalman_base import make_cluster_information_csv


def make_PCA(data, n_component):
    plt.cla()
    pca = PCA(n_components=n_component)
    pca.fit(data)
    cluster_pca = pca.transform(data)
    print(data.shape)

    pca_columns = ["x", "y"]
    df_custer_pca = pd.DataFrame(cluster_pca, columns=pca_columns)

    centroid = df_custer_pca.loc[0:0]
    predict = df_custer_pca.loc[1:1]
    time_centroid = df_custer_pca.loc[2:]
    return centroid, predict, time_centroid


def scala_visualization_2D(
    custompath, predict, scala_measurements, args, clusternumber, mode
):
    # print(predict.shape) (30,1), pandas
    # print(scala_measurements.shape) (30,enddate-startdate), nparray

    time_color = sns.color_palette("pastel", args.n_cluster).as_hex()
    predict_color = sns.color_palette("bright", args.n_cluster).as_hex()

    timeseries = list(scala_measurements[clusternumber])
    years = [i for i in range(args.start_date, args.end_date + 1)]
    years = years + [years[-1]]
    # years[-2] = str(years[-2]) + "\nActual"
    # years[-1] = str(years[-1]) + "\nPredict"
    for i in range(len(years)):
        years[i] = "" if i % 5 != 0 else years[i]
    years[-2] = str(args.end_date) + "\nA"
    years[-1] = str(args.end_date) + "\nP"
    timeseries.append(predict.iloc[clusternumber]["predict"])

    x = np.arange(len(years)) * 2
    colors = [time_color[0] for i in range(len(years) - 1)] + [predict_color[0]]

    plt.xlabel(f"Forecast of next year's document volume | cluster {clusternumber}")
    # plt.bar(x, timeseries, color=colors, width=1.8)
    plt.plot(x, timeseries)
    plt.xticks(x, years)
    plt.tick_params(
        axis="x",
        length=2,
        pad=6,
        labelsize=5,
    )

    # plt.savefig(f"./scala_{clusternumber}.png")
    if mode == "VAR":
        plt.savefig(f"./results/images/scala/scala_{clusternumber}_VAR.png")
    else:
        plt.savefig(f"./results/images/scala/scala_{clusternumber}.png")
    plt.cla()


def centroid_visualization_2D(
    custompath, predict, vector_measurements, args, clusternumber, mode
):
    """
    clusternumber = 몇번째 cluster에 대한 centroid 이동을 print할 예정인지
    """
    plt.cla()
    line_color = sns.color_palette("pastel", args.n_cluster).as_hex()
    point_color = sns.color_palette("bright", args.n_cluster).as_hex()

    _, whole_centroid_list = make_cluster_information_csv(
        custompath, args.datapath, args.n_cluster
    )
    centroid = whole_centroid_list[clusternumber]

    centroid = centroid.reshape((1,) + centroid.shape)
    time_centroid = vector_measurements[clusternumber]
    predict = np.array(predict[clusternumber])
    predict = predict.reshape((1,) + predict.shape)

    centroid, predict, time_centroid = make_PCA(
        np.concatenate((centroid, predict, time_centroid), axis=0), 2
    )
    plt.plot(
        time_centroid["x"],
        time_centroid["y"],
        color=line_color[1],
        linestyle="dotted",
        alpha=0.6,
    )
    plt.scatter(
        time_centroid["x"],
        time_centroid["y"],
        color=point_color[1],
        s=4**2,
    )
    plt.scatter(
        predict["x"],
        predict["y"],
        color=point_color[2],
        s=4**2,
    )
    plt.scatter(
        centroid["x"],
        centroid["y"],
        marker="*",
        color=point_color[0],
        s=9**2,
    )
    start_point = time_centroid.loc[2:2]
    end_point = time_centroid.loc[time_centroid.shape[0] + 1]

    plt.text(start_point["x"], start_point["y"], "start")
    plt.text(predict["x"], predict["y"], "predict")
    plt.text(end_point["x"], end_point["y"], "actual")
    # plt.text(time_centroid["x"][-1], time_centroid["y"][-1], "actual")

    plt.xlabel(f"Forecast of next year's cluster centroid | cluster {clusternumber}")

    if mode == "VAR":
        plt.savefig(f"./results/images/vector/vector_{clusternumber}_VAR.png")
    else:
        plt.savefig(f"./results/images/vector/vector_{clusternumber}.png")
