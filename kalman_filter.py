import operator
import pandas as pd
import numpy as np
import pickle
from utils.csv_to_numpy import csv_to_numpy
from utils.make_kalman_base import make_base
import statistics


def euclidean_distance(pt1, pt2):
    distance = 0
    for i in range(len(pt1)):
        pt1[i] = float(pt1[i])
        pt2[i] = float(pt2[i])
        distance += (pt1[i] - pt2[i]) ** 2
    return distance**0.5


def cosine_similarity(A, B):
    return np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))


def measurement_update(
    estimated_mean, estimated_var, measurement_mean, measurement_var
):
    new_mean = (measurement_var * estimated_mean + estimated_var * measurement_mean) / (
        estimated_var + measurement_var
    )
    new_var = estimated_var * measurement_var / (estimated_var + measurement_var)
    return new_mean, new_var


def state_prediction(
    estimated_mean, estimated_var, motion_control_mean, motion_control_var
):
    new_mean = estimated_mean + motion_control_mean
    new_var = estimated_var + motion_control_var
    return new_mean, new_var


def kalman_filter_scala(scala_motion_controls, scala_measurements):
    # scala_motion_controls = np.array(scala_motion_controls)
    # scala_measurements = np.array(scala_measurements)

    cluster_sum = scala_measurements.sum(axis=1)  # 각 cluste 문서의 합계(2011~2021)
    kalman_scala_result = []
    n_cluster = scala_motion_controls.shape[0]
    for i in range(len(scala_measurements)):
        measurements = scala_measurements[i]
        measurement_var = np.var(measurements)

        motion_control = scala_motion_controls[i]
        motion_control_var = np.var(motion_control)

        mu = 0  # 확률분포 평균, 기댓값 (=구하려는 기댓값)
        sig = 0  # sigma, 분산

        for j in range(len(motion_control)):
            mu, sig = state_prediction(
                mu, sig, motion_control[j], motion_control_var
            )  # predict
            mu, sig = measurement_update(
                mu, sig, measurements[j], measurement_var
            )  # update
        # print(
        #     f"predict next cluster rate of {i} is {round(mu * 100, 4)}, actually {round(measurements[-1] * 100, 4)}."
        # )
        # print(f"{round(mu * 100, 4)}, {round(measurements[-1] * 100, 4)}")
        # print(f"{round(mu, 4)}")
        kalman_scala_result.append(round(mu, 4))
    kalman_scala_result = pd.DataFrame(kalman_scala_result, columns=["predict"])
    kalman_scala_result.to_csv(
        f"./results/kalman_scala_{n_cluster}_result.csv", index=False
    )


def kalman_filter_vector(vector_motion_controls, vector_measurements):
    total_cos = 0
    kalman_vector_result = []
    n_cluster = vector_motion_controls.shape[0]
    for i in range(len(vector_measurements)):
        measurements = vector_measurements[i]
        # measurement_var = np.var(measurements, axis=0)
        measurement_var = np.cov(measurements)[0, 1]  # tmp

        motion_control = vector_motion_controls[i]
        # motion_control_var = np.var(motion_control, axis=0)
        motion_control_var = np.cov(motion_control)[0, 1]  # tmp

        mu = 0
        sig = 0
        for j in range(len(motion_control)):
            mu, sig = state_prediction(mu, sig, motion_control[j], motion_control_var)
            # print("predict: [%f %f]" % (mu, sig))
            mu, sig = measurement_update(mu, sig, measurements[j], measurement_var)
        cos_sim = cosine_similarity(mu, measurements[-1])
        total_cos += cos_sim
        kalman_vector_result.append(round(cos_sim, 4))
    kalman_vector_result = pd.DataFrame(kalman_vector_result, columns=["predict"])
    kalman_vector_result.to_csv(
        f"./results/kalman_vector_{n_cluster}_result.csv", index=False
    )
    # print(total_cos / len(vector_measurements))  # 비율로 print 해야할 때


def average(measurements, years, mode):
    if mode == "scala":
        for measurement in measurements:
            measurement = measurement[-years - 1 : -1]
            # print(round(sum(measurement) / years, 4))
    else:
        total_cos = 0
        for i, measurement in enumerate(measurements):
            centroid = np.array(measurement[-1])
            measurement = measurement[-years - 1 : -1]
            mean = np.mean(measurement, axis=0)
            cos_sim = cosine_similarity(mean, centroid)
            total_cos += cos_sim
            # print(round(cos_sim, 4))
        # print(round(total_cos / len(measurements), 4))


if __name__ == "__main__":
    (
        scala_motion_controls,
        scala_measurements,
        vector_motion_controls,
        vector_measurements,
    ) = make_base(15, 2011, 2021)
    kalman_filter_scala(scala_motion_controls, scala_measurements)
    kalman_filter_vector(vector_motion_controls, vector_measurements)
