import sklearn.cluster as cluster


def clustering(df, methods, n_cluster):
    if methods == "kmeans":
        print("A task that takes time to execute. Hold on a minute, please.")
        model = cluster.KMeans(n_cluster, n_init=100, max_iter=1000)
        label = model.fit_predict(df)
        # print(f"{n_cluster} of cluster score : {model.score(df)}")
        centroids = model.cluster_centers_
        return label, centroids, n_cluster


def find_label_centroid(df, methods, n_cluster):
    label, centroids, _ = clustering(df, methods, n_cluster)
    return centroids
