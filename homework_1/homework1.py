from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris


LOWER_LIMIT = 1
UPPER_LIMIT = 7


def elbow_method(data: np.ndarray, lower_limit: int, upper_limit: int) -> list:
    """
    Метод локтя для определения оптимального числа кластеров.
    """
    distances_to_cluster_centroids = []
    for cluster in range(lower_limit, upper_limit + 1):
        kmeans = KMeans(n_clusters=cluster)
        kmeans.fit(data)
        distances_to_cluster_centroids.append(kmeans.inertia_)
    return distances_to_cluster_centroids


def assigning_cluster_points(data: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """
    Назначение точек к ближайшим центроидам.
    """
    clusters = []
    for element in data:
        distance = np.linalg.norm(element - centroids, axis=1)
        clusters.append(np.argmin(distance))
    return np.array(clusters)


def update_centroids(data: np.ndarray, clusters: np.ndarray, k: int) -> np.ndarray:
    """
    Обновление положения центроидов как среднее значение точек в кластере.
    """
    centroids = np.zeros((k, data.shape[1]))
    for cluster in range(k):
        points = data[clusters == cluster]
        if points.size > 0:  # Проверка, чтобы избежать деления на ноль
            centroids[cluster] = np.mean(points, axis=0)
        else:
            centroids[cluster] = np.random.rand(data.shape[1])  # Случайная инициализация
    return centroids


def k_means(data: np.ndarray, k_value: int, centroids: np.ndarray, max_iterations: int):
    """
    Основной цикл k-means.
    """
    for _ in range(max_iterations):
        clusters_value = assigning_cluster_points(data, centroids)
        new_centroids = update_centroids(data, clusters_value, k_value)

        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids
        yield centroids, clusters_value


def plot_step(data: np.ndarray, clusters: np.ndarray, centroids: np.ndarray, k: int, step: int):
    """
    Визуализация всех проекций данных на текущем шаге.
    """
    feature_combinations = list(combinations(range(data.shape[1]), 2))
    for f1, f2 in feature_combinations:
        plt.figure(figsize=(6, 5))
        for cluster in range(k):
            plt.scatter(data[clusters == cluster, f1], data[clusters == cluster, f2])
        plt.scatter(centroids[:, f1], centroids[:, f2], s=300, c='red')
        plt.title(f"Step {step}: Features {f1} vs {f2}")
        plt.xlabel(f"Feature {f1}")
        plt.ylabel(f"Feature {f2}")
        plt.show()


def plot_projections(data: np.ndarray, clusters: np.ndarray, k: int):
    """
    Финальная визуализация всех проекций данных.
    """
    feature_combinations = list(combinations(range(data.shape[1]), 2))
    for number, (f1, f2) in enumerate(feature_combinations, 1):
        plt.figure(figsize=(6, 5))
        for cluster in range(k):
            plt.scatter(data[clusters == cluster, f1], data[clusters == cluster, f2])
        plt.title(f"Final Projection: Features {f1} vs {f2}")
        plt.xlabel(f"Feature {f1}")
        plt.ylabel(f"Feature {f2}")
        plt.show()


def main():
    iris = load_iris()
    iris_data = iris.data

    j_c = elbow_method(iris_data, LOWER_LIMIT, UPPER_LIMIT)
    plt.plot(range(LOWER_LIMIT, UPPER_LIMIT + 1), j_c, marker='o')
    plt.title("Elbow Method")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia")
    plt.show()

    while True:
        try:
            k_value = int(input('Введите значение k (от {} до {}): '.format(LOWER_LIMIT, UPPER_LIMIT)))
            if LOWER_LIMIT <= k_value <= UPPER_LIMIT:
                break
            else:
                print(f"Значение должно быть между {LOWER_LIMIT} и {UPPER_LIMIT}.")
        except ValueError:
            print("Введите целое число.")

    while True:
        try:
            max_iterations = int(input('Введите максимальное количество итераций: '))
            if max_iterations > 0:
                break
            else:
                print("Максимальное количество итераций должно быть больше 0.")
        except ValueError:
            print("Введите целое число.")

    centroids = iris_data[np.random.choice(iris_data.shape[0], k_value, replace=False)]

    for step, (centroids, clusters) in enumerate(k_means(iris_data, k_value, centroids, max_iterations), 1):
        plot_step(iris_data, clusters, centroids, k_value, step)

    plot_projections(iris_data, clusters, k_value)


if __name__ == '__main__':
    main()