import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split


def main():
    dataset = pd.read_csv('AmesHousing.csv')
    dataset = dataset.drop(columns=['Order'])

    numeric_dataset = dataset.select_dtypes(include=['number'])

    correlation_matrix = numeric_dataset.corr()
    high_corr_columns = []

    for i in range(len(correlation_matrix.columns)):
        for j in range(i + 1, len(correlation_matrix.columns)):
            if abs(correlation_matrix.iloc[i, j]) > 0.8:
                col1 = correlation_matrix.columns[i]
                col2 = correlation_matrix.columns[j]
                high_corr_columns.append((col1, col2, correlation_matrix.iloc[i, j]))

    print("Пары столбцов с корреляцией > 0.8:")
    for pair in high_corr_columns:
        print(f"Столбцы: {pair[0]} и {pair[1]}, Корреляция: {pair[2]:.2f}")

    first_elements = [pair[0] for pair in high_corr_columns]  # массив столбцов которые нужно убрать
    normal_dataset = numeric_dataset.drop(columns=first_elements, axis=1)


    # Разделение данных на признаки (X) и целевую переменную (y)
    X, y = normal_dataset.drop(columns=['SalePrice'], axis=1), normal_dataset['SalePrice']

    X = X.fillna(X.mean())  # Заполнение средним значением
    y = y.fillna(y.mean())

    # Разбиение данных на обучающую и тестовую выборки
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, train_size=0.2, random_state=42)

    lr = LinearRegression()
    lr.fit(X_train, Y_train)

    predict = lr.predict(X_test)

    # Вычисление RMSE
    rmse = np.sqrt(mean_squared_error(Y_test, predict))
    print(f"RMSE для линейной регрессии: {rmse:.2f}")

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(
        X_pca[:, 0],  # Первая главная компонента
        X_pca[:, 1],  # Вторая главная компонента
        y,  # Целевая переменная (ось Z)
        c=y,  # Цвет точек зависит от значения целевой переменной
        cmap='viridis'
    )

    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('SalePrice')
    ax.set_title('3D Scatter Plot of PCA-transformed Data')

    cbar = plt.colorbar(scatter, shrink=0.5, aspect=10)
    cbar.set_label('SalePrice')

    plt.show()


if __name__ == "__main__":
    main()