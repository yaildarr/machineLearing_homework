import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def main() :
    #предобрабботка файлов
    data = pd.read_csv('AmesHousing.csv')
    data.drop(["Order", "PID"], axis=1, inplace=True)
    numeric_data = data.select_dtypes(include='number')
    corr_matrix = numeric_data.corr()

    plt.figure(figsize=(20, 16))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
    plt.show()

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    upper_triangle = corr_matrix.where(mask)
    high_corr_features = {
        column
        for column in upper_triangle.columns
        if any(upper_triangle[column] > 0.79)
    }
    numeric_data.drop(columns=high_corr_features, inplace=True)

    if "SalePrice" not in numeric_data.columns:
        numeric_data["SalePrice"] = data["SalePrice"]

    plt.figure(figsize=(20, 16))
    sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm")
    plt.show()

    numeric_data = numeric_data.dropna()

    #построение графика
    X = numeric_data.drop("SalePrice", axis=1)
    y = numeric_data["SalePrice"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_pca[:, 0], X_pca[:, 1], y, c=y, cmap='viridis')
    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    ax.set_zlabel('SalePrice')
    plt.show(block=True)

    #Разбейте данные на x_train, y_train, x_test и y_test для оценки точности работы алгоритма.
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y)

    #Посчитайте метрику RMSE.
    alphas = np.logspace(-4, 1, 30)
    errors = []
    for alpha in alphas:
        model = Lasso(alpha=alpha, max_iter=10000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        errors.append(rmse)

    plt.figure(figsize=(10, 5))
    plt.plot(alphas, errors, marker='o')
    plt.xlabel("Коэффициент регуляризации (alpha)")
    plt.ylabel("RMSE")
    plt.title("Зависимость ошибки от alpha (Lasso)")
    plt.grid(True)
    plt.show()

    best_alpha = alphas[np.argmin(errors)]
    print(f"Лучшее значение alpha: {best_alpha:.5f}, RMSE: {min(errors):.2f}")

    model = Lasso(alpha=best_alpha)
    model.fit(X_train, y_train)

    coefficients = pd.Series(model.coef_, index=X.columns)
    top_features = coefficients.abs().sort_values(ascending=False).head(10)

    print("Топ-10 признаков по влиянию на SalePrice:")
    print(top_features)
    top_features.plot(kind='barh', title="Влияние признаков (Lasso)")
    plt.gca().invert_yaxis()
    plt.xlabel("Коэффициент")
    plt.show()


if __name__ == '__main__':
    main()