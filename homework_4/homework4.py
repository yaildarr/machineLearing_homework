import json
import requests
import numpy as np
from deap import base, creator, tools, algorithms
import folium
import webbrowser
import os
import random

POP_SIZE = 100  # Размер популяции
NGEN = 40  # Количество поколений
CXPB = 0.7  # Вероятность кроссовера
MUTPB = 0.2  # Вероятность мутации


# Функция загрузки точек из JSON
def load_points_from_file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    points = []
    names = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("Каждая точка в JSON должна быть словарем")
        lat = float(item.get("lat"))
        lng = float(item.get("lng"))
        priority = int(item.get("priority"))
        name = str(item.get("name"))
        points.append({"lat": lat, "lng": lng, "priority": priority})
        names.append(name)
    if not points:
        raise ValueError("JSON-файл не содержит точек")
    return points, names


# Функция для получения матрицы времени и расстояний через OSRM
def get_time_distance_matrix(points, mode):
    n = len(points)
    time_matrix = np.zeros((n, n))
    distance_matrix = np.zeros((n, n))
    osrm_mode = {"1": "foot", "2": "bike", "3": "car"}[mode]
    coordinates = ";".join(f"{p['lng']},{p['lat']}" for p in points)
    url = f"http://router.project-osrm.org/table/v1/{osrm_mode}/{coordinates}?annotations=duration,distance"

    response = requests.get(url)
    data = response.json()
    durations = data["durations"]
    distances = data["distances"]
    for i in range(n):
        for j in range(n):
            if durations[i][j] is None or distances[i][j] is None:
                raise ValueError(f"Маршрут между точками {i} и {j} недоступен")
            time_matrix[i][j] = durations[i][j] / 60.0
            distance_matrix[i][j] = distances[i][j] / 1000.0
    return time_matrix, distance_matrix


# Функция оценки качества маршрута для генетического алгоритма, fitness
def evaluate(individual, time_matrix, points, max_time):
    total_time = 0.0
    total_priority = points[0]["priority"]  # Учитываем начальную точку
    route = [0] + individual + [0]  # Начинаем и заканчиваем в точке 0

    #общее время поездки по маршруту
    for i in range(len(route) - 1):
        total_time += time_matrix[route[i]][route[i + 1]]

    # если лимимт превышен, выкидываем точку
    if total_time > max_time:
        return (-float('inf'),)

    # считаю приоритет маршрута
    for idx in individual:
        total_priority += points[idx]["priority"]
    return (total_priority,)


# кроссовер для перестановок, скрещивания двух родителей
def custom_crossover(ind1, ind2):
    size = len(ind1)
    if size < 2:
        return ind1[:], ind2[:]  # Если маршрут слишком короткий, возвращаем без изменений

    # Выбираем точки кроссовера, чтобы вырезать эту часть из родителя
    start = random.randint(0, size - 1)
    end = random.randint(start + 1, size)

    # Создаем пустых потомков
    child1 = creator.Individual([None] * size)
    child2 = creator.Individual([None] * size)

    # Копируем сегмент из ind1 в child1 и из ind2 в child2
    for i in range(start, end):
        child1[i] = ind1[i]
        child2[i] = ind2[i]

    # Заполняем оставшиеся позиции из ind2 для child1
    ind2_iter = iter([x for x in ind2 if x not in child1[start:end]])
    for i in range(size):
        if child1[i] is None:
            child1[i] = next(ind2_iter)

    # Заполняем оставшиеся позиции из ind1 для child2
    ind1_iter = iter([x for x in ind1 if x not in child2[start:end]])
    for i in range(size):
        if child2[i] is None:
            child2[i] = next(ind1_iter)

    return child1, child2


# Функция ввода параметров пользователем
def get_user_input():
    print("Выберите способ передвижения:")
    print("1 - Пешком")
    print("2 - На велосипеде")
    print("3 - На автомобиле")
    while True:
        choice = input("Введите номер: ").strip()
        if choice in ['1', '2', '3']:
            break
        print("Неверный ввод. Пожалуйста, введите 1, 2 или 3.")

    while True:
        try:
            max_time = float(input("Введите максимальное время поездки в часах: "))
            if max_time > 0:
                break
            print("Время должно быть положительным числом.")
        except ValueError:
            print("Пожалуйста, введите число.")
    return choice, max_time * 60  # Переводим в минуты


# Функция для получения координат маршрута через OSRM
def get_route_coordinates(points, route, mode):
    osrm_mode = {"1": "foot", "2": "bike", "3": "car"}[mode]
    coordinates = ";".join(f"{points[i]['lng']},{points[i]['lat']}" for i in route) #собираем строку координат
    url = f"http://router.project-osrm.org/route/v1/{osrm_mode}/{coordinates}?overview=full&geometries=geojson"

    try:
        response = requests.get(url)
        data = response.json()
        if data["code"] != "Ok":
            raise Exception(f"OSRM Error: {data.get('message', 'Unknown error')}")
        return data["routes"][0]["geometry"]["coordinates"]  # Список [lng, lat]
    except Exception as e:
        print(f"Ошибка получения маршрута: {e}")


# Функция создания карты
def create_map(points, names, route, mode):
    map_center = [points[route[0]]["lat"], points[route[0]]["lng"]]
    my_map = folium.Map(location=map_center, zoom_start=14)

    for idx, point in enumerate(points):
        color = 'red' if idx in route else 'blue'
        folium.Marker(
            location=[point["lat"], point["lng"]],
            popup=f"{names[idx]}\nПриоритет: {point['priority']}",
            icon=folium.Icon(color=color, icon='info-sign')
        ).add_to(my_map)

    coords = get_route_coordinates(points, route, mode)
    path_coords = [[lat, lng] for lng, lat in coords]
    folium.PolyLine(path_coords, color="green", weight=2.5, opacity=1).add_to(my_map)

    map_file = "optimized_route.html"
    my_map.save(map_file)
    return map_file


# Основная функция
def main():
    mode, max_time = get_user_input()

    points, names = load_points_from_file("trip.json")

    time_matrix, distance_matrix = get_time_distance_matrix(points, mode)

    # Проверяем, не превышает ли время минимальный маршрут
    min_route_time = time_matrix[0][1] + time_matrix[1][0]
    if min_route_time > max_time:
        print(
            f" Даже минимальный маршрут туда и обратно занимает {min_route_time:.2f} минут, что превышает {max_time:.2f} минут.")
        # Создаем минимальный маршрут из двух точек
        best_route = [0, 1, 0]
        total_time = time_matrix[0][1] + time_matrix[1][0]
        total_distance = distance_matrix[0][1] + distance_matrix[1][0]
        total_priority = points[0]["priority"] + points[1]["priority"]
        print("\n Минимальный маршрут:")
        for idx in best_route[:-1]:
            print(f"- {names[idx]} (Приоритет: {points[idx]['priority']})")
        print(f"- {names[0]} (возврат)")
        print(f"\nОбщая длина маршрута: {total_distance:.2f} км")
        print(f"Время на дорогу: {int(total_time // 60)} ч {int(total_time % 60)} мин")
        print(f"Суммарный приоритет: {total_priority}")
        map_file = create_map(points, names, best_route, mode)
        webbrowser.open(f"file://{os.path.abspath(map_file)}")
        return

    # Инициализация генетического алгоритма
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    # Генерируем индивидуумы как полные перестановки всех точек (кроме начальной)
    toolbox.register("indices", random.sample, range(1, len(points)), len(points) - 1) # случайная перестановка всех точек
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices) #генератор одного индивидуума
    toolbox.register("population", tools.initRepeat, list, toolbox.individual) #генератор всей популяции
    toolbox.register("mate", custom_crossover)  # скрещивание
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.2)  # перемешивание индексов, мутация
    toolbox.register("select", tools.selTournament, tournsize=3)  # отбор
    toolbox.register("evaluate", evaluate, time_matrix=time_matrix, points=points, max_time=max_time)  #функция оценки особи

    # создаем начальную популяцию
    pop = toolbox.population(n=POP_SIZE)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("max", np.max) # объект для сбора статистики
    stats.register("avg", np.mean)

    # запускаем алгоритм
    algorithms.eaSimple(pop, toolbox, cxpb=CXPB, mutpb=MUTPB, ngen=NGEN, stats=stats, halloffame=hof, verbose=True)

    best = hof[0]
    best_route = [0] + best + [0]
    total_priority = evaluate(best, time_matrix, points, max_time)[0]
    total_time = sum(time_matrix[best_route[i]][best_route[i + 1]] for i in range(len(best_route) - 1))
    total_distance = sum(distance_matrix[best_route[i]][best_route[i + 1]] for i in range(len(best_route) - 1))

    print("\nЛучший маршрут:")
    for idx in best_route[:-1]:
        print(f"- {names[idx]} (Приоритет: {points[idx]['priority']})")
    print(f"- {names[0]} (возврат)")
    print(f"Общая длина маршрута: {total_distance:.2f} км")
    print(f"Время на дорогу: {int(total_time // 60)} ч {int(total_time % 60)} мин")
    print(f"Суммарный приоритет: {total_priority}")

    map_file = create_map(points, names, best_route, mode)
    webbrowser.open(f"file://{os.path.abspath(map_file)}")


if __name__ == "__main__":
    main()
