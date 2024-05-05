"""
Код основан на алгоритме из статьи:
"Модифицированные алгоритмы построения нейронной сети SOFM"
Ссылка: https://cyberleninka.ru/article/n/modifitsirovannye-algoritmy-postroeniya-neyronnoy-seti-sofm
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2

def initialize_grid(size):
    array = np.zeros((size, size, 2))
    for i in range(size):
        for j in range(size):
            array[i, j] = [i / size * 300, j / size * 300]
    return array

def find_bmu(grid, point):
    distances = np.linalg.norm(grid - point, axis=2)
    indices = np.unravel_index(np.argmin(distances), distances.shape)
    return indices

def point_in_figure(figure):
    height, width = figure.shape
    while True:
        rand_point = (np.random.randint(0, width), np.random.randint(0, height))
        if figure[rand_point[1], rand_point[0]] != 0:
            break
    return rand_point

def update_weights(grid, bmu, point, learning_rate, radius):
    # Обновление весов сетки на основе найденного BMU
    # Здесь нужно добавить реализацию обновления весов сетки
    distance = np.linalg.norm(np.array(bmu)[:, np.newaxis, np.newaxis] - np.indices(grid.shape[:2]), axis=0)
    # influence = np.exp(-distance**2 / (2.0 * (radius**2)))
    influence = np.power(0.00001, np.power(distance, 2) / (np.power(radius, 2)))

    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            dif_length = point - grid[i, j]
            # length = np.linalg.norm(dif_length)
            
            grid[i, j] += influence[i, j] * learning_rate * dif_length
    return grid

def adaptive_grid_algorithm(figure, data, grid_size, image_center, num_iterations, initial_radius):
    # Основной алгоритм
    grid = initialize_grid(grid_size)
    grid += (image_center - 50, image_center - 150)
    
    s_0 = 20000# * grid_size
    s_1 = num_iterations * 15# * grid_size
    radius = initial_radius + ((initial_radius * 0.4 - initial_radius)/(s_0)) * ( s_0 - 1 )
    a_n1 = initial_radius * 0.4
    print(radius)
    # shuffled_data = np.random.permutation(data)[:s_0]
    for index in range(1, s_0):
        rand_point = point_in_figure(figure)
        bmu = find_bmu(grid, rand_point)
        learning_rate = np.power(index, -0.2) * (1 - np.exp((5 * (index - (s_0 + s_1)))/(s_0 + s_1)))
        radius = initial_radius + ((a_n1 - initial_radius)/(s_0)) * ( s_0 - index ) #a(n0) = a1 * 0.2
        # radius = initial_radius * np.power(index, -0.2)
        grid = update_weights(grid, bmu, rand_point, learning_rate, radius)
        if index%100 == 0:
            print(index)

    num_iterations+=s_0
    for iteration in range(1 + s_0, num_iterations):
        shuffled_data = np.random.permutation(data)[:10]
        for point_coords in shuffled_data:
            bmu = find_bmu(grid, point_coords)
            while True:
                if 0 in bmu or grid_size - 1 in bmu:
                    break
                else:
                    border_grid = grid.copy()
                    border_grid[1:grid_size-1, 1:grid_size-1] = (image_center, image_center + 100)
                    bmu = find_bmu(border_grid, grid[bmu])
                    break
            learning_rate = np.power(iteration, -0.2) * (1 - np.exp(5 * (iteration - num_iterations)/num_iterations))
            radius = 2.0 + a_n1 * (1- np.exp((5 * (iteration - num_iterations - s_0))/(num_iterations + s_0))) * np.power(0.005, ((iteration - s_0)/(num_iterations - s_0)))
            grid = update_weights(grid, bmu, point_coords, learning_rate, radius)
        
        for _ in range(1, 20):
            rand_point = point_in_figure(figure)
            bmu = find_bmu(grid, rand_point)
            while True:
                if 0 in bmu or grid_size - 1 in bmu:
                    bmu = find_bmu(grid[1:grid_size-1, 1:grid_size-1], grid[bmu])
                    bmu = (bmu[0] + 1, bmu[1] + 1)
                    break
                else:
                    break
            learning_rate = np.power(iteration, -0.2) * (1 - np.exp(5 * (iteration - num_iterations)/num_iterations))
            radius = 2.0 + a_n1 * (1 - np.exp((5 * (iteration - num_iterations - s_0))/(num_iterations + s_0))) * np.power(0.005, ((iteration - s_0)/(num_iterations - s_0)))
            # radius = initial_radius * np.power(iteration, -0.2)
            grid = update_weights(grid, bmu, rand_point, learning_rate, radius)
        if iteration%100 == 0:
            print(iteration - s_0)
    
    return grid

def insert_points(contour, threshold_scale=0.2):
    distances = []
    for i in range(len(contour)):
        start_point = contour[i][0]
        end_point = contour[(i+1) % len(contour)][0]
        distance = np.linalg.norm(end_point - start_point)
        distances.append(distance)

    average_distance = np.mean(distances)
    threshold_distance = average_distance * threshold_scale

    new_contour = []
    for i in range(len(contour)):
        start_point = contour[i][0]
        new_contour.append([start_point])
        end_point = contour[(i+1) % len(contour)][0]

        if distances[i] > threshold_distance:
            num_points_to_insert = int(distances[i] // average_distance)
            for j in range(1, num_points_to_insert + 1):
                interpolated_point = start_point + (end_point - start_point) * j / (num_points_to_insert + 1)
                new_contour.append([interpolated_point.astype(int)])

    return np.array(new_contour, dtype=np.int32)

# Загрузка изображения
image = cv2.imread('Shape_nn_1.png', cv2.IMREAD_GRAYSCALE)
image_height, image_width = image.shape[:2]
image_center = min(image_height, image_width)//2

# Применение пороговой обработки для выделения фигуры
_, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)

# Находим контуры фигуры
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Выбираем самый большой контур
shape_contour = max(contours, key=cv2.contourArea)

# Аппроксимируем контур до более низкого числа точек с сохранением точности
epsilon = 0.0002 * cv2.arcLength(shape_contour, True)
approximated_contour = cv2.approxPolyDP(shape_contour, epsilon, True)
refined_contour = insert_points(approximated_contour)


# plt.imshow(image, cmap='gray')
# # plt.plot(refined_contour[:, 0, 0], refined_contour[:, 0, 1], color='blue', label='Shape Contour')
# plt.scatter(refined_contour[:, 0, 0], refined_contour[:, 0, 1], color='green', s=10, label='Refined Points')

# plt.show()

# Рразмер сетки и количество итераций
grid_size = 40
num_iterations = 2000 # *30 = 30000
initial_radius = grid_size * 0.5 #0.8

# Применение алгоритма
adaptive_grid = adaptive_grid_algorithm(binary, approximated_contour.reshape(-1, 2), grid_size, image_center, num_iterations, initial_radius)


plt.imshow(image, cmap='gray')
plt.plot(approximated_contour[:, 0, 0], approximated_contour[:, 0, 1], color='blue', label='Shape Contour')
for i in range(adaptive_grid.shape[0]):
    plt.plot(adaptive_grid[i, :, 0], adaptive_grid[i, :, 1], color='yellow', linewidth=0.3)
for j in range(adaptive_grid.shape[1]):
    plt.plot(adaptive_grid[:, j, 0], adaptive_grid[:, j, 1], color='yellow', linewidth=0.3)
plt.scatter(adaptive_grid[:, :, 0], adaptive_grid[:, :, 1], color='red', label='Adaptive Grid Nodes', s=1)
plt.title('Adaptive Grid with Shape Contour')
plt.show()
