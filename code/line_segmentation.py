import numpy as np
import os
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle
from PyQt5.QtGui import QImage, QColor
from PyQt5.QtGui import QPainter, QPen, QColor
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImageReader
import sys
import cv2

from PyQt5.QtWidgets import QApplication, QLabel
from PyQt5.QtGui import QPixmap
from scipy.spatial import Delaunay

global input_name, file_name, file_path, width, height, binary_image
y_diff_global_end = []
angles_in_triangulations = []
y_diff_for_merging = []
colors = [
    (255, 0, 0),  # Красный
    (255, 165, 0),  # Оранжевый
    (255, 255, 0),  # Желтый
    (0, 128, 0),  # Зеленый
    (173, 216, 230),  # Голубой
    (0, 0, 255),  # Синий
    (128, 0, 128)  # Фиолетовый
]
def create_point_image(x, y, colour=(255, 0, 0)):
    painter = QPainter()
    painter.begin(binary_image)

    # Устанавливаем перо для рисования контуров точек
    pen = QPen()
    pen.setWidth(5)  # Устанавливаем толщину линии (размер точки)
    pen.setColor(QColor(*colour))  # Красный цвет
    painter.setPen(pen)

    # Рисуем точки
    for i in range(len(x)):
        painter.drawPoint(int(x[i]), int(height - y[i]))

    painter.end()

def create_binary_image(x, y, image, color=(0,0,0)):
    painter = QPainter()
    painter.begin(image)

    # Устанавливаем перо для рисования контуров фигур
    pen = QPen()
    pen.setWidth(2)  # Устанавливаем толщину линии
    pen.setColor(QColor(*color))  # Черный цвет
    painter.setPen(pen)
    for i in range(len(x) - 1):
        if x[i] and y[i]:
            painter.drawLine(int(x[i]), int(height - y[i]), int(x[i + 1]), int(height - y[i + 1]))

    painter.end()

def get_image_dimensions(file_path):
    image_reader = QImageReader(file_path)
    size = image_reader.size()
    width = size.width()
    height = size.height()
    return width, height

def init_image(name):
    global input_name, file_name, file_path, width, height, binary_image
    input_name = name
    file_path = f'/Users/victoriasmirnova/PycharmProjects/pythonProject6/for_triangle/SK_jpgs/{input_name}.jpg'
  # Путь к вашему изображению
    width, height = get_image_dimensions(file_path)

    binary_image = QImage(width, height, QImage.Format_RGB32)
    binary_image.fill(QColor(255, 255, 255))



def get_coordinates():
    x_all, y_all, mean_points, sort_points = [], [], [], []
    frame = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    frame = cv2.medianBlur(frame, 3)
    img = cv2.adaptiveThreshold(
        frame,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=9,
        C=11
    )


    def sobel_edge_detector(img):
        grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0)
        grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1)
        grad = np.sqrt(grad_x ** 2 + grad_y ** 2)
        grad_norm = (grad * 255 / grad.max()).astype(np.uint8)
        return grad_norm

    binary_image = sobel_edge_detector(img)
    binary_image = cv2.dilate(binary_image, kernel=np.ones((1, 1), np.uint8), iterations=1)
    #binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    median_points, mean_points, cm_points, mean_frame_points = [], [],[],[]
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # объединение контуров
    diff_y_coords, diff_x_coords = [], []
    for idx, contour in enumerate(contours):
        if len(contour) > 10:
            x_coords = contour[:, 0, 0]
            y_coords = height - contour[:, 0, 1]
            diff_y_coords.append(abs(max(y_coords) - min(y_coords)))
            diff_x_coords.append(abs(max(x_coords) - min(x_coords)))
            # median
            median_points.append([np.median(x_coords), np.median(y_coords)])# центр масс или медиана и рассмотреть вместе графику, для случаев со слившимися строками рядом а в стровке
            #mean
            mean_points.append([np.mean(x_coords), np.mean(y_coords)])
            #centre_mass
            moments = cv2.moments(contour)
            # Центр масс
            cx = moments['m10'] / moments['m00']
            cy = height - moments['m01'] / moments['m00']
            cm_points.append([cx, cy])
            #среднее фрейма
            mean_frame_points.append([abs(max(x_coords) + min(x_coords)) // 2, abs(max(y_coords) + min(y_coords))//2])
            sort_points.append(np.max(y_coords))
            x_all.append(x_coords)
            y_all.append(y_coords)
    mean_points = np.array(mean_points)
    median_points = np.array(median_points)
    cm_points = np.array(cm_points)
    mean_frame_points = np.array(mean_frame_points)
    diff_y_coords = np.array(diff_y_coords)
    diff_x_coords = np.array(diff_x_coords)
    #отбираю по длине для триангуляций
    mask = ((diff_x_coords >= np.quantile(diff_x_coords, 0.25)) &
    (diff_y_coords >= np.quantile(diff_y_coords, 0.15)))
    '''mean_points = mean_points[
    (diff_x_coords >= np.quantile(diff_x_coords, 0.25)) &
    (diff_y_coords >= np.quantile(diff_y_coords, 0.15)) ]'''

    new_inds = np.argsort(sort_points)[::-1]
    sorted_x_all = [x_all[i] for i in new_inds]
    sorted_y_all = [y_all[i] for i in new_inds]

    return sorted_x_all, sorted_y_all, median_points[mask], median_points

def draw_center_points(points):
    painter = QPainter()
    painter.begin(new_image)
    pen = QPen()
    pen.setWidth(10)
    colors = [(255, 0, 0), (255, 165, 0), (0, 0, 255), (128, 0, 128)]
    # Красный, Оранжевый, Синий, Фиолетовый
    #[cm_points[mask], median_points[mask], mean_points[mask], mean_frame_points[mask]]
    for (p,c) in zip(points, colors):
        pen.setColor(QColor(*c))
        painter.setPen(pen)
        for (px, py) in p:
            painter.drawPoint(int(px), int(height - py))
def draw_contiurs(x, y, mean_points=None, image=None):
    #draw_center_points(mean_points)
    for i in range(len(x)):
        #get_tangle(x[i], y[i])
        create_binary_image(x[i], y[i], image, (0, 0, 0))


def get_tangle(x, y,  color=(0,255,0)):
    max_y, min_y = np.max(y), np.min(y)
    max_x, min_x = np.max(x), np.min(x)
    painter = QPainter()
    painter.begin(new_image)
    x_all = [[min_x, min_x], [max_x, max_x], [min_x, max_x], [min_x, max_x]]
    y_all = [[min_y, max_y], [min_y, max_y], [min_y, min_y], [max_y, max_y]]
    # Устанавливаем перо для рисования контуров фигур
    pen = QPen()
    pen.setWidth(2)  # Устанавливаем толщину линии
    pen.setColor(QColor(0,255,0))
    painter.setPen(pen)
    for j, x in enumerate(x_all):
        for i in range(len(x) - 1):
            painter.drawLine(int(x[i]), int(height - y_all[j][i]), int(x[i + 1]), int(height - y_all[j][i + 1]))


def triangulation_draw(points_for_triangle):
    #new_image = QImage(file_path).convertToFormat(QImage.Format_RGB32)
    points_for_triangle = np.array(np.concatenate([points_for_triangle, [[0,0], [width-1, 0], [width-1, height - 1], [0, height - 1]]]))
    tri = Delaunay(points_for_triangle)
    tris = tri.simplices
    edges = set([])
    for tri in tris:
        for k in range(3):
            i, j = tri[k], tri[(k + 1) % 3]
            i, j = min(i, j), max(i, j)
            edges.add((i, j))

    def calculate_angle(p1, p2):
        delta_y = p2[1] - p1[1]
        delta_x = p2[0] - p1[0]
        angle = np.degrees(np.arctan2(delta_y, delta_x))
        return abs(angle)

    angle_threshold = 10
    filtered_lines = []
    triangulation_points = []
    # Перебор всех рёбер триангуляции
    edge_points =[[0,0], [width-1, 0], [width-1, height - 1], [0, height - 1]]
    for simplex in tris:
        for i in range(3):
                p1 = points_for_triangle[simplex[i]]
                p2 = points_for_triangle[simplex[(i + 1) % 3]]
                if list(p1) not in edge_points and list(p2) not in edge_points:
                    triangulation_points.append(p1)
                    triangulation_points.append(p2)
                    angle = calculate_angle(p1, p2)
                    if abs(angle) < angle_threshold or abs(angle) > 180 - angle_threshold:
                        filtered_lines.append((p1, p2))



    #result = merge_lines(filtered_lines)
    #print(result)
    painter = QPainter(new_image)
    pen = QPen(QColor(0, 0, 255))  # Синий цвет линий
    pen.setWidth(2)  # Толщина линии
    painter.setPen(pen)

    # Рисуем рёбра
    for i, j in edges:
        xi, yi = points_for_triangle[i]
        xj, yj = points_for_triangle[j]
        painter.drawLine(int(xi), height - int(yi), int(xj), height - int(yj))

    pen.setColor(QColor(0, 128, 255))
    pen.setWidth(8)
    painter.setPen(pen)
    sorted_filters_lines = sorted(filtered_lines, key = lambda point: (point[0][1] + point[1][1]) // 2, reverse=True)

    for line in sorted_filters_lines:
        p1, p2 = line
        painter.drawLine(int(p1[0]), height - int(p1[1]), int(p2[0]), height - int(p2[1]))

    # Рисуем точки
    pen.setColor(QColor(255, 0, 0))  # Красный цвет точек
    pen.setWidth(10)  # Толщина точек
    painter.setPen(pen)
    for x, y in points_for_triangle:
        painter.drawPoint(int(x), height - int(y))

    painter.end()

    new_image.save(f"/Users/victoriasmirnova/PycharmProjects/pythonProject6/for_triangle/SK_exp_pict/{input_name}_triang_new_str.jpg")
    return sorted_filters_lines, triangulation_points

def cluster_lines(triangle_lines):
    all_lines = []
    all_lines_mean = []
    new_line = [triangle_lines[0]]
    center_points_y = [(line[1][1] + line[0][1]) //2 for line in triangle_lines]
    y_diff = list(map(int, abs(np.diff(center_points_y))))
    #print(y_diff)
    y_diff_global_end.append(y_diff)
    #for i in range(len(triangle_lines) - 1):
        #y_diff.append(abs(triangle_lines[i][1][1] - triangle_lines[i+1][0][1]))
    #y_const = np.quantile(y_diff, 0.9)
    y_const = np.quantile(y_diff, 0.9)
    #print(len(triangle_lines))
    #print(y_const)
    #(np.mean(y_diff))
    for line in triangle_lines[1::]:
        #крайняя точка
        if abs((line[0][1] + line[1][1]) // 2 - (new_line[-1][1][1] + new_line[-1][0][1]) // 2) > y_const:
            all_lines.append(new_line)
            all_lines_mean.append(int(np.mean([(elem[0][1]+ elem[1][1])//2 for elem in new_line])))
            new_line = [line]
        else:
            new_line.append(line)
    if new_line:
        all_lines.append(new_line)
        all_lines_mean.append(int(np.mean([(elem[0][1] + elem[1][1]) // 2 for elem in new_line])))
    diff_lines_y = abs(np.diff(all_lines_mean))
    to_merge_coef = np.quantile(diff_lines_y, 0.3)
    y_diff_for_merging.append(list(map(int, abs(np.diff(all_lines_mean)))))
    need_merging = diff_lines_y < to_merge_coef
    merged_all_lines = []
    new_line = [all_lines[0]]
    merged_lines_mean = []
    for i, is_merging in enumerate(need_merging):
        #print(len(all_lines[i]))
        if len(all_lines[i]) < 12:
            if is_merging:
                    new_line.append(all_lines[i])
            else:
                if new_line:
                    merged_all_lines.append(np.concatenate(new_line))
                    merged_lines_mean.append(np.mean([(line[0][1] + line[1][1]) // 2 for line in merged_all_lines[-1]]))
                    new_line = []
                else:
                    merged_all_lines.append(all_lines[i])
                    merged_lines_mean.append(np.mean([(line[0][1] + line[1][1]) // 2 for line in merged_all_lines[-1]]))
        else:
            if new_line:
                merged_all_lines.append(np.concatenate(new_line))
                new_line = []
                merged_lines_mean.append(np.mean([(line[0][1] + line[1][1]) // 2 for line in merged_all_lines[-1]]))
            else:
                merged_all_lines.append(all_lines[i])
                merged_lines_mean.append(np.mean([(line[0][1] + line[1][1]) // 2 for line in merged_all_lines[-1]]))
    if new_line:
        merged_all_lines.append(np.concatenate(new_line))
        merged_lines_mean.append(np.mean([(line[0][1] + line[1][1]) // 2 for line in merged_all_lines[-1]]))

    else:
        merged_all_lines.append(all_lines[-1])
        merged_lines_mean.append(np.mean([(line[0][1] + line[1][1]) // 2 for line in merged_all_lines[-1]]))
    '''new_merged_lines = []
    new_line = [merged_all_lines[0]]
    for i, line in enumerate(merged_all_lines[1:]):
        merging_coeff = np.mean(merged_all_lines[i:i+5])
        if abs((new_line[-1][0][1] + new_line[-1][1][1]) // 2 - merged_lines_mean[i+1])'''
    painter = QPainter(new_image)
    print(len(merged_all_lines), len(merged_lines_mean))

    for i, group_line in enumerate(merged_all_lines):
        pen = QPen(QColor(*colors[i % 7]))
        pen.setWidth(10)  # Толщина линии
        painter.setPen(pen)
        for line in group_line:
            p1, p2 = line
            painter.drawLine(int(p1[0]), height - int(p1[1]), int(p2[0]), height - int(p2[1]))
    painter.end()
    new_image.save(f"/Users/victoriasmirnova/PycharmProjects/pythonProject6/for_triangle/SK_exp_pict/{input_name}new_strs.jpg")
    y_coords_max = np.array([np.max([(elem[0][1]+elem[1][1])//2 for elem in group_line]) for group_line in merged_all_lines])
    y_coords_min = np.array([np.min([(elem[0][1]+elem[1][1])//2 for elem in group_line]) for group_line in merged_all_lines])
    y_coords = (y_coords_max + y_coords_min) // 2
    q10_y_coords = np.mean(np.abs(np.diff(y_coords))) *0.9
    adjusted_coords = []
    adjusted_coords_min = []
    i = 0
    while i < len(y_coords):
        if i < len(y_coords) - 1 and abs(y_coords[i] - y_coords[i + 1]) < q10_y_coords:
            avg = np.max([y_coords[i], y_coords[i + 1]])
            adjusted_coords.append(avg)  # Записываем среднее
            i += 2  # Пропускаем следующий элемент, так как он уже учтен
        else:
            adjusted_coords.append(y_coords[i])  # Оставляем без изменений
            i += 1
    return adjusted_coords, merged_all_lines, merged_lines_mean

def draw_lines_on_pict(list_of_lines):
    image_strs = QImageReader(file_path).read().convertToFormat(QImage.Format_RGB32)
    painter = QPainter(image_strs)
    for i, group_line in enumerate(list_of_lines):
        pen = QPen(QColor(*colors[i % 7]))
        pen.setWidth(10)  # Толщина линии
        painter.setPen(pen)
        y_coord = group_line

        p1 = [0, y_coord]
        p2 = [width - 1, y_coord]
        painter.drawLine(int(p1[0]), height - int(p1[1]), int(p2[0]), height - int(p2[1]))
    painter.end()
    image_strs.save(f"/Users/victoriasmirnova/PycharmProjects/pythonProject6/for_triangle/SK_exp_pict/{input_name}_init_with_lines.jpg")



def approx_lines(points_array):
    """
    Обрабатывает массив точек, выбирает крайние левые и правые точки, а также 25-й и 75-й квантили (если точек достаточно).
    Если в массиве только две уникальные точки, третьей точкой будет среднее арифметическое этих двух.
    Выполняет аппроксимацию квадратичной функцией методом наименьших квадратов.

    Параметры:
        points_array (numpy.ndarray): Массив точек формы (N, 2, 2).

    Возвращает:
        selected_points (numpy.ndarray): Выбранные точки для аппроксимации (4, 2) или меньше.
        coefficients (tuple): Коэффициенты квадратичной функции (a, b, c).
    """
    # 1. Извлечение всех уникальных точек
    all_points = points_array.reshape(-1, 2)  # Преобразуем в форму (N, 2)
    unique_points = np.unique(all_points, axis=0)

    num_unique = len(unique_points)

    if num_unique < 2:
        raise ValueError("Недостаточно уникальных точек для аппроксимации.")

    # 2. Определение крайних точек
    left_point = unique_points[np.argmin(unique_points[:, 0])]
    right_point = unique_points[np.argmax(unique_points[:, 0])]

    if num_unique >= 4:
        # 25-й и 75-й квантили
        quantile_25 = np.percentile(unique_points[:, 0], 40)
        quantile_75 = np.percentile(unique_points[:, 0], 60)

        # Найти ближайшие точки к квантилям
        point_25 = unique_points[np.argmin(np.abs(unique_points[:, 0] - quantile_25))]
        point_75 = unique_points[np.argmin(np.abs(unique_points[:, 0] - quantile_75))]

        # Выбранные точки
        selected_points = np.array([left_point, point_25, point_75, right_point])
    elif num_unique == 3:
        # Если только три точки, включаем левую, правую и среднюю
        median_x = np.median(unique_points[:, 0])
        middle_point_index = np.argmin(np.abs(unique_points[:, 0] - median_x))
        middle_point = unique_points[middle_point_index]
        selected_points = np.array([left_point, middle_point, right_point])
    elif num_unique == 2:
        # Если только две точки, используем среднее для третьей точки
        middle_point = (left_point + right_point) / 2
        selected_points = np.array([left_point, middle_point, right_point])

    # 3. Подготовка данных для аппроксимации
    x = selected_points[:, 0]
    y = selected_points[:, 1]

    # 4. Аппроксимация квадратичной функцией методом наименьших квадратов
    # Создание матрицы X с тремя столбцами: x^2, x, 1
    X = np.vstack((x ** 2, x, np.ones(len(x)))).T

    # Вектор y
    Y = y

    # Решение системы методом наименьших квадратов
    coefficients, residuals, rank, s = np.linalg.lstsq(X, Y, rcond=None)
    a, b, c = coefficients

    return selected_points, (a, b, c)

def draw_quadratic_lines(image_strs, selected_points, coefficients, color_selected=QColor(255, 0, 0), color_curve=QColor(0, 0, 255)):
    """
    Рисует выбранные точки и аппроксимирующую квадратичную кривую на QImage.

    Параметры:
        qimage (QImage): Изображение, на котором будет происходить отрисовка.
        selected_points (numpy.ndarray): Выбранные точки для аппроксимации (3, 2).
        coefficients (tuple): Коэффициенты квадратичной функции (a, b, c).
        color_selected (QColor): Цвет для выбранных точек.
        color_curve (QColor): Цвет для аппроксимирующей кривой.
    """
    a, b, c = coefficients

    painter = QPainter(image_strs)
    painter.setRenderHint(QPainter.Antialiasing)

    # Рисование выбранных точек
    pen = QPen(color_selected)
    pen.setWidth(8)
    painter.setPen(pen)
    for point in selected_points:
        x, y = point
        painter.drawPoint(int(x), height - int(y))

    # Рисование квадратичной кривой
    pen_curve = QPen(color_curve)
    pen_curve.setWidth(2)
    painter.setPen(pen_curve)

    # Генерация точек для кривой
    x_min, x_max = selected_points[:, 0].min(), selected_points[:, 0].max()
    x_fit = np.linspace(x_min, x_max, 400)  # Расширение области для лучшего отображения
    y_fit = a * x_fit**2 + b * x_fit + c

    # Преобразование координат в целочисленные
    curve_points = [(int(x), int(y)) for x, y in zip(x_fit, y_fit)]

    # Рисование кривой, соединяя последовательные точки
    for i in range(len(curve_points) - 1):
        painter.drawLine(curve_points[i][0], height - curve_points[i][1],
                         curve_points[i+1][0], height - curve_points[i+1][1])

    painter.end()
    image_strs.save(f"/Users/victoriasmirnova/PycharmProjects/pythonProject6/for_triangle/approx/{input_name}_approx.jpg")


'''def calculate_y_threshold(main_points, all_points, margin=0.5):

    # Преобразование данных в массивы
    main_points = np.array(main_points)  # Упорядоченные строки по y
    all_points = np.array(all_points)
    print(abs(np.diff(main_points)) )
    # 1. Вычисляем расстояния между соседними строками в main_points
    if len(main_points) > 1:
        main_distances = abs(np.diff(main_points))  # Разности между соседними строками
        median_main_distance = np.median(main_distances)  # Медианное расстояние
    else:
        median_main_distance = 10  # Если одна строка, задаём минимальное расстояние по умолчанию
    # 2. Минимальные расстояния от all_points до строк main_points
    all_y = all_points[:, 1]
    min_distances = np.min(np.abs(all_y[:, None] - main_points[None, :]), axis=1)
    median_point_distance = np.median(min_distances)  # Медианное расстояние точек до строк
    # 3. Итоговое значение y_threshold
    y_threshold = max(median_main_distance, median_point_distance) * (1 + margin)
    return y_threshold'''


def merge_points_to_lines(main_points, all_points):
    """
    Присоединяет точки к ближайшему кластеру (строке) на основе вертикальной координаты (y).

    Параметры:
        main_points (list): Список координат y кластеров (средние значения y для строк).
        all_points (list): Список точек для присоединения (формат: [[x, y], ...]).

    Возвращает:
        merged_lines (list of lists): Обновлённые строки, каждая строка — список точек.
    """
    # Преобразуем main_points в массив для удобства работы
    main_points = np.array(main_points)
    all_points = np.array(all_points)

    # Инициализация списка строк
    merged_lines = [[] for _ in range(len(main_points))]

    # Обрабатываем каждую точку из all_points
    for point in all_points:
        x_new, y_new = point
        closest_cluster_idx = None
        min_distance = float('inf')

        # Находим ближайший кластер (строку) по координате y
        for i in range(len(main_points)):
            distance = abs(y_new - main_points[i])
            if distance < min_distance:
                min_distance = distance
                closest_cluster_idx = i

        # Присоединяем точку к ближайшему кластеру
        merged_lines[closest_cluster_idx].append((x_new, y_new))

    # Сортируем точки внутри каждой строки по x
    for line in merged_lines:
        line.sort(key=lambda point: point[0])

    return merged_lines


names = ['438-1-219 л2об', '438-1-219 л7об', '438-1-219 л11', '438-1-219 л15', '438-1-219 л17', '438-1-219 л17об', '438-1-219 л21', '438-1-219 л18', '438-1-219 л18об', '438-1-219 л19', '438-1-219 л19об', '438-1-219 л20', '438-1-219 л20об']
#names = ['438-1-219 л2об', '438-1-219 л7об', '438-1-219 л11', '438-1-219 л15', '438-1-219 л17', '438-1-219 л17об', '438-1-219 л21']
for name in names:
    init_image(name)
    x_all, y_all, mean_points, median_points_all = get_coordinates()

    new_image = QImage(width, height, QImage.Format_RGB32)
    new_image.fill(QColor(255, 255, 255))
    draw_contiurs(x_all, y_all, mean_points, new_image)

    new_image.save(f"/Users/victoriasmirnova/PycharmProjects/pythonProject6/for_triangle//SK_exp_pict/{input_name}_frame_points.jpg")

    sorted_filters_lines, triangulation_points = triangulation_draw(mean_points)
    list_of_lines, merged_all_lines, merged_lines_mean = cluster_lines(sorted_filters_lines)
    print(list_of_lines)
    merged_all_lines = merge_points_to_lines(list_of_lines, mean_points)
    print("merged_all_lines", merged_all_lines)
    selected_points, coefs = [], []
    for line_points in merged_all_lines:
        x, (a, b, c) = approx_lines(np.array(line_points))
        selected_points.append(x)
        coefs.append((a, b, c))

    image_strs = QImageReader(file_path).read().convertToFormat(QImage.Format_RGB32)
    for x, (a, b, c) in zip(selected_points, coefs):
        draw_quadratic_lines(image_strs, np.array(x), (a, b, c), color_selected=QColor(255, 0, 0),
                             color_curve=QColor(0, 0, 255))
    #draw_lines_on_pict(list_of_lines)


#########################
'''y_diff_global_end_np = np.concatenate(y_diff_global_end)
y_diff_global_end_np = y_diff_global_end_np[y_diff_global_end_np != 0]
plt.hist(y_diff_global_end_np, bins=100, edgecolor='black')  # bins - количество столбцов
quantiles = [0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

quantile_values = np.quantile(y_diff_global_end_np, quantiles)
# Добавление квантилей на график
for i, (q, value) in enumerate(zip(quantiles, quantile_values)):
    plt.axvline(value, color=np.array(colors[i % 7])/255, linestyle='--', label=f'{int(q * 100)}-й квантиль')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())

plt.xlabel('Разница')
plt.ylabel('Частота')
plt.title('Гистограмма абсолютной разницы высот ребер триангуляций')
plt.savefig("y_diff_global_end.jpg", format="jpg")

plt.show()
plt.hist(np.concatenate(y_diff_for_merging), bins=100, edgecolor='black')  # bins - количество столбцов
quantiles = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75]

quantile_values = np.quantile(np.concatenate(y_diff_for_merging), quantiles)
print(colors)
# Добавление квантилей на график

for i, (q, value) in enumerate(zip(quantiles, quantile_values)):
    plt.axvline(value, color=np.array(colors[i % 7])/255, linestyle='--', label=f'{int(q * 100)}-й квантиль')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())

plt.xlabel('Разница')
plt.ylabel('Частота')
plt.title('Гистограмма абсолютной разницы высот строк')
plt.savefig("y_diff_for_merging.jpg", format="jpg")

plt.show()'''

