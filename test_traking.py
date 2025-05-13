from math import sqrt
import cv2
import numpy as np
import time

x1 = 586
y1 = 474
x2 = 650
y2 = 606

def remove_small_particles(img, min_size=150):
    # Проверяем, цветное ли изображение (3 канала)
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Конвертируем в grayscale
    else:
        gray = img  # Если уже grayscale, оставляем как есть

    # Бинаризация (если изображение не бинарное)
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # Находим контуры
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Создаем маску для крупных объектов
    mask = np.zeros_like(binary)

    for contour in contours:
        # Если площадь контура больше min_size, добавляем его в маску
        if cv2.contourArea(contour) > min_size:
            cv2.drawContours(mask, [contour], -1, 255, -1)

    # Применяем маску к оригинальному изображению
    if len(img.shape) == 3:
        # Если исходное изображение цветное, применяем маску к каждому каналу
        result = cv2.bitwise_and(img, img, mask=mask)
    else:
        # Если исходное изображение grayscale, просто применяем маску
        result = cv2.bitwise_and(img, img, mask=mask)

    return result

def line_intersection(p1, p2, q1, q2):
    """Вычисляет точку пересечения отрезков p1-p2 и q1-q2, если она существует."""
    def det(a, b):
        return a[0]*b[1] - a[1]*b[0]

    xdiff = (p1[0] - p2[0], q1[0] - q2[0])
    ydiff = (p1[1] - p2[1], q1[1] - q2[1])

    div = det(xdiff, ydiff)
    if div == 0:
        return None  # Линии параллельны

    d = (det(p1, p2), det(q1, q2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div

    # Проверка, что точка пересечения лежит на обоих отрезках
    if (min(p1[0], p2[0]) <= x <= max(p1[0], p2[0]) and
        min(p1[1], p2[1]) <= y <= max(p1[1], p2[1]) and
        min(q1[0], q2[0]) <= x <= max(q1[0], q2[0]) and
        min(q1[1], q2[1]) <= y <= max(q1[1], q2[1])):
        return (x, y)
    return None

# Пример использования
# Загрузка изображения и получение контура
image = cv2.imread('Image.png', cv2.IMREAD_GRAYSCALE)
image = remove_small_particles(image)
# Задание линии
pt1 = (x1, y1)
pt2 = (x2, y2)

cv2.line(image, pt1, pt2, (255, 0, 0), 1)

_, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
intersections = []

start_time = time.time()
for contour in contours:
    data = contour.reshape(-1, 2)  # Самый быстрый способ убрать лишние уровни
    length = len(data)
    for i in range(length):
        p1 = data[i]
        p2 = data[(i + 1) if i + 1 < length else 0]
        # Расчёт на голых int-ах — пуля быстрее
        x1_, y1_ = p1
        x2_, y2_ = p2
        intersect = line_intersection((x1_, y1_), (x2_, y2_), pt1, pt2)
        if intersect:
            intersections.append(intersect)
            break

print("The code was executed for ", time.time()-start_time, " seconds")

cv2.imshow("Image", image)
cv2.waitKey(0)

print("Изначальная длина: ", (sqrt((x2-x1)**2+(y2-y1)**2)))

for pt in intersections:
    print(f"{float(pt[0])}   {float(pt[1])}")
    print("Длина до точки пересечения: ", (sqrt((x1-float(pt[0]))**2+(y1-float(pt[1]))**2)))

print("\nКонтур", contour)