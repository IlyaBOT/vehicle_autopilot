import cv2
import numpy as np

# Размер изображения (можно изменить под нужное)
image = cv2.imread('Image.png', cv2.IMREAD_GRAYSCALE)

# Координаты центра квадрата
center_x, center_y = 500, 334  # заменяй на нужные координаты

# Размер квадрата
size = 32
half = size // 2

# Вычисление координат углов квадрата
top_left = (center_x - half, center_y - half)
bottom_right = (center_x + half, center_y + half)

# Копия изображения для отрисовки (чтобы не испортить оригинал)
image_with_square = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # переводим в цветное, чтобы нарисовать цветной квадрат

# Рисуем квадрат (зелёный, толщина 2)
cv2.rectangle(image_with_square, top_left, bottom_right, (0, 255, 0), 2)

# Показываем результат
cv2.imshow("Square on Image", image_with_square)
cv2.waitKey(0)
cv2.destroyAllWindows()