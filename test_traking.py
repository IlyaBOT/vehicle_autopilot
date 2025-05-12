import cv2
import numpy as np
from scipy.spatial import distance as dist
from collections import deque


class HighAccuracyTracker:
    def __init__(self):
        # Initialize multiple tracking methods for robustness
        self.tracker = cv2.TrackerCSRT_create()  # Primary tracker
        self.feature_tracker = cv2.legacy.MultiTracker_create()  # For feature points
        self.orb = cv2.ORB_create(1000)  # Feature detector
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Angle estimation variables
        self.reference_points = None
        self.reference_features = None
        self.reference_descriptors = None
        self.angle = 0
        self.angle_history = deque(maxlen=10)  # For angle smoothing
        self.kalman = cv2.KalmanFilter(4, 2)  # For position smoothing
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)

        # Parameters
        self.min_matches = 10
        self.ratio_test_threshold = 0.75

    def init(self, frame, bbox):
        # Initialize primary tracker
        self.tracker.init(frame, bbox)

        # Initialize Kalman filter
        self.kalman.statePre = np.array([[bbox[0]], [bbox[1]], [0], [0]], dtype=np.float32)
        self.kalman.statePost = np.array([[bbox[0]], [bbox[1]], [0], [0]], dtype=np.float32)

        # Extract reference points and features
        x, y, w, h = [int(v) for v in bbox]
        roi = frame[y:y + h, x:x + w]

        # Store reference points (corners of bounding box)
        self.reference_points = np.array([
            [x, y],  # Top-left
            [x + w, y],  # Top-right
            [x + w, y + h],  # Bottom-right
            [x, y + h]  # Bottom-left
        ], dtype=np.float32)

        # Detect and store reference features
        self.reference_features, self.reference_descriptors = self.orb.detectAndCompute(roi, None)

        # Initialize feature point tracker
        kp = self.orb.detect(roi)
        for p in kp[:10]:  # Track top 10 features
            pt = (x + int(p.pt[0]), y + int(p.pt[1]))
            self.feature_tracker.add(cv2.legacy.TrackerKCF_create(), frame, (pt[0] - 5, pt[1] - 5, 10, 10))

    def update(self, frame):
        # Update primary tracker
        success, bbox = self.tracker.update(frame)

        if not success:
            return False, None, None

        # Get current bounding box coordinates
        x, y, w, h = [int(v) for v in bbox]
        center = (x + w // 2, y + h // 2)

        # Update Kalman filter
        prediction = self.kalman.predict()
        measurement = np.array([[np.float32(center[0])], [np.float32(center[1])]])
        estimated = self.kalman.correct(measurement)

        # Use Kalman-filtered position
        smoothed_x = int(estimated[0][0])
        smoothed_y = int(estimated[1][0])
        smoothed_center = (smoothed_x, smoothed_y)

        # Feature-based angle estimation (more accurate)
        current_roi = frame[y:y + h, x:x + w]
        current_features, current_descriptors = self.orb.detectAndCompute(current_roi, None)

        if (self.reference_descriptors is not None and current_descriptors is not None and
                len(self.reference_descriptors) > self.min_matches and
                len(current_descriptors) > self.min_matches):

            # Match features
            matches = self.bf.match(self.reference_descriptors, current_descriptors)
            matches = sorted(matches, key=lambda x: x.distance)

            # Apply ratio test
            good_matches = []
            for m in matches[:50]:
                if m.distance < 30:  # Absolute distance threshold
                    good_matches.append(m)

            if len(good_matches) > self.min_matches:
                # Get matching points
                ref_pts = np.float32([self.reference_features[m.queryIdx].pt for m in good_matches])
                ref_pts[:, 0] += x  # Adjust for ROI offset
                ref_pts[:, 1] += y

                curr_pts = np.float32([current_features[m.trainIdx].pt for m in good_matches])
                curr_pts[:, 0] += x
                curr_pts[:, 1] += y

                # Estimate affine transformation
                M, _ = cv2.estimateAffinePartial2D(ref_pts, curr_pts)

                if M is not None:
                    # Extract angle from transformation matrix
                    angle_rad = np.arctan2(M[1, 0], M[0, 0])
                    self.angle = np.degrees(angle_rad)
                    self.angle_history.append(self.angle)

                    # Calculate median angle for smoothing
                    if len(self.angle_history) > 0:
                        self.angle = np.median(self.angle_history)

        return True, (smoothed_x - w // 2, smoothed_y - h // 2, w, h), self.angle


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


def find_non_black_rectangles(img, black_threshold=1, min_area=150):
    # Конвертация в grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Создание маски для не-черных областей (пиксели ярче чем black_threshold)
    _, mask = cv2.threshold(gray, black_threshold, 255, cv2.THRESH_BINARY)

    # Поиск контуров
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Фильтрация контуров по площади и форме (ищем прямоугольники)
    rectangles = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        print("Area: {}".format(area))
        if area < min_area:
            continue

        # Аппроксимация контура многоугольником
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        # Если у многоугольника 4 угла - считаем его прямоугольником
        if len(approx) >= 4:
            # rectangles.append(approx)
            # Иначе берем ограничивающий прямоугольник
            x, y, w, h = cv2.boundingRect(cnt)

            # Создаем прямоугольник из 4 точек
            rect = np.array([
                [x, y],
                [x + w, y],
                [x + w, y + h],
                [x, y + h]
            ], dtype=np.int32)
            rectangles.append(rect)
            print(rectangles)

    # Отрисовка найденных прямоугольников
    result = img.copy()
    for rect in rectangles:
        cv2.drawContours(result, [rect], -1, (0, 255, 0), 3)

    # Показать результат
    # cv2.imshow('Non-black Rectangles', result)

    return result

# Initialize video capture
# video = cv2.VideoCapture('video.mp4')

# Read first frame
frame = cv2.imread('screenshot_cam0.webp')

lower = np.array([0, 0, 0])
upper = np.array([179, 30, 187])

# Convert to HSV format and color threshold
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, lower, upper)
frame = cv2.bitwise_and(frame, frame, mask=mask)
# ret, frame = video.read()
frame = remove_small_particles(frame)
# Select ROI (region of interest)
cv2.imshow("Select ROI", frame)
bbox = cv2.selectROI("Select ROI", frame, False)
cv2.destroyWindow("Select ROI")

# Initialize our high accuracy tracker
tracker = HighAccuracyTracker()
tracker.init(frame, bbox)

# Create output window
cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)

tray = find_non_black_rectangles(frame)
# cv2.imshow("Rectangles", tray)


while True:
    frame = cv2.imread('screenshot_cam0.webp')

    lower = np.array([0, 0, 0])
    upper = np.array([179, 30, 187])

    # Convert to HSV format and color threshold
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    frame = cv2.bitwise_and(frame, frame, mask=mask)
    frame = remove_small_particles(frame)
    tray = find_non_black_rectangles(frame)
    cv2.imshow("Rectangle", tray)
    # ret, frame = video.read()
    # if not ret:
    #     break

    # Update tracker
    success, bbox, angle = tracker.update(frame)

    # Draw results if tracking was successful
    if success:
        x, y, w, h = [int(v) for v in bbox]
        center = (x + w // 2, y + h // 2)

        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Draw center point
        cv2.circle(frame, center, 5, (255, 0, 0), -1)

        # Draw orientation arrow
        angle_rad = np.radians(angle)
        end_point = (int(center[0] + 50 * np.cos(angle_rad)),
                     int(center[1] + 50 * np.sin(angle_rad)))
        cv2.arrowedLine(frame, center, end_point, (0, 0, 255), 2, tipLength=0.3)

        # Draw info
        info = [
            f"Position: ({center[0]}, {center[1]})",
            f"Angle: {angle:.1f}°",
            f"Size: {w}x{h}",
            "Press Q to quit"
        ]

        for i, text in enumerate(info):
            cv2.putText(frame, text, (10, 30 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    else:
        cv2.putText(frame, "Tracking lost", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow("Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# video.release()
cv2.destroyAllWindows()
