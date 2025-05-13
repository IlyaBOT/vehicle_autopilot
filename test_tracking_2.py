import cv2
import numpy as np
from vehicle.Vehicle import Vehicle

class RobotTracker:
    def __init__(self, vehicle):
        self.vehicle = vehicle
        self.tracker = cv2.legacy.TrackerCSRT_create()
        self.angle = 0

    def init_tracking(self, frame):
        bbox = cv2.selectROI("Select ROI", frame, False)
        cv2.destroyWindow("Select ROI")
        self.tracker.init(frame, bbox)

    def process_frame(self, frame):
        success, bbox = self.tracker.update(frame)
        if success:
            x, y, w, h = [int(v) for v in bbox]
            center = (x + w // 2, y + h // 2)

            # Calculate robot orientation (assumed already provided externally)
            angle_rad = np.radians(self.angle)

            # Front of the robot
            front_point = (
                int(center[0] + (w // 2) * np.cos(angle_rad)),
                int(center[1] + (h // 2) * np.sin(angle_rad))
            )

            short_end = (
                int(front_point[0] + 50 * np.cos(angle_rad)),
                int(front_point[1] + 50 * np.sin(angle_rad))
            )
            long_end = (
                int(front_point[0] + 80 * np.cos(angle_rad)),
                int(front_point[1] + 80 * np.sin(angle_rad))
            )

            # Mask for line detection
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask_line = cv2.inRange(hsv_frame, np.array([0, 54, 157]), np.array([60, 255, 210]))

            def is_on_line(point):
                x, y = point
                return 0 <= x < mask_line.shape[1] and 0 <= y < mask_line.shape[0] and mask_line[y, x] > 0

            short_on_line = all(is_on_line((int(front_point[0] + i * np.cos(angle_rad)),
                                            int(front_point[1] + i * np.sin(angle_rad)))) for i in range(1, 51, 5))

            long_on_line = all(is_on_line((int(front_point[0] + i * np.cos(angle_rad)),
                                           int(front_point[1] + i * np.sin(angle_rad)))) for i in range(1, 81, 5))

            if short_on_line and long_on_line:
                self.vehicle.setMotorPower(100, 100)
                status = "Full speed (100%)"
            elif short_on_line and not long_on_line:
                self.vehicle.setMotorPower(15, 15)
                status = "Slowing down (15%)"
            else:
                self.vehicle.setMotorPower(0, 0)
                self.vehicle.rotate(-30)
                status = "Stopping and turning left (-30°)"

            # Visualization
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(frame, center, 5, (255, 0, 0), -1)
            cv2.line(frame, front_point, short_end, (255, 0, 0), 2)
            cv2.line(frame, front_point, long_end, (255, 0, 0), 2)
            cv2.putText(frame, status, (10, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            blended = cv2.addWeighted(frame, 0.7, cv2.cvtColor(mask_line, cv2.COLOR_GRAY2BGR), 0.3, 0)
            cv2.imshow("Robot Tracking", blended)
        else:
            cv2.putText(frame, "Tracking lost", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Robot Tracking", frame)


def main():
    vehicle = Vehicle()
    tracker = RobotTracker(vehicle)

    frame = cv2.imread('camera1.webp')
    tracker.init_tracking(frame)

    while True:
        frame = cv2.imread('camera1.webp')
        tracker.process_frame(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
