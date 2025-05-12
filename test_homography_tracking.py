import cv2
import numpy as np

# Initialize video capture
# cap = cv2.VideoCapture(0)  # or video file path

# Read first frame
frame = cv2.imread('screenshot_cam0.webp')


# Let user select ROI
bbox = cv2.selectROI("Select Object to Track", frame, False)
cv2.destroyWindow("Select Object to Track")

# Extract ROI from frame
x, y, w, h = [int(i) for i in bbox]
ref_frame = frame[y:y + h, x:x + w]
ref_gray = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)

# Initialize ORB detector
orb = cv2.ORB_create()

# Find keypoints and descriptors in ROI
ref_kp, ref_des = orb.detectAndCompute(ref_gray, None)

# Define reference corners (in ROI coordinates)
ref_corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)

# FLANN parameters and matcher
FLANN_INDEX_LSH = 6
index_params = dict(algorithm=FLANN_INDEX_LSH,
                    table_number=6,
                    key_size=1,
                    multi_probe_level=1)
search_params = dict(checks=1)
flann = cv2.FlannBasedMatcher(index_params, search_params)

while True:
    frame = cv2.imread('screenshot_cam0.webp')

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find keypoints and descriptors in current frame
    kp, des = orb.detectAndCompute(gray, None)
    print('len(kp)', len(kp))

    if des is not None and len(kp) > 3:
        # Match descriptors
        matches = flann.knnMatch(ref_des, des, k=2)

        # Lowe's ratio test
        good = []
        try:
            for m, n in matches:
                print('m.distance', m.distance, 'n.distance', n.distance)
                if m.distance < 0.9 * n.distance:
                    good.append(m)
            print('len(good)', len(good))
        except Exception as e:
            print(e)
        if len(good) > 2:
            # Get matching points
            src_pts = np.float32([ref_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            # Find homography with RANSAC
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if M is not None:
                # Transform reference corners to current frame
                # Note: We need to adjust for the original ROI position
                adjusted_ref_corners = ref_corners + np.float32([[[x, y]]])  # Add original ROI offset
                current_corners = cv2.perspectiveTransform(adjusted_ref_corners, M)

                # Draw bounding box
                frame = cv2.polylines(frame, [np.int32(current_corners)], True, (0, 255, 0), 3)

                # Display number of good matches
                cv2.putText(frame, f'Matches: {len(good)}', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Homography Tracking', frame)
    cv2.imshow('ref_frame', ref_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
