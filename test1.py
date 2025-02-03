import cv2
import numpy as np

video = cv2.VideoCapture("radar.mp4")

detected_triangles = []

while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break  # Video bittiğinde döngüyü kır

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 25, 255])

    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    masked_frame = cv2.bitwise_and(frame, frame, mask=white_mask)

    edges = cv2.Canny(masked_frame, 30, 100)

    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros_like(frame)

    current_triangles = []

    for contour in contours:
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 3:  
            area = cv2.contourArea(contour)
            if area > 500:  
                cv2.drawContours(mask, [approx], 0, (255, 255, 255), -1)
                current_triangles.append(approx)

    for triangle in detected_triangles:
        cv2.drawContours(mask, [triangle], 0, (255, 255, 255), -1)

    detected_triangles = current_triangles

    result = cv2.bitwise_and(frame, mask)

    resized_result = cv2.resize(result, (640, 360))

    cv2.imshow("White Triangles Only", resized_result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
