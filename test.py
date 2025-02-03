import cv2
import numpy as np

video_path = 'radar3.mp4'
cap = cv2.VideoCapture(video_path)

ret, frame = cap.read()
if not ret:
    print("Video okunamadı.")
    exit()

height, width, _ = frame.shape

roi_width = width // 6
roi_height = height // 6

roi_x = (width - roi_width) // 2  
roi_y = height - roi_height - 26  

cv2.namedWindow("Video with ROI", cv2.WINDOW_NORMAL) 
cv2.resizeWindow("Video with ROI", 800, 600)  

while True:
    ret, frame = cap.read()
    if not ret:
        break

    roi_frame = frame[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]
    masked_frame = np.zeros_like(frame) 
    masked_frame[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width] = roi_frame

    roi_gray = cv2.cvtColor(masked_frame[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width], cv2.COLOR_BGR2GRAY)
    roi_gray_colored = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2BGR)  # Gri tonlamayı tekrar renkliye çeviriyoruz
    masked_frame[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width] = roi_gray_colored

    new_roi_width = roi_width * 2  # ROI'yi iki katına çıkartıyoruz
    new_roi_height = roi_height * 2

    roi_resized = cv2.resize(roi_gray_colored, (new_roi_width, new_roi_height))

    roi_end_x = roi_x + new_roi_width
    roi_end_y = roi_y + new_roi_height

    if roi_end_x > width:
        roi_end_x = width
        roi_resized = cv2.resize(roi_gray_colored, (width - roi_x, new_roi_height))  

    if roi_end_y > height:
        roi_end_y = height
        roi_resized = cv2.resize(roi_gray_colored, (new_roi_width, height - roi_y))  

    masked_frame[roi_y:roi_end_y, roi_x:roi_end_x] = roi_resized

    cv2.imshow("Video with ROI", masked_frame)

    # 'q' tuşuna basarak çıkış yapabiliriz
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Video kaynağını serbest bırak ve pencereleri kapat
cap.release()
cv2.destroyAllWindows()
