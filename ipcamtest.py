import cv2
import numpy as np

# IP kameradan video akışını al
cap = cv2.VideoCapture('rtsp://admin:MHYwKe@192.168.1.26:8556')

# Sarı rengi tespit etmek için renk aralığı (HSV formatında)
lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([30, 255, 255])

# Morfolojik işlemler için kernel
kernel = np.ones((5, 5), np.uint8)

# Arka plan çıkarıcıyı oluştur
back_sub = cv2.createBackgroundSubtractorMOG2()

# Pencereyi oluştur ve boyutlandır
cv2.namedWindow('Sadece Top', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Sadece Top', 800, 600)  # İstediğiniz boyutları ayarlayın

while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break

    # Gürültü azaltma
    frame = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)

    # Görüntüyü HSV formatına çevirme
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Gaussian Blur ile gürültüyü azaltma
    blurred = cv2.GaussianBlur(hsv, (15, 15), 0)

    # Sarı rengi maskeyle tespit etme
    mask = cv2.inRange(blurred, lower_yellow, upper_yellow)

    # Arka plan çıkarma
    fg_mask = back_sub.apply(frame)

    # Maskeyi birleştirme (hem sarı renk maskesi hem de arka plan maskesi)
    combined_mask = cv2.bitwise_and(mask, mask, mask=fg_mask)

    # Maskeyi işlemek için morfolojik işlemler
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

    # Maskeyi kullanarak sadece sarı nesneleri gösterme
    result = cv2.bitwise_and(frame, frame, mask=combined_mask)

    # Topun pozisyonunu işaretleme
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
        if len(approx) >= 12:  # + işareti olarak kabul edilebilecek bir köşe sayısı
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.drawContours(result, [approx], 0, (0, 255, 0), 2)
            cv2.circle(result, (x + w // 2, y + h // 2), 5, (0, 0, 255), -1)

    # Görüntüyü gösterme
    cv2.imshow('Sadece Top', result)

    # Çıkmak için 'q' tuşuna basın
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
