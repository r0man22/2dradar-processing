import cv2
import numpy as np

# Video dosyasını yükle
cap = cv2.VideoCapture('radar.mp4')

# Sarı rengi tespit etmek için renk aralığı (HSV formatında)
lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([30, 255, 255])

# Morfolojik işlemler için kernel
kernel = np.ones((5, 5), np.uint8)

# Pencereyi oluştur ve boyutlandır
cv2.namedWindow('Sadece Top', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Sadece Top', 800, 600)  # İstediğiniz boyutları ayarlayın

while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break
    
    # Görüntüyü HSV formatına çevirme
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Gaussian Blur ile gürültüyü azaltma
    blurred = cv2.GaussianBlur(hsv, (15, 15), 0)
    
    # Sarı rengi maskeyle tespit etme
    mask = cv2.inRange(blurred, lower_yellow, upper_yellow)
    
    # Morfolojik işlemlerle gürültüyü temizleme
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Maskeyi kullanarak sadece sarı nesneleri gösterme
    result = cv2.bitwise_and(frame, frame, mask=mask)
    
    # Topun pozisyonunu işaretleme
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
        if len(approx) >= 12:  # + işareti olarak kabul edilebilecek bir köşe sayısı
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.drawContours(result, [approx], 0, (0, 255, 0), 2)
            cv2.circle(result, (x + w // 2, y + h // 2), 5, (0, 0, 255), -1)
    
    # Frame'i göster
    cv2.imshow('Sadece Top', result)
    
    # Çıkmak için 'q' tuşuna basın
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
