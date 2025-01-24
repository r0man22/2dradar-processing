import cv2
import numpy as np

# IP kameranın RTSP URL'si
ip_camera_url = 'rtsp://admin:MHYwKe@192.168.1.26:8556'

# IP kameradan veri al
cap = cv2.VideoCapture(ip_camera_url)

# Sarı rengi tespit etmek için renk aralığı (HSV formatında)
lower_yellow = np.array([15, 100, 100])  # Aralığı biraz daha genişlettik
upper_yellow = np.array([35, 255, 255])

# Morfolojik işlemler için kernel
kernel = np.ones((5, 5), np.uint8)

# Pencereyi oluştur ve boyutlandır
cv2.namedWindow('Sadece Top', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Sadece Top', 800, 600)  # İstediğiniz boyutları ayarlayın

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Kameradan görüntü alınamadı.")
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
        # Konturları birleştir ve daha düzgün hale getir
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)

        # Artı işaretine benzeyen şekilleri tespit et
        if len(approx) >= 12:  # Artı işareti için genellikle çok sayıda kenar vardır
            (x, y, w, h) = cv2.boundingRect(contour)

            # Yükseklik ve genişlik oranlarına bakarak şeklin artı işareti olup olmadığını kontrol et
            aspect_ratio = w / h
            if aspect_ratio > 0.8 and aspect_ratio < 1.2:  # Yüksekliği ve genişliği yakın olan nesneleri tercih ederiz
                # Artı işaretini yeşil renkle çizin
                cv2.drawContours(result, [approx], 0, (0, 255, 0), 2)
                # Merkeze kırmızı bir nokta ekleyin
                cv2.circle(result, (x + w // 2, y + h // 2), 5, (0, 0, 255), -1)
    
    # Frame'i göster
    cv2.imshow('Sadece Top', result)
    
    # Çıkmak için 'q' tuşuna basın
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
