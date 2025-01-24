import cv2
import numpy as np

# Video dosyasını aç
video = cv2.VideoCapture("radar.mp4")

# Geçmişte algılanan üçgenlerin koordinatlarını tutmak için bir liste
detected_triangles = []

while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break  # Video bittiğinde döngüyü kır

    # Gri tonlamalı hale getir
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Gürültüyü azaltmak için bulanıklaştırma
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Beyaz rengin bulunduğu bölgeleri tespit etmek için HSV renk uzayına geç
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Beyaz renk için HSV aralığı
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 25, 255])

    # Beyaz renkli bölgeleri maskele
    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    # Maskeyi orijinal görüntüyle birleştir
    masked_frame = cv2.bitwise_and(frame, frame, mask=white_mask)

    # Maskelenmiş görüntüde kenarları tespit et
    edges = cv2.Canny(masked_frame, 30, 100)

    # Konturları bul
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Yeni bir maske oluştur
    mask = np.zeros_like(frame)

    current_triangles = []

    for contour in contours:
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 3:  # Eğer köşe sayısı 3 ise (üçgen)
            area = cv2.contourArea(contour)
            if area > 500:  # Çok küçük konturları filtrele
                # Üçgeni çiz ve koordinatlarını kaydet
                cv2.drawContours(mask, [approx], 0, (255, 255, 255), -1)
                current_triangles.append(approx)

    # Eski üçgenlerle birleştir ve süreklilik sağla
    for triangle in detected_triangles:
        cv2.drawContours(mask, [triangle], 0, (255, 255, 255), -1)

    # Yeni tespit edilen üçgenleri geçmiş listesine ekle
    detected_triangles = current_triangles

    # Orijinal kareyi sadece üçgenlerin olduğu maskeye göre filtrele
    result = cv2.bitwise_and(frame, mask)

    # Çıktıyı yeniden boyutlandır (pencereyi küçültmek için)
    resized_result = cv2.resize(result, (640, 360))

    # Çerçeveyi göster
    cv2.imshow("White Triangles Only", resized_result)

    # 'q' tuşuna basarak çıkış yap
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kaynakları serbest bırak
video.release()
cv2.destroyAllWindows()
