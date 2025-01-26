import cv2
import numpy as np

# Görüntüyü yükle
image = cv2.imread('radar.png')

# Görüntüyü gri tonlamaya çevir
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Görüntüyü bulanıklaştır
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Eşikleme ile görüntüyü binary hale getir
_, thresholded = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

# Konturları bul
contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Sadece üçgen nesneleri tutmak için boş bir siyah görüntü oluştur
output_image = np.zeros_like(image)

# Üçgenleri tespit et ve çiz
for contour in contours:
    # Konturu düzgünleştir
    epsilon = 0.04 * cv2.arcLength(contour, True)  # Yaklaşık değer
    approx = cv2.approxPolyDP(contour, epsilon, True)  # Konturu düzgünleştir
    
    # Eğer şekil 3 kenarlıysa, bu bir üçgen olabilir
    if len(approx) == 3:
        cv2.drawContours(output_image, [approx], -1, (0, 255, 0), 2)  # Üçgeni yeşil renkte çiz

# Sonuç görüntüsünü göster
cv2.imshow("Üçgenler", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
