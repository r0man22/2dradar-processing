import cv2

# Video dosyasını aç (Daha önce indirdiğimiz veya YouTube'dan aldığımız video)
cap = cv2.VideoCapture("video.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Video sona erdi veya okuma başarısız!")
        break

    # Görüntü boyutlarını al
    height, width, _ = frame.shape

    # ROI tanımla (alt kısmın ortasında 1/20'lik bir alan)
    roi_height = height // 6  # 1/20 yüksekliği
    roi_width = width // 6  # Görüntünün genişliğinin 1/5'i
    x_start = (width - roi_width) // 2  # ROI'nin sol kenar başlangıcı
    y_start = height - roi_height - 16  # ROI'nin üst kenar başlangıcı

    # ROI bölgesini kes
    roi = frame[y_start:y_start + roi_height, x_start:x_start + roi_width]

    # ROI üzerinde bir işlem (örneğin gri tonlama)
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi_colored = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2BGR)

    # Orijinal görüntüye geri yerleştirme (işlenmiş ROI'yi görüntüye koyma)
    frame[y_start:y_start + roi_height, x_start:x_start + roi_width] = roi_colored

    # ROI'yi göstermek için bir çerçeve çiz (isteğe bağlı)
    cv2.rectangle(frame, (x_start, y_start), (x_start + roi_width, y_start + roi_height), (0, 255, 0), 2)

    # Görüntüyü ekranda göster
    cv2.imshow("Video with ROI", frame)

    # İşlenmiş ROI'yi ayrıca göstermek isterseniz:
    # cv2.imshow("Processed ROI", roi_colored)

    # 'q' tuşuna basarak çıkış yap
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
