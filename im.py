import cv2
import sys

# Video kaynağını aç
video = cv2.VideoCapture("radar.mp4")
if not video.isOpened():
    print("Video açılmadı.")
    sys.exit()

# İlk kareyi oku
success, frame = video.read()
if not success:
    print('Video okuma hatası')
    sys.exit()

# Pencereyi oluştur
cv2.namedWindow("Video", cv2.WINDOW_NORMAL)

# Takip edilecek nesneyi seç
bbox_object = cv2.selectROI("Video", frame, fromCenter=False, showCrosshair=True)

# İzleme bölgesini seç
bbox_zone = cv2.selectROI("Video", frame, fromCenter=False, showCrosshair=True)
cv2.destroyAllWindows()

# Nesne takipçisini oluştur
tracker = cv2.TrackerCSRT_create()
ok = tracker.init(frame, bbox_object)

# Video çalıştır ve nesneyi takip et
while True:
    success, frame = video.read()
    if not success:
        break

    ok, bbox_object = tracker.update(frame)

    # Nesneyi çerçeve içine al ve izleme bölgesini göster
    if ok:
        (x, y, w, h) = [int(v) for v in bbox_object]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2, 1)
        cv2.rectangle(frame, (int(bbox_zone[0]), int(bbox_zone[1])), (int(bbox_zone[0] + bbox_zone[2]), int(bbox_zone[1] + bbox_zone[3])), (0, 255, 0), 2, 1)

        # Nesne izleme bölgesine girerse bildirim ver
        if (x > bbox_zone[0] and x + w < bbox_zone[0] + bbox_zone[2] and y > bbox_zone[1] and y + h < bbox_zone[1] + bbox_zone[3]):
            cv2.putText(frame, "Nesne Bolgede!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            print("Bildirim: Nesne izleme bölgesine girdi!")

    cv2.imshow("Video", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC tuşu ile çıkış
        break

video.release()
cv2.destroyAllWindows()
