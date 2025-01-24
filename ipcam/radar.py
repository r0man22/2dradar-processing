import cv2
import numpy as np
import streamlink

def stream_to_url(url, quality='best'):
    streams = streamlink.streams(url)
    if streams:
        return streams[quality].to_url()
    else:
        raise ValueError("No streams were available")

def main(url, quality='best', fps=30.0):
    stream_url = stream_to_url(url, quality)
    cap = cv2.VideoCapture(stream_url)

    if not cap.isOpened():
        raise ValueError("Failed to open video stream")

    # Gerçek FPS değerini al
    real_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Real FPS: {real_fps}")

    # Eğer gerçek FPS çok düşükse, bunu kullanabiliriz. Ancak genellikle gerçek FPS'i kullanmak daha doğru olur.
    frame_time = int((1.0 / real_fps) * 1000.0)

    # Ekran boyutlarını al
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ROI ayarları
    roi_height = height // 6
    roi_width = width // 7  # Ekranın ortasında 1/7'lik genişlik

    # ROI'nin koordinatlarını hesapla (alt kısımda ortalanmış)
    roi_top = height - roi_height - 30  # ROI'yi 30px yukarı kaydırıyoruz
    roi_left = (width - roi_width) // 2

    # Pencereyi başlatma
    window_name = "Twitch Stream"
    cv2.namedWindow(window_name)

    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Video çözünürlüğünü üç katına çıkarma (örneğin 960x720)
            resized_frame = cv2.resize(frame, (960, 720))  # Frame'i 960x720 boyutlarına büyült

            # ROI'yi kesme
            if roi_top + roi_height <= resized_frame.shape[0] and roi_left + roi_width <= resized_frame.shape[1]:
                roi_frame = resized_frame[roi_top:roi_top + roi_height, roi_left:roi_left + roi_width]
            else:
                print("Invalid ROI coordinates")
                continue  # Geçersiz ROI koordinatları, bir sonraki frame'e geç

            # Sarı renk aralığını tanımlama (HSV formatında)
            hsv_frame = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
            lower_yellow = np.array([20, 100, 100])  # Alt sarı renk sınırı
            upper_yellow = np.array([30, 255, 255])  # Üst sarı renk sınırı

            # Sarı renk maskesini oluşturma
            yellow_mask = cv2.inRange(hsv_frame, lower_yellow, upper_yellow)

            # Maskeyi frame üzerine uygulama
            yellow_objects = cv2.bitwise_and(roi_frame, roi_frame, mask=yellow_mask)

            # Sarı nesnelerin etrafına çerçeve çekme (sadece sarı nesneleri gösterecek)
            contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) > 500:  # Nesnenin yeterince büyük olduğundan emin olun
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(roi_frame, (x, y), (x + w, y + h), (0, 255, 255), 2)  # Sarı çerçeve

            # ROI'nin üzerine yeşil çerçeve ekleme
            cv2.rectangle(resized_frame, (roi_left, roi_top), (roi_left + roi_width, roi_top + roi_height), (0, 255, 0), 2)

            # ROI'yi tam ekran yerine sadece ROI kısmını göstereceğiz
            resized_frame[roi_top:roi_top + roi_height, roi_left:roi_left + roi_width] = roi_frame

            # Frame'i ekranda göster
            cv2.imshow(window_name, resized_frame)

            # FPS'yi daha doğru bir şekilde kontrol et
            if cv2.waitKey(frame_time) & 0xFF == ord('q'):
                break
        except Exception as e:
            print(f"Error: {e}")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    url = "https://www.twitch.tv/hxneymar"  # Twitch kanal URL'si
    main(url)


