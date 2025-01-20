from pytube import YouTube

# İndirmek istediğiniz YouTube videosunun URL'sini buraya yazın
video_url = 'https://www.youtube.com/watch?v=8oPBdQhQuNY'

# YouTube video nesnesi oluşturma
yt = YouTube(video_url)

# En yüksek çözünürlükteki video akışını seçme
video_stream = yt.streams.filter(only_video=True, file_extension='mp4').get_highest_resolution()

# Videoyu indirme
video_stream.download(filename='downloaded_video.mp4')

print("Video başarıyla indirildi!")
