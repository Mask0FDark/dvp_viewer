# DVP RTSP/UDP Viewer

Программа для просмотра RTSP/UDP потоков с использованием GStreamer + OpenCV.

## Возможности
- Поддержка RTSP и UDP
- Декодирование H264/H265
- Медианная фильтрация (Canny)
- Масштабируемое окно
- Контроль FPS и отображение контуров

## Требования
- Python 3.10+
- PyGObject (gi)
- OpenCV
- GStreamer runtime (если запускаешь exe, включено в zip)

## Запуск
```bash
python dvp.py
