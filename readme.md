## VideoLogger

### Описание проекта:

API-сервис для трекинга людей на видеопотоке.

Основан на детекторе YOLOX, предварительно обученном на датасете Microsoft COCO, и трекере BYTE.
Использует PyTorch для реализации детекции/трекинга и FastAPI для предоставления API.

На данный момент находится в стадии активной разработки.


### Источники:
* [Детектор YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
* [Трекер BYTE](https://github.com/ifzhang/ByteTrack)