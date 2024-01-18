## VideoEventWatcher


### Описание проекта:

Простой API-сервис для отслеживания событий на видеопотоке.

Основан на детекторе YOLOX, предварительно обученном на датасете Microsoft COCO, и трекере BYTE.
Использует PyTorch для реализации детекции/трекинга и FastAPI для предоставления API.


### Установка:

1. Клонировать репозиторий:  
   `git clone https://github.com/mvp280102/VideoEventWatcher`
2. Перейти в директорию проекта:  
   `cd VideoEventWatcher`
3. Создать и активировать виртуальное окружение:  
   `python -m venv venv`  
   Для Linux/Mac: `source venv/bin/activate`  
   Для Windows: `venv/Scripts/activate.bat`
4. Установить зависимости:  
   `pip install -r requirements.txt`

Для корректной работы сервиса необходим интерпретатор Python версии не ниже 3.8,
а также брокер сообщений RabbitMQ и сервер СУБД PostgreSQL.
Также рекомендован инструмент администрирования баз данных pgAdmin 4.


### Первый запуск:
1. В СУБД PostgreSQL создать базу данных с названием `videoeventwatcher`.
2. В директории проекта создать поддиректории с именами `inputs`, `outputs`, `frames`, `events`, `tracks`.
3. Загрузить веса модели детекции из официального репозитория YOLOX (см. раздел Источники).


### Использование:

1. Запустить сервер FastAPI:  
   `uvicorn main:app --reload`  
   Сервер будет доступен по адресу http://127.0.0.1:8000.
2. Задать конфигурационные данные в файлах директории `config`.
3. Загрузить файл с видео для обработки и конфигурационные файлы по соответствующим адресам.
4. Запустить обнаружение и обработку событий.


### Результат:

TODO:


### Источники:
* [Веб-фреймворк FastAPI](https://fastapi.tiangolo.com/)
* [Детектор YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
* [Трекер BYTE](https://github.com/ifzhang/ByteTrack)
