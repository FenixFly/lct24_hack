# lct24_hack


## Запуск процедуры геопривязки с помощью скрипта `main.py`

Для запуска процедуры получения координат воспользуйтесь скриптом `main.py`:
 ```bash
python main.py --crop_name "D:\satellite_hackathon\18. Sitronics\1_20\crop_0_0_0000.tif" --layout_name "D:\satellite_hackathon\18. Sitronics\layouts\layout_2021-08-16.tif"
```

В папке `result`, рядом с файлом `main.py`, создастся файл `coords.csv` в требуемом формате.

## Запуск с помощью docker
Сначала необходимо собрать образ с помощью команды:
```bash
docker build -t satt_app .
```

Затем, необходимые подложки поместить в папку `layouts` и необходимые сцены поместить в папку `crops`.
После этого необходимо запустить контейнер следующим образом:

Для докера в `Windows`:
```bash
docker run --shm-size 8G --name satt -v .\:/app -t satt_app:latest --crop_name crops/crop_0_0_0000.tif --layout_name layouts/layout_2021-08-16.tif
```

Для докера в `Linux`:
```bash
docker run --shm-size 16G --name satt -v ./:/app -t satt_app:latest --crop_name crops/crop_0_0_0000.tif --layout_name layouts/layout_2021-08-16.tif
```

## Запуск процедуры поиска испорченых пикселей с помощью скрипта `pixels_detection.py`

Для запуска процедуры получения координат испорченных пикселей и исправленного изображения воспользуйтесь скриптом `pixels_detection.py`:

`python pixels_detection.py --crop_name "D:\satellite_hackathon\18. Sitronics\1_20\crop_0_0_0000.tif" [--output_tif output_0_0.tiff --output_txt output_0_0.txt]`

