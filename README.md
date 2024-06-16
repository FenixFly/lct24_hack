# lct24_hack

Для запуска процедуры получения координат воспользуйтесь скриптом `main.py`:
`python main.py --crop_name "D:\satellite_hackathon\18. Sitronics\1_20\crop_0_0_0000.tif" --layout_name "D:\satellite_hackathon\18. Sitronics\layouts\layout_2021-08-16.tif" [--output_csv coord.csv]`

В папке рядом с файлом `main.py` создастся файл `coords.csv` в требуемом формате.

Для запуска процедуры получения координат испорченных пикселей и исправленного изображения воспользуйтесь скриптом `pixels_detection.py`:

`python pixels_detection.py --crop_name "D:\satellite_hackathon\18. Sitronics\1_20\crop_0_0_0000.tif" [--output_tif output_0_0.tiff --output_txt output_0_0.txt]`
