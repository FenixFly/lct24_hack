import os
from PIL import Image
import numpy as np
import rasterio as rs
from matplotlib import pyplot as plt
import seaborn_image as isns
import cv2
from rasterio.plot import reshape_as_image, reshape_as_raster
import argparse
import pandas as pd
import datetime
from pathlib import Path


def cli_argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--crop_name',
                        type=str,
                        dest='crop_name',
                        help='Full path crop .tif file',
                        required=True)
    parser.add_argument('--layout_name',
                        type=str,
                        dest='layout_name',
                        help='Full path layout .tif file',
                        required=True)
    parser.add_argument('--output_csv',
                        type=str,
                        help='Path to output file .csv file',
                        default='result/coords.csv',
                        required=False)
    args = parser.parse_args()

    return args


def calculate_satellite_matching(img1, img2):

    alg = cv2.SIFT_create()
    thresholds = [0.7, 0.8, 0.9, 0.95, 0.99]

    kp1, des1 = alg.detectAndCompute(img1, None)
    kp2, des2 = alg.detectAndCompute(img2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)

    for threshold in thresholds:
        good = []
        for m,n in matches:
            if m.distance < threshold * n.distance:
                good.append([m])
        
        if len(good) >=4:
            break
   
    return kp1, des1, kp2, des2, good


def calc_output_coordinates(perspective_transform, raster, scale_factor):
    tmp = perspective_transform.squeeze()[:,::-1]/scale_factor
    res = []
    res.append(raster.xy(tmp[0][0],tmp[0][1]))
    res.append(raster.xy(tmp[3][0],tmp[3][1]))
    res.append(raster.xy(tmp[2][0],tmp[2][1]))
    res.append(raster.xy(tmp[1][0],tmp[1][1]))
    return res


def resizing(img, factor=1/4):
    height, width, channels = img.shape
    height2, width2 = int(height * factor), int(width * factor)
    return cv2.resize(img, (height2, width2), interpolation = cv2.INTER_AREA)


def get_crop_image(crop):
    img = reshape_as_image(crop[:3])
    scaled_img = cv2.convertScaleAbs(img, alpha=(255.0/np.max(img)))
    img_norm = cv2.normalize(scaled_img, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    img_u = img_norm.astype(np.uint8)
    img_u[:,:,0] = cv2.equalizeHist(img_u[:,:,0])
    img_u[:,:,1] = cv2.equalizeHist(img_u[:,:,1])
    img_u[:,:,2] = cv2.equalizeHist(img_u[:,:,2])
    return img_u


def main(layout_name, crop_name, output_csv, debug=True):

    scale_factor = 1/4

    # Read  images
    raster = rs.open(layout_name)
    full_raster = raster.read()
    raster1 = rs.open(crop_name)
    patch_raster = raster1.read()

    start_time = datetime.datetime.now()

    # Подготавливаем патч
    crop = patch_raster
    crop_img = get_crop_image(crop)
    crop_img_gray = cv2.cvtColor(crop_img, cv2.COLOR_RGB2GRAY)

    # Подготавливаем подложку
    full_img = reshape_as_image(full_raster)
    full_img_resize = resizing(full_img, scale_factor)
    full_img_resize_u8 = get_crop_image(reshape_as_raster(full_img_resize))
    full_img_resize_gray = cv2.cvtColor(full_img_resize_u8, cv2.COLOR_RGB2GRAY)

    # Находим матчинг между патчем и подложкой
    kp1, des1, kp2, des2, good_matches = calculate_satellite_matching(crop_img_gray, full_img_resize_gray)
    good_sort = sorted(good_matches, key=lambda x: x[0].distance)

    # Находим ограничивающую рамку патча на подложке
    src_pts = np.float32([ kp1[m[0].queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m[0].trainIdx].pt for m in good_matches ]).reshape(-1,1,2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()
    h,w = crop_img_gray.shape
    pts = np.float32([ [0,0],[0,h],[w,h],[w,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)

    if debug==True:
        print(M)


    if M[0][0] < 0 or M[1][1] < 0:
        # Если гомография не получилась
        
        # TODO: сделать повторный запуск матчинга на небольшом куске, близком к центру матчинга
        # сейчас запускается просто на более маленьком маштабе
        scale_factor = scale_factor / 2
        full_img = reshape_as_image(full_raster)
        full_img_resize = resizing(full_img, scale_factor)
        full_img_resize_u8 = get_crop_image(reshape_as_raster(full_img_resize))
        full_img_resize_gray = cv2.cvtColor(full_img_resize_u8, cv2.COLOR_RGB2GRAY)
        kp1, des1, kp2, des2, good_matches = calculate_satellite_matching(crop_img_gray, full_img_resize_gray)
        good_sort = sorted(good_matches, key=lambda x: x[0].distance)
        src_pts = np.float32([ kp1[m[0].queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m[0].trainIdx].pt for m in good_matches ]).reshape(-1,1,2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()
        h,w = crop_img_gray.shape
        pts = np.float32([ [0,0],[0,h],[w,h],[w,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)

    end_time = datetime.datetime.now()

    if debug==True:
        # Отрисовываем матчинг между патчем и подложкой
        full_img_resize_gray = cv2.polylines(full_img_resize_gray,[np.int32(dst)],True,255,3, cv2.LINE_AA)
        img3 = cv2.drawMatchesKnn(crop_img_gray, kp1, 
                              full_img_resize_gray, kp2, 
                              good_matches, 
                              None, 
                             )
        isns.imgplot(img3).set_title(crop_name)
        plt.gcf().set_size_inches(20, 20)

    # Подсчитываем координаты
    output_coordinates = calc_output_coordinates(dst, raster, scale_factor)

    # Добавляем координаты в csv
    res = {
        'layout_name':layout_name,
        'crop_name':crop_name,
        'ul':f'{output_coordinates[0][0]:.3f}_{output_coordinates[0][1]:.3f}',
        'ur':f'{output_coordinates[1][0]:.3f}_{output_coordinates[1][1]:.3f}',
        'br':f'{output_coordinates[2][0]:.3f}_{output_coordinates[2][1]:.3f}',
        'bl':f'{output_coordinates[3][0]:.3f}_{output_coordinates[3][1]:.3f}',
        'crs':f'EPSG:32637',
        'start': f"{start_time.strftime('%Y-%m-%dT%H:%M:%S')}",
        'end': end_time.strftime('%Y-%m-%dT%H:%M:%S')
    }
    
    df = pd.DataFrame(data = res, index=[0])
    df.to_csv(output_csv, index=False, header=True)

if __name__=='__main__':
    args = cli_argument_parser()
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    main(args.layout_name, args.crop_name, args.output_csv, debug=True)

