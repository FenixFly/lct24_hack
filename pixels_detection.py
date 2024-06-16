import numpy as np
import rasterio as rs
import cv2
import pandas as pd
import argparse

def cli_argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--crop_name',
                        type=str,
                        dest='crop_name',
                        help='Full path crop .tif file',
                        required=True)
    parser.add_argument('--output_tif',
                        type=str,
                        help='Path to output .tif file with fixed pixels',
                        default='output.tif',
                        required=False)
    parser.add_argument('--output_txt',
                        type=str,
                        help='Path to output .txt file with error pixels',
                        default='output.txt',
                        required=False)
    args = parser.parse_args()

    return args


def search_error_pixels_in_slice(slice, slice_id, k_plus = 1.5, k_minus = 1.2):
    avg = np.median(slice)
    blur = cv2.medianBlur(slice, 3)
    h,w = slice.shape
    
    error_pixels_formatted = []
    
    for iy in range(h):
        for ix in range(w):
            
            if slice[iy,ix] > blur[iy,ix] + avg * k_plus:
                print('upper:', iy,ix, slice[iy,ix], blur[iy,ix])
                error_pixels_formatted.append( (iy, ix, slice_id, slice[iy,ix], blur[iy,ix]) )
                slice[iy,ix] = blur[iy,ix]
                
            
            if slice[iy,ix] < blur[iy,ix] - avg * k_minus:
                
                print('lower:',iy,ix, slice[iy,ix], blur[iy,ix])    
                error_pixels_formatted.append( (iy, ix, slice_id, slice[iy,ix], blur[iy,ix]) )
                slice[iy,ix] = blur[iy,ix]
    
    return slice, error_pixels_formatted


def save_slices_list_to_tiff(slices_list, raster_meta, filename):
    np_array = np.array(slices_list)
    
    with rs.open(fp=filename, mode='w',**raster_meta) as dst:
        dst.write(np_array)


def save_error_pixels_information(error_pixels_formatted, filename):
    df = pd.DataFrame(error_pixels_formatted, columns=[0,1,2,3,4])
    print(df)
    df.to_csv(filename, sep=';', header=False, index=False)


def search_error_pixels_in_image(cutted_patch_raster):
    
    slices = []
    # Обработать каждый слой
    slice0, pixels0 = search_error_pixels_in_slice(cutted_patch_raster[0], 0, 1.5, 1.1)
    slice1, pixels1 = search_error_pixels_in_slice(cutted_patch_raster[1], 1, 1.5, 1.0)
    slice2, pixels2 = search_error_pixels_in_slice(cutted_patch_raster[2], 2, 1.5, 1.4)
    slice3, pixels3 = search_error_pixels_in_slice(cutted_patch_raster[3], 3, 1.7, 0.8)

    slices_list = [slice0, slice1, slice2, slice3]
    error_pixels_formatted_list = []+pixels0+pixels1+pixels2+pixels3
    
    return slices_list, error_pixels_formatted_list



def main(crop_name, output_tif, output_txt):
    
    # Читаем tiff 
    raster = rs.open(crop_name)
    patch_raster = raster.read()
    raster_meta = raster.meta.copy()

    # Запускаем поиск неправильных пикселей
    slices_list, error_pixels_formatted_list = search_error_pixels_in_image(patch_raster)

    # Сохранить в файлы
    save_slices_list_to_tiff(slices_list, raster_meta, output_tif)
    save_error_pixels_information(error_pixels_formatted_list, output_txt)


if __name__=='__main__':
    args = cli_argument_parser()
    main(args.crop_name, args.output_tif, args.output_txt)
