import os
from osgeo import gdal
import rasterio
import pandas as pd
from tqdm import tqdm
import re
import numpy as np
import math

file_dir = r"./avng_data/L2"
save_file = r"./avng_L2_allinfo.csv"

file_list = os.listdir(file_dir)

filename_list = []
wavelength_list = []
fwhm_list = []
GSD_list = []
id_list = []

for file in tqdm(file_list):
    if file.endswith(".dat") or file.endswith(".tif") or file.endswith(".tiff"):
        try:
            with rasterio.open(os.path.join(file_dir, file)) as ds:                    
                # 获取基本信息
                id_ = file.split(".")[0]
                transform = ds.transform
                GSD = round(math.sqrt(transform.a ** 2 + transform.b ** 2), 1)

                GSD_list.append(GSD)
                id_list.append(id_)

                metadata = ds.tags(ns='ENVI')

                # 获取波长信息
                wavelength_array = None
                if 'wavelength' in metadata or "_wavelength" in metadata:
                    key_wavelength = "wavelength" if 'wavelength' in metadata else "_wavelength"
            
                    wavelength_str = metadata[key_wavelength].strip("{}")
                    wavelengths = [round(float(w.strip()), 5) for w in wavelength_str.split(',')]
                    wavelength_array = np.array(wavelengths).tolist()
                    
                # 获取FWHM信息
                fwhm = None
                if 'fwhm' in metadata or "____fwhm" in metadata:
                    key_fwhm = "fwhm" if 'fwhm' in metadata else "____fwhm"
                    fwhm_str = metadata[key_fwhm].strip("{}")
                    fwhm_values = [round(float(f.strip()), 5) for f in fwhm_str.split(',')]
                    fwhm = np.array(fwhm_values).tolist()
                
                # if len(fwhm) != nb:
                #     print(f"fwhm length is not equal to nb: {file}, len(fwhm): {len(fwhm)}, nb: {nb}") if fwhm is not None else None
                # if len(wavelength_array) != nb:
                #     print(f"wavelength length is not equal to nb: {file}, len(wavelength_array): {len(wavelength_array)}, nb: {nb}") if wavelength_array is not None else None

                # 保存获取的信息
                filename_list.append(file)
                wavelength_list.append(wavelength_array)
                fwhm_list.append(fwhm)
            
        except Exception as e:
            print(f"处理文件 {file} 时出错: {str(e)}")

# 创建数据框并保存
df = pd.DataFrame({
    "id": id_list,
    "filename": filename_list,
    "GSD": GSD_list,
    "wavelength": wavelength_list,
    "fwhm": fwhm_list,
})

# df按照id排序
df = df.sort_values(by="id")
df.to_csv(save_file, index=False)
print(f"save to {save_file}")
