# for nc files

import os
from osgeo import gdal
import netCDF4 as nc
import pandas as pd
from tqdm import tqdm


nc_dir = r"./av3_origin_data/"
save_file = r"./av3_L2_allinfo.csv"

nc_list = os.listdir(nc_dir)

filename_list = []
wavelength_list = []
fwhm_list = []
for nc_file in tqdm(nc_list):
    if nc_file.endswith(".nc"):
        try:
            with nc.Dataset(os.path.join(nc_dir, nc_file)) as ds:
                # wavelengths = ds["radiance"]["wavelength"][:]
                # fwhm = ds["radiance"]["fwhm"][:]
                wavelengths = ds["reflectance"]["wavelength"][:]
                fwhm = ds["reflectance"]["fwhm"][:]
                wavelengths = list(wavelengths)
                fwhm = list(fwhm)
                filename_list.append(nc_file)
                wavelength_list.append(wavelengths)
                fwhm_list.append(fwhm)
        except Exception as e:
            print(f"处理文件 {nc_file} 时出错: {str(e)}")

df = pd.DataFrame({"filename": filename_list, "wavelength": wavelength_list, "fwhm": fwhm_list})
df.to_csv(save_file, index=False)
