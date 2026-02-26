# Hyper-400K

## Data list
All data used to construct the Hyper-400K dataset were sourced from the official AVIRIS data portals.

- [AVIRIS-C L1.txt](./data_list/AVIRIS-C_L1.txt) 
- [AVIRIS-C L2.txt](./data_list/AVIRIS-C_L2.txt) 
- [AVIRIS-NG L1.txt](./data_list/AVIRIS-NG_L1.txt) 
- [AVIRIS-NG L2.txt](./data_list/AVIRIS-NG_L2.txt) 
- [AVIRIS-3 L1.txt](./data_list/AVIRIS-3_L1.txt) 
- [AVIRIS-3 L2.txt](./data_list/AVIRIS-3_L2.txt) 

The original data can be accessed from the following official portals:

- AVIRIS Classic: [https://popo.jpl.nasa.gov/avcl/](https://popo.jpl.nasa.gov/avcl/)
- AVIRIS-NG: [https://popo.jpl.nasa.gov/avng/](https://popo.jpl.nasa.gov/avng/)
- AVIRIS-3: [https://popo.jpl.nasa.gov/av3/](https://popo.jpl.nasa.gov/av3/)
- NASA Earthdata Portal: [https://search.earthdata.nasa.gov](https://search.earthdata.nasa.gov/search?q=AV3_L1B&ac=true)

Please note that the data collection used in this project is current as of January, 2025. The AVIRIS team continues to acquire new airborne hyperspectral data, and additional datasets may become available after this date.


## Data prepare
- `split_image.py`: Split the original hyperspectral images into smaller patches.
- `read_base_info_GDAL.py`: Read base information from GDAL-supported hyperspectral images.
- `read_base_info_NC.py`: Read base information from NC format hyperspectral images.


## Example data
An example subset is available [here](https://r2.3sobs.top/share/SpecAware/example_data/example_data.zip), showcasing cropped patches from multiple HSI sensors.


## Acknowledgement
We sincerely thank the AVIRIS team for their sustained efforts in collecting, curating, and sharing airborne hyperspectral imagery.
