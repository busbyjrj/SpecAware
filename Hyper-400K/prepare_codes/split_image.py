import os
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm
import rasterio
import numpy as np


def read_image(fileName):
    with rasterio.open(fileName) as dataset:
        image_MATRIX = dataset.read()  # (C,H,W)
        image_MATRIX = image_MATRIX.transpose(1, 2, 0)  # --> cols, rows, bands
    return image_MATRIX


def split_Image(image_filepath, save_imagepath, patch_size):
    print(f"cut {image_filepath} to {save_imagepath}")
    save_name = os.path.splitext(os.path.basename(image_filepath))[0]
    os.makedirs(save_imagepath, exist_ok=True)

    image_MATRIX = read_image(image_filepath)
    rows, cols = image_MATRIX.shape[0], image_MATRIX.shape[1]

    c_rows = patch_size
    c_cols = patch_size

    number_rows = int(np.ceil(rows / c_rows))
    number_cols = int(np.ceil(cols / c_cols))

    for i in range(number_rows):
        for j in range(number_cols):
            start_row = i * c_rows
            end_row = start_row + c_rows
            start_col = j * c_cols
            end_col = start_col + c_cols
            c_MATRIX = image_MATRIX[start_row:end_row, start_col:end_col, :]

            nb_selected = c_MATRIX.shape[2]

            ignore_value = -9999
            save_MATRIX = np.ones((c_rows, c_cols, nb_selected)) * ignore_value
            save_MATRIX[:c_MATRIX.shape[0], :c_MATRIX.shape[1], :] = c_MATRIX

            # 数据检查，否有任一通道值超过max_data
            # AV3_L1:100, AV3_L2:2.5, AVC_L1:30, AVC_L2:2.5, AVNG_L1:100, AVNG_L2:2.5
            max_data = 100

            # 对于L2数据，还需去掉水汽波段
            # AVNG_L2
            # all_bands = np.arange(0, 425)
            # # 屏蔽1，195-211，281-319, 421-425
            # bad_bands = np.concatenate((np.array([0]), np.arange(194, 211), np.arange(280, 319), np.arange(420, 425)))
            # save_MATRIX[:, :, bad_bands] = 0

            # AVC_L2
            # all_bands = np.arange(0, 224)
            # #  屏蔽104-114波段  屏蔽153-168波段
            # bad_bands = np.concatenate([np.arange(103, 114), np.arange(152, 168)])
            # save_MATRIX[:, :, bad_bands] = 0

            # AV3_L2
            # all_bands = np.arange(0, 284)
            # # 屏蔽130-141波段、191-213波段
            # bad_bands = np.concatenate([np.arange(129, 141), np.arange(190, 213)])
            # save_MATRIX[:, :, bad_bands] = 0

            mask = np.any(save_MATRIX > max_data, axis=2)  # (H,W)

            # 统计无效值比例（<0 或 nan/inf 或 >max_data）
            invalid_mask = (save_MATRIX < 0) | ~np.isfinite(save_MATRIX) | mask[:, :, np.newaxis]
            zero_ratio = np.sum(invalid_mask) / save_MATRIX.size

            if zero_ratio > 0.2:
                print(f"跳过块 {save_name}_r{start_row}_c{start_col}：无效值比例过高 ({zero_ratio:.2%})")
                continue

            # 处理小于0和nan、inf、-inf等值
            save_MATRIX[mask] = 0
            save_MATRIX = np.nan_to_num(save_MATRIX,
                                            nan=0,
                                            posinf=0,
                                            neginf=0)
            save_MATRIX[save_MATRIX < 0] = 0

            # 保存为CHW格式
            save_MATRIX = save_MATRIX.transpose(2, 0, 1).astype(np.float16)
            saveName = os.path.join(save_imagepath, save_name + '_r' + str(start_row) + '_c' + str(start_col) + '.npz')
            np.savez_compressed(saveName, save_MATRIX)
                

def split_Image_with_pbar(args):
    image_filepath, save_imagepath, patch_size, pbar = args
    try:
        split_Image(image_filepath, save_imagepath, patch_size)
    except Exception as e:
        print(f"处理文件 {image_filepath} 时出错: {str(e)}")
    finally:
        pbar.update(1)

if __name__ == "__main__":
    image_root = r"./src_dir"
    save_root = r"./dst_dir"

    os.makedirs(save_root) if not os.path.exists(save_root) else None
        
    image_filepaths = []
    save_imagepaths = []

    image_list = os.listdir(image_root)
    for image in image_list:
        image_path = os.path.join(image_root, image)
        save_path = os.path.join(save_root, image.split('.')[0])

        if not (image.endswith(".tiff") or image.endswith(".tif") or image.endswith(".img")
                or image.endswith(".dat") or image.endswith("_img") or image.endswith("_ORT")):
            continue
        print(image_path, save_path)
        if os.path.exists(save_path):
            continue
        image_filepaths.append(image_path)
        save_imagepaths.append(save_path)

        # split_Image(image_path, save_path, patch_size=256)


    # 限制最大线程数
    max_threads = 20  # 对于IO密集型任务，可以使用更多的线程

    # 使用多线程处理
    if len(image_filepaths) > 0:
        # 创建进度条
        with tqdm(total=len(image_filepaths), desc="Cutting images") as pbar:
            # 准备参数列表
            args_list = [(image_filepaths[i], save_imagepaths[i], 256, pbar) for i in range(len(image_filepaths))]

            # 创建线程池
            with ThreadPoolExecutor(max_workers=max_threads) as executor:
                # 提交任务并等待完成
                list(executor.map(split_Image_with_pbar, args_list))

            print(f"All {len(image_filepaths)} Done!")
