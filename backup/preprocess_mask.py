import os

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__()

import cv2
import gc
from glob import glob
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed
from utils import get_mask_npy_df
import time
import multiprocessing


# def process_image(index, row, image_dir, mask_target_path, img_target_path, tile_size):
#     image_id = row.image_id
#     image_path = image_dir / (image_id + ".png")
#     mask_path = row.mask_file_paths
#     is_tma = row.is_tma

#     img = cv2.imread(str(image_path))

#     height, width = img.shape[:2]


#     mask = np.load(str(mask_path))
#     mask = cv2.resize(mask, (width, height),interpolation=cv2.INTER_NEAREST)

#     rows = height // tile_size
#     cols = width // tile_size

#     if is_tma:
#         print("is tma")
#     else:
#         for i in range(rows):
#             for j in range(cols):

#                 tile_mask = mask[
#                     i * tile_size : (i + 1) * tile_size,
#                     j * tile_size : (j + 1) * tile_size,
#                 ]
#                 true_percentage = tile_mask.sum() / tile_mask.size
#                 if true_percentage < 0.5:
#                     continue

#                 mask_file_name = f"{image_id}_{i}_{j}.npy"
#                 mask_file_path = mask_target_path / mask_file_name

#                 np.save(str(mask_file_path), tile_mask)

#                 tile = img[
#                     i * tile_size: (i + 1) * tile_size,
#                     j * tile_size: (j + 1) * tile_size,
#                 ]


#                 tile_filename = f"{image_id}_{i}_{j}.png"
#                 tile_path = img_target_path / tile_filename
#                 cv2.imwrite(str(tile_path), tile)

#                 del tile


import numpy as np
import pyvips


def process_file(seg_file, target_path):
    """
    Processes a single segmentation file, loads the mask, saves it to a numpy file.

    Args:
      seg_file: Path to the segmentation file.
      target_path: Path to the directory where the numpy files should be saved.

    Returns:
      None
    """
    mask = pyvips.Image.new_from_file(seg_file, access="sequential").numpy()
    mask_target_file_name = (
        str(target_path) + "/" + seg_file.split("/")[-1].split(".")[0] + ".npy"
    )
    np.save(mask_target_file_name, mask)


def mask2label(mask: np.ndarray):
    """
    modify the color mask to label mask in place.

    Args:
        mask: (H, W), np.uint8

    Returns:
        mask: (H, W), np.uint8, value: {'background': 0, 'tumor': 1, 'stroma': 2, 'necrosis': 3}
    """
    bg = mask.sum(axis=-1) == 0
    mask = mask.argmax(axis=-1).astype(np.uint8) + 1
    mask[bg] = 0

    return mask.astype(np.uint8)


if __name__ == "__main__":
    data_dir = Path("dataset")
    train_df = pd.read_csv(data_dir / "train.csv", dtype={"image_id": str})

    segmentation_files = glob(str(data_dir / "masks/*.png"))
    target_path = data_dir / "masks_np"
    target_path.mkdir(exist_ok=True)
    
    start_time = time.time()
    
    cpu_count = 8
    
    # Optionally limit the number of parallel processes to use (default is all CPUs)
    n_jobs = min(cpu_count, len(segmentation_files))
    
    results = Parallel(n_jobs=n_jobs)(delayed(process_file)(seg_file, target_path) for seg_file in tqdm(segmentation_files))
    
    # for index, seg_file in tqdm(enumerate(segmentation_files)):
    #     print(seg_file)
    #     mask = pyvips.Image.new_from_file(seg_file, access="sequential").numpy()
    #     mask_target_file_name = (
    #         target_path + "/" + seg_file.split("/")[-1].split(".")[0] + ".npy"
    #     )
    #     print(mask_target_file_name)
    #     np.save(mask_target_file_name, mask)
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Execution time: {total_time:.3f} seconds")
