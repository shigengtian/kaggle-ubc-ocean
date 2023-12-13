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


def process_image(index, row, image_dir, mask_target_path, img_target_path, tile_size):
    image_id = row.image_id
    image_path = image_dir / (image_id + ".png")
    mask_path = row.mask_file_paths
    is_tma = row.is_tma

    img = cv2.imread(str(image_path))

    height, width = img.shape[:2]
    
    
    mask = np.load(str(mask_path))
    mask = cv2.resize(mask, (width, height),interpolation=cv2.INTER_NEAREST)
    
    rows = height // tile_size
    cols = width // tile_size

    if is_tma:
        print("is tma")
    else:
        for i in range(rows):
            for j in range(cols):

                tile_mask = mask[
                    i * tile_size : (i + 1) * tile_size,
                    j * tile_size : (j + 1) * tile_size,
                ]
                true_percentage = tile_mask.sum() / tile_mask.size
                if true_percentage < 0.5:
                    continue
                
                mask_file_name = f"{image_id}_{i}_{j}.npy" 
                mask_file_path = mask_target_path / mask_file_name
                
                np.save(str(mask_file_path), tile_mask)
                               
                tile = img[
                    i * tile_size: (i + 1) * tile_size,
                    j * tile_size: (j + 1) * tile_size,
                ]

                
                tile_filename = f"{image_id}_{i}_{j}.png"
                tile_path = img_target_path / tile_filename
                cv2.imwrite(str(tile_path), tile)

                del tile
                

if __name__ == "__main__":

    data_dir = Path("dataset")
    train_df = pd.read_csv(data_dir / "train.csv", dtype={"image_id": str})
    test_df = pd.read_csv(data_dir / "test.csv")
    img_target_path = data_dir / "tile_2048"
    img_target_path.mkdir(parents=True, exist_ok=True)
    mask_target_path = data_dir / "mask_2048"
    mask_target_path.mkdir(parents=True, exist_ok=True)
    
    mask_dir = data_dir / "masks_np"
    mask_npy_df = get_mask_npy_df(mask_dir)
    print(mask_npy_df.head())

    train_df= mask_npy_df.merge(train_df, on="image_id", how="left")
    print(train_df.head())

    tile_size = 2048
    # Define the number of parallel processes
    num_processes = 6  # Adjust as needed
    image_dir = data_dir / "train_images"
    
    # Use joblib to parallelize image processing
    Parallel(n_jobs=num_processes)(
        delayed(process_image)(index, row, image_dir, mask_target_path, img_target_path, tile_size)
        for index, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Tile images")
    )
    
    # train_df = pd.read_csv(data_dir / "train.csv", dtype={"image_id": str})
    # test_df = pd.read_csv(data_dir / "test.csv")
    # target_path = data_dir / "tile_2048"
    # target_path.mkdir(parents=True, exist_ok=True)

    # mask_dir = data_dir / "masks_np"
    # mask_npy_df = get_mask_npy_df(mask_dir)
    # print(mask_npy_df.head())

    # train_df= mask_npy_df.merge(train_df, on="image_id", how="left")
    # print(train_df.head())

    # tile_size = 2048
    # # Define the number of parallel processes
    # num_processes = 6  # Adjust as needed
    # image_dir = data_dir / "train_images"
    
    # # Use joblib to parallelize image processing
    # Parallel(n_jobs=num_processes)(
    #     delayed(process_image)(index, row, image_dir, target_path, tile_size)
    #     for index, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Tile images")
    # )
    
    
 