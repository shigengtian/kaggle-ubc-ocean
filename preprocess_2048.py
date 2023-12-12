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


def process_image(index, row, image_dir, target_path, tile_size):
    image_id = row.image_id
    image_path = image_dir / (image_id + ".png")
    is_tma = row.is_tma

    img = cv2.imread(str(image_path))

    height, width = img.shape[:2]
    rows = height // tile_size
    cols = width // tile_size

    if is_tma:
        print("is tma")
    else:
        for i in range(rows):
            for j in range(cols):
                tile = img[
                    i * tile_size: (i + 1) * tile_size,
                    j * tile_size: (j + 1) * tile_size,
                ]

                tile_filename = f"{image_id}_{i}_{j}.png"
                tile_path = target_path / tile_filename
                cv2.imwrite(str(tile_path), tile)

                del tile
                

if __name__ == "__main__":
    data_dir = Path("dataset")
    train_df = pd.read_csv(data_dir / "train.csv", dtype={"image_id": str})
    test_df = pd.read_csv(data_dir / "test.csv")
    target_path = data_dir / "tile_2048"
    target_path.mkdir(parents=True, exist_ok=True)

    tile_size = 2048
        # Define the number of parallel processes
    num_processes = 4  # Adjust as needed
    image_dir = data_dir / "train_images"
    
    # Use joblib to parallelize image processing
    Parallel(n_jobs=num_processes)(
        delayed(process_image)(index, row, image_dir, target_path, tile_size)
        for index, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Tile images")
    )
    
    
    # print(train_df.head())
    # print(train_df.columns)

    #

    # for index, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Tile images"):
    #     image_id = row.image_id
    #     image_path = image_dir / (image_id + ".png")
    #     is_tma = row.is_tma

    #     img = cv2.imread(str(image_path))

    #     height, width = img.shape[:2]
    #     rows = height // tile_size
    #     cols = width // tile_size


    #     if is_tma:
    #         print("is tma")
    #     else:
    #         for i in range(rows):
    #             for j in range(cols):

    #                 tile = img[
    #                        i * tile_size: (i + 1) * tile_size,
    #                        j * tile_size: (j + 1) * tile_size,
    #                        ]

    #                 tile_filename = f"{image_id}_{i}_{j}.png"
    #                 tile_path = f"{str(target_path)}/{tile_filename}"
    #                 cv2.imwrite(tile_path, tile)

    #                 # _img_ids.append(image_id)
    #                 # _tile_file_paths.append(tile_path)
    #                 del tile
    #                 gc.collect()

