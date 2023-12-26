import os
import gc
import cv2
import glob
import zipfile
import time
import threading
import pyvips
import random as rn
from tqdm import tqdm
from PIL import Image

# from threading import Thread
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from pathlib import Path
from glob import glob
from joblib import Parallel, delayed
from PIL import Image
import shutil


def process_image(row, data_dir, tile_path):
    tma = row["is_tma"]
    image_id = row["image_id"]
    img_path = data_dir / "train_images" / f"{row.image_id}.png"

    img = pyvips.Image.new_from_file(img_path)

    height = img.height
    width = img.width

    if tma:
        pass
    else:
        for y in range(0, height, tile_size):
            for x in range(0, width, tile_size):
                # Extract a tile from the image and mask
                img_tile = img.crop(
                    x, y, min(tile_size, width - x), min(tile_size, height - y)
                ).numpy()

                gray = cv2.cvtColor(img_tile, cv2.COLOR_RGB2GRAY)
                _, binary_image = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

                # black area ratio threshold start from here
                black_pixels = np.count_nonzero(binary_image == 0)
                total_pixels = np.prod(binary_image.shape[:2])
                black_area_ratio = black_pixels / total_pixels

                if black_area_ratio > 0.1:
                    # print(f"black_area_ratio: {black_area_ratio}")
                    continue

                tile_save_path = tile_path / f"{image_id}_tile_{x}_{y}.png"

                Image.fromarray(img_tile).resize(new_size).save(tile_save_path)


if __name__ == "__main__":
    tile_size = 2048
    white_thr = 240
    drop_thr = 0.6
    new_size = (512, 512)

    data_dir = Path("dataset")
    train_df = pd.read_csv(data_dir / "train.csv", dtype={"image_id": str})

    print(train_df.head())

    tile_path = data_dir / f"train_tiles_full_{tile_size}"
    shutil.rmtree(str(tile_path), ignore_errors=True)
    tile_path.mkdir(exist_ok=True)

    num_processes = 20
    Parallel(n_jobs=num_processes, backend="multiprocessing")(
        delayed(process_image)(row, data_dir, tile_path)
        for _, row in tqdm(train_df.iterrows(), total=len(train_df))
    )
