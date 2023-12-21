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


# def vips_read_image(image_path, longest_edge):
#     """
#     Read image using libvips

#     Parameters
#     ----------
#     image_path: str
#         Path of the image

#     Returns
#     -------
#     image: numpy.ndarray of shape (height, width, 3)
#         Image array
#     """

#     image_thumbnail = pyvips.Image.thumbnail(image_path, longest_edge)

#     return np.ndarray(
#         buffer=image_thumbnail.write_to_memory(),
#         dtype=np.uint8,
#         shape=[image_thumbnail.height, image_thumbnail.width, image_thumbnail.bands],
#     )


# def process_image(row, data_dir, target_path):
#     tma = row["is_tma"]
#     if not tma:
#         img_path = data_dir / "train_images" / f"{row.image_id}.png"
#         img = vips_read_image(str(img_path), longest_edge=5000)
#         cv2.imwrite(str(target_path / f"{row.image_id}.png"), img)
#     else:
#         img_path = data_dir / "train_images" / f"{row.image_id}.png"
#         img = cv2.imread(str(img_path))
#         cv2.imwrite(str(target_path / f"{row.image_id}.png"), img)


# def convert_rgb_to_labels(img_path: str, folder: str):
#     name = os.path.basename(img_path)
#     img = np.array(Image.open(img_path))
#     # plt.imshow(img)
#     bg = np.ones((img.shape[0], img.shape[1], 1)) * 128
#     stack = np.concatenate((bg, img), axis=2)
#     mask = np.argmax(stack, axis=2).astype(np.uint8)
#     img_path = os.path.join(folder, name)
#     Image.fromarray(mask).save(img_path)
#     return img_path


def process_image(row, data_dir, tile_path):
    tma = row["is_tma"]
    image_id = row["image_id"]
    img_path = data_dir / "train_images" / f"{row.image_id}.png"
    # mask_path = row["mask_path"]

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

                gray = cv2.cvtColor(img_tile, cv2.COLOR_BGR2GRAY)
                _, binary_image = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

                # black area ratio threshold start from here
                black_pixels = np.count_nonzero(binary_image == 0)
                total_pixels = np.prod(binary_image.shape[:2])
                black_area_ratio = black_pixels / total_pixels

                # black_pixels = total_pixels - np.count_nonzero(binary_image)

                # black_area_ratio = black_pixels / total_pixels

                if black_area_ratio > 0.3:
                    # print(f"black_area_ratio: {black_area_ratio}")
                    continue
                # black area ratio threshold end here

                # white area ratio threshold start from here
                # white_pixels = np.count_nonzero(binary_image == 255)
                # white_area_ratio = white_pixels / total_pixels

                # if white_area_ratio > 0.7:
                #     print(f"white_area_ratio: {white_area_ratio}")
                #     continue
                # white area ratio threshold end here

                # mask_tile = mask.crop(
                #     x, y, min(tile_size, width - x), min(tile_size, height - y)
                # ).numpy()


                tile_save_path = tile_path / f"{image_id}_tile_{x}_{y}.png"
                Image.fromarray(img_tile).resize(new_size, Image.LANCZOS).save(
                    tile_save_path
                )

if __name__ == "__main__":
    tile_size = 2048
    white_thr = 240
    drop_thr = 0.6
    new_size = (512, 512)
    
    data_dir = Path("dataset")
    train_df = pd.read_csv(data_dir / "train.csv", dtype={"image_id": str})

    print(train_df.head())
    
    tile_path = data_dir / "train_tiles"
    shutil.rmtree(str(tile_path))
    tile_path.mkdir(exist_ok=True)
    
    num_processes = 20
    Parallel(n_jobs=num_processes, backend="multiprocessing")(
        delayed(process_image)(row, data_dir, tile_path)
        for _, row in tqdm(train_df.iterrows(), total=len(train_df))
    )


    # for index, row in train_df.iterrows():
    #     print(row)
    #     tma = row["is_tma"]
    #     tma = row["is_tma"]
    #     image_id = row["image_id"]
    #     img_path = data_dir / "train_images" / f"{row.image_id}.png"
    #     if tma:
    #         pass
    #     else:
    #         print(row["image_id"])
    #     break
    
    
    # train_masks = sorted(glob(str(data_dir / "masks/*.png")))
    # train_mask_df = pd.DataFrame({"mask_path": train_masks})
    # train_mask_df["image_id"] = train_mask_df["mask_path"].apply(
    #     lambda x: x.split("/")[-1].split(".")[0]
    # )

    # train_df = train_mask_df.merge(train_df, on="image_id", how="left")



    # tile_2048_path = data_dir / "train_tiles_2048"
    # shutil.rmtree(str(tile_2048_path))
    # tile_2048_path.mkdir(exist_ok=True)

    # tile_2048_mask_path = data_dir / "train_tiles_2048_mask"
    # shutil.rmtree(str(tile_2048_mask_path))
    # tile_2048_mask_path.mkdir(exist_ok=True)

