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


def convert_rgb_to_labels(img_path: str, folder: str):
    name = os.path.basename(img_path)
    img = np.array(Image.open(img_path))
    # plt.imshow(img)
    bg = np.ones((img.shape[0], img.shape[1], 1)) * 128
    stack = np.concatenate((bg, img), axis=2)
    mask = np.argmax(stack, axis=2).astype(np.uint8)
    img_path = os.path.join(folder, name)
    Image.fromarray(mask).save(img_path)
    return img_path


if __name__ == "__main__":
    data_dir = Path("dataset")
    train_images = sorted(glob(str(data_dir / "train_images/*.png")))
    train_df = pd.read_csv(data_dir / "train.csv", dtype={"image_id": str})

    train_masks = sorted(glob(str(data_dir / "masks/*.png")))
    train_mask_df = pd.DataFrame({"mask_path": train_masks})
    train_mask_df["image_id"] = train_mask_df["mask_path"].apply(
        lambda x: x.split("/")[-1].split(".")[0]
    )

    train_df = train_mask_df.merge(train_df, on="image_id", how="left")

    #    target_path = data_dir / "train_thumbnails_3500"
    #    target_path.mkdir(exist_ok=True)
    #
    #    num_processes = 12
    #    Parallel(n_jobs=num_processes, backend="multiprocessing")(
    #        delayed(process_image)(row, data_dir, target_path)
    #        for _, row in tqdm(train_df.iterrows(), total=len(train_df))
    #    )

    tile_size = 2048
    white_thr = 240
    drop_thr = 0.6
    tile_2048_path = data_dir / "train_tiles_2048"
    tile_2048_path.mkdir(exist_ok=True)

    tile_2048_mask_path = data_dir / "train_tiles_2048_mask"
    tile_2048_mask_path.mkdir(exist_ok=True)

    for index, row in tqdm(train_df.iterrows(), total=len(train_df)):
        tma = row["is_tma"]
        image_id = row["image_id"]
        img_path = data_dir / "train_images" / f"{row.image_id}.png"
        mask_path = row["mask_path"]

        img = pyvips.Image.new_from_file(img_path)
        # mask = pyvips.Image.new_from_file(mask_path)

        height = img.height
        width = img.width

        if ~tma:
            # Iterate over the image in tiles
            for y in range(0, height, tile_size):
                for x in range(0, width, tile_size):
                    # Extract a tile from the image and mask
                    img_tile = img.crop(
                        x, y, min(tile_size, width - x), min(tile_size, height - y)
                    ).numpy()

                    # mask_tile = mask.crop(
                    #     x, y, min(tile_size, width - x), min(tile_size, height - y)
                    # ).numpy()

                    # img_tile_numpy = img_tile.numpy()

                    black_bg = np.sum(img_tile, axis=2) == 0
                    img_tile[black_bg, :] = 255
                    mask_bg = np.mean(img_tile, axis=2) > white_thr
                    if np.sum(mask_bg) >= (np.prod(mask_bg.shape) * drop_thr):
                        # print(f"skip almost empty tile: {k:06}_{int(x_ / w)}-{int(y_ / h)}")
                        continue

                    new_size = (512, 512)
                    tile_save_path = tile_2048_path / f"{image_id}_tile_{x}_{y}.png"
                    Image.fromarray(img_tile).resize(new_size, Image.LANCZOS).save(
                        tile_save_path
                    )

                    # img_tile_numpy = img_tile.numpy()
                    # if np.max(img_tile_numpy) == 0:
                    #     print("black tile")
                    #     continue

                    # use first channel check black area
                    # img_first_channel_mask = img_tile_numpy[:, :, 0] > 0
                    # not_black_precentage = np.sum(img_first_channel_mask) / (
                    #     img_tile_numpy.shape[0] * img_tile_numpy.shape[1]
                    # )

                    # if not_black_precentage > 0.8:
                    #     print("black area > 0.8")
                    #     continue

                    # white_bg = np.mean(img_tile_numpy, axis=2) > white_thr
                    # if np.sum(white_bg) >= (np.prod(white_bg.shape) * white_drop_thr):
                    #     print(f"skip almost empty tile")
                    #     continue

                    # true_percentage = tile_mask.sum() / tile_mask.size
                    # if true_percentage < 0.8:
                    #     continue
                    # Process the tiles as needed
                    # For example, you can save the tiles to a new file
                    # tile_save_path = tile_2048_path / f"{image_id}_tile_{x}_{y}.png"
                    # img_tile.write_to_file(tile_save_path)

                    # tile_mask_save_path = (
                    #     tile_2048_mask_path / f"{image_id}_tile_{x}_{y}.png"
                    # )

                    # mask_tile.write_to_file(tile_mask_save_path)
            # pass
            # img_path = data_dir / "train_images" / f"{row.image_id}.png"
            # img = vips_read_image(str(img_path), longest_edge=5000)
            # cv2.imwrite(str(target_path / f"{row.image_id}.png"), img)
        else:
            pass
            # img_path = data_dir / "train_images" / f"{row.image_id}.png"
            # img = cv2.imread(str(img_path))
            # cv2.imwrite(str(target_path / f"{row.image_id}.png"), img)

        # break
