import os
import gc
import cv2
import glob
import zipfile
import time
import threading

os.environ["VIPS_CONCURRENCY"] = "4"
os.environ["VIPS_DISC_THRESHOLD"] = "15gb"

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


def vips_read_image(image_path, longest_edge):
    """
    Read image using libvips

    Parameters
    ----------
    image_path: str
        Path of the image

    Returns
    -------
    image: numpy.ndarray of shape (height, width, 3)
        Image array
    """

    image_thumbnail = pyvips.Image.thumbnail(image_path, longest_edge)

    return np.ndarray(
        buffer=image_thumbnail.write_to_memory(),
        dtype=np.uint8,
        shape=[image_thumbnail.height, image_thumbnail.width, image_thumbnail.bands],
    )


def process_image(row, data_dir, target_path):
    tma = row["is_tma"]
    if not tma:
        img_path = data_dir / "train_images" / f"{row.image_id}.png"
        img = vips_read_image(str(img_path), longest_edge=5000)
        cv2.imwrite(str(target_path / f"{row.image_id}.png"), img)
    else:
        img_path = data_dir / "train_images" / f"{row.image_id}.png"
        img = cv2.imread(str(img_path))
        cv2.imwrite(str(target_path / f"{row.image_id}.png"), img)


if __name__ == "__main__":
    data_dir = Path("dataset")
    train_images = sorted(glob(str(data_dir / "train_images/*.png")))
    train_df = pd.read_csv(data_dir / "train.csv", dtype={"image_id": "str"})

    mask_file = sorted(glob(str(data_dir / "masks/*.png")))
    print(mask_file)
    print(len(mask_file))

    mask_df = pd.DataFrame(mask_file, columns=["mask_path"])
    mask_df["image_id"] = mask_df["mask_path"].apply(
        lambda x: x.split("/")[-1].split(".")[0]
    )
    train_df = mask_df.merge(train_df, on="image_id", how="left")

    print(train_df)

    # white_thr = 240
    # drop_thr = 0.8

    tile_size = 2048
    for index, row in train_df.iterrows():
        print(row)
        img_path = data_dir / "train_images" / f"{row.image_id}.png"
        mask_path = data_dir / "masks" / f"{row.image_id}.png"

        img = pyvips.Image.new_from_file(img_path)
        img = img.colourspace("srgb")
        mask = pyvips.Image.new_from_file(mask_path)
        mask = mask.colourspace("srgb")

        width, height = img.width, img.height

        print(f"image_weight: {width}, image_height: {height}")

        maks_width, mask_height = mask.width, mask.height
        print(f"mask_weight: {maks_width}, mask_height: {mask_height}")

        rows = height // tile_size
        cols = width // tile_size

        for i in range(rows):
            for j in range(cols):
                # https://libvips.github.io/pyvips/vimage.html#pyvips.Image.crop
                left = i * tile_size
                top = j * tile_size

                tile_width = tile_size
                tile_height = tile_size

                # Adjust bounds
                if left + tile_width > width:
                    tile_width = width - left

                if top + tile_height > height:
                    tile_height = height - top

                if tile_width <= 0 or tile_height <= 0:
                    # Handle the case where either width or height is not positive
                    continue

                tile = img.crop(left, top, tile_width, tile_height)
                tile_mask = mask.crop(left, top, tile_width, tile_height)

        #             pil_image = Image.fromarray(tile.numpy())
        #             pil_image.save("test.png", format="png")
        #             break

        # black_bg = np.sum(tile, axis=2) == 0
        # tile[black_bg, :] = 255
        # mask_bg = np.mean(tile, axis=2) > white_thr

        # if np.sum(mask_bg) >= (np.prod(mask_bg.shape) * drop_thr):
        #     print(f"skip almost empty tile")
        #     # print(f"skip almost empty tile: {k:06}_{int(x_ / w)}-{int(y_ / h)}")
        #     continue

        # if tile.shape[:2] != (h, w):
        #         tile_ = tile
        #         tile_size = (h, w) if tile.ndim == 2 else (h, w, tile.shape[2])
        #         tile = np.zeros(tile_size, dtype=tile.dtype)
        #         tile[:tile_.shape[0], :tile_.shape[1], ...] = tile_

        # files = []
        # for y, y_, x, x_ in idxs:
        #     # https://libvips.github.io/pyvips/vimage.html#pyvips.Image.crop
        #     tile = im.crop(x, y, min(w, im.width - x), min(h, im.height - y)).numpy()[..., :3]
        #     if tile.shape[:2] != (h, w):
        #         tile_ = tile
        #         tile_size = (h, w) if tile.ndim == 2 else (h, w, tile.shape[2])
        #         tile = np.zeros(tile_size, dtype=tile.dtype)
        #         tile[:tile_.shape[0], :tile_.shape[1], ...] = tile_

        #     black_bg = np.sum(tile, axis=2) == 0
        #     tile[black_bg, :] = 255
        #     mask_bg = np.mean(tile, axis=2) > white_thr

        #     if np.sum(mask_bg) >= (np.prod(mask_bg.shape) * drop_thr):
        #         #print(f"skip almost empty tile: {k:06}_{int(x_ / w)}-{int(y_ / h)}")
        #         continue
        # target_path = data_dir / "tile_2048" / f"{row.image_id}"
        # p_img = os.path.join(folder, f"{int(x_ / w)}-{int(y_ / h)}.png")
        # # print(tile.shape, tile.dtype, tile.min(), tile.max())
        # new_size = int(size * scale), int(size * scale)
        # Image.fromarray(tile).resize(new_size, Image.LANCZOS).save(p_img)
        # files.append(p_img)
        # need to set counter check as some empty tiles could be skipped earlier
        # if len(files) >= max_samples:
        #     break

        break

    # target_path = data_dir / "train_thumbnails_custom"
    # target_path.mkdir(exist_ok=True)

    # num_processes = 12
    # Parallel(n_jobs=num_processes, backend="multiprocessing")(
    #     delayed(process_image)(row, data_dir, target_path)
    #     for _, row in tqdm(train_df.iterrows(), total=len(train_df))
    # )

    # for index, row in tqdm(train_df.iterrows(), total=len(train_df)):
    #     tma = row["is_tma"]
    #     if ~tma:
    #         img_path = data_dir / "train_images" / f"{row.image_id}.png"
    #         img = vips_read_image(str(img_path), longest_edge=5000)
    #         cv2.imwrite(str(target_path / f"{row.image_id}.png"), img)
    #     else:
    #         img_path = data_dir / "train_images" / f"{row.image_id}.png"
    #         img = cv2.imread(str(img_path))
    #         cv2.imwrite(str(target_path / f"{row.image_id}.png"), img)
