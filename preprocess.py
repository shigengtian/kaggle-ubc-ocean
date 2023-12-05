import os

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__()
import pandas as pd
import numpy as np
from pathlib import Path
import pyvips
import cv2
from glob import glob

# def extract_image_tiles(
#     p_img, folder, size: int = 2048, scale: float = 0.5,
#     drop_thr: float = 0.85, inds = None
# ) -> list:
#     name, _ = os.path.splitext(os.path.basename(p_img))
#     im = pyvips.Image.new_from_file(p_img)
#     w = h = size
#     if not inds:
#         # https://stackoverflow.com/a/47581978/4521646
#         inds = [(y, y + h, x, x + w)
#                 for y in range(0, im.height, h)
#                 for x in range(0, im.width, w)]
#     files, idxs, k = [], [], 0
#     for idx in inds:
#         y, y_, x, x_ = idx
#         # https://libvips.github.io/pyvips/vimage.html#pyvips.Image.crop
#         tile = im.crop(x, y, min(w, im.width - x), min(h, im.height - y)).numpy()[..., :3]
#         if drop_thr is not None:
#             mask_bg = np.sum(tile, axis=2) == 0
#             if np.sum(mask_bg) >= (np.prod(mask_bg.shape) * drop_thr):
#                 #print(f"skip almost empty tile: {k:06}_{int(x_ / w)}-{int(y_ / h)}")
#                 continue
#         if tile.shape[:2] != (h, w):
#             tile_ = tile
#             tile_size = (h, w) if tile.ndim == 2 else (h, w, tile.shape[2])
#             tile = np.zeros(tile_size, dtype=tile.dtype)
#             tile[:tile_.shape[0], :tile_.shape[1], ...] = tile_
#         p_img = os.path.join(folder, f"{k:05}_{int(x_ / w)}-{int(y_ / h)}.png")
#         # print(tile.shape, tile.dtype, tile.min(), tile.max())
#         new_size = int(size * scale), int(size * scale)
#         Image.fromarray(tile).resize(new_size, Image.LANCZOS).save(p_img)
#         files.append(p_img)
#         idxs.append(idx)
#         k += 1
#     return files, idxs


def extract_image_tiles(img_path, size=2048):
    img = cv2.imread(str(img_path))
    h, w = size, size  # Set the size of the tiles
    height, width = img.shape[:2]
    print(img.shape)
    
    save_path = "tiles" 

    # Calculate the number of rows and columns
    rows = height // h
    cols = width // w

    # Extract tiles from the image
    # tiles = []
    for i in range(rows):
        for j in range(cols):
            tile = img[i * h: (i + 1) * h, j * w: (j + 1) * w]
            tile_filename = f"tile_{i}_{j}.jpg"
            # print(123123)
            # tile_path = os.path.join(save_path, tile_filename)
            tile_path = f"tiles/{tile_filename}"
            cv2.imwrite(tile_path, tile)
            
            # tiles.append(tile)

    # return tiles

    
    

    # if not inds:
    #     # https://stackoverflow.com/a/47581978/4521646
    #     inds = [(y, y + h, x, x + w)
    #             for y in range(0, im.height, h)
    #             for x in range(0, im.width, w)]
    # files, idxs, k = [], [], 0
    # for idx in inds:
    #     y, y_, x, x_ = idx
    #     # https://libvips.github.io/pyvips/vimage.html#pyvips.Image.crop
    #     tile = im.crop(x, y, min(w, im.width - x), min(h, im.height - y)).numpy()[..., :3]
    #     if drop_thr is not None:
    #         mask_bg = np.sum(tile, axis=2) == 0
    #         if np.sum(mask_bg) >= (np.prod(mask_bg.shape) * drop_thr):
    #             #print(f"skip almost empty tile: {k:06}_{int(x_ / w)}-{int(y_ / h)}")
    #             continue
    #     if tile.shape[:2] != (h, w):
    #         tile_ = tile
    #         tile_size = (h, w) if tile.ndim == 2 else (h, w, tile.shape[2])
    #         tile = np.zeros(tile_size, dtype=tile.dtype)
    #         tile[:tile_.shape[0], :tile_.shape[1], ...] = tile_
    #     p_img = os.path.join(folder, f"{k:05}_{int(x_ / w)}-{int(y_ / h)}.png")
    #     # print(tile.shape, tile.dtype, tile.min(), tile.max())
    #     new_size = int(size * scale), int(size * scale)
    #     Image.fromarray(tile).resize(new_size, Image.LANCZOS).save(p_img)
    #     files.append(p_img)
    #     idxs.append(idx)
    #     k += 1
    # return files, idxs


def get_thumbnails(train_thumbnails_dir):
    """
    Get train_thumbnails_files_df
    """
    train_thumbnails_files = list(train_thumbnails_dir.glob("*.png"))
    print(f"train_thumbnails_files: {len(train_thumbnails_files)}")

    train_thumbnails_files_df = pd.DataFrame(
        train_thumbnails_files, columns=["train_thumbnails_file_paths"]
    )

    train_thumbnails_files_df["image_id"] = train_thumbnails_files_df[
        "train_thumbnails_file_paths"
    ].apply(lambda x: str(x).split("/")[-1].split("_")[0])
    return train_thumbnails_files_df


def get_mask_df(mask_dir):
    mask_files = list(mask_dir.glob("*.png"))
    print(f"mask_files: {len(mask_files)}")
    mask_files_df = pd.DataFrame(mask_files, columns=["mask_file_paths"])
    mask_files_df["image_id"] = mask_files_df["mask_file_paths"].apply(
        lambda x: str(x).split("/")[-1].split(".")[0]
    )

    return mask_files_df


if __name__ == "__main__":
    data_dir = Path("dataset")
    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    target_path = data_dir / "tile_512"
    target_path.mkdir(parents=True, exist_ok=True)

    data_dir = Path("dataset")
    mask_dir = data_dir / "masks"
    train_thumbnails_dir = data_dir / "train_thumbnails"

    train_thumbnails_files_df = get_thumbnails(train_thumbnails_dir)
    mask_files_df = get_mask_df(mask_dir)

    train_thumbnails_files_df = train_thumbnails_files_df.merge(
        mask_files_df, on="image_id", how="left"
    ).dropna()

    # print(train_thumbnails_files_df.head())
    # extract_image_tiles("dataset/train_thumbnails/5251_thumbnail.png")
    
    
    extract_image_tiles("dataset/train_images/5251.png")
    
    # for index, row in train_thumbnails_files_df.iterrows():
    #     thumbnails_fils_path = row["train_thumbnails_file_paths"]
    #     mask_file_path = row["mask_file_paths"]
    #     print(thumbnails_fils_path, mask_file_path)
    #     break

    # train_df = pd.read_csv(data_dir / "train.csv", dtype={"image_id": str})

    # train_thumbnails_files_df = train_thumbnails_files_df.merge(
    #     train_df, on="image_id", how="left"
    # )

    # print(train_thumbnails_files_df.head())
