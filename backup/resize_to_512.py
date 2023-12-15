import os

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__()
import pandas as pd
import numpy as np
from pathlib import Path
import pyvips
import cv2
from glob import glob
from joblib import Parallel, delayed

def extract_image_tiles(img_id, img_path, size=2048):
    img = cv2.imread(str(img_path))
    h, w = size, size  # Set the size of the tiles
    height, width = img.shape[:2]

    save_path = "tiles"

    # Calculate the number of rows and columns
    rows = height // h
    cols = width // w

    for i in range(rows):
        for j in range(cols):
            tile = img[i * h : (i + 1) * h, j * w : (j + 1) * w]
            tile_filename = f"{img_id}_{i}_{j}.jpg"

            tile_path = f"dataset/tiles/{tile_filename}"
            cv2.imwrite(tile_path, tile)


def get_thumbnails_df(thumbnails_dir):
    """
    Get train_thumbnails_files_df
    """
    thumbnails_files = list(thumbnails_dir.glob("*.png"))
    print(f"thumbnails_files: {len(thumbnails_files)}")

    thumbnails_files_df = pd.DataFrame(
        thumbnails_files, columns=["thumbnails_file_paths"]
    )

    thumbnails_files_df["image_id"] = thumbnails_files_df[
        "thumbnails_file_paths"
    ].apply(lambda x: str(x).split("/")[-1].split("_")[0])
    return thumbnails_files_df


def get_mask_df(mask_dir):
    mask_files = list(mask_dir.glob("*.png"))
    print(f"mask_files: {len(mask_files)}")
    mask_files_df = pd.DataFrame(mask_files, columns=["mask_file_paths"])
    mask_files_df["image_id"] = mask_files_df["mask_file_paths"].apply(
        lambda x: str(x).split("/")[-1].split(".")[0]
    )

    return mask_files_df


def get_train_img_df(train_img_dir):
    img_files = list(train_img_dir.glob("*.png"))
    print(f"mask_files: {len(img_files)}")
    train_img_df = pd.DataFrame(img_files, columns=["img_file_paths"])

    train_img_df["image_id"] = train_img_df["img_file_paths"].apply(
        lambda x: str(x).split("/")[-1].split(".")[0]
    )

    return train_img_df



def process_image(img_file, data_dir):
    img = cv2.imread(str(img_file))
    target_img = cv2.resize(img, (512, 512))
    img_name = str(img_file).split("/")[-1]
    target_path = data_dir / "train_images_512" / img_name
    cv2.imwrite(str(target_path), target_img)
    

if __name__ == "__main__":
    data_dir = Path("dataset")
    train_img_dir = data_dir / "train_images"
    img_files = list(train_img_dir.glob("*.png"))
     
     
    # Specify the number of parallel processes (adjust as needed)
    num_processes = 8

    # Use joblib to parallelize image processing
    Parallel(n_jobs=num_processes)(
        delayed(process_image)(img_file, data_dir) for img_file in img_files
    )
    
    
    # for img_file in img_files:
    #     img = cv2.imread(str(img_file))
    #     target_img = cv2.resize(img, (512, 512))
    #     img_name = str(img_file).split("/")[-1]
    #     target_path = data_dir / "train_images_512" / img_name
    #     cv2.imwrite(str(target_path), target_img)
    #     # break
     
    # train_df = pd.read_csv(data_dir / "train.csv")
    # test_df = pd.read_csv(data_dir / "test.csv")
    # target_path = data_dir / "tile_512"
    # target_path.mkdir(parents=True, exist_ok=True)

    # data_dir = Path("dataset")
    # mask_dir = data_dir / "masks"
    # train_thumbnails_dir = data_dir / "train_thumbnails"
    # train_img_dir = data_dir / "train_images"

    # train_thumbnails_files_df = get_thumbnails_df(train_thumbnails_dir)
    # mask_files_df = get_mask_df(mask_dir)
    # train_img_df = get_train_img_df(train_img_dir)

    # train_df = mask_files_df.merge(train_thumbnails_files_df, on="image_id", how="left")
    # train_df = train_df.merge(train_img_df, on="image_id", how="left")
    
    # train_csv_df = pd.read_csv(data_dir / "train.csv", dtype=str)
    
    # train_df = train_df.merge(train_csv_df, on="image_id", how="left")

    # print(train_df[train_df["is_tma"] == "True"].head())
    
    # print(train_csv_df.head())
    # print(len(train_df))
    # print(train_df.head())

    # tile_size = 2048
    # save_path = "dataset/tiles"

    # for index, row in train_df.iterrows():
    #     img_id = row["image_id"]
    #     img_path = row["img_file_paths"]
    #     mask_path = row["mask_file_paths"]
    #     img = cv2.imread(str(img_path))

    #     height, width = img.shape[:2]

    #     mask = cv2.imread(str(mask_path))
    #     mask = mask[:, :, 2]
    #     mask = (mask > 0).astype(int)

    #     rows = height // tile_size
    #     cols = width // tile_size

    #     for i in range(rows):
    #         for j in range(cols):
    #             tile_mask = mask[
    #                 i * tile_size : (i + 1) * tile_size,
    #                 j * tile_size : (j + 1) * tile_size,
    #             ]
    #             true_percentage = tile_mask.sum() / tile_mask.size
    #             print(tile_mask.size)
    #             if true_percentage < 0.5:
    #                 continue

    #             tile = img[
    #                 i * tile_size : (i + 1) * tile_size,
    #                 j * tile_size : (j + 1) * tile_size,
    #             ]

    #             tile_filename = f"{img_id}_{i}_{j}.jpg"
    #             tile_path = f"{save_path}/{tile_filename}"
    #             cv2.imwrite(tile_path, tile)