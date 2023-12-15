import os

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__()
import pandas as pd
import numpy as np
from pathlib import Path
import pyvips
import cv2
from glob import glob


if __name__ == "__main__":
    data_dir = Path("dataset")
    train_df = pd.read_csv(data_dir / "train.csv")
    

    train_df = train_df[train_df["is_tma"] == True]
    print(train_df)
   
    tile_size = (512, 512)  # Specify the size of each tile
    overlap = 50  # Specify the overlap between tiles
    for index, row in train_df.iterrows():
        img_id = row["image_id"]
        img_path = f"dataset/train_images/{img_id}.png"
        img = cv2.imread(str(img_path))
        height, width, channels = img.shape

        num_tiles_y = int(np.ceil(height / (tile_size[0] - overlap)))
        num_tiles_x = int(np.ceil(width / (tile_size[1] - overlap)))

        # Create an empty canvas for the tiled image
        tiled_image = np.zeros((num_tiles_y * tile_size[0] - (num_tiles_y - 1) * overlap,
                                num_tiles_x * tile_size[1] - (num_tiles_x - 1) * overlap, channels), dtype=np.uint8)

        # Tile the image
        for i in range(num_tiles_y):
            for j in range(num_tiles_x):
                start_y = i * (tile_size[0] - overlap)
                end_y = start_y + tile_size[0]
                start_x = j * (tile_size[1] - overlap)
                end_x = start_x + tile_size[1]

                # Adjust the end indices to handle the last tiles
                end_y = min(end_y, height)
                end_x = min(end_x, width)

                # Place the tile on the canvas
                tile = tiled_image[start_y:end_y, start_x:end_x, :] = img[start_y:end_y, start_x:end_x, :]
                cv2.imwrite(f"dataset/tile_tma/{img_id}_{i}_{j}.png", tile)
                
                
        # for i in range(num_tiles_y):
        #     for j in range(num_tiles_x):
        #         start_y = i * (tile_size[0] - overlap)
        #         end_y = start_y + tile_size[0]
        #         start_x = j * (tile_size[1] - overlap)
        #         end_x = start_x + tile_size[1]

        #         tiled_image[start_y:end_y, start_x:end_x, :] = img[start_y:end_y, start_x:end_x, :]
        #         cv2.imwrite(f"dataset/tile_tma/{img_id}_{i}_{j}.png", tiled_image[start_y:end_y, start_x:end_x, :])
                
                
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
        
    # print(train_df)
    # print(train_df[train_df["is_tma"] == True].head())
    
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

    #             tile_filename = f"{img_id}_{i}_{j}.png"
    #             tile_path = f"{save_path}/{tile_filename}"
    #             cv2.imwrite(tile_path, tile)