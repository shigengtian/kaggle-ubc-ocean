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