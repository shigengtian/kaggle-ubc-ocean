import pandas as pd
import numpy as np
from pathlib import Path


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

    
    
# tiles_img, idxs = extract_image_tiles(
#     os.path.join(DATASET_IMAGES, "1020.png"),
#     "/kaggle/temp/images", size=1024, scale=0.5,
# )
# tiles_seg, _ = extract_image_tiles(
#     os.path.join(DATASET_MASKS, "1020.png"),
#     "/kaggle/temp/annotations", size=1024, scale=0.5,
#     drop_thr=None, inds=idxs
# )
# print(f"tiles_img={len(tiles_img)}")
# print(f"tiles_seg={len(tiles_seg)}")

# !ls -lh /kaggle/temp/images

if __name__ == '__main__':
    data_dir = Path('dataset')
    train_df = pd.read_csv(data_dir / 'train.csv')
    test_df = pd.read_csv(data_dir / 'test.csv')
    
    mask_files = sorted(list((data_dir / 'ubc-ovarian-cancer-competition-supplemental-masks').glob('*.png')))
    # print(mask_files)
    # print(train_df.head())
    # print(test_df.head())
