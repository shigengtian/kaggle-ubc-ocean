import pandas as pd
import numpy as np
from pathlib import Path


if __name__ == "__main__":
    datadir = Path("dataset")
    train_df = pd.read_csv(datadir / "train.csv")
    
    is_tma_count = len(train_df[ train_df["is_tma"] == True])
    isnot_tma_count = len(train_df[ train_df["is_tma"] == False])
    
    print("is_tma_count", is_tma_count)
    print("isnot_tma_count", isnot_tma_count)
    print(len(train_df))

    
    print(train_df[ train_df["is_tma"] == True])