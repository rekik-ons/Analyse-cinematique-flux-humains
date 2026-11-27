"""Convertit data/points/*_points.csv en data/keypoints/<video>.npy"""
from pathlib import Path
import numpy as np
import pandas as pd

out_dir = Path("data/keypoints")
out_dir.mkdir(parents=True, exist_ok=True)
for csv in sorted(Path("data/points").glob("*_points.csv")):
    d = pd.read_csv(csv)
    if len(d) == 0:
        print(f"{csv.name} : VIDE (en-tête seulement), relancer l'extraction")
        continue
    arr = d.iloc[:, 3:].to_numpy(dtype=np.float32).reshape(len(d), 33, 3)
    np.save(out_dir / csv.name.replace("_points.csv", ".npy"), arr)
    print(csv.name, arr.shape)