import sys
import numpy as np
if len(sys.argv) < 2:
    sys.exit("Usage : python check_points.py data/keypoints/FastA.npy")

kp = np.load(sys.argv[1])                       # (n, 33, 3)
x = kp[:, [23, 24], 0].mean(axis=1)
y = kp[:, [23, 24], 1].mean(axis=1)
saut = np.hypot(np.diff(x, prepend=np.nan), np.diff(y, prepend=np.nan))
mauvais = saut > 0.10
print(f"{len(kp)} images | détectées : {np.isfinite(x).mean():.0%} | sauts > 0.10 : {np.nansum(mauvais)}")
print("images concernées :", np.where(mauvais)[0].tolist()[:30])
print("déplacement médian :", round(float(np.nanmedian(saut)), 4),
      "| moyen :", round(float(np.nanmean(saut)), 4),
      "| p90 :", round(float(np.nanpercentile(saut, 90)), 4))