import numpy as np, pandas as pd, sys

d = pd.read_csv(sys.argv[1])
x = d[["x23", "x24"]].mean(axis=1)
y = d[["y23", "y24"]].mean(axis=1)
saut = np.hypot(x.diff(), y.diff())          # déplacement entre deux images (échelle 0-1)
mauvais = saut > 0.10                        # seuil à ajuster
print(f"{len(d)} images | détectées : {x.notna().mean():.0%} | sauts > 0.10 : {mauvais.sum()}")
print("images concernées :", d.loc[mauvais, "frame"].tolist()[:30])
print("déplacement médian par image :", round(saut.median(), 4))