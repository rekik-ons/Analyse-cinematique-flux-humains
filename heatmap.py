import cv2
import numpy as np

class Heatmap:
    def __init__(self,width,height): self.grid=np.zeros((height,width),np.float32)
    def add(self,point,dt):
        if point is None: return
        h,w=self.grid.shape
        x,y=(np.asarray(point)*[w-1,h-1]).astype(int)
        if 0<=x<w and 0<=y<h: self.grid[y,x]+=max(0,dt)
    def save(self,path):
        grid=cv2.GaussianBlur(self.grid,(0,0),sigmaX=max(2,min(self.grid.shape)/40))
        image=cv2.applyColorMap(cv2.normalize(grid,None,0,255,cv2.NORM_MINMAX).astype(np.uint8),cv2.COLORMAP_JET)
        if not cv2.imwrite(str(path),image): raise OSError('Heatmap non sauvegardee')
