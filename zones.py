import cv2
import json
import numpy as np
from pathlib import Path

def in_zone(point, polygon):
    if point is None or len(polygon)<3: return False
    return cv2.pointPolygonTest(np.asarray(polygon,dtype=np.float32),tuple(map(float,point)),False)>=0

class ZoneManager:
    def __init__(self, path, source_id):
        self.path=Path(path); self.source_id=source_id
        self.points=[]; self.validated=False; self.width=1; self.height=1
        if self.path.exists():
            try:
                d=json.loads(self.path.read_text(encoding='utf-8'))
                if d.get('source_id')==source_id and self.valid(d['polygon']):
                    self.points=d['polygon']; self.validated=True
            except (ValueError,KeyError,TypeError,OSError): print('Configuration invalide.')
    @staticmethod
    def valid(points):
        p=np.asarray(points,dtype=np.float32)
        return p.ndim==2 and p.shape[1]==2 and len(p)>=3 and np.isfinite(p).all() and ((p>=0)&(p<=1)).all() and cv2.contourArea(p)>0.0001
    def mouse_callback(self,event,x,y,flags,param):
        if self.validated: return
        if event==cv2.EVENT_LBUTTONDOWN and 0<=x<self.width and 0<=y<self.height:
            self.points.append([x/max(1,self.width-1),y/max(1,self.height-1)])
        elif event==cv2.EVENT_RBUTTONDOWN and self.points: self.points.pop()
    def save(self):
        if not self.valid(self.points): print('Zone invalide : 3 sommets minimum.'); return
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.path.write_text(json.dumps({'source_id':self.source_id,'name':'restricted_zone','type':'forbidden','coordinate_system':'normalized','polygon':self.points},indent=2),encoding='utf-8')
        self.validated=True
    def reset(self):
        self.points=[]; self.validated=False
        if self.path.exists(): self.path.unlink()
    def contains(self,point): return self.validated and in_zone(point,self.points)
    def draw(self,frame,inside=False):
        self.height,self.width=frame.shape[:2]
        if not self.points: return frame
        p=(np.asarray(self.points)*[self.width-1,self.height-1]).astype(np.int32)
        color=(0,0,255) if inside else (0,220,0)
        if self.validated:
            overlay=frame.copy(); cv2.fillPoly(overlay,[p],color); cv2.addWeighted(overlay,.15,frame,.85,0,frame)
        if len(p)>=2: cv2.polylines(frame,[p],self.validated,color,2)
        for pt in p: cv2.circle(frame,tuple(pt),4,color,-1)
        return frame
