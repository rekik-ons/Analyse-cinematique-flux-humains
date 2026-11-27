import cv2
import numpy as np

def draw_annotation(frame,points,visibility,hip,status,fps):
    h,w=frame.shape[:2]
    color=(0,0,255) if status=='DANS ZONE' else (0,220,0) if status=='HORS ZONE' else (0,220,255)
    if points is not None:
        valid=(visibility>=.5)&np.isfinite(points).all(axis=1)&((points>=0)&(points<=1)).all(axis=1)
        p=points[valid]*[w-1,h-1]
        if len(p)>=2:
            a=np.maximum(p.min(axis=0).astype(int)-5,0); b=np.minimum(p.max(axis=0).astype(int)+5,[w-1,h-1])
            cv2.rectangle(frame,tuple(a),tuple(b),color,1)
        if hip is not None: cv2.circle(frame,tuple((hip*[w-1,h-1]).astype(int)),6,color,-1)
    cv2.putText(frame,f'FPS: {fps:.1f}',(8,20),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
    cv2.putText(frame,status,(8,40),cv2.FONT_HERSHEY_SIMPLEX,.45,color,1)
    cv2.putText(frame,'S:save R:reset P:pause Q:quit',(5,h-8),cv2.FONT_HERSHEY_SIMPLEX,.35,(255,255,255),1)
