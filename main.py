import argparse
import hashlib
import time
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np
from pose import PoseDetector
from zones import ZoneManager
from annotation import draw_annotation
from alerts import draw_alert
from logger import EventLogger
from heatmap import Heatmap

ROOT=Path(__file__).parent

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--video')
    args=parser.parse_args()
    source=str(Path(args.video).resolve()) if args.video else 0
    source_id=hashlib.sha256(str(source).encode()).hexdigest()[:16]
    cap=cv2.VideoCapture(source)
    if not cap.isOpened(): cap.release(); print('Source inaccessible'); return
    detector=writer=logger=heat=None
    out=ROOT/'outputs'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True)
    name='Human Flow Analysis - C'
    try:
        detector=PoseDetector(); zone=ZoneManager(ROOT/'config'/f'zones_{source_id}.json',source_id)
        logger=EventLogger(out/'events.csv')
        cv2.namedWindow(name,cv2.WINDOW_AUTOSIZE); cv2.setMouseCallback(name,zone.mouse_callback)
        source_fps=cap.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(source_fps) or source_fps<=0: source_fps=30
        start=time.perf_counter(); previous=start; last_ms=-1; index=0; old_status=None; processing_seconds=0
        while True:
            begin=time.perf_counter()
            ret,frame=cap.read()
            if not ret: break
            h,w=frame.shape[:2]
            timestamp=max(last_ms+1,int(index*1000/source_fps if args.video else (begin-start)*1000))
            dt=(timestamp-last_ms)/1000 if last_ms>=0 else 0
            last_ms=timestamp; index+=1
            if heat is None:
                heat=Heatmap(w,h)
                writer=cv2.VideoWriter(str(out/'annotated_video.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),source_fps,(w,h))
                if not writer.isOpened(): raise RuntimeError('Impossible de creer la video MP4')
            points=detector.extract_keypoints(frame,timestamp); hip=detector.get_hip_center(points)
            inside=zone.contains(hip)
            status='POSE ABSENTE' if points is None else 'HANCHES NON VISIBLES' if hip is None else 'ZONE NON CONFIGUREE' if not zone.validated else 'DANS ZONE' if inside else 'HORS ZONE'
            # C journalise uniquement les changements d'etat geometrique.
            # Les evenements metier de B seront ajoutes via logger.log().
            if status!=old_status: logger.log(timestamp/1000,'zone_state', 'C',status); old_status=status
            heat.add(hip,dt)
            display=np.zeros_like(frame); zone.draw(display,inside); detector.draw_skeleton(display,points)
            fps=1/max(time.perf_counter()-previous,1e-6); previous=time.perf_counter()
            draw_annotation(display,points,detector.visibility,hip,status,fps)
            draw_alert(display,'PRESENCE EN ZONE' if inside else None)
            writer.write(display); cv2.imshow(name,display)
            processing_seconds+=time.perf_counter()-begin
            delay=max(1,int(1000/source_fps-(time.perf_counter()-begin)*1000)) if args.video else 1
            key=cv2.waitKey(delay)&255
            if key==ord('p'):
                while True:
                    key=cv2.waitKey(30)&255
                    if key==ord('s'): zone.save()
                    elif key==ord('r'): zone.reset()
                    elif key in (ord('p'),ord('q')): break
                    # Rafraichit la zone pendant la pause, sans enregistrer d'image supplementaire.
                    paused=np.zeros_like(frame); zone.draw(paused,zone.contains(hip)); detector.draw_skeleton(paused,points); draw_annotation(paused,points,detector.visibility,hip,'PAUSE',0); cv2.imshow(name,paused)
                # La pause ne constitue pas du temps de presence observe.
                if not args.video: start+=time.perf_counter()-previous
                previous=time.perf_counter()
            if key==ord('q'): break
            if key==ord('s'): zone.save()
            elif key==ord('r'): zone.reset()
        if index: print(f'FPS traitement moyen: {index/max(processing_seconds,1e-6):.2f}')
    finally:
        cap.release()
        if writer is not None: writer.release()
        if detector is not None: detector.close()
        if logger is not None: logger.close()
        if heat is not None: heat.save(out/'heatmap.png')
        cv2.destroyAllWindows(); print('Resultats:',out)

if __name__=='__main__': main()
