import cv2

def draw_alert(frame, message):
    if not message: return
    h,w=frame.shape[:2]
    cv2.rectangle(frame,(1,1),(w-2,h-2),(0,0,255),3)
    cv2.putText(frame,message,(8,60),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,0,255),1)
