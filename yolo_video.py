from ultralytics import YOLO
import cvzone
import cv2
import math


cap = cv2.VideoCapture("./videos/traffic.mp4")

# cap = cv2.VideoCapture(0)
# cap.set(3, 960)
# cap.set(4, 540)


model = YOLO("yolo-weights/yolo26n.pt")

classNames = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]


while True:
    success, img = cap.read()
    img = cv2.resize(img, (960, 540))    
    result = model(img, stream=True)
    
    for r in result:
        boxes = r.boxes
        
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,255), 3)
            
            w, h = x2-x1, y2-y1
            bbox = x1, y1, w, h
            
            cvzone.cornerRect(img, bbox)
            
            conf = box.conf[0]
            conf = math.ceil(conf*100)/100
            
            cls = int(box.cls[0])
            
            cvzone.putTextRect(img, f"{classNames[cls]} {conf}", (max(0,x1), max(30, y1-20)), scale=0.8)
        
    cv2.imshow("Object Detection in Video", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    
    
cap.release()
cv2.destroyAllWindows()


            
            
            
        
        
        
        
    
    
