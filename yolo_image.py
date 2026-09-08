from ultralytics import YOLO
import cv2

model = YOLO("./yolo-weights/yolo26l.pt")

result = model("./Images/bus2.jpg")

annotated_frame = result[0].plot()

cv2.imshow("Object Detection in Image", annotated_frame)

cv2.waitKey(0)
cv2.destroyAllWindows()

