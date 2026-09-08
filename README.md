# Real time Object Detection Program

It has two versions:
yolo_image.py  - for object detection in image
yolo_video.py  - for object detection in video

You can add image files in images folder and video files in videos folder


## yolo_image.py
To use your own image:
Put your image in images folder then change path in line 6

## yolo_video.py
Here you can either use already set video or use your webcam for real time object detection through your camera.

To use webcam: 
1. Uncomment the lines 9, 10, 11
2. Comment line 7

To use your own video:
Put your video in videos folder then change path in line 7

# To leave the video or image process, enter "q" on keyboard

# For better accuracy use large version of yolo model. i.e. change "yolo26n.pt" to "yolo26l.pt" 
## But this will take more time or feel laggy if you do not have good gpu.