import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import numpy as np
import tensorflow as tf
from utils import detect_image, detect_realtime, detect_video, Load_Yolo_model, detect_video_realtime_mp
from configs import *

file="5.png"
image_path = f"./IMAGES/testing/{file}"
video_path = "/home/aaryaman_bhardwaj/Documents/ML/Yolo-v3/IMAGES/road_cone_1.mp4"

yolo = Load_Yolo_model()

# detect_image(yolo, image_path, f"./IMAGES/detect_v4_{file}", input_size=YOLO_INPUT_SIZE, show=True, iou_threshold=0.4,  CLASSES=TRAIN_CLASSES, score_threshold=0.5, rectangle_colors=(255, 0, 0))

detect_video(yolo, video_path, '/home/aaryaman_bhardwaj/Documents/ML/Yolo-v3/IMAGES/road_cone_2.mp4', input_size=YOLO_INPUT_SIZE, show=True, CLASSES=TRAIN_CLASSES, score_threshold=0.35, rectangle_colors=(255,0,0))


# detect_realtime(yolo, '', input_size=YOLO_INPUT_SIZE, show=True, CLASSES=TRAIN_CLASSES, rectangle_colors=(255, 0, 0))

# detect_video_realtime_mp(video_path, "Output.mp4", input_size=YOLO_INPUT_SIZE, show=True, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0), realtime=False)
