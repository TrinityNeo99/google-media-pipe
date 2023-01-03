'''
Descripttion: 自动描述，请修改
Author: Wei Jiangning
version: 
Date: 2022-11-21 11:13:26
LastEditors: Wei Jiangning
LastEditTime: 2023-01-03 18:48:46
'''
import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
from tqdm import tqdm
import time
from util import *

class Api():
  def static_image(self):
    # For static images:
    IMAGE_FILES = []
    BG_COLOR = (192, 192, 192) # gray
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=True,
        min_detection_confidence=0.5) as pose:
      for idx, file in enumerate(IMAGE_FILES):
        image = cv2.imread(file)
        image_height, image_width, _ = image.shape
        # Convert the BGR image to RGB before processing.
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not results.pose_landmarks:
          continue
        print(
            f'Nose coordinates: ('
            f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image_width}, '
            f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * image_height})'
        )

        annotated_image = image.copy()
        # Draw segmentation on the image.
        # To improve segmentation around boundaries, consider applying a joint
        # bilateral filter to "results.segmentation_mask" with "image".
        condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
        bg_image = np.zeros(image.shape, dtype=np.uint8)
        bg_image[:] = BG_COLOR
        annotated_image = np.where(condition, annotated_image, bg_image)
        # Draw pose landmarks on the image.
        mp_drawing.draw_landmarks(
            annotated_image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)
        # Plot pose world landmarks.
        mp_drawing.plot_landmarks(
            results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
  
  def camera(self, path = ''):
    # For webcam input:
    if len(path) > 0:
      cap = cv2.VideoCapture(path)
    else:
      cap = cv2.VideoCapture(0)
    frame_num = cap.get(7)
    out_put_frames = []
    vout = get_vout("test-mediapipe-getKeyPoints", 960, 1080)
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1) as pose:
      with tqdm(total=frame_num) as pbar:
        t1 = time.time()
        while cap.isOpened():
          success, image = cap.read()
          
          # cv2.imshow("test", image)
          # cv2.waitKey(0)
          if not success:
            print("Ignoring empty camera frame.")
            break
            # If loading a video, use 'break' instead of 'continue'.
            continue
          # image = image[0:1080,0:960]
          # To improve performance, optionally mark the image as not writeable to
          # pass by reference.
          image.flags.writeable = False
          image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
          results = pose.process(image)
          # print(results)
          # break

          # Draw the pose annotation on the image.
          image.flags.writeable = True
          image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
          mp_drawing.draw_landmarks(
              image,
              results.pose_landmarks,
              mp_pose.POSE_CONNECTIONS,
              landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
          # Flip the image horizontally for a selfie-view display.
          # cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
          out_put_frames.append(image)
          # vout.write(image)
          if cv2.waitKey(5) & 0xFF == 27:
            break
          pbar.update(1)
    t2 = time.time()
    avg_fps = frame_num / (t2 - t1)
    print("average fps: ", avg_fps)
    cap.release()
    return out_put_frames