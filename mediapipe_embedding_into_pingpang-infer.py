'''
Descripttion: 自动描述，请修改
Author: Wei Jiangning
version: 
Date: 2022-12-03 11:25:31
LastEditors: Wei Jiangning
LastEditTime: 2022-12-03 17:08:57
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
from draw import *

# 导出关键点的参考资料 https://blog.csdn.net/qq_64605223/article/details/125606507

def mideapipe_keypoints(path):
	pose_model = mp_pose.Pose( min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1)
	cap = cv2.VideoCapture(path)
	frame_num = cap.get(7)
	w = int(cap.get(3))
	h = int(cap.get(4))
	vout = get_vout("test-mediapipe-getKeyPoints", w, h)
	all_kps = []
	with tqdm(total=frame_num) as pbar:
		t1 = time.time()
		cnt = 1
		while cap.isOpened():
			success, image = cap.read()
			if not success:
				print("Ignoring empty camera frame.")
				break
			# image = image[0:1080,0:960]
			# To improve performance, optionally mark the image as not writeable to
			# pass by reference.
			image.flags.writeable = False
			image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
			results = pose_model.process(image)

			# Draw the pose annotation on the image.
			image.flags.writeable = True
			image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
   
			cur_kps = keyponts_convert(results.pose_landmarks.landmark, w, h, cnt)
			cur_kps_pairs = kps_line2pairs(cur_kps)
			image_sk = draw_skeleton_kps_on_origin(cur_kps_pairs, image)
			vout.write(image_sk)
			all_kps.append(cur_kps)
			cnt += 1
			pbar.update(1)
	t2 = time.time()
	avg_fps = frame_num / (t2 - t1)
	print("average fps: ", avg_fps)
	cap.release()
	return all_kps

def keyponts_convert(frame_results, frame_w, frame_h, frame_cnt):
    nose = frame_results[mp_pose.PoseLandmark.NOSE]
    keypoints = []
    nose_x = nose.x * frame_w
    nose_y = nose.y * frame_h
    keypoints.append(frame_cnt)
    keypoints.append(nose_x)
    keypoints.append(nose_y)
    keypoints.extend([-1]*8)
    left_shoulder = frame_results[mp_pose.PoseLandmark.LEFT_SHOULDER]
    keypoints.append(left_shoulder.x * frame_w)
    keypoints.append(left_shoulder.y * frame_h)
    right_shoulder = frame_results[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    keypoints.append(right_shoulder.x * frame_w)
    keypoints.append(right_shoulder.y * frame_h)
    left_elbow = frame_results[mp_pose.PoseLandmark.LEFT_ELBOW]
    keypoints.append(left_elbow.x * frame_w)
    keypoints.append(left_elbow.y * frame_h)
    right_elbow = frame_results[mp_pose.PoseLandmark.RIGHT_ELBOW]
    keypoints.append(right_elbow.x * frame_w)
    keypoints.append(right_elbow.y * frame_h)
    left_wrist = frame_results[mp_pose.PoseLandmark.LEFT_WRIST]
    keypoints.append(left_wrist.x * frame_w)
    keypoints.append(left_wrist.y * frame_h)
    right_wrist = frame_results[mp_pose.PoseLandmark.RIGHT_WRIST]
    keypoints.append(right_wrist.x * frame_w)
    keypoints.append(right_wrist.y * frame_h)
    left_hip = frame_results[mp_pose.PoseLandmark.LEFT_HIP]
    keypoints.append(left_hip.x * frame_w)
    keypoints.append(left_hip.y * frame_h)
    right_hip = frame_results[mp_pose.PoseLandmark.RIGHT_HIP]
    keypoints.append(right_hip.x * frame_w)
    keypoints.append(right_hip.y * frame_h)
    left_knee = frame_results[mp_pose.PoseLandmark.LEFT_KNEE]
    keypoints.append(left_knee.x * frame_w)
    keypoints.append(left_knee.y * frame_h)
    right_knee = frame_results[mp_pose.PoseLandmark.RIGHT_KNEE]
    keypoints.append(right_knee.x * frame_w)
    keypoints.append(right_knee.y * frame_h)
    left_ankle = frame_results[mp_pose.PoseLandmark.LEFT_ANKLE]
    keypoints.append(left_ankle.x * frame_w)
    keypoints.append(left_ankle.y * frame_h)
    right_ankle = frame_results[mp_pose.PoseLandmark.RIGHT_ANKLE]
    keypoints.append(right_ankle.x * frame_w)
    keypoints.append(right_ankle.y * frame_h)
    # print(keypoints)
    # print(len(keypoints))
    
    return keypoints
    
def kps_line2pairs(kps):
	ret = []
	for i in range(1, 35, 2):
		ret.append([kps[i], kps[i + 1]])
	return ret

if __name__ == "__main__":
	path = r"test\01-test-half.avi"
	all_kps = mideapipe_keypoints(path)
	# print(all_kps)