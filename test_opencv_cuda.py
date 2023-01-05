'''
Descripttion: 自动描述，请修改
Author: Wei Jiangning
version: 
Date: 2023-01-05 13:26:49
LastEditors: Wei Jiangning
LastEditTime: 2023-01-05 13:41:46
'''
import cv2
from tqdm import tqdm

count = cv2.cuda.getCudaEnabledDeviceCount()
print(count)
if count > 0:
    print("using cuda for opencv")
else:
    print("not using cuda for opencv")


path = r"D:\Project\google-media-pipe\test\01-test-half.avi"
cap = cv2.VideoCapture(path)
with tqdm(total=cap.get(7)) as pbar:
	while cap.isOpened():
		ret, frame = cap.read()
		if ret == False:
			break
		i = 0
		while i < 10:
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			i+= 1
		pbar.update(1)