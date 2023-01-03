'''
Descripttion: 自动描述，请修改
Author: Wei Jiangning
version: 
Date: 2022-11-21 11:46:52
LastEditors: Wei Jiangning
LastEditTime: 2023-01-03 18:49:22
'''
import cv2
import sys

def video2frames(video_path):
    vidcap = cv2.VideoCapture(video_path)
    print("vidcap: ", sys.getsizeof(vidcap))
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    # frame_width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # frame_height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("video: ", video_path.split("\\")[-1], " fps: ", fps)
    ret, frame = vidcap.read()
    frame_list = []
    while ret:
        # color convert for HRnet to get KeyPoints
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_list.append(cv2.flip(frame, -1))
        ret, frame = vidcap.read()
        # print(u'add frame：%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))
    return frame_list

def frames2Video(frame_list):
    width = len(frame_list[0][0])
    height = len(frame_list[0])
    print(width, height)
    skip_frame_cnt = 1 # 原始帧率和默认帧率相同
    out_dir = "./test"
    video_name = "test-mediapipe-getKeyPoints"
    outcap = cv2.VideoWriter(
        '{}/{}.mp4'.format(out_dir, video_name),
        cv2.VideoWriter_fourcc('M', 'P', '4', 'V'), 30, (width, height))
    for img in frame_list:
        outcap.write(img)
    outcap.release()
    
def get_vout(video_raw_name, width, height):
    outcap = cv2.VideoWriter(
        '{}.avi'.format(video_raw_name),
        cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (width, height))
    return outcap