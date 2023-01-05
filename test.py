'''
Descripttion: 自动描述，请修改
Author: Wei Jiangning
version: 
Date: 2022-11-21 11:40:36
LastEditors: Wei Jiangning
LastEditTime: 2023-01-05 13:44:06
'''
from pose_api import *
from util import *
count = cv2.cuda.getCudaEnabledDeviceCount()
print(count)
if count > 0:
    print("using cuda for opencv")
else:
    print("not using cuda for opencv")

if __name__ == "__main__":
    path = r"C:\Users\weiji\OneDrive - bupt.edu.cn\文档\宽广实验室\2022-正手动作分析论文\论文素材\01.mp4"
    path = r"F:\pingpang-all-data\体育课数据\2022_10_24-第五周乒乓球教学\实验组粗剪\2022_10_24-实验组01-边硕-张瑞\01_1.mp4"
    path = r"F:\pingpang-all-data\体育课数据\2022_10_24-第五周乒乓球教学\实验组粗剪\2022_10_17-实验组09-肖宇杭-严泽宇\09_3.mp4"
    path = r"D:\Project\google-media-pipe\test\01-test-half.avi"
    pose = Api()
    out_frames = pose.camera(path)
    frames2Video(out_frames)
    
    