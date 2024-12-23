""" 
# @Author: Youbin Yao 
# @Date: 2024-09-01 13:24:30
# @Last Modified by:   Youbin Yao 
# @Last Modified time: 2024-09-01 13:24:30  
""" 
import sys
import os
sys.path.append(os.getcwd())
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(parent_dir, 'RoboticsToolBox'))
sys.path.append(os.path.join(parent_dir, 'Visualization'))
import numpy as np
import cv2
import json
from RoboticsToolBox import Bestman_Real_Elephant
# from Visualization import calibration_eye_to_hand, Transform
from urdf_parser_py.urdf import URDF
from scipy.spatial.transform import Rotation as R
# from ikpy.chain import Chain
# from ikpy.link import OriginLink, URDFLink
 
# 读取 JSON 文件中的相机参数
json_file_path = 'Visualization/camera_params.json'
with open(json_file_path, 'r') as file:
    camera_params = json.load(file)


connections = [[1, (663, 398), (580, 350)], [2, (658, 309), (680, 350)], [3, (656, 214), (680, 250)], [4, (657, 140), (580, 250)]]



# 提取相机参数
mtx = np.array(camera_params['mtx']).reshape((3, 3))
dist = np.array(camera_params['dist'])
rvecs = np.array(camera_params['rvecs'])
tvecs = np.array(camera_params['tvecs'])

# 假设图像中目标点的像素坐标 (u, v) 和深度 Z
u, v= 656, 214 # 假设中心点
Z = 689 # 深度
 
# 提取内参矩阵(中的参数
mtx = np.array(
[[644.7208252,    0.,         631.20788574],
 [  0. ,        643.23901367, 372.19293213],
 [  0.,           0.,           1.        ]])
fx = mtx[0, 0]
fy = mtx[1, 1]
cx = mtx[0, 2]
cy = mtx[1, 2]
print(mtx)
# 将图像坐标 (u, v, z) 转换为相机坐标 (X, Y, Z)
X = (u - cx) * Z / fx
Y = (v - cy) * Z / fy
Z = Z
P_camera = np.array([X, Y, Z, 1.0])
print(P_camera)
 # arm 1
T_Cam2Robot_arm1 =np.array(
[[ 9.99787698e-01,  1.98010749e-02, -5.69876868e-03, 4.45405209e+02],
 [ 1.94412345e-02, -9.98158053e-01, -5.74677223e-02,  7.45542670e-01],
 [-6.82619452e-03,  5.73447307e-02, -9.98331100e-01,  8.85434233e+02],
 [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
 # arm 2
T_Cam2Robot_arm2 =np.array(
 [[-9.98701938e-01, 6.31164355e-03, -5.05430824e-02,  4.94098017e+02],
 [ 3.88014936e-03, 9.98836829e-01,  4.80617570e-02,  1.83141849e+01],
 [ 5.07876409e-02, 4.78032551e-02, -9.97564767e-01,  8.80432493e+02],
 [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
 
point_base = np.dot(T_Cam2Robot_arm2, P_camera)
x_base, y_base, z_base = point_base[0], point_base[1], point_base[2]

print(f"目标点在机械臂基座坐标系下的位置: ({x_base}, {y_base}, {z_base})")
bestman = Bestman_Real_Elephant("192.168.43.243", 5001)
# bestman.state_on()
bestman.get_current_cartesian()
bestman.get_current_joint_values()
# 定义垂直向下的欧拉角
# bestman.set_arm_joint_values([0.0, -120.0, 120.0, -90.0, -90.0, -0.0],speed=500)
bestman.set_arm_coords([x_base,y_base,220, -175,0,60],speed=800)
bestman.get_current_cartesian()
bestman.get_current_joint_values() 
 