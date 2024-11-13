"""Environment class."""
from concurrent.futures import ThreadPoolExecutor
from multiprocess import Process
import os
import sys
import time
import hydra
import yaml
sys.path.append(os.getcwd())
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(parent_dir, 'RoboticsToolBox'))
sys.path.append(os.path.join(parent_dir, 'Montion_Planning'))
import cv2
import gym
import numpy as np
import pybullet as p
from collections import namedtuple
from Visualization import sim_camera as cameras
import imageio
import ompl
from ompl import base as ob
from ompl import geometric as og
from RoboticsToolBox import Bestman_Real_Elephant
from RoboticsToolBox import utility
from Motion_Planning import MotionPlanner
PLACE_STEP = 0.0003
PLACE_DELTA_THRESHOLD = 0.005

# URDF Paths
UR5_URDF_PATH = '/home/robot/Desktop/BestMan_Elephant/Asset/elephant_630pro/urdf/elephant_630pro-1.urdf'
PLANE_URDF_PATH = 'plane/plane.urdf'
UR5_WORKSPACE_URDF_PATH = 'ur5/workspace_real.urdf'

def load_urdf(pybullet_client, file_path, *args, **kwargs):
    """Loads the given URDF filepath."""
    # Handles most general file open case.
    try:
        return pybullet_client.loadURDF(file_path, *args, **kwargs)
    except pybullet_client.error:
        pass


class Environment(gym.Env):
    """OpenAI Gym-style environment class for UR5 robot."""

    def __init__(self, assets_root, task=None, disp=False, shared_memory=False, hz=240, record_cfg=None):
        self.assets_root = assets_root
        self.task = task
        self.disp = disp
        self.hz = hz
        self.record_cfg = record_cfg
        self.obj_ids = {'fixed': [], 'rigid': [], 'deformable': []}
        self.joints = {}
        self.joints1 = {}
        self.homej = np.array([0, -0.5, 0.5, -0.5, -0.5, 0]) * np.pi
        self.initialize_pybullet()
        self.save_video = False
        # Define observation and action spaces
        self.setup_spaces()
        self.step_counter = 0
        self.step_counter1 = 0
    def initialize_pybullet(self):
        """Initialize PyBullet physics engine."""
        disp_option = p.GUI if self.disp else p.DIRECT
        client = p.connect(disp_option)
        p.loadPlugin('fileIOPlugin', physicsClientId=client)
        p.setAdditionalSearchPath(self.assets_root)
        p.setTimeStep(1. / self.hz)
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        # self.load_urdfs()

    def load_urdfs(self,arm1_joint_values, arm2_joint_values):
        """Load URDF models into the simulation."""
        load_urdf(p, PLANE_URDF_PATH, [0, 0, -0.001])
        self.ur5 = load_urdf(p, UR5_URDF_PATH, [-0.25, 0.255, 0], useFixedBase=True)
        self.ur5_1 = load_urdf(
            p,UR5_URDF_PATH, [0.25, -0.28, 0], baseOrientation=[0.,0.,1.,0.], useFixedBase=True) #
        n_joints = p.getNumJoints(self.ur5)
        joints = [p.getJointInfo(self.ur5, i) for i in range(n_joints)]
        self.joints = [j[0] for j in joints if j[2] == p.JOINT_REVOLUTE]  
        self.joints1 = [j[0] for j in joints if j[2] == p.JOINT_REVOLUTE] 
        self.ee_tip = 6  # Link ID of suction cup.
        self.ee1_tip = 6  # Link ID of suction cup.
        # for i in range(len(self.joints)):
        #     p.resetJointState(self.ur5, self.joints[i], self.homej[i])
        # for i in range(len(self.joints1)):
        #     p.resetJointState(self.ur5_1, self.joints1[i], self.homej[i])
        for i in range(len(self.joints)):
            p.resetJointState(self.ur5, self.joints[i], arm1_joint_values[i])
        for i in range(len(self.joints1)):
            p.resetJointState(self.ur5_1, self.joints1[i], arm2_joint_values[i])
        self.num_joints = p.getNumJoints(self.ur5)
    def setup_spaces(self):
        """Define observation and action spaces for the environment."""
        color_tuple = [gym.spaces.Box(0, 255, (config['image_size'][0], config['image_size'][1], 3), dtype=np.uint8)
                       for config in cameras.RealSenseD415.CONFIG]
        depth_tuple = [gym.spaces.Box(0.0, 20.0, config['image_size'], dtype=np.float32)
                       for config in cameras.RealSenseD415.CONFIG]
        
        self.observation_space = gym.spaces.Dict({'color': gym.spaces.Tuple(color_tuple), 'depth': gym.spaces.Tuple(depth_tuple)})
        self.action_space = gym.spaces.Dict({
            'pose0': gym.spaces.Tuple((gym.spaces.Box(0.25, 0.75, shape=(3,), dtype=np.float32),
                                       gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32))),
            'pose1': gym.spaces.Tuple((gym.spaces.Box(0.25, 0.75, shape=(3,), dtype=np.float32),
                                       gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)))
        })
        
    def step_simulation(self):
        p.stepSimulation()
        self.step_counter += 1

        # if self.save_video and self.step_counter % 5 == 0:
        #     self.add_video_frame()
    def reset(self, arm1_joint_values, arm2_joint_values):
        """Reset the environment to an initial state."""
        self.obj_ids = {'fixed': [], 'rigid': [], 'deformable': []}
        # p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)
        p.setGravity(0, 0, -9.8)

        # Temporarily disable rendering to load scene faster.
        # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

        load_urdf(p, os.path.join(self.assets_root, PLANE_URDF_PATH),
                                 [0, 0, -0.001])
        load_urdf(
        p, os.path.join(self.assets_root, UR5_WORKSPACE_URDF_PATH), [0, 0, 0])
        p.resetSimulation()
        self.load_urdfs(arm1_joint_values, arm2_joint_values)
        # Set initial joint states or other reset logic
        return self._get_obs()

    def step(self, action):
        """Take a step in the environment based on the action provided."""
        # Implement action execution logic
        return self._get_obs()  # Return new observations

    def move_to_joint_config(self, ur5, target_joint_config, speed=0.01, timeout=20):
        """Move the specified UR5 to the target joint configuration."""
        for _ in range(timeout):
            current_joint_config = [p.getJointState(ur5, j)[0] for j in self.joints[ur5]]
            if np.allclose(current_joint_config, target_joint_config, atol=1e-2):
                return
            # Control logic
            p.setJointMotorControlArray(ur5, self.joints[ur5], p.POSITION_CONTROL, target_joint_config)
            p.stepSimulation()

    def _get_obs(self):
        """Retrieve observations from the environment."""
        obs = {'color': (), 'depth': ()}
        for config in cameras.RealSenseD415.CONFIG:
            color, depth, _ = self.render_camera(config)
            obs['color'] += (color,)
            obs['depth'] += (depth,)
        return obs

    def render_camera(self, config, image_size=None, shadow=1):
        """Render RGB-D image with specified camera configuration."""
        if not image_size:
            image_size = config['image_size']

        # OpenGL camera settings.
        lookdir = np.float32([0, 0, 1]).reshape(3, 1)
        updir = np.float32([0, -1, 0]).reshape(3, 1)
        rotation = p.getMatrixFromQuaternion(config['rotation'])
        rotm = np.float32(rotation).reshape(3, 3)
        lookdir = (rotm @ lookdir).reshape(-1)
        updir = (rotm @ updir).reshape(-1)
        lookat = config['position'] + lookdir
        focal_len = config['intrinsics'][0]
        znear, zfar = config['zrange']
        viewm = p.computeViewMatrix(config['position'], lookat, updir)
        fovh = (image_size[0] / 2) / focal_len
        fovh = 180 * np.arctan(fovh) * 2 / np.pi

        # Notes: 1) FOV is vertical FOV 2) aspect must be float
        aspect_ratio = image_size[1] / image_size[0]
        projm = p.computeProjectionMatrixFOV(fovh, aspect_ratio, znear, zfar)

        # Render with OpenGL camera settings.
        _, _, color, depth, segm = p.getCameraImage(
            width=image_size[1],
            height=image_size[0],
            viewMatrix=viewm,
            projectionMatrix=projm,
            shadow=shadow,
            flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
            renderer=p.ER_BULLET_HARDWARE_OPENGL)

        # Get color image.
        color_image_size = (image_size[0], image_size[1], 4)
        color = np.array(color, dtype=np.uint8).reshape(color_image_size)
        color = color[:, :, :3]  # remove alpha channel
        if config['noise']:
            color = np.int32(color)
            color += np.int32(self._random.normal(0, 3, image_size))
            color = np.uint8(np.clip(color, 0, 255))

        # Get depth image.
        depth_image_size = (image_size[0], image_size[1])
        zbuffer = np.array(depth).reshape(depth_image_size)
        depth = (zfar + znear - (2. * zbuffer - 1.) * (zfar - znear))
        depth = (2. * znear * zfar) / depth
        if config['noise']:
            depth += self._random.normal(0, 0.003, depth_image_size)

        # Get segmentation image.
        segm = np.uint8(segm).reshape(depth_image_size)

        return color, depth, segm
    def solve_ik(self, pose):
        """Calculate joint configuration with inverse kinematics."""
        joints = p.calculateInverseKinematics(
            bodyUniqueId=self.ur5,
            endEffectorLinkIndex=self.ee_tip,
            targetPosition=pose[0],
            targetOrientation=pose[1],
            lowerLimits = [
                -np.pi,                # J1
                -3 * np.pi / 2,       # J2
                -5 * np.pi / 6,       # J3
                -260 * np.pi / 180,   # J4
                -168 * np.pi / 180,   # J5
                -174 * np.pi / 180    # J6
                ],

            upperLimits = [
                    np.pi,                # J1
                    90 * np.pi / 180,     # J2
                    5 * np.pi / 6,        # J3
                    80 * np.pi / 180,     # J4
                    168 * np.pi / 180,    # J5
                    174 * np.pi / 180     # J6
                ],
            jointRanges=[np.pi, 2.3562, 34, 34, 34, 34],  # * 6,
            restPoses=np.float32(self.homej).tolist(),
            maxNumIterations=100,
            residualThreshold=1e-5)
        joints = np.float32(joints)
        joints[2:] = (joints[2:] + np.pi) % (2 * np.pi) - np.pi
        # print('joints', joints)
        return joints

    def solve_ik1(self, pose):
        """Calculate joint configuration with inverse kinematics."""
        joints = p.calculateInverseKinematics(
            bodyUniqueId=self.ur5_1,
            endEffectorLinkIndex=self.ee1_tip,
            targetPosition=pose[0],
            targetOrientation=pose[1],
            lowerLimits = [
                -np.pi,                # J1
                -3 * np.pi / 2,       # J2
                -5 * np.pi / 6,       # J3
                -260 * np.pi / 180,   # J4
                -168 * np.pi / 180,   # J5
                -174 * np.pi / 180    # J6
                ],

            upperLimits = [
                np.pi,                # J1
                90 * np.pi / 180,     # J2
                5 * np.pi / 6,        # J3
                80 * np.pi / 180,     # J4
                168 * np.pi / 180,    # J5
                174 * np.pi / 180     # J6
            ],
            jointRanges=[np.pi, 2.3562, 34, 34, 34, 34],  # * 6,
            restPoses=np.float32(self.homej).tolist(),
            maxNumIterations=100,
            residualThreshold=1e-5)
        joints = np.float32(joints)
        joints[2:] = (joints[2:] + np.pi) % (2 * np.pi) - np.pi
        return joints
    
    def movej(self, targj, speed=0.01, timeout=20):
        """Move UR5 to target joint configuration."""
        if self.save_video:
            timeout = timeout * 5

        t0 = time.time()
        while (time.time() - t0) < timeout:
            currj = [p.getJointState(self.ur5, i)[0] for i in self.joints]
            currj = np.array(currj)
            # nptargj = [list(t) for t in targj]
            # nptargj = [element for sublist in nptargj for element in sublist]

            # print(currj, targj)
            diffj = targj - currj
            if all(np.abs(diffj) < 1e-2):
                return False

            # Move with constant velocity
            norm = np.linalg.norm(diffj)
            v = diffj / norm if norm > 0 else 0
            stepj = currj + v * speed
            gains = np.ones(len(self.joints))
            p.setJointMotorControlArray(
                bodyIndex=self.ur5,
                jointIndices=self.joints,
                controlMode=p.POSITION_CONTROL,
                targetPositions=stepj,
                positionGains=gains)
            self.step_counter += 1
            self.step_simulation()

        print(f'Warning: movej exceeded {timeout} second timeout. Skipping.')
        return True

    def movej1(self, targj, speed=0.01, timeout=20):
        """Move UR5 to target joint configuration."""
        if self.save_video:
            timeout = timeout * 5

        t0 = time.time()
        while (time.time() - t0) < timeout:
            currj = [p.getJointState(self.ur5_1, i)[0] for i in self.joints1]
            currj = np.array(currj)
            targj = np.array(targj)
            diffj = targj - currj
            if all(np.abs(diffj) < 1e-2):
                return False

            # Move with constant velocity
            norm = np.linalg.norm(diffj)
            v = diffj / norm if norm > 0 else 0
            stepj = currj + v * speed
            gains = np.ones(len(self.joints1))
            p.setJointMotorControlArray(
                bodyIndex=self.ur5_1,
                jointIndices=self.joints1,
                controlMode=p.POSITION_CONTROL,
                targetPositions=stepj,
                positionGains=gains)
            self.step_counter1 += 1
            self.step_simulation()

        print(f'Warning: movej1 exceeded {timeout} second timeout. Skipping.')
        return True
    def movep(self, pose, speed=0.01):
        """Move UR5 to target end effector pose."""
        targj = self.solve_ik(pose)
        # print('targj', targj)
        return self.movej(targj, speed)
    def movep1(self, pose, speed=0.01):
        """Move UR5 to target end effector pose."""
        targj = self.solve_ik1(pose)
        # print('targj', targj)
        return self.movej1(targj, speed)
    
    def movej_two_rrt(self, targj, targj1, speed=0.01, timeout=20):
        """Move UR5 to target joint configuration."""
        # 这里是针对动作list的
        if self.save_video:
            timeout = timeout * 10

        t0 = time.time()
        # print(targj)
        # print(targj1)
        # print(len(targj), len(targj1))
        for i in range(len(targj)):
            # print('targ',targj[i], targj1[i])
            while (time.time() - t0) < timeout:
                currj = [p.getJointState(self.ur5, i)[0] for i in self.joints]
                currj = np.array(currj)
                diffj = targj[i] - currj

                # Move with constant velocity
                norm = np.linalg.norm(diffj)
                v = diffj / norm if norm > 0 else 0
                stepj = currj + v * speed
                gains = np.ones(len(self.joints))

                currj1 = [p.getJointState(self.ur5_1, i)[0] for i in self.joints1]
                currj1 = np.array(currj1)
                diffj1 = targj1[i] - currj1
                
                norm1 = np.linalg.norm(diffj1)
                v1 = diffj1 / norm1 if norm1 > 0 else 0
                stepj1 = currj1 + v1 * speed
                gains1 = np.ones(len(self.joints1))
                if all(np.abs(diffj) < 1e-2) and all(np.abs(diffj1) < 1e-2):
                    # print(np.abs(diffj), np.abs(diffj1))
                    break
 
                    
                p.setJointMotorControlArray(
                    bodyIndex=self.ur5_1,
                    jointIndices=self.joints1,
                    controlMode=p.POSITION_CONTROL,
                    targetPositions=stepj1,
                    positionGains=gains1)
                # print(1)
                p.setJointMotorControlArray(
                    bodyIndex=self.ur5,
                    jointIndices=self.joints,
                    controlMode=p.POSITION_CONTROL,
                    targetPositions=stepj,
                    positionGains=gains)
                # print(2)
                self.step_counter += 1
                self.step_simulation()
            if (time.time() - t0) > timeout:
                print(f'Warning: movej_two_rrt exceeded {timeout} second timeout. Skipping.')
                return True

        return False
 
def get_current_pose(robot_id, num_joints):
    # 获取所有关节的角度
    joint_angles = []
    for joint_index in range(num_joints):
        joint_state = p.getJointState(robot_id, joint_index)
        joint_angles.append(joint_state[0])  # 将角度添加到列表中

    return joint_angles

def main():
    # 读取 YAML 文件
    with open('Examples/sim_real/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # get_real
    bestman1 = Bestman_Real_Elephant("192.168.43.38", 5001)
    bestman2 = Bestman_Real_Elephant("192.168.43.243", 5001)
    # bestman.state_on()
    # bestman.power_off ()
    bestman1.check_error()
    bestman2.check_error()

    # bestman1.set_arm_joint_values([0.0, -90.0, 90.0, -90.0, -90.0, -0.0])
    # bestman2.set_arm_joint_values([0.0, -90.0, 90.0, -90.0, -90.0, -0.0])
    joint_values1 = [10.0, -70.0, 90.0, -90.0, -90.0, -0.0]
    joint_values2 = [10.0, -70.0, 90.0, -90.0, -90.0, -0.0]
    joint_values1 = [0.0, -90.0, 90.0, -90.0, -90.0, -0.0]
    joint_values2 = [0.0, -90.0, 90.0, -90.0, -90.0, -0.0]
    # 定义运动函数
    def move_arm(arm, joint_values):
        arm.set_arm_joint_values(joint_values)
 
    # 创建进程
    p1 = Process(target=move_arm, args=(bestman1, joint_values1))
    p2 = Process(target=move_arm, args=(bestman2, joint_values2))
    # 启动进程
    p1.start()
    p2.start()

    # 等待两个进程完成
    p1.join()
    p2.join()
    # current_cartesian1 = bestman1.get_current_cartesian()
    # joint_values1 = bestman1.get_current_joint_values()
    # current_cartesian2 = bestman2.get_current_cartesian()
    # joint_values2 = bestman2.get_current_joint_values()
    # print('real', current_cartesian1, joint_values1)
    # # Unify to radians
    # arm1_joint_values  = utility.degrees_to_radians(np.array(joint_values1))
    # arm2_joint_values  = utility.degrees_to_radians(np.array(joint_values2))
    # current_posture = utility.degrees_to_radians(np.array(current_cartesian1)[3:])
    # print('real pos', current_posture)

    # # Initialize environment and task.
    # env = Environment(
    #     config['assets_root'],
    #     disp=config['disp'],
    #     shared_memory=config['shared_memory'],
    #     hz=480,
    #     record_cfg=config['record']
    # )
    # obs = env.reset(arm1_joint_values, arm2_joint_values)
    
    # link_state_1 = p.getLinkState(env.ur5, 5)
    # current_position_1 = link_state_1[4]  # 第一个机械臂的末端执行器位置
    # current_orientation_1 = link_state_1[5]  # 第一个机械臂的末端执行器姿态（四元数)
    # print(current_position_1, current_orientation_1)
    # # target_orientation = p.getQuaternionFromEuler([0, -3.14159 , 0])  # 垂直向下的姿态
    # target_orientation = p.getQuaternionFromEuler(current_posture)  # 垂直向下的姿态

    # arm1_current_joint = get_current_pose(env.ur5, 6)
    # arm2_current_joint = get_current_pose(env.ur5_1, 6)
    # # euler_to_quaternion input: radians
    # prepick_to_pick1 = ((0.0, 0.1, 0.4408505856990814), target_orientation)
    # prepick_to_pick2 = ((0.0, 0.0, 0.4408505856990814), target_orientation)
   
    # arm1_goal_joint = env.solve_ik(prepick_to_pick1)
    # arm2_goal_joint = env.solve_ik(prepick_to_pick2)
    # # timeout = env.movep(prepick_to_pick1)
    # # timeout = env.movep1(prepick_to_pick2)
    # dual_arm_current = [
    #         arm1_current_joint,
    #         arm2_current_joint
    #     ]
    # dual_arm_goal = [
    #     list(arm1_goal_joint),
    #     list(arm2_goal_joint)
    # ]
    #     # 创建 MotionPlanner 实例并运行运动规划
    # planner = MotionPlanner(arm_count=2, vMax=0.2, solve_time=10.0)
    # action1, action2 = planner.plan(dual_arm_current, dual_arm_goal)
    # # timeout = env.movep_two(prepick_to_pick1, prepick_to_pick2)
    # timeout = env.movej_two_rrt(action1, action2)
    
    # # print(arm1_goal_joint)
    # arm1_goal_values = utility.radians_to_degrees(arm1_goal_joint)
    # print(arm1_goal_joint)
    # print(arm1_goal_values)
    # for action in action1:
    #     # print('action',action)
    #     action = utility.radians_to_degrees(np.array(action))
    #     bestman1.set_arm_joint_values(action)
    # for action in action2:
    #     action = utility.radians_to_degrees(np.array(action))
    #     bestman2.set_arm_joint_values(action)
    # while True :
    #     pass
if __name__ == '__main__':
    main()


