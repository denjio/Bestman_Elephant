"""Environment class."""
from multiprocess import Process
import os
import sys
import time
import yaml
sys.path.append(os.getcwd())
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(parent_dir, 'RoboticsToolBox'))
sys.path.append(os.path.join(parent_dir, 'Montion_Planning'))
import numpy as np
import pybullet as p
from RoboticsToolBox import Bestman_Real_Elephant
from RoboticsToolBox import utility
from Motion_Planning import MotionPlanner, MotionPlanner_Collision
from enviroment import Environment

PLACE_STEP = 0.0003
PLACE_DELTA_THRESHOLD = 0.005


 # 创建进程
def buid_prcess(function1, function2, args1, args2):
    # 创建进程
    p1 = Process(target=function1, args=args1)
    p2 = Process(target=function2, args=args2)

    # 启动进程
    p1.start()
    p2.start()

    # 等待两个进程完成
    p1.join()
    p2.join()

 # 定义运动函数
 
def get_current_pose(robot_id, num_joints):
    # 获取所有关节的角度
    joint_angles = []
    for joint_index in range(num_joints):
        joint_state = p.getJointState(robot_id, joint_index)
        joint_angles.append(joint_state[0])  # 将角度添加到列表中

    return joint_angles

def move_with_action_list(action_list,  arm):
    for action in action_list:
        # print('action',action)
        action = utility.radians_to_degrees(np.array(action))
        arm._set_arm_joint_values(action)


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
    # open gripper
    buid_prcess(bestman1.close_gripper, bestman2.close_gripper, (100, 0), (100, 0))
    bestman1.wait_move_done()
    bestman2.wait_move_done()
 
    homej = [10.0, -70.0, 90.0, -90.0, -90.0, 0.0]
    homej = [0.0, -90.0, 90.0, -90.0, -90.0, 0.0]
    homej = [0.0, -120.0, 120.0, -90.0, -90.0, -0.0]
   
    # move_homej
    buid_prcess(bestman1._set_arm_joint_values, bestman2._set_arm_joint_values, (homej,), (homej,))
    bestman1.wait_move_done()
    bestman2.wait_move_done()

    current_cartesian1 = bestman1.get_current_cartesian()
    joint_values1 = bestman1.get_current_joint_values()
    current_cartesian2 = bestman2.get_current_cartesian()
    joint_values2 = bestman2.get_current_joint_values()
    print('real', current_cartesian1, joint_values1)
    # Unify to radians
    arm1_joint_values  = utility.degrees_to_radians(np.array(joint_values1))
    arm2_joint_values  = utility.degrees_to_radians(np.array(joint_values2))
    current_posture = utility.degrees_to_radians(np.array(current_cartesian1)[3:])
    print('real pos', current_posture)

    # Initialize environment and task.
    env = Environment(
        config['assets_root'],
        disp=config['disp'],
        shared_memory=config['shared_memory'],
        hz=480,
        record_cfg=config['record']
    )
    obs = env.reset(arm1_joint_values, arm2_joint_values)
    
    link_state_1 = p.getLinkState(env.ur5, 5)
    current_position_1 = link_state_1[4]  # 第一个机械臂的末端执行器位置
    current_orientation_1 = link_state_1[5]  # 第一个机械臂的末端执行器姿态（四元数)
    print(current_position_1, current_orientation_1)
    target_orientation = p.getQuaternionFromEuler([0, -3.14159 , 0])  # 垂直向下的姿态
    # target_orientation = p.getQuaternionFromEuler(current_posture)  # 垂直向下的姿态

    arm1_current_joint = get_current_pose(env.ur5, 6)
    arm2_current_joint = get_current_pose(env.ur5_1, 6)
    # euler_to_quaternion input: radians
    prepick_to_pick1 = ((0.0, -0.1, 0.4408505856990814),target_orientation)
    prepick_to_pick2 = ((-0.1, -0.2, 0.4408505856990814),target_orientation)
   
    arm1_goal_joint = env.solve_ik(prepick_to_pick1)
    arm2_goal_joint = env.solve_ik(prepick_to_pick2)
    # timeout = env.movep(prepick_to_pick1)
    # timeout = env.movep1(prepick_to_pick2)
    dual_arm_current = [
            arm1_current_joint,
            arm2_current_joint
        ]
    dual_arm_goal = [
        list(arm1_goal_joint),
        list(arm2_goal_joint)
    ]
    # 创建 MotionPlanner 实例并运行运动规划 method: BITstar, RRTstar, RRTConnect
    robot_ids = [env.ur5, env.ur5_1]
    env_obj_ids = [env.plane_id, env.workspace_id]
    planner = MotionPlanner_Collision(method='BITstar', arm_count=2, robot_ids=robot_ids, solve_time=15.0, env_obj_ids=env_obj_ids)
    action1, action2 = planner.plan(dual_arm_current, dual_arm_goal)
    # timeout = env.movep_two(prepick_to_pick1, prepick_to_pick2)
    timeout = env.movej_two_rrt(action1, action2)
    
    arm1_goal_values = utility.radians_to_degrees(arm1_goal_joint)
    print(arm1_goal_joint)
    print(arm1_goal_values)
    
  
    # move with ompl_plan
    buid_prcess(move_with_action_list, move_with_action_list, (action1, bestman1), (action2, bestman2))
    bestman1.wait_move_done()
    bestman2.wait_move_done()

    z1 = np.array(bestman1.get_current_cartesian())[2]
    z2 = np.array(bestman2.get_current_cartesian())[2]

    # open gripper
    buid_prcess(bestman1.open_gripper, bestman2.open_gripper, (100, 40), (100, 40))
    bestman1.wait_move_done()
    bestman2.wait_move_done()

    # grasp down
    buid_prcess(bestman1.set_single_coord, bestman2.set_single_coord,(2, 150, 2000), (2, 150, 2000))
    bestman1.wait_move_done()
    bestman2.wait_move_done()

    # close gripper
    buid_prcess(bestman1.close_gripper, bestman2.close_gripper, (100, 0), (100, 0))
    bestman1.wait_move_done()
    bestman2.wait_move_done()

    # grasp up
    buid_prcess(bestman1.set_single_coord, bestman2.set_single_coord, (2, z1, 2000), (2, z2, 2000))
    bestman1.wait_move_done()
    bestman2.wait_move_done()

    # z1 = np.array(bestman1.get_current_cartesian())[2]
    # z2 = np.array(bestman2.get_current_cartesian())[2]
    # # move_homej
    # buid_prcess(bestman1._set_arm_joint_values, bestman2._set_arm_joint_values, (homej,), (homej,))
    # bestman1.wait_move_done()
    # bestman2.wait_move_done()
 
    # # grasp down
    # buid_prcess(bestman1.set_single_coord, bestman2.set_single_coord,(2, 180, 2000), (2, 180, 2000))
    # bestman1.wait_move_done()
    # bestman2.wait_move_done()

    # # open gripper
    # buid_prcess(bestman1.open_gripper, bestman2.open_gripper, (100, 65), (100, 65))
    # bestman1.wait_move_done()
    # bestman2.wait_move_done()

    # time.sleep(1)
    #  # grasp up
    # buid_prcess(bestman1.set_single_coord, bestman2.set_single_coord, (2, z1, 2000), (2, z2, 2000))
    # bestman1.wait_move_done()
    # bestman2.wait_move_done()

    # # close gripper
    # buid_prcess(bestman1.close_gripper, bestman2.close_gripper, (100, 0), (100, 0))
    # bestman1.wait_move_done()
    # bestman2.wait_move_done()

    while True :
        pass
if __name__ == '__main__':
    main()

   
        
  