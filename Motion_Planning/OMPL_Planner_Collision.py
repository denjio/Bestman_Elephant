from os.path import abspath, dirname, join
import sys
import math
import numpy as np
from functools import partial

try:
    from ompl import util as ou
    from ompl import base as ob
    from ompl import geometric as og
except ImportError:
    sys.path.insert(0, join(dirname(dirname(abspath(__file__))), 'py-bindings'))
    from ompl import util as ou
    from ompl import base as ob
    from ompl import geometric as og

import pybullet as p


def check_collision_with_environment(robot_id, link_indices, env_obj_ids, distance_threshold=0.01):
    """
    检查机械臂是否与环境中的其他物体发生碰撞。
    :param robot_id: 机械臂的 PyBullet ID
    :param link_indices: 机械臂的连杆索引列表
    :param env_obj_ids: 环境中所有物体的 ID 列表
    :param distance_threshold: 碰撞检测距离阈值
    :return: True 如果发生碰撞，否则 False
    """
    for link_index in link_indices:
        if link_index == -1:  # 跳过基座
            continue
        for obj_id in env_obj_ids:
            closest_points = p.getClosestPoints(
                bodyA=robot_id, bodyB=obj_id,
                linkIndexA=link_index, distance=distance_threshold
            )
            if len(closest_points) > 0:  # 如果检测到靠近的点
                for point in closest_points:
                    contact_distance = point[8]  # 获取 contactDistance
                    if contact_distance < 0:  # 仅当距离为负值（即真正碰撞）时返回 True
                        print('真实碰撞点:', obj_id, robot_id, point)
                        return True
    return False


def check_collision(robot_id1, robot_id2):
    """
    检查两个机器人是否碰撞。
    :param robot_id1: 第一个机器人 ID
    :param robot_id2: 第二个机器人 ID
    :return: True 如果发生碰撞，否则 False
    """
    collision_points = p.getClosestPoints(bodyA=robot_id1, bodyB=robot_id2, distance=0.01)
    return len(collision_points) > 0

def check_self_collision(robot_id, link_indices, distance_threshold=0.01):
    """
    检查机械臂自身是否发生碰撞。
    :param robot_id: 机械臂的 PyBullet ID
    :param link_indices: 机械臂的连杆索引列表
    :param distance_threshold: 碰撞检测距离阈值
    :return: True 如果发生碰撞，否则 False
    """
    num_links = len(link_indices)
    for i in range(num_links):
        for j in range(i + 1, num_links):
            # 跳过固定基座的连杆
            if link_indices[i] == -1 or link_indices[j] == -1:
                continue
            
            # 获取两个连杆之间的最近点信息
            closest_points = p.getClosestPoints(
                bodyA=robot_id, bodyB=robot_id,
                linkIndexA=link_indices[i], linkIndexB=link_indices[j],
                distance=distance_threshold
            )
            if len(closest_points) > 0:  # 如果检测到靠近的点
                for point in closest_points:
                    contact_distance = point[8]  # 获取 contactDistance
                    if contact_distance < 0:  # 仅当距离为负值（即真正碰撞）时返回 True
                        print('真实碰撞点:',robot_id, point)
                        return True
    return False

def isStateValid(spaceInformation, state, robot_ids, env_obj_ids, num_joints_per_arm):
    """
    检查给定状态是否有效（无碰撞）。
    :param spaceInformation: OMPL 的空间信息对象
    :param state: 当前状态（OMPL 的 State 对象）
    :param robot_ids: 机器人 ID 列表
    :param env_obj_ids: 环境中物体的 ID 列表
    :param num_joints_per_arm: 每个机械臂的关节数
    :return: True 如果状态无碰撞，否则 False
    """
    num_arms = len(robot_ids)

    # 为每个机械臂设置关节状态
    for arm_index in range(num_arms):
        start_joint = arm_index * num_joints_per_arm
        end_joint = start_joint + num_joints_per_arm
        joint_positions = [
            state[i]  # 直接通过索引访问状态值
            for i in range(start_joint, end_joint)
        ]
        for i, joint_value in enumerate(joint_positions):
            p.resetJointState(robot_ids[arm_index], i, joint_value)

    # # 检查自碰撞
    # for robot_id in robot_ids:
    #     link_indices = [j for j in range(p.getNumJoints(robot_id))]
    #     if check_self_collision(robot_id, link_indices):
    #         print(1)
    #         return False

    # 检查机械臂与环境中物体的碰撞
    for robot_id in robot_ids:
        link_indices = [j for j in range(p.getNumJoints(robot_id))]
        if check_collision_with_environment(robot_id, link_indices, env_obj_ids):
            print(2)
            return False

    # 检查机械臂之间的碰撞
    for i in range(len(robot_ids)):
        for j in range(i + 1, len(robot_ids)):
            if check_collision(robot_ids[i], robot_ids[j]):
                print(3)
                return False

    return True

class SpaceMotionValidator(ob.MotionValidator):
    def __init__(self, si, robot_ids, env_obj_ids, num_joints_per_arm):
        super().__init__(si)
        self.si = si
        self.robot_ids = robot_ids
        self.num_joints_per_arm = num_joints_per_arm
        self.env_obj_ids = env_obj_ids
    def checkMotion(self, s1, s2):
        """
        检查从状态 s1 到 s2 的路径是否无碰撞。
        :param s1: 起始状态
        :param s2: 目标状态
        :return: True 如果路径无碰撞，否则 False
        """
        steps = 10  # 将路径分为 10 步进行插值检测
        state_space = self.si.getStateSpace()
        num_dimensions = state_space.getDimension()

        for i in range(steps + 1):
            # 插值生成中间状态
            interp_state = ob.State(state_space)
            for j in range(num_dimensions):
                interp_state[j] = s1[j] + (s2[j] - s1[j]) * i / steps

            # 检查插值状态是否有效
            if not isStateValid(self.si, interp_state, self.robot_ids, self.env_obj_ids, self.num_joints_per_arm, ):
                return False
        return True

class MotionPlanner_Collision:
    def __init__(self, method='RRTstar', arm_count=2, solve_time=10.0, robot_ids=None, env_obj_ids=None):
        self.arm_count = arm_count
        self.joints_per_arm = 6
        self.Dof = self.arm_count * self.joints_per_arm
        self.solve_time = solve_time
        self.robot_ids = robot_ids  # PyBullet 中的机器人 ID
        self.space = self.setup_bounds()
        self.si = ob.SpaceInformation(self.space)
        self.env_obj_ids = env_obj_ids
        # 设置碰撞检测
        self.si.setMotionValidator(SpaceMotionValidator(self.si, self.robot_ids, self.env_obj_ids, self.joints_per_arm))
        self.si.setStateValidityChecker(ob.StateValidityCheckerFn(
            partial(isStateValid, self.si, robot_ids=robot_ids, env_obj_ids=self.env_obj_ids, num_joints_per_arm=self.joints_per_arm)
        ))
        self.method = method


    def setup_bounds(self):
        lower_limits_arm = [
            -np.pi, -3 * np.pi / 2, -5 * np.pi / 6, -260 * np.pi / 180,
            -168 * np.pi / 180, -174 * np.pi / 180
        ]
        upper_limits_arm = [
            np.pi, 90 * np.pi / 180, 5 * np.pi / 6, 80 * np.pi / 180,
            168 * np.pi / 180, 174 * np.pi / 180
        ]

        Lowbound = lower_limits_arm * self.arm_count
        Highbound = upper_limits_arm * self.arm_count

        vector_space = ob.RealVectorStateSpace(self.Dof)
        bounds = ob.RealVectorBounds(self.Dof)
        
        for i in range(self.Dof):
            bounds.setLow(i, Lowbound[i])
            bounds.setHigh(i, Highbound[i])
            
        vector_space.setBounds(bounds)
        return vector_space

    def setup_problem(self, start_config, goal_config):
        pdef = ob.ProblemDefinition(self.si)
        
        start = ob.State(self.space)
        goal = ob.State(self.space)
        
        for i in range(self.Dof):
            start[i] = float(start_config[i])
            goal[i] = float(goal_config[i])
        # 检查起始状态是否有效
        if not isStateValid(self.si, start, self.robot_ids, self.env_obj_ids, self.joints_per_arm):
            raise ValueError("起始状态无效（发生碰撞或超出限制）。")
        
        # 检查目标状态是否有效
        if not isStateValid(self.si, goal, self.robot_ids, self.env_obj_ids, self.joints_per_arm):
            raise ValueError("目标状态无效（发生碰撞或超出限制）。")
    
        pdef.setStartAndGoalStates(start, goal)
        if self.method == 'RRTstar':
            planner = og.RRTstar(self.si)
        elif self.method == 'BITstar':
            planner = og.BITstar(self.si)
        elif self.method == 'RRTConnect':
            planner = og.RRTConnect(self.si)
        # planner.setRange(self.vMax)
        planner.setProblemDefinition(pdef)
        planner.setup()
        
        return planner, pdef

    def plan(self, arm_start, arm_goal):
        start_config = sum(arm_start, [])  # 展平起始位置列表
        goal_config = sum(arm_goal, [])    # 展平目标位置列表
        print("Starting configuration:", start_config)
        print("Goal configuration:", goal_config)
        
        planner, pdef = self.setup_problem(start_config, goal_config)
        result = planner.solve(self.solve_time)
        
        if result:
            path = pdef.getSolutionPath()
            print("Found path of length", path.length())
            
            # 获取并处理路径为动作列表
            actions = self.get_actions_from_path(path)
            if self.arm_count == 2:
                return actions[0], actions[1]  # 分别返回双臂动作
            else:
                return actions[0]  # 返回单臂动作
        else:
            print("No solution found.")
            return None, None if self.arm_count == 2 else None

    def get_actions_from_path(self, path):
        """将路径转换为动作列表，并按机械臂数量划分动作"""
        actions = [[] for _ in range(self.arm_count)]
        for i in range(path.getStateCount()):
            state = path.getState(i)
            for arm_index in range(self.arm_count):
                joint_start = arm_index * self.joints_per_arm
                joint_end = joint_start + self.joints_per_arm
                actions[arm_index].append([state[j] for j in range(joint_start, joint_end)])
        return actions

 