""" 
# @Author: Youbin Yao 
# @Date: 2024-11-12 17:36:39
# @Last Modified by:   Youbin Yao 
# @Last Modified time: 2024-11-12 17:36:39  
""" 
import os
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
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(parent_dir, 'Motion_Planning'))

def isStateValid(spaceInformation, state):
    p = state[0]
    return True  # 可以根据需求在这里设置状态的有效性条件

class SpaceTimeMotionValidator(ob.MotionValidator):
    def __init__(self, si):
        super().__init__(si)
        self.si = si

    def checkMotion(self, s1, s2):
        if not self.si.isValid(s2):
            return False
        delta_pos = self.si.getStateSpace().distance(s1, s2)
        return True

class MotionPlanner:
    def __init__(self, arm_count=2, vMax=0.2, solve_time=10.0):
        self.arm_count = arm_count
        self.joints_per_arm = 6
        self.Dof = self.arm_count * self.joints_per_arm
        self.vMax = vMax
        self.solve_time = solve_time
        self.space = self.setup_bounds()
        self.si = ob.SpaceInformation(self.space)
        self.si.setMotionValidator(SpaceTimeMotionValidator(self.si))
        self.si.setStateValidityChecker(ob.StateValidityCheckerFn(
            partial(isStateValid, self.si)
        ))

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
        
        pdef.setStartAndGoalStates(start, goal)
        
        planner = og.RRTstar(self.si)
        planner.setRange(self.vMax)
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

def main():
    # 定义单臂和双臂的起始和目标关节位置
    single_arm_current = [[0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]]
    single_arm_goal = [[-1.5458, -2.2948, 1.9372, -1.2128, -1.5893, 1.5112]]

    dual_arm_current = [
        [0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0],
        [0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]
    ]
    dual_arm_goal = [
        [-1.5458, -2.2948, 1.9372, -1.2128, -1.5893, 1.5112],
        [-1.5352, -1.9108, 1.7586, -1.4182, -1.5893, 1.5227]
    ]

    # 单臂规划
    print("Single-arm planning:")
    single_arm_planner = MotionPlanner(arm_count=1, vMax=0.2, solve_time=10.0)
    single_arm_actions = single_arm_planner.plan(single_arm_current, single_arm_goal)
    print("Single-arm actions:", single_arm_actions)

    # 双臂规划
    print("\nDual-arm planning:")
    dual_arm_planner = MotionPlanner(arm_count=2, vMax=0.2, solve_time=10.0)
    action1, action2 = dual_arm_planner.plan(dual_arm_current, dual_arm_goal)
    print("Dual-arm actions (Arm 1):", action1)
    print("Dual-arm actions (Arm 2):", action2)

# if __name__ == "__main__":
#     main()
