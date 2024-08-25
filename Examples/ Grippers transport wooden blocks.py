""" 
# @Author: Youbin Yao 
# @Date: 2024-08-23 11:31:25
# @Last Modified by:   Youbin Yao 
# @Last Modified time: 2024-08-23 11:31:25  
""" 
import os
import sys
sys.path.append(os.getcwd())

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(parent_dir, 'RoboticsToolBox'))

from pymycobot import ElephantRobot
from RoboticsToolBox import Bestman_Real_Elephant


if __name__=='__main__':
    # 标准姿态 [0.0, -120.0, 120.0, -90.0, -90.0, -0.0]
    # 零位 [0,-90,0,-90,0,0]
    bestman = Bestman_Real_Elephant("192.168.43.38", 5001)
    # bestman.state_on()
    bestman.check_error()
    target_trajectory = [
            [0.5729730725288391, -0.05068958178162575, 0.19427451491355896, 0.2969912886619568, -0.6381924748420715, 0.6559267044067383, -0.27251917123794556],
            [0.5860132575035095, -0.03786855563521385, 0.23408854007720947, 0.33398041129112244, -0.6333060264587402, 0.6339964866638184, -0.29228243231773376],
            [0.5932722687721252, -0.00427872221916914, 0.2615146338939667, -0.36070218682289124, 0.6458043456077576, -0.6061388254165649, 0.2922779619693756],
            [0.5901758074760437, 0.06993594765663147, 0.2914247214794159, -0.3982785642147064, 0.6554641127586365, -0.5877817869186401, 0.25739768147468567],
            [0.5728914141654968, 0.16257865726947784, 0.3004828095436096, -0.4288429319858551, 0.6845492124557495, -0.5475476384162903, 0.21835222840309143],
            [0.5349975824356079, 0.2612200081348419, 0.29423990845680237, -0.4434780478477478, 0.7227771282196045, -0.5031689405441284, 0.1665576845407486],
            [0.502289354801178, 0.32907742261886597, 0.24723584949970245, -0.3830461800098419, 0.7727231979370117, -0.4946635961532593, 0.1071559339761734],
            [0.49048033356666565, 0.33556684851646423, 0.1762292981147766, -0.33498552441596985, 0.7840493321418762, -0.5154750943183899, 0.0856550931930542],
            [0.49339544773101807, 0.334873765707016, 0.12699848413467407, -0.3229883313179016, 0.7895045876502991, -0.5138758420944214, 0.09106409549713135],
            [0.4951143264770508, 0.2605552673339844, 0.10749507695436478, -0.29603084921836853, 0.7468422651290894, -0.5826343894004822, 0.1230020746588707],
            [0.5105777978897095, 0.17725078761577606, 0.0976296067237854, -0.291313499212265, 0.7007455229759216, -0.6215708255767822, 0.19427280128002167],
            [0.5435065627098083, 0.06313088536262512, 0.11161068826913834, -0.297743558883667, 0.6461387276649475, -0.6455997824668884, 0.2775867283344269],
            [0.5594496726989746, -0.03314505144953728, 0.1478213518857956, 0.3072815537452698, -0.579055905342102, 0.6743392944335938, -0.33990997076034546],
            [0.5756168365478516, -0.06275233626365662, 0.2009924054145813, 0.34431394934654236, -0.5344999432563782, 0.6804036498069763, -0.36442917585372925],
            [0.5784454345703125, -0.12293380498886108, 0.2624003291130066, 0.3557124137878418, -0.48897919058799744, 0.6842485666275024, -0.4076419472694397],
            [0.5636951923370361, -0.21520644426345825, 0.3116268813610077, 0.3457571268081665, -0.44096311926841736, 0.6897113919258118, -0.45858660340309143],
            [0.5404208302497864, -0.31854310631752014, 0.24892762303352356, 0.2763579487800598, -0.5303931832313538, 0.6992200016975403, -0.391664057970047],
            [0.5138706564903259, -0.29581376910209656, 0.1351354867219925, 0.24635033309459686, -0.6483463048934937, 0.6676037907600403, -0.2706727385520935],
            [0.5513743758201599, -0.2239120602607727, 0.08924050629138947, -0.3200412094593048, 0.6891423463821411, -0.6101202964782715, 0.2245209813117981],
            [0.5742518901824951, -0.1305057555437088, 0.09171643108129501, -0.3716944754123688, 0.69478440284729, -0.5891148447990417, 0.17905732989311218],
            [0.5915540456771851, -0.039267927408218384, 0.10550501197576523, -0.4135972261428833, 0.6998876333236694, -0.5652957558631897, 0.1397687941789627],
            [0.6060796976089478, 0.025441067293286324, 0.11670827120542526, -0.42734822630882263, 0.7028806209564209, -0.5518678426742554, 0.1370190531015396],
            [0.6179891228675842, 0.0633757933974266, 0.13044734299182892, -0.41930848360061646, 0.6576049327850342, -0.593339741230011, 0.19920873641967773],
            [0.6067739725112915, 0.06295245885848999, 0.18496927618980408, 0.40669456124305725, -0.5851887464523315, 0.6297315955162048, -0.3091791570186615],
            [0.6095121502876282, 0.056209687143564224, 0.23338516056537628, 0.41905108094215393, -0.5604040026664734, 0.6283918023109436, -0.33980485796928406],
            [0.6121770143508911, 0.04117409512400627, 0.2725994288921356, 0.4189033806324005, -0.5498623847961426, 0.6234208345413208, -0.36540088057518005]
        ]

    bestman.move_arm_follow_target_trajectory(target_trajectory,trajectory_type='pose')
    # bestman.get_current_joint_values()
    # # bestman.open_gripper()
    # bestman.set_arm_joint_values([0.0, -120.0, 120.0, -90.0, -90.0, -0.0])
    # bestman.robot.get_coords()
    # bestman.get_current_joint_values()
    # # bestman.close_gripper()
    # bestman.power_off()
    # for i in range(1):
    #     "机器人关节运动到安全点,设定关节角度"
    #     bestman.set_arm_joint_values([94.828,-143.513,135.283,-82.969,-87.257,-44.033])

    #     "机器人笛卡尔运动到码垛抓取过渡点,设定位姿"
    #     bestman.set_arm_coords([-130.824,256.262,321.533,176.891,-0.774,-128.700], 3000)

    #     "机器人以当前坐标位置往Z轴负方向整体运动100mm,到达抓取位置"
    #     bestman.set_jog_relative("Z",-100,1500,1)

    #     "控制夹爪闭合到30mm"
    #     bestman.close_gripper(30,100)

    #     "机器人以当前坐标位置往Z轴正方向整体运动100mm,到达抓取过渡点"
    #     bestman.set_jog_relative("Z",100,1500,1)

    #     "机器人以当前坐标位置往Y轴正方向整体运动300mm,到达放置过渡点"
    #     bestman.set_jog_relative("Y",300,1500,1)

    #     "机器人以当前坐标位置往Z轴负方向整体运动100mm,到达放置位置"
    #     bestman.set_jog_relative("Z",-100,1500,1)

    #     "控制夹爪完全张开"
    #     bestman.open_gripper(100,100)
        

    #     "机器人以当前坐标位置往Z轴正方向整体运动100mm,到达放置过渡点"
    #     bestman.set_jog_relative("Z",100,1500,1)

    #     "机器人关节运动到安全点"
    #     bestman.set_arm_joint_values([94.828,-143.513,135.283,-82.969,-87.257,-44.033])