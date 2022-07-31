"""
Provide PyBullet planning scene to check IK, collisions, etc.
and implementations of task actions
"""
from robot import Robot
from workspace import Workspace
from camera import Camera
from motion_planner import MotionPlanner

import pybullet as p
import rospkg
import json
import os
import numpy as np
import copy
import gc

class PlanningSystem():
    def __init__(self, scene_name):
        """
        Create a PyBullet scene including workspace, robot and camera
        """
        # load scene definition file
        pid = p.connect(p.DIRECT)
        # pid = p.connect(p.GUI)

        f = open(scene_name+".json", 'r')
        scene_dict = json.load(f)

        rp = rospkg.RosPack()
        package_path = rp.get_path('vbcpm_execution_system')
        urdf_path = os.path.join(package_path,scene_dict['robot']['urdf'])
        joints = [0.] * 16

        ll = [-1.58, \
            -3.13, -1.90, -2.95, -2.36, -3.13, -1.90, -3.13, \
            -3.13, -1.90, -2.95, -2.36, -3.13, -1.90, -3.13] +  \
            [0.0, -0.8757, 0.0, 0.0, -0.8757, 0.0]
        ### upper limits for null space
        ul = [1.58, \
            3.13, 1.90, 2.95, 2.36, 3.13, 1.90, 3.13, \
            3.13, 1.90, 2.95, 2.36, 3.13, 1.90, 3.13] + \
            [0.8, 0.0, 0.8757, 0.81, 0.0, 0.8757]
        ### joint ranges for null space
        jr = [1.58*2, \
            6.26, 3.80, 5.90, 4.72, 6.26, 3.80, 6.26, \
            6.26, 3.80, 5.90, 4.72, 6.26, 3.80, 6.26] + \
            [0.8, 0.8757, 0.8757, 0.81, 0.8757, 0.8757]

        robot = Robot(urdf_path, scene_dict['robot']['pose']['pos'], scene_dict['robot']['pose']['ori'], 
                        ll, ul, jr, 'motoman_left_ee', 0.3015, pid, [1,1,1,1,1,1,1,1,0,0,0,0,0,0,0])


        joints = [0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ,0.0,  # left (suction)
                1.75, 0.8, 0.0, -0.66, 0.0, 0.0 ,0.0,  # right
                ]
        robot.set_joints(joints)  # 

        workspace_low = scene_dict['workspace']['region_low']
        workspace_high = scene_dict['workspace']['region_high']
        padding = scene_dict['workspace']['padding']
        workspace = Workspace(scene_dict['workspace']['pos'], scene_dict['workspace']['ori'], \
                                scene_dict['workspace']['components'], workspace_low, workspace_high, padding, \
                                pid)
        workspace_low = workspace.region_low
        workspace_high = workspace.region_high
        # camera: using information published from execution_scene
        camera = Camera()
        motion_planner = MotionPlanner(robot, workspace)\
        
        self.robot = robot
        self.workspace = workspace
        self.camera = camera
        self.motion_planner = motion_planner

        self.robot.set_motion_planner(motion_planner)
    
    def plan_to_suction_pose(self, obj, suction_pose_in_obj, suction_joint, start_joint_dict):
        # self.motion_planner = motion_planner.MotionPlanner(self.robot, self.workspace)
        suction_joint_dict_list = self.motion_planner.suction_plan(start_joint_dict, obj.transform.dot(suction_pose_in_obj), 
                                                                    suction_joint, self.robot, workspace=self.workspace)
        if len(suction_joint_dict_list) == 0:
            return [], []
            # return [], []
        # lift up
        relative_tip_pose = np.eye(4)
        relative_tip_pose[:3,3] = np.array([0,0,0.05]) # lift up by 0.05
        # print('#######################################')
        # print('start joint angle: ')
        # print(self.robot.joint_dict_to_vals(suction_joint_dict_list[-1]))
        joint_dict_list = self.motion_planner.straight_line_motion(suction_joint_dict_list[-1], obj.transform.dot(suction_pose_in_obj), 
                                                                relative_tip_pose, self.robot, workspace=self.workspace)
        # print('straight-line motion, len(joint_dict_list): ', len(joint_dict_list))
        # input('waiting...')
        if len(joint_dict_list) == 0:
            return [], []

        return suction_joint_dict_list, joint_dict_list

    def plan_to_intermediate_pose(self, move_obj_idx, obj, suction_pose_in_obj, x_dist,
                                    intermediate_obj_pose, intermediate_joint, start_joint_dict):

        intermediate_joint_dict_list_1 = [start_joint_dict]
        if not obj.sensed:
            # first time to move the object
            current_tip_pose = self.robot.get_tip_link_pose(start_joint_dict)
            relative_tip_pose = np.eye(4)

            relative_tip_pose[:3,3] = -x_dist
            # relative_tip_pose[:3,3] = tip_pose[:3,3] - current_tip_pose[:3,3]
            relative_tip_pose[1:3,3] = 0 # only keep the x value

            # self.motion_planner.clear_octomap()
            joint_dict_list = self.motion_planner.straight_line_motion(start_joint_dict, current_tip_pose, relative_tip_pose, self.robot,
                                                                    collision_check=False, workspace=self.workspace)


            intermediate_joint_dict_list_1 = joint_dict_list

        # reset collision env: to remove the object to be moved

        # self.set_collision_env(list(self.prev_occluded_dict.keys()), [move_obj_idx], [move_obj_idx], padding=3)

        if len(intermediate_joint_dict_list_1) == 0:
            return []
        joint_dict_list = self.motion_planner.suction_with_obj_plan(intermediate_joint_dict_list_1[-1], suction_pose_in_obj, 
                                                                    intermediate_joint, self.robot, 
                                                                    obj)

        gc.collect()            

        if len(joint_dict_list) == 0:
            return []
        intermediate_joint_dict_list = intermediate_joint_dict_list_1 + joint_dict_list

        return intermediate_joint_dict_list


    def obj_sense_plan(self, obj, joint_angles, tip_pose_in_obj):

        joint_dict_list = self.motion_planner.suction_with_obj_plan(self.robot.joint_dict, tip_pose_in_obj, joint_angles, self.robot, obj)

        return joint_dict_list

    def plan_to_placement_pose(self, obj, tip_pose_in_obj, 
                                intermediate_joint, intermediate_joint_dict_list, 
                                lift_up_joint_dict_list, suction_joint_dict_list):
        # ** loop until the object is put back
        object_put_back = False
        while True:
            placement_joint_dict_list = []
            reset_joint_dict_list = []
            # if move_obj_transform is None:
            if True:
                # input('valid pose is not found...')
                # * step 1: plan a path to go to the intermediate pose
                # obtain the start tip transform

                # do a motion planning to current sense pose

                joint_dict_list = self.motion_planner.suction_with_obj_plan(self.robot.joint_dict, tip_pose_in_obj, intermediate_joint, self.robot, obj)
                if len(joint_dict_list) == 0:
                    continue
                placement_joint_dict_list = joint_dict_list + intermediate_joint_dict_list[::-1] + lift_up_joint_dict_list[::-1]
                reset_joint_dict_list = suction_joint_dict_list[::-1]
                object_put_back = True
                break
        return placement_joint_dict_list, reset_joint_dict_list
    
    def wrap_angle(self, angle, ll, ul):
        angle = angle % (np.pi*2)
        if angle > np.pi:
            angle = angle - np.pi*2
        if angle < ll:
            angle = ll
        if angle > ul:
            angle = ul
        return angle
    
    def generate_rot_traj(self, start_joint_dict, waypoint, dtheta=5*np.pi/180):
        start_angle = start_joint_dict[self.robot.joint_names[7]]
        change = waypoint - start_angle
        ntheta = int(np.ceil(np.abs(change) / dtheta))
        dtheta = change / ntheta

        traj = []
        joint_dict = copy.deepcopy(start_joint_dict)
        traj.append(joint_dict)
        angle = start_angle
        for i in range(ntheta):
            joint_dict = copy.deepcopy(joint_dict)
            joint_dict[self.robot.joint_names[7]] += dtheta
            traj.append(joint_dict)

        return traj

