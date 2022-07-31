"""
This implements the robot object, which loads the robot description file,
and provides several helper functions to plan/control the robot
"""
import os
import pybullet as p

class Robot():
    def __init__(self, pose, ori, urdf, cid):
        """
        load the robot description file, and construct the robot model
        """
        robot_id = p.loadURDF(urdf, pose, ori, useFixedBase=True, physicsClientId=cid)
        self.robot_id = robot_id

    def gripper_open(self):
        pass
    def gripper_close(self):
        pass