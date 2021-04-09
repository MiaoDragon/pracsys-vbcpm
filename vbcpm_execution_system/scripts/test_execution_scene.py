"""
This defines the setup in PyBullet for the Vision-Based Constrained Placement Task.
Taking input of a problem configuration (in the format of a json file), we load each
component into the PyBullet scene. We then set up the services and topics to communicate
with the planner, including a trajectory tracker service, a camera-related topic, and
a robot-related topic.
"""
import argparse
import utility
import rospy
import rospkg
import actionlib
import os
import time
import cv2
import numpy as np
from sensor_msgs.msg import JointState
import control_msgs.msg
from vbcpm_execution_system.srv import RobotiqControl, RobotiqControlResponse

def main():
    rospy.init_node('test_execution')
    rospy.wait_for_service('robotiq_controller')
    print('after wait_for_service.')
    robotiq_controller = rospy.ServiceProxy('robotiq_controller', RobotiqControl)
    res = robotiq_controller(["finger_joint", "left_inner_knuckle_joint", "left_inner_finger_joint", \
                            "right_outer_knuckle_joint", "right_inner_knuckle_joint", "right_inner_finger_joint"], 0.5, 0., 0., 1.)
    print('service done.')
    rospy.spin()


main()