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
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from vbcpm_execution_system.srv import RobotiqControl, RobotiqControlResponse
import json


def traj_track_compare(pos_traj_dict):
    arm_joint = rospy.wait_for_message("joint_states", JointState)    
    # ** get joint names from dict
    # create a dict: name => position
    joint_state_dict = {}
    for i in range(len(arm_joint.name)):
        joint_state_dict[arm_joint.name[i]] = arm_joint.position[i]

    traj_joint_name = list(pos_traj_dict[0]['joint'].keys())
    
    cur_joint = []
    for i in range(len(traj_joint_name)):
        cur_joint.append(joint_state_dict[traj_joint_name[i]])

    # ** get start joint from traj
    start_joint = []
    for i in range(len(traj_joint_name)):
        start_joint.append(pos_traj_dict[0]['joint'][traj_joint_name[i]])

    # ** track from current state to the start state
    cur_to_start_traj = JointTrajectory()
    cur_to_start_traj.joint_names = traj_joint_name
    points = []
    start_joint = np.array(start_joint)
    cur_joint = np.array(cur_joint)
    step_sz = 4. * np.pi / 180.
    dif = start_joint - cur_joint
    dif = np.absolute(dif)
    max_dif = np.max(dif)
    num_steps = int(np.ceil(max_dif / step_sz))
    step = (start_joint - cur_joint) / num_steps
    cur_to_start_np = []
    print('num_step: ')
    print(num_steps)
    for i in range(num_steps+1):
        pos = cur_joint + step * i
        cur_to_start_np.append(pos)
        point = JointTrajectoryPoint()
        point.positions = pos
        point.time_from_start = rospy.Duration(1. * i)
        points.append(point)
    cur_to_start_np = np.array(cur_to_start_np)
    # print('current to start:')
    # print(cur_to_start_np)
    cur_to_start_traj.points = points
    rospy.loginfo('before going to start position...')

    traj_track(cur_to_start_traj)
    rospy.loginfo('reached start point.')
    try:
        input("Press Enter to Continue...")
    except:
        pass
    print('here!')
    # ** track from start using the traj
    # get position trajectory
    pos_traj = []
    print(len(pos_traj_dict))
    for i in range(len(pos_traj_dict)):
        pos_traj_i = []
        for name in traj_joint_name:
            pos_traj_i.append(pos_traj_dict[i]['joint'][name])
        pos_traj.append(pos_traj_i)
    pos_traj = np.array(pos_traj)

    traj = JointTrajectory()
    traj.joint_names = traj_joint_name
    points = []

    # below is for QCQP approach
    for i in range(len(pos_traj)):
        point = JointTrajectoryPoint()
        point.positions = pos_traj[i]
        points.append(point)
    
    traj.points = points

    result, tracked_traj, tracked_time_traj = traj_track(traj)
    rospy.loginfo('after tracking.')

    tracked_time_traj = np.array(tracked_time_traj) - tracked_time_traj[0]
    # ** plot the tracking error
done = False
def traj_track(traj):
    global done
    client = actionlib.SimpleActionClient("geometric_trajectory_server", FollowJointTrajectoryAction)
    # Waits until the action server has started up and started
    # listening for goals.
    client.wait_for_server()

    action_goal = FollowJointTrajectoryGoal()
    #goal.trajectory = left_joint_trajectory
    action_goal.trajectory = traj
    feedbacks = []
    done = False
    def feedback_cb(feedback):
        rospy.loginfo('receiving feedback...')
        feedbacks.append(feedback)
    def done_cb(state, result):
        global done
        rospy.loginfo("Action server is done. State: %s, result: %s" % (str(state), str(result)))
        done = True
    
    # Sends the goal to the action server.
    client.send_goal(action_goal, feedback_cb=feedback_cb, done_cb=done_cb)

    # # Waits for the server to finish performing the action.
    #client.wait_for_result()
    rate = rospy.Rate(10)
    joint_names = traj.joint_names
    tracked_traj = []
    tracked_time_traj = []
    while not done:
        print('not done still...')
        # obtain the joint trajectory during tracking
        arm_joint = rospy.wait_for_message("joint_states", JointState)    
        tracked_traj_i = {}
        for i in range(len(arm_joint.name)):
            tracked_traj_i[arm_joint.name[i]] = arm_joint.position[i]
        tracked_traj.append(tracked_traj_i)
        tracked_time_traj.append(arm_joint.header.stamp.secs)
        #print('done: ')
        #print(done)
        time.sleep(0.1)
    # # Prints out the result of executing the action
    result = client.get_result()
    print('result:')
    print(result)
    print("traj_track: finished tracking.")
    return result, tracked_traj, tracked_time_traj


def main():
    rospy.init_node('test_execution')
    rospy.wait_for_service('robotiq_controller')
    print('after wait_for_service.')
    robotiq_controller = rospy.ServiceProxy('robotiq_controller', RobotiqControl)
    res = robotiq_controller(["finger_joint", "left_inner_knuckle_joint", "left_inner_finger_joint", \
                            "right_outer_knuckle_joint", "right_inner_knuckle_joint", "right_inner_finger_joint"], 0.5, 0., 0., 1.)
    print('service done.')
    

    # * Test trajectory tracker

    traj_client = actionlib.SimpleActionClient('geometric_trajectory_server', control_msgs.msg.FollowJointTrajectoryAction)

    client = traj_client
    client.wait_for_server()
    print('after waiting for action server...')

    f = open('traj1.json', 'r')
    joint_traj_dict = json.load(f) # [{joint: {joint_name: joint_val}, pose: {link_name: {pos: pos_val, ori: ori_val}}}]
    ### lower limits for null space
    ll = [-3.13, -1.90, -2.95, -2.36, -3.13, -1.90, -3.13, -3.13, -1.90, -2.95, -2.36, -3.13, -1.90, -3.13]
    ll = {'arm_left_joint_1_s': ll[0], 'arm_left_joint_2_l': ll[1], 'arm_left_joint_3_e': ll[2], 'arm_left_joint_4_u': ll[3],
          'arm_left_joint_5_r': ll[4], 'arm_left_joint_6_b': ll[5], 'arm_left_joint_7_t': ll[6]}
    ### upper limits for null space
    ul = [3.13, 1.90, 2.95, 2.36, 3.13, 1.90, 3.13, 3.13, 1.90, 2.95, 2.36, 3.13, 1.90, 3.13]
    ul = {'arm_left_joint_1_s': ul[0], 'arm_left_joint_2_l': ul[1], 'arm_left_joint_3_e': ul[2], 'arm_left_joint_4_u': ul[3],
          'arm_left_joint_5_r': ul[4], 'arm_left_joint_6_b': ul[5], 'arm_left_joint_7_t': ul[6]}

    joint_names = joint_traj_dict[0]['joint'].keys()
    start_i = len(joint_traj_dict)//2
    end_i = len(joint_traj_dict)//2
    for i in range(start_i, -1, -1):
        valid = True
        for name in joint_names:
            joint_val = joint_traj_dict[i]['joint'][name]
            if joint_val < ll[name] or joint_val > ul[name]:
                valid = False
                print('joint name: %s, joint val: %f, lower bound: %f, upper bound: %f' % (name, joint_val, ll[name], ul[name]))
                break
        if not valid:
            print('invalid, i=%d' % (i))
            break
    i += 1
    start_i = i
    print('i: ')
    print(i)

    for i in range(end_i, len(joint_traj_dict)):
        valid = True
        for name in joint_names:
            joint_val = joint_traj_dict[i]['joint'][name]
            if joint_val < ll[name] or joint_val > ul[name]:
                valid = False
                print('joint name: %s, joint val: %f, lower bound: %f, upper bound: %f' % (name, joint_val, ll[name], ul[name]))
                break
        if not valid:
            print('invalid, i=%d' % (i))
            break
    i -= 1
    end_i = i
    print('i: ')
    print(i)

    joint_traj_dict = joint_traj_dict[start_i:end_i+1]
    print('start_i: ')
    print(start_i)
    print('end_i:')
    print(end_i)

    traj_track_compare(joint_traj_dict[:20])





    action_goal = FollowJointTrajectoryGoal()
    #goal.trajectory = left_joint_trajectory
    action_goal.trajectory = traj
    feedbacks = []
    self.done = False
    def feedback_cb(feedback):
        rospy.loginfo('receiving feedback...')
        feedbacks.append(feedback)
    def done_cb(state, result):
        rospy.loginfo("Action server is done. State: %s, result: %s" % (str(state), str(result)))
        self.done = True
    
    # Sends the goal to the action server.
    client.send_goal(action_goal, feedback_cb=feedback_cb, done_cb=done_cb)
    print('goal is sent.')
    # # Waits for the server to finish performing the action.
    #client.wait_for_result()
    rate = rospy.Rate(10)
    joint_names = traj.joint_names
    tracked_traj = []
    tracked_time_traj = []
    while not self.done:
        # obtain the joint trajectory during tracking
        arm_joint = rospy.wait_for_message("joint_states", JointState)    
        tracked_traj_i = {}
        for i in range(len(arm_joint.name)):
            tracked_traj_i[arm_joint.name[i]] = arm_joint.position[i]
        tracked_traj.append(tracked_traj_i)
        tracked_time_traj.append(arm_joint.header.stamp.secs)
        #print('done: ')
        #print(done)
        time.sleep(0.1)

    # # Prints out the result of executing the action
    result = client.get_result()
    print('result:')
    print(result)

    rospy.spin()


main()