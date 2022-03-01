"""
Provide a ROS node that does the following:

- publish:
  - joint state of robot
  - camera images (through ROS service)

- subscribe:
  - joint trajectory for tracking

PyBullet:
- usage:
  randomly generate a problem, 
  or load a previously generated problem instance
- load robot, and objects in the scene. Generate a scene graph to represent the problem
"""

import rospy
import object_retrieval_prob_generation as prob_gen
from sensor_msgs.msg import Image, JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from cv_bridge import CvBridge

class ExecutionSystem():
    def __init__(self, load=None, scene_name='scene1'):
        # if load is None, then randomly generate a problem instance
        # otherwise use the load as a filename to load the previously generated problem
        if load is not None:
            print('loading file: ', load)
            # load previously generated object
            f = open(load, 'rb')
            data = pickle.load(f)
            f.close()
            scene_f, obj_poses, obj_pcds, obj_shapes, obj_sizes, \
                target_pose, target_pcd, target_obj_shape, target_obj_size = data
            data = prob_gen.load_problem(scene_f, obj_poses, obj_pcds, obj_shapes, obj_sizes, 
                                target_pose, target_pcd, target_obj_shape, target_obj_size)
            pid, scene_f, robot, workspace, camera, \
                obj_poses, obj_pcds, obj_ids, \
                target_pose, target_pcd, target_obj_id = data
            f = open(scene_f, 'r')
            scene_dict = json.load(f) 
            
            print('loaded')
        else:
            scene_f = scene_name+'.json'
            data = prob_gen.random_one_problem(scene=scene_f, level=1, num_objs=7, num_hiding_objs=1)
            pid, scene_f, robot, workspace, camera, obj_poses, obj_pcds, obj_ids, obj_shapes, obj_sizes, \
                target_pose, target_pcd, target_obj_id, target_obj_shape, target_obj_size = data
            data = (scene_f, obj_poses, obj_pcds, obj_shapes, obj_sizes, target_pose, target_pcd, target_obj_shape, target_obj_size)
            save = input('save current scene? 0 - no, 1 - yes...')
            f = open(scene_f, 'r')
            scene_dict = json.load(f)
            f.close()
            if int(save) == 1:
                save_f = input('save name: ').strip()
                f = open(save_f + '.pkl', 'wb')
                pickle.dump(data, f)
                f.close()
                print('saved')

        self.pid = pid
        self.scene_dict = scene_dict
        self.robot = robot
        self.workspace = workspace
        self.camera = camera
        self.obj_poses = obj_poses
        self.obj_pcds = obj_pcds
        self.obj_ids = obj_ids
        self.obj_shapes = obj_shapes
        self.obj_sizes = obj_sizes
        self.target_pose = target_pose
        self.target_pcd = target_pcd
        self.target_obj_id = target_obj_id
        self.target_obj_shape = target_obj_shape
        self.target_obj_size = target_obj_size

        self.bridge = CvBridge()

        self.attached_obj_id = None
        # * initialize ROS services
        # - robot trajectory tracker
        rospy.Service("execute_trajectory", vbcpm_integrated_system.srv.ExecuteTrajectory, 
                      self.execute_trajectory)
        rospy.Service("attach_object", vbcpm_integrated_system.srv.AttachObject,
                      self.attach_object)



        # * initialize ROS pubs and subs
        # - camera
        # - robot_state_publisher
        self.rgb_cam_pub = rospy.Publisher('rgb_image', Image)
        self.depth_cam_pub = rospy.Publisher('depth_image', Image)
        self.seg_cam_pub = rospy.Publisher('seg_image', Image)
        self.rs_pub = rospy.Publisher('robot_state_publisher', JointState)

    def execute_trajectory(self, req):
        """
        PyBullet:
        if object attached, move arm and object at the same time
        """
        traj = req.trajectory # sensor_msgs/JointTrajectory
        # joint_dict_list = []
        joint_names = traj.joint_names
        points = traj.points

        if self.attached_obj_id is not None:
            # compute initial transformation of object and robot link
            transform = self.robot.get_tip_link_pose()

            link_state = p.getBasePositionAndOrientation(self.attached_obj_id, physicsClientId=self.robot.pybullet_id)
            pos = link_state[0]
            ori = link_state[1]
            obj_transform = tf.quaternion_matrix([ori[3],ori[0],ori[1],ori[2]])
            obj_transform[:3,3] = pos

        for i in range(len(points)):
            pos = points[i].positions
            time_from_start = points[i].time_from_start
            joint_dict = {joint_names[j]: pos[j] for j in range(len(pos))}
            self.robot.set_joint_from_dict(joint_dict)

            if self.attached_obj_id is not None:
                new_transform = self.robot.get_tip_link_pose()
                rel_transform = new_transform.dot(np.linalg.inv(transform))
                new_obj_transform = rel_transform.dot(obj_transform)
                quat = tf.quaternion_from_matrix(new_obj_transform) # w x y z
                p.resetBasePositionAndOrientation(self.attached_obj_id, new_obj_transform[:3,3], [quat[1],quat[2],quat[3],quat[0]], 
                                                    physicsClientId=self.robot.pybullet_id)
                transform = new_transform
                obj_transform = new_obj_transform
            

    def attach_object(self):
        """
        attach the object closest to the robot
        """
        min_dist = 100.0
        min_i = -1
        for i in range(len(self.obj_ids)):
            points = p.getClosestPoints(self.robot.pybullet_id, self.obj_ids[i], 0.03)
            if len(points) == 0:
                continue
            for j in range(len(points)):
                if points[j][8] < min_dist:
                    min_dist = points[j][8]
                    min_i = i
        if min_i != -1:
            self.attached_obj_id = min_i
        else:
            print('attach_object cannot find a cloest point')
            raise RuntimeError

    def publish_image(self):
        """
        obtain image from PyBullet and publish
        """
        rgb_img, depth_img, seg_img = self.camera.sense()
        msg = self.bridge.cv2_to_imgmsg(rgb_img, 'passthrough')
        self.rgb_cam_pub.publish(msg)

        msg = self.bridge.cv2_to_imgmsg(depth_img, 'passthrough')
        self.depth_cam_pub.publish(msg)

        msg = self.bridge.cv2_to_imgmsg(seg_img, 'passthrough')
        self.seg_cam_pub.publish(msg)
        
    def publish_robot_state(self):
        """
        obtain joint state from PyBullet and publish
        """
        msg = JointState()
        for name, val in self.robot.joint_dict:
            msg.name.append(name)
            msg.position.append(val)
        self.rs_pub.publish(msg)

    def run(self):
        """
        keep spinning and publishing to the ROS topics
        """
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            self.publish_robot_state()
            self.publish_image()
            rate.sleep()



import json
import random
import numpy as np
import rospkg
import pybullet as p
from workspace import Workspace
from robot import Robot
import os
import time
from camera import Camera

import open3d as o3d
from perception_pipeline import PerceptionPipeline

import transformations as tf

from visual_utilities import *
from object_retrieval_partial_obs_prob_generation import load_problem, random_one_problem
from pipeline_baseline import PipelineBaseline
from prob_planner import ProbPlanner
from motion_planner import MotionPlanner

import sys
import pickle



def main():
    if int(sys.argv[1]) > 0:
        print('argv: ', sys.argv)
        # load previously generated object
        f = open('saved_problem.pkl', 'rb')
        data = pickle.load(f)
        f.close()
        scene_f, obj_poses, obj_pcds, obj_shapes, obj_sizes, target_pose, target_pcd, target_obj_shape, target_obj_size = data
        data = load_problem(scene_f, obj_poses, obj_pcds, obj_shapes, obj_sizes, 
                            target_pose, target_pcd, target_obj_shape, target_obj_size)
        pid, scene_f, robot, workspace, camera, obj_poses, obj_pcds, obj_ids, target_pose, target_pcd, target_obj_id = data
        f = open(scene_f, 'r')
        scene_dict = json.load(f)        
        
        print('loaded')
    else:
        scene_f = 'scene1.json'
        data = random_one_problem(scene=scene_f, level=1, num_objs=7, num_hiding_objs=1)
        pid, scene_f, robot, workspace, camera, obj_poses, obj_pcds, obj_ids, obj_shapes, obj_sizes, \
            target_pose, target_pcd, target_obj_id, target_obj_shape, target_obj_size = data
        data = (scene_f, obj_poses, obj_pcds, obj_shapes, obj_sizes, target_pose, target_pcd, target_obj_shape, target_obj_size)
        save = input('save current scene? 0 - no, 1 - yes...')
        f = open(scene_f, 'r')
        scene_dict = json.load(f)
        f.close()
        if int(save) == 1:
            f = open('saved_problem.pkl', 'wb')
            pickle.dump(data, f)
            f.close()
            print('saved')

    problem_def = {}
    problem_def['pid'] = pid
    problem_def['scene_dict'] = scene_dict
    problem_def['robot'] = robot
    problem_def['workspace'] = workspace
    problem_def['camera'] = camera
    problem_def['obj_pcds'] = obj_pcds
    problem_def['obj_ids'] = obj_ids
    problem_def['target_obj_pcd'] = target_pcd
    problem_def['target_obj_id'] = target_obj_id
    problem_def['obj_poses'] = obj_poses
    problem_def['target_obj_pose']= target_pose
    motion_planner = MotionPlanner(robot, workspace)
    problem_def['motion_planner'] = motion_planner
    
    construct_occlusion_graph(obj_ids, obj_poses, camera, pid)
    # input('after constrcuting occlusion graph')

    robot.set_motion_planner(motion_planner)

    pipeline = PipelineBaseline(problem_def)

    # pipeline.solve(ProbPlanner())
    pipeline.run_pipeline(10)    

if __name__ == "__main__":
    main()

