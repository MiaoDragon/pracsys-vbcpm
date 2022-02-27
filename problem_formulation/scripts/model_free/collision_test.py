import motion_planning.collision_check_utils as cc_utils
import open3d as o3d

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



"""
Construct a planning environment, sample poses and check collisions
"""
f = open('saved_problem.pkl', 'rb')
data = pickle.load(f)
f.close()
scene_f, obj_poses, obj_pcds, obj_shapes, obj_sizes, target_pose, target_pcd, target_obj_shape, target_obj_size = data
data = load_problem(scene_f, obj_poses, obj_pcds, obj_shapes, obj_sizes, 
                    target_pose, target_pcd, target_obj_shape, target_obj_size)
pid, scene_f, robot, workspace, camera, obj_poses, obj_pcds, obj_ids, target_pose, target_pcd, target_obj_id = data
f = open(scene_f, 'r')
scene_dict = json.load(f)


cc_utils.robot_collision_with_voxel_env(joint_vals, robot, collision_transform, collision_voxel, voxel_resol)