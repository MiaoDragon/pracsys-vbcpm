
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
from object_retrieval_partial_obs_prob_generation import random_one_problem
from pipeline_baseline import PipelineBaseline
from prob_planner import ProbPlanner
from motion_planner import MotionPlanner


if __name__ == "__main__":

    

    pid, scene_dict, robot, workspace, camera, obj_poses, obj_pcds, obj_ids, target_pose, target_pcd, target_obj_id = \
            random_one_problem(scene='scene1.json', level=1, num_objs=7, num_hiding_objs=1)

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
    
    robot.set_motion_planner(motion_planner)

    pipeline = PipelineBaseline(problem_def)

    pipeline.solve(ProbPlanner())