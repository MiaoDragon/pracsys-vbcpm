
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
