"""
the baseline of the occlusion-aware manipulation problem.
Takes the created problem instance where relavent information
is created, and use the constructed planner to solve the problem.
"""
import numpy as np
import pybullet as p
import transformations as tf
import cv2
from visual_utilities import *
import cam_utilities
import open3d as o3d
from evaluation import performance_eval
LOG = 1

class PipelineBaseline():
    def __init__(self, problem_def):
        """
        problem_def: a dict containing useful information of the problem
        including: 
        - pid
        - scene_dict
        - robot
        - workspace
        - camera
        - occlusion
        - obj_pcds
        - target_obj_pcd
        - obj_ids
        - target_obj_id
        """
        self.problem_def = problem_def
        obj_pcds = self.problem_def['obj_pcds']
        target_obj_pcd = self.problem_def['target_obj_pcd']
        obj_ids = self.problem_def['obj_ids']
        target_obj_id = self.problem_def['target_obj_id']
        occlusion = self.problem_def['occlusion']
        camera = self.problem_def['camera']
        pid = self.problem_def['pid']
        scene_dict = self.problem_def['scene_dict']
        robot = self.problem_def['robot']
        workspace = self.problem_def['workspace']
        self.obj_pcds = obj_pcds
        self.target_obj_pcd = target_obj_pcd
        self.obj_ids = obj_ids
        self.target_obj_id = target_obj_id
        self.occlusion = occlusion
        self.camera = camera
        self.pid = pid
        self.scene_dict = scene_dict
        self.robot = robot
        self.workspace = workspace

        # obtain initial occlusion
        rgb_img, depth_img, _, obj_poses, target_obj_pose = camera.sense(obj_pcds, target_obj_pcd, obj_ids, target_obj_id)        

        cv2.imwrite('start_img.jpg', rgb_img)

        occluded = occlusion.scene_occlusion(depth_img, None, camera.info['extrinsics'], camera.info['intrinsics'])
        occlusion_label, occupied_label, occluded_list = occlusion.label_scene_occlusion(occluded, camera.info['extrinsics'], camera.info['intrinsics'],
                                                                            obj_poses, obj_pcds)
        self.prev_occluded = occluded
        self.prev_occlusion_label = occlusion_label
        self.prev_occupied_label = occupied_label
        self.prev_occluded_list = occluded_list
        self.prev_depth_img = depth_img
        self.prev_obj_poses = obj_poses
        self.prev_target_obj_pose = target_obj_pose

    def solve(self, planner_func, iter_n=10):
        # given the planner function: planning_info -> move_object_idx, move_object_pose
        step = 0
        actions = []
        occlusions = []
        planner_samples = []
        success = 0
        for iter_i in range(iter_n):
            # until the target object becomes visible
            # set up planning info
            planning_info = dict(self.problem_def)
            planning_info['obj_poses'] = self.prev_obj_poses
            planning_info['occlusion_label'] = self.prev_occlusion_label
            planning_info['occupied_label'] = self.prev_occupied_label
            planning_info['occluded_list'] = self.prev_occluded_list
            planning_info['depth_img'] = self.prev_depth_img
            planning_info['max_sample_n'] = 1000
            if LOG:
                move_obj_idx, move_obj_pose, planner_sample_i = planner_func(planning_info)
            else:
                move_obj_idx, move_obj_pose = planner_func(planning_info)

            if move_obj_pose is None:
                # the plan has failed, try next iteration
                continue

            # in PyBullet move the object to the desired location
            rot_mat = np.zeros((4,4))
            rot_mat[:3,:3] = move_obj_pose[:3,:3]
            rot_mat[3,3] = 1
            quat = tf.quaternion_from_matrix(rot_mat)  # w x y z
            input("after planning... step...")
            p.resetBasePositionAndOrientation(self.obj_ids[move_obj_idx], move_obj_pose[:3,3], [quat[1],quat[2],quat[3],quat[0]], physicsClientId=self.pid)
            input("after planning... step...")

            # sense
            rgb_img, depth_img, _, new_obj_poses, new_target_obj_pose = self.camera.sense(self.obj_pcds, self.target_obj_pcd, self.obj_ids, self.target_obj_id)
            step += 1
            cv2.imwrite('step_%d.jpg' % (step), rgb_img)

            obj_poses = []

            # if new_target_obj_pose is not None:
            #     break

            # merge the two obj_poses
            for i in range(len(new_obj_poses)):
                if new_obj_poses[i] is None:
                    obj_poses.append(self.prev_obj_poses[i])
                else:
                    obj_poses.append(new_obj_poses[i])

            # new occlusion
            occluded = self.occlusion.scene_occlusion(depth_img, None, self.camera.info['extrinsics'], self.camera.info['intrinsics'])
            # occluded = occlusion.get_occlusion_single_obj(camera.info['extrinsics'], obj_poses[-1], obj_pcds[-1])
            occlusion_label, occupied_label, occluded_list = self.occlusion.label_scene_occlusion(occluded, self.camera.info['extrinsics'], self.camera.info['intrinsics'],
                                                                                obj_poses, self.obj_pcds)
            # merge the previous occlusion and the current occlusion
            occlusion_label, occupied_label, occluded_list = self.occlusion.update_occlusion(self.prev_occlusion_label, self.prev_occupied_label, self.prev_occluded_list, 
                                                                                            occlusion_label, occupied_label, occluded_list)
            vis_voxel = visualize_voxel(self.occlusion.voxel_x, self.occlusion.voxel_y, self.occlusion.voxel_z, occlusion_label>0, [1,0,0])
            env_pcd, _ = cam_utilities.pcd_from_depth(self.camera.info['intrinsics'], self.camera.info['extrinsics'], depth_img, None)

            env_pcd = np.concatenate([env_pcd, np.ones(env_pcd.shape[0]).reshape((-1,1))],axis=1)
            # transform the ray vector into the voxel_grid space
            # notice that we have to divide by the resolution vector
            env_pcd = np.linalg.inv(self.occlusion.transform).dot(env_pcd.T).T
            env_pcd = env_pcd[:,:3] / self.occlusion.resol
            vis_pcd = visualize_pcd(env_pcd, [0.,0.,0.])

            o3d.visualization.draw_geometries([vis_pcd] + [vis_voxel])


            # log important information for evaluation
            actions.append((move_obj_idx, self.prev_obj_poses[move_obj_idx], move_obj_pose))
            occlusions.append((self.prev_occlusion_label>0, occlusion_label>0))
            planner_samples.append(planner_sample_i)

            # update data
            self.prev_obj_poses = obj_poses
            self.prev_occlusion_label = occlusion_label
            self.prev_occupied_label = occupied_label
            self.prev_occluded_list = occluded_list

            if new_target_obj_pose is not None:
                success = 1
                break
    
        if LOG:
            scene_difficulty = {}
            scene_difficulty['object_n'] = len(self.prev_obj_poses) + 1
            fname = "experiments/evaluation.csv"
            performance_eval(actions, occlusions, planner_samples, success, scene_difficulty, fname)