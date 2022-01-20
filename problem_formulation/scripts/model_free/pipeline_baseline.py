"""
the baseline of the occlusion-aware manipulation problem.
Takes the created problem instance where relavent information
is created, and use the constructed planner to solve the problem.
"""
from re import T
import numpy as np
from numpy.core.shape_base import block
from transformations.transformations import rotation_matrix
import pybullet as p
import transformations as tf
import cv2
from visual_utilities import *
import cam_utilities
import open3d as o3d
# from evaluation import performance_eval
from perception_pipeline import PerceptionPipeline
import pipeline_utils
import pose_generation
import rearrangement_plan
import matplotlib.pyplot as plt

import gc

import motion_planner

LOG = 1
SNAPSHOT = 0

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
        - obj_poses
        - target_obj_pose
        """
        self.problem_def = problem_def
        # obj_pcds = self.problem_def['obj_pcds']
        target_obj_pcd = self.problem_def['target_obj_pcd']
        obj_poses = self.problem_def['obj_poses']
        target_obj_pose = self.problem_def['target_obj_pose']
        obj_ids = self.problem_def['obj_ids']
        target_obj_id = self.problem_def['target_obj_id']
        camera = self.problem_def['camera']
        pid = self.problem_def['pid']
        scene_dict = self.problem_def['scene_dict']
        robot = self.problem_def['robot']
        workspace = self.problem_def['workspace']
        motion_planner = self.problem_def['motion_planner']
        # self.obj_pcds = obj_pcds
        self.target_obj_pcd = target_obj_pcd
        self.pybullet_obj_poses = obj_poses
        self.target_obj_pose = target_obj_pose
        self.pybullet_obj_ids = obj_ids
        self.target_obj_id = target_obj_id
        self.camera = camera
        self.pid = pid
        self.scene_dict = scene_dict
        self.robot = robot
        self.workspace = workspace
        self.motion_planner = motion_planner

        # obj_poses into a dictionary: obj_id to obj_pose
        obj_poses = {}
        for i in range(len(self.pybullet_obj_ids)):
            obj_poses[self.pybullet_obj_ids[i]] = self.pybullet_obj_poses[i]
        self.pybullet_obj_poses = obj_poses

        # obtain initial occlusion
        # rgb_img, depth_img, _, obj_poses, target_obj_pose = camera.sense(obj_pcds, target_obj_pcd, obj_ids, target_obj_id)        
        rgb_img, depth_img, _ = camera.sense()
        cv2.imwrite('start_img.jpg', rgb_img)


        # workspace_low = scene_dict['workspace']['region_low']
        # workspace_high = scene_dict['workspace']['region_high']
        workspace_low = workspace.region_low
        workspace_high = workspace.region_high

        resol = np.array([0.01,0.01,0.01])

        world_x = workspace_high[0]-workspace_low[0]
        world_y = workspace_high[1]-workspace_low[1]
        world_z = workspace_high[2]-workspace_low[2]
        x_base = workspace_low[0]
        y_base = workspace_low[1]
        z_base = workspace_low[2]
        x_vec = np.array([1.0,0.,0.])
        y_vec = np.array([0.,1,0.])
        z_vec = np.array([0,0,1.])

        occlusion_params = {'world_x': world_x, 'world_y': world_y, 'world_z': world_z, 'x_base': x_base, 'y_base': y_base,
                            'z_base': z_base, 'resol': resol, 'x_vec': x_vec, 'y_vec': y_vec, 'z_vec': z_vec}
        object_params = {'resol': resol, 'scale': 0.01}  # scale is used to scale the depth, not the voxel
        target_params = {'target_pybullet_id': target_obj_id}

        perception_pipeline = PerceptionPipeline(occlusion_params, object_params, target_params)


        perception_pipeline.pipeline_sim(camera, [robot.robot_id], workspace.component_ids)

        occluded = perception_pipeline.slam_system.filtered_occluded
        
        # occluded = occlusion.scene_occlusion(depth_img, None, camera.info['extrinsics'], camera.info['intrinsics'])

        occluded_dict = perception_pipeline.slam_system.filtered_occluded_dict
        occlusion_label  = perception_pipeline.slam_system.filtered_occlusion_label
        occupied_label = perception_pipeline.slam_system.occupied_label_t
        occupied_dict = perception_pipeline.slam_system.occupied_dict_t
        # occlusion_label, occupied_label, occluded_list = occlusion.label_scene_occlusion(occluded, camera.info['extrinsics'], camera.info['intrinsics'],
        #                                                                     obj_poses, obj_pcds)

        self.prev_occluded = occluded
        # self.prev_occlusion_label = occlusion_label
        self.prev_occupied_label = occupied_label
        self.prev_occlusion_label = occlusion_label
        self.prev_occluded_dict = occluded_dict
        self.prev_occupied_dict = occupied_dict
        self.prev_depth_img = depth_img

        self.perception = perception_pipeline
        # self.prev_objects = objects
        # self.prev_obj_poses = obj_poses
        # self.prev_target_obj_pose = target_obj_pose



    def transform_obj_pybullet(self, transform, move_obj_pybullet_id):
        # move the object in the pybullet using the given relative transform
        # update the pose of the object in the recording dict
        pybullet_obj_pose = transform.dot(self.pybullet_obj_poses[move_obj_pybullet_id])
        self.pybullet_obj_poses[move_obj_pybullet_id] = pybullet_obj_pose
        quat = tf.quaternion_from_matrix(pybullet_obj_pose) # w x y z
        p.resetBasePositionAndOrientation(move_obj_pybullet_id, pybullet_obj_pose[:3,3], [quat[1],quat[2],quat[3],quat[0]], physicsClientId=self.pid)

    def transform_obj_from_pose_both(self, pose, obj_id, obj_pybullet_id):
        prev_obj_voxel_pose = self.perception.slam_system.objects[obj_id].transform
        transform = pose.dot(np.linalg.inv(prev_obj_voxel_pose))
        self.perception.slam_system.objects[obj_id].update_transform_from_relative(transform)  # set transform
        self.transform_obj_pybullet(transform, obj_pybullet_id)


    def execute_traj(self, joint_dict_list):
        # given a joint_dict_list, set the robot joint values at those locations
        for i in range(len(joint_dict_list)):
            self.robot.set_joint_from_dict(joint_dict_list[i])
            # input("next point...")

    def execute_traj_with_obj(self, joint_dict_list, move_obj_idx, move_obj_pybullet_id):
        """
        obtain the relative transform of the object and the end-effector link, and the relative transform
        of the object reconstruction with end-effector link
        set the robot joint angles to different joint states, and transform object in pybullet accordingly
        at the end, set the object reconstruction transform according to the relative transform
        """
        # obtain the current robot transform
        ee_idx = self.robot.total_link_name_ind_dict[self.robot.tip_link_name]
        link_state = p.getLinkState(self.robot.robot_id, ee_idx, physicsClientId=self.robot.pybullet_id)
        pos = link_state[0]
        ori = link_state[1]
        transform = tf.quaternion_matrix([ori[3],ori[0],ori[1],ori[2]])
        transform[:3,3] = pos
        # obtain the relative transform
        pybullet_obj_in_link = np.linalg.inv(transform).dot(self.pybullet_obj_poses[move_obj_pybullet_id])
        obj_recon_in_link = np.linalg.inv(transform).dot(self.perception.slam_system.objects[move_obj_idx].transform)
        # trasnform the robot and object together in pybullet
        for i in range(len(joint_dict_list)):
            self.robot.set_joint_from_dict(joint_dict_list[i])
            link_state = p.getLinkState(self.robot.robot_id, ee_idx, physicsClientId=self.robot.pybullet_id)
            pos = link_state[0]
            ori = link_state[1]
            transform = tf.quaternion_matrix([ori[3],ori[0],ori[1],ori[2]])
            transform[:3,3] = pos

            obj_recon_transform = transform.dot(obj_recon_in_link)

            self.transform_obj_from_pose_both(obj_recon_transform, move_obj_idx, move_obj_pybullet_id)
            # # transform the object
            # pybullet_obj_transform = transform.dot(pybullet_obj_in_link)
            # quat = tf.quaternion_from_matrix(pybullet_obj_transform) # w x y z
            # p.resetBasePositionAndOrientation(move_obj_pybullet_id, pybullet_obj_transform[:3,3], [quat[1],quat[2],quat[3],quat[0]], physicsClientId=self.pid)
            # # transform the object in scene
            # obj_recon_transform = transform.dot(obj_recon_in_link)
            # input("next point...")
            
    def pipeline_sim(self):
        # sense & perceive
        self.perception.pipeline_sim(self.camera, [self.robot.robot_id], self.workspace.component_ids)

        # update data
        self.prev_occluded = self.perception.slam_system.filtered_occluded
        self.prev_occlusion_label = self.perception.slam_system.filtered_occlusion_label
        self.prev_occupied_label = self.perception.slam_system.occupied_label_t
        self.prev_occluded_dict = self.perception.slam_system.filtered_occluded_dict
        self.prev_occupied_dict = self.perception.slam_system.occupied_dict_t


    def sense_object(self, obj_id, obj_pybullet_id, center_position):
        # sense the object until uncertainty region is None

        # naive method: randomly sample one angle with max uncertainty: uncertainty is defined by the 
        # number of pixels with unseen parts or boundary
        # pixel is unseen if 
        # 1. tsdf<=min is seen first
        # 2. tsdf>=max is seen first then directly jump to tsdf<=min

        obj = self.perception.slam_system.objects[obj_id]        
        n_samples = 10
        max_uncertainty = 0
        max_angle = 0.0
        max_net_transform = obj.transform
        for i in range(n_samples):
            # sample an orientation
            angle = np.random.normal(loc=0,scale=1,size=[2])
            angle = angle / np.linalg.norm(angle)
            angle = np.arcsin(angle[1])
            # rotate around z axis
            rot_mat = tf.rotation_matrix(angle, [0,0,1])

            transform = rot_mat
            # transform[:3,:3] = rot_mat
            transform[:3,3] = center_position

            net_transform = obj.get_net_transform_from_center_frame(obj.sample_conservative_pcd(), transform)
            # check uncertainty of the object at the net transform given the camera model
            camera_extrinsics = self.camera.info['extrinsics']
            camera_intrinsics = self.camera.info['intrinsics']
            uncertainty = self.perception.slam_system.occlusion.obtain_object_uncertainty(obj, net_transform, 
                                                                                        camera_extrinsics, camera_intrinsics,
                                                                                        self.prev_depth_img.shape)
            if uncertainty > max_uncertainty:
                max_uncertainty = uncertainty
                max_angle = angle
                max_net_transform = net_transform
            print('sample %d uncertainty: ' % (i), uncertainty)
    
        # move the object
        self.transform_obj_from_pose_both(max_net_transform, obj_id, obj_pybullet_id)
        self.perception.sense_object(obj_id, self.camera, [self.robot.robot_id], self.workspace.component_ids)
        obj.set_sensed()

    def mask_pcd_with_padding(self, occ_filter, pcd_indices, padding=1):
        """
        given the transformed pcd indices in occlusion transform, add padding to the pcd and mask it as valid
        in occ_filter
        """
        valid_filter = (pcd_indices[:,0] >= padding) & (pcd_indices[:,0] < occ_filter.shape[0]-padding) & \
                        (pcd_indices[:,1] >= padding) & (pcd_indices[:,1] < occ_filter.shape[1]-padding) & \
                        (pcd_indices[:,2] >= 0) & (pcd_indices[:,2] < occ_filter.shape[2])
        pcd_indices = pcd_indices[valid_filter]
        if len(pcd_indices) == 0:
            return occ_filter
        masked_occ_filter = np.array(occ_filter)
        masked_occ_filter[pcd_indices[:,0]-1,pcd_indices[:,1],pcd_indices[:,2]] = 0
        masked_occ_filter[pcd_indices[:,0]+1,pcd_indices[:,1],pcd_indices[:,2]] = 0
        masked_occ_filter[pcd_indices[:,0],pcd_indices[:,1]-1,pcd_indices[:,2]] = 0
        masked_occ_filter[pcd_indices[:,0],pcd_indices[:,1]+1,pcd_indices[:,2]] = 0
        masked_occ_filter[pcd_indices[:,0]-1,pcd_indices[:,1]-1,pcd_indices[:,2]] = 0
        masked_occ_filter[pcd_indices[:,0]+1,pcd_indices[:,1]-1,pcd_indices[:,2]] = 0
        masked_occ_filter[pcd_indices[:,0]-1,pcd_indices[:,1]+1,pcd_indices[:,2]] = 0
        masked_occ_filter[pcd_indices[:,0]+1,pcd_indices[:,1]+1,pcd_indices[:,2]] = 0

        return masked_occ_filter

    def mask_pcd_xy_with_padding(self, occ_filter, pcd_indices, padding=1):
        """
        given the transformed pcd indices in occlusion transform, add padding to the pcd and mask it as valid
        in occ_filter
        filter out all z axis since we assume objects won't be stacked on top of each other
        """
        masked_occ_filter = np.array(occ_filter)
        valid_filter = (pcd_indices[:,0] >= 0) & (pcd_indices[:,0] < occ_filter.shape[0]) & \
                        (pcd_indices[:,1] >= 0) & (pcd_indices[:,1] < occ_filter.shape[1])
        pcd_indices = pcd_indices[valid_filter]
        masked_occ_filter[pcd_indices[:,0],pcd_indices[:,1],:] = 0

        valid_filter = (pcd_indices[:,0] >= padding) & (pcd_indices[:,0] < occ_filter.shape[0]-padding) & \
                        (pcd_indices[:,1] >= padding) & (pcd_indices[:,1] < occ_filter.shape[1]-padding)
        # valid_filter_2 = (pcd_indices[:,0] >= padding) & (pcd_indices[:,0] < occ_filter.shape[0]-padding) & \
        #                 (pcd_indices[:,1] >= padding) & (pcd_indices[:,1] < occ_filter.shape[1]-padding)                        
        pcd_indices = pcd_indices[valid_filter]
        if len(pcd_indices) == 0:
            return masked_occ_filter
        for padding_i in range(0,padding+1):
            for padding_j in range(0,padding+1):
                masked_occ_filter[pcd_indices[:,0]-padding_i,pcd_indices[:,1]-padding_j,:] = 0
                masked_occ_filter[pcd_indices[:,0]-padding_i,pcd_indices[:,1]+padding_j,:] = 0
                masked_occ_filter[pcd_indices[:,0]+padding_i,pcd_indices[:,1]-padding_j,:] = 0
                masked_occ_filter[pcd_indices[:,0]+padding_i,pcd_indices[:,1]+padding_j,:] = 0

        del valid_filter
        del pcd_indices

        return masked_occ_filter

    def set_collision_env(self, occlusion_obj_list, ignore_occlusion_list, ignore_occupied_list, padding=0):
        """
        providing the object list to check collision and the ignore list, set up the collision environment
        """
        # occlusion_filter = np.zeros(self.prev_occluded.shape).astype(bool)
        occupied_filter = np.zeros(self.prev_occupied_label.shape).astype(bool)

        # occlusion_filter = self.prev_occluded
        occlusion_filter = np.array(self.prev_occluded)


        for id in occlusion_obj_list:
            # if id == move_obj_idx:
            #     continue
            # should include occlusion induced by this object

            if id not in ignore_occlusion_list:
                occlusion_filter = occlusion_filter | self.prev_occluded_dict[id]
            if id not in ignore_occupied_list:
                occupied_filter = occupied_filter | (self.prev_occupied_dict[id])

        # occlusion_filter[self.prev_occluded_dict[move_obj_idx]] = 0

        # for id in ignore_occlusion_list:
        #     occlusion_filter[self.prev_occluded_dict[id]] = 0
        # for id in ignore_occupied_list:
        #     occupied_filter[self.prev_occupied_dict[id]] = 0

        # mask out the ignored obj        
        if padding > 0:
            for id in ignore_occupied_list:
                pcd = self.perception.slam_system.objects[id].sample_conservative_pcd()
                obj_transform = self.perception.slam_system.objects[id].transform
                pcd = obj_transform[:3,:3].dot(pcd.T).T + obj_transform[:3,3]
                transform = self.perception.slam_system.occlusion.transform
                transform = np.linalg.inv(transform)
                pcd = transform[:3,:3].dot(pcd.T).T + transform[:3,3]
                pcd = pcd / self.perception.slam_system.occlusion.resol

                pcd = np.floor(pcd).astype(int)
                
                

                occlusion_filter = self.mask_pcd_xy_with_padding(occlusion_filter, pcd, padding)
                occupied_filter = self.mask_pcd_xy_with_padding(occupied_filter, pcd, padding)
                del pcd

        self.motion_planner.set_collision_env(self.perception.slam_system.occlusion, 
                                            occlusion_filter, occupied_filter)
        del occlusion_filter
        del occupied_filter

    def plan_to_suction_pose(self, obj_id, suction_pose_in_obj, suction_joint, start_joint_dict):
        self.motion_planner = motion_planner.MotionPlanner(self.robot, self.workspace)
        self.set_collision_env(list(self.prev_occluded_dict.keys()), [obj_id], [obj_id], padding=5)
        obj = self.perception.slam_system.objects[obj_id]
        suction_joint_dict_list = self.motion_planner.suction_plan(start_joint_dict, obj.transform.dot(suction_pose_in_obj), suction_joint, self.robot)
        if len(suction_joint_dict_list) == 0:
            return [], []
            # return [], []
        # lift up
        relative_tip_pose = np.eye(4)
        relative_tip_pose[:3,3] = np.array([0,0,0.05]) # lift up by 0.05
        joint_dict_list = self.motion_planner.straight_line_motion(suction_joint_dict_list[-1], obj.transform.dot(suction_pose_in_obj), relative_tip_pose, self.robot)
        if len(joint_dict_list) == 0:
            return [], []


        return suction_joint_dict_list, joint_dict_list

    def plan_to_intermediate_pose(self, move_obj_idx, suction_pose_in_obj, 
                                    intermediate_obj_pose, intermediate_joint, start_joint_dict):
        gc.collect()            

        intermediate_joint_dict_list_1 = [start_joint_dict]
        if not self.perception.slam_system.objects[move_obj_idx].sensed:
            # first time to move the object
            current_tip_pose = self.robot.get_tip_link_pose(start_joint_dict)
            relative_tip_pose = np.eye(4)

            # get the distance from workspace

            # voxel = visualize_voxel(self.perception.slam_system.occlusion.voxel_x,
            #                 self.perception.slam_system.occlusion.voxel_y,
            #                 self.perception.slam_system.occlusion.voxel_z,
            #                 self.prev_occupied_label>0, [1,0,0])
            # o3d.visualization.draw_geometries([voxel])
            # voxel2 = visualize_voxel(self.perception.slam_system.occlusion.voxel_x,
            #                 self.perception.slam_system.occlusion.voxel_y,
            #                 self.perception.slam_system.occlusion.voxel_z,
            #                 self.prev_occluded_dict[move_obj_idx], [1,0,0])
            # o3d.visualization.draw_geometries([voxel2])


            occupied_filter = self.prev_occupied_dict[move_obj_idx]
            # occupied_filter = self.prev_occupied_label == (move_obj_idx+1)
            x_dist = self.perception.slam_system.occlusion.voxel_x[occupied_filter].max() + 1
            x_dist = x_dist * self.perception.slam_system.occlusion.resol[0]
            relative_tip_pose[:3,3] = -x_dist
            # relative_tip_pose[:3,3] = tip_pose[:3,3] - current_tip_pose[:3,3]
            relative_tip_pose[1:3,3] = 0 # only keep the x value
            self.motion_planner.clear_octomap()
            self.motion_planner.wait(2.0)
            
            joint_dict_list = self.motion_planner.straight_line_motion(start_joint_dict, current_tip_pose, relative_tip_pose, self.robot,
                                                                    collision_check=True)

            intermediate_joint_dict_list_1 = joint_dict_list

        # reset collision env: to remove the object to be moved
        self.set_collision_env(list(self.prev_occluded_dict.keys()), [move_obj_idx], [move_obj_idx], padding=5)


        joint_dict_list = self.motion_planner.suction_with_obj_plan(intermediate_joint_dict_list_1[-1], suction_pose_in_obj, 
                                                                    intermediate_joint, self.robot, 
                                                                    move_obj_idx, self.perception.slam_system.objects)

        # joint_dict_list = self.motion_planner.suction_plan(self.robot.joint_dict, tip_pose, joint_angles, self.robot)
        # given the list of [{name->val}], execute in pybullet
        # self.execute_traj_with_obj(joint_dict_list, move_obj_idx, move_obj_pybullet_id)
        del occupied_filter

        if len(joint_dict_list) == 0:
            return []
        intermediate_joint_dict_list = intermediate_joint_dict_list_1 + joint_dict_list

        return intermediate_joint_dict_list


    def obj_sense_plan(self, move_obj_idx, move_obj_pybullet_id, tip_pose_in_obj):
        planning_info = dict()
        planning_info['obj_i'] = move_obj_idx
        planning_info['pybullet_obj_i'] = move_obj_pybullet_id
        planning_info['pybullet_obj_pose'] = self.pybullet_obj_poses[move_obj_pybullet_id]

        planning_info['objects'] = self.perception.slam_system.objects
        planning_info['occlusion'] = self.perception.slam_system.occlusion
        planning_info['workspace'] = self.workspace
        planning_info['selected_tip_in_obj'] = tip_pose_in_obj
        planning_info['joint_dict'] = self.robot.joint_dict

        planning_info['robot'] = self.robot
        # planning_info['motion_planner'] = self.motion_planner
        planning_info['occluded_label'] = self.prev_occlusion_label
        planning_info['occupied_label'] = self.prev_occupied_label
        planning_info['seg_img'] = self.perception.seg_img==move_obj_pybullet_id
        planning_info['camera'] = self.camera

        sense_pose, selected_tip_in_obj, tip_pose, start_joint_angles, joint_angles = \
            pipeline_utils.sample_sense_pose(**planning_info)

        self.set_collision_env(list(self.prev_occluded_dict.keys()), [move_obj_idx], [move_obj_idx])

        joint_dict_list = self.motion_planner.suction_with_obj_plan(self.robot.joint_dict, tip_pose_in_obj, joint_angles, self.robot, 
                                                move_obj_idx, self.perception.slam_system.objects)
        return joint_dict_list

    def plan_to_placement_pose(self, move_obj_idx, move_obj_pybullet_id, tip_pose_in_obj, start_obj_pose, 
                                intermediate_obj_pose, intermediate_joint, intermediate_joint_dict_list, 
                                lift_up_joint_dict_list, suction_joint_dict_list):
        # ** loop until the object is put back
        object_put_back = False
        while True:
            planning_info = dict()
            planning_info['objects'] = self.perception.slam_system.objects
            planning_info['occupied_label'] = self.prev_occupied_label
            planning_info['occlusion'] = self.perception.slam_system.occlusion
            planning_info['occlusion_label'] = self.prev_occlusion_label
            planning_info['occluded_dict'] = self.prev_occluded_dict
            # planning_info['depth_img'] = self.prev_depth_img
            planning_info['workspace'] = self.workspace

            planning_info['robot'] = self.robot
            planning_info['gripper_tip_in_obj'] = tip_pose_in_obj

            # select where to place
            planning_info['obj_i'] = move_obj_idx
            move_obj_transform, back_joint_angles = pipeline_utils.placement_pose_generation(**planning_info)
            # voxel1 = visualize_voxel(self.perception.slam_system.occlusion.voxel_x, 
            #                         self.perception.slam_system.occlusion.voxel_y,
            #                         self.perception.slam_system.occlusion.voxel_z,
            #                         self.perception.slam_system.filtered_occluded, [1,0,0])
            # o3d.visualization.draw_geometries([voxel1])

            placement_joint_dict_list = []
            reset_joint_dict_list = []
            # if move_obj_transform is None:
            if True:
                # input('valid pose is not found...')
                if SNAPSHOT:
                    transform = np.linalg.inv(start_obj_pose).dot(self.perception.slam_system.objects[move_obj_idx].transform)
                    self.transform_obj_pybullet(transform, move_obj_pybullet_id)
                    self.perception.slam_system.objects[move_obj_idx].update_transform_from_relative(transform)  # set transform
                    input("valid pose is not found...")
                else:
                    # * step 1: plan a path to go to the intermediate pose
                    # obtain the start tip transform
                    tip_start_pose = self.perception.slam_system.objects[move_obj_idx].transform.dot(tip_pose_in_obj)
                    target_object_pose = intermediate_obj_pose
                    # do a motion planning to current sense pose

                    occlusion_filter = np.zeros(self.prev_occluded.shape).astype(bool)
                    # occlusion_filter = np.array(self.prev_occluded)
                    for id, occlusion in self.prev_occluded_dict.items():
                        if id == move_obj_idx:
                            continue
                        # should include occlusion induced by this object
                        occlusion_filter = occlusion_filter | occlusion
                    # occlusion_filter[self.prev_occluded_dict[move_obj_idx]] = 0
                    occlusion_filter[self.prev_occupied_dict[move_obj_idx]] = 0

                    self.motion_planner.set_collision_env(self.perception.slam_system.occlusion, 
                                                        occlusion_filter, (self.prev_occupied_label>0)&(self.prev_occupied_label!=move_obj_idx+1))
                    joint_dict_list = self.motion_planner.suction_with_obj_plan(self.robot.joint_dict, tip_pose_in_obj, intermediate_joint, self.robot, 
                                                            move_obj_idx, self.perception.slam_system.objects)
                    if len(joint_dict_list) == 0:
                        continue
                    placement_joint_dict_list = joint_dict_list + intermediate_joint_dict_list[::-1] + lift_up_joint_dict_list[::-1]
                    reset_joint_dict_list = suction_joint_dict_list[::-1]
                object_put_back = True
                break

            input('valid pose is found...')

            # we have a target transform
            if SNAPSHOT:
                self.transform_obj_pybullet(move_obj_transform, move_obj_pybullet_id)
                self.perception.slam_system.objects[move_obj_idx].update_transform_from_relative(move_obj_transform)  # set transform
                object_put_back = True
            else:
                # do motion planning to get to the target pose
                # plan a path to the lifted pose, then put down
                # * step 1: plan a path to the lifted pose then to target
                target_transform = move_obj_transform.dot(self.perception.slam_system.objects[move_obj_idx].transform)
                pre_transform = np.array(target_transform)
                pre_transform[2,3] = pre_transform[2,3] + 0.05
                final_tip_pose = target_transform.dot(tip_pose_in_obj)
                tip_pose = pre_transform.dot(tip_pose_in_obj)
                quat = tf.quaternion_from_matrix(tip_pose)
                
                # plan the lifting motion
                relative_tip_pose = np.eye(4)
                relative_tip_pose[:3,3] = np.array([0,0,0.05]) # lift up by 0.05
                lifting_joint_dict_list = self.motion_planner.straight_line_motion(self.robot.joint_vals_to_dict(back_joint_angles), 
                                                                                    final_tip_pose, relative_tip_pose, self.robot)
                lifting_joint_dict_list = lifting_joint_dict_list[::-1]
                pre_joint_angles = []
                # convert dict to list
                for i in range(len(self.robot.joint_names)):
                    pre_joint_angles.append(lifting_joint_dict_list[0][self.robot.joint_names[i]])

                occlusion_filter = np.zeros(self.prev_occluded.shape).astype(bool)
                # occlusion_filter = np.array(self.prev_occluded)
                for id, occlusion in self.prev_occluded_dict.items():
                    if id == move_obj_idx:
                        continue
                    # should include occlusion induced by this object
                    occlusion_filter = occlusion_filter | occlusion
                # occlusion_filter[self.prev_occluded_dict[move_obj_idx]] = 0
                occlusion_filter[self.prev_occupied_label==move_obj_idx+1] = 0

                self.motion_planner.set_collision_env(self.perception.slam_system.occlusion, 
                                                    occlusion_filter, (self.prev_occupied_label>0)&(self.prev_occupied_label!=move_obj_idx+1))



                joint_dict_list = self.motion_planner.suction_with_obj_plan(self.robot.joint_dict, tip_pose, pre_joint_angles, self.robot, 
                                                        move_obj_idx, self.perception.slam_system.objects)

                if len(joint_dict_list) > 0:
                    object_put_back = True
                else:
                    # resample
                    continue
                
                placement_joint_dict_list = joint_dict_list + lifting_joint_dict_list
                # self.execute_traj_with_obj(joint_dict_list + lifting_joint_dict_list, move_obj_idx, move_obj_pybullet_id)                
                

                if not SNAPSHOT:
                    # * step 2: reset the arm: reverse of suction plan
                    joint_dict_list = self.motion_planner.suction_plan(self.robot.init_joint_dict, tip_pose, self.robot.joint_vals, self.robot)                        
                    reset_joint_dict_list = joint_dict_list
                    # self.execute_traj(joint_dict_list[::-1])
            if object_put_back:
                break

        return placement_joint_dict_list, reset_joint_dict_list

    def move_and_sense(self, move_obj_idx, move_obj_pybullet_id, moved_objects):
        """
        move the valid object out of the workspace, sense it and the environment, and place back
        """
        _, suction_poses_in_obj, suction_joints = self.pre_move(move_obj_idx, moved_objects)

        # # obj_id, obj, robot, workspace, occlusion, occlusion_label, occupied_label
        # planning_info = dict()
        # planning_info['obj_id'] = move_obj_idx
        # planning_info['obj'] = self.perception.slam_system.objects[move_obj_idx]
        # planning_info['robot'] = self.robot
        # planning_info['workspace'] = self.workspace
        # planning_info['occlusion'] = self.perception.slam_system.occlusion
        # planning_info['occlusion_label'] = self.prev_occlusion_label
        # planning_info['occupied_label'] = self.prev_occupied_label
        # planning_info['sample_n'] = 20

        # _, suction_poses_in_obj, suction_joints = pipeline_utils.generate_start_poses(**planning_info)  # suction pose in object frame



        if len(suction_joints) == 0:  # no valid suction joint now
            return False

        start_obj_pose = self.perception.slam_system.objects[move_obj_idx].transform

        # obj_i, pybullet_obj_i, pybullet_obj_pose, objects, 
        # seg_img, occlusion, occluded_label, occupied_label, 
        # gripper_tip_poses_in_obj, suction_joints, camera, robot, workspace
        planning_info = dict()
        planning_info['obj_i'] = move_obj_idx
        planning_info['pybullet_obj_i'] = move_obj_pybullet_id
        planning_info['pybullet_obj_pose'] = self.pybullet_obj_poses[move_obj_pybullet_id]

        planning_info['objects'] = self.perception.slam_system.objects
        planning_info['occlusion'] = self.perception.slam_system.occlusion
        planning_info['workspace'] = self.workspace
        planning_info['gripper_tip_poses_in_obj'] = suction_poses_in_obj
        planning_info['suction_joints'] = suction_joints

        planning_info['robot'] = self.robot
        # planning_info['motion_planner'] = self.motion_planner
        planning_info['occluded_label'] = self.prev_occlusion_label
        planning_info['occupied_label'] = self.prev_occupied_label
        planning_info['seg_img'] = self.perception.seg_img==move_obj_pybullet_id
        planning_info['camera'] = self.camera
        intermediate_pose, suction_poses_in_obj, suction_joints, intermediate_joints = \
            pipeline_utils.generate_intermediate_poses(**planning_info)
        # generate intermediate pose for the obj with valid suction pose


        for i in range(len(suction_poses_in_obj)):
            suction_pose_in_obj = suction_poses_in_obj[i]
            suction_joint = suction_joints[i]
            intermediate_joint = intermediate_joints[i]

            pick_joint_dict_list, lift_joint_dict_list = \
                self.plan_to_suction_pose(move_obj_idx, suction_pose_in_obj, suction_joint, self.robot.joint_dict)  # internally, plan_to_pre_pose, pre_to_suction, lift up



            if len(pick_joint_dict_list) == 0:
                continue

            retreat_joint_dict_list = self.plan_to_intermediate_pose(move_obj_idx, suction_pose_in_obj, 
                                    intermediate_pose, intermediate_joint, lift_joint_dict_list[-1])
            if len(retreat_joint_dict_list) == 0:
                continue

            # TODO below
            self.execute_traj(pick_joint_dict_list)
            self.execute_traj_with_obj(lift_joint_dict_list, move_obj_idx, move_obj_pybullet_id)
            self.execute_traj_with_obj(retreat_joint_dict_list, move_obj_idx, move_obj_pybullet_id)

            self.pipeline_sim()  # sense the environmnet
            
            for k in range(8):
                obj_sense_joint_dict_list = self.obj_sense_plan(move_obj_idx, move_obj_pybullet_id, suction_pose_in_obj)
                self.execute_traj_with_obj(obj_sense_joint_dict_list, move_obj_idx, move_obj_pybullet_id)
                
                self.perception.sense_object(move_obj_idx, self.camera, [self.robot.robot_id], self.workspace.component_ids)
                self.perception.slam_system.objects[move_obj_idx].set_sensed()
                self.pipeline_sim()
                # input("after sensing")        

            planning_info = dict()
            planning_info['move_obj_idx'] = move_obj_idx
            planning_info['move_obj_pybullet_id'] = move_obj_pybullet_id

            planning_info['tip_pose_in_obj'] = suction_pose_in_obj
            planning_info['start_obj_pose'] = start_obj_pose

            planning_info['intermediate_obj_pose'] = intermediate_pose
            # planning_info['motion_planner'] = self.motion_planner
            planning_info['intermediate_joint'] = intermediate_joint
            planning_info['intermediate_joint_dict_list'] = retreat_joint_dict_list
            planning_info['lift_up_joint_dict_list'] = lift_joint_dict_list
            planning_info['suction_joint_dict_list'] = pick_joint_dict_list

            placement_joint_dict_list, reset_joint_dict_list = self.plan_to_placement_pose(**planning_info)
            self.execute_traj_with_obj(placement_joint_dict_list, move_obj_idx, move_obj_pybullet_id)
            self.execute_traj(reset_joint_dict_list)
            print('length of reset_joint_dict_list: ', len(reset_joint_dict_list))
            return True
        return False


    def pre_move(self, target_obj_i, moved_objects):
        """
        before moving the object, check reachability constraints. Rearrange the blocking objects
        """
        # * check reachability constraints by sampling grasp poses
        target_obj = self.perception.slam_system.objects[target_obj_i]
        occlusion = self.perception.slam_system.occlusion
        valid_pts, valid_orientations, valid_joints = \
            pose_generation.grasp_pose_generation(target_obj_i, target_obj, self.robot, self.workspace, 
                                                  self.perception.slam_system.occlusion, 
                                                  self.prev_occlusion_label, self.prev_occupied_label, sample_n=10)

        # * check each sampled pose to see if it's colliding with any objects. Create blocking object set
        total_blocking_objects = []
        total_blocking_object_nums = []
        joint_indices = []
        transformed_rpcds = []
        # input('before checking grasp poses... number of valid_orientations: %d' %(len(valid_orientations)))

        res_pts = []
        res_orientations = []
        res_joints = []

        # get the target object pcd
        target_pcd = target_obj.sample_conservative_pcd()
        target_pcd = target_obj.transform[:3,:3].dot(target_pcd.T).T + target_obj.transform[:3,3]
        target_pcd = occlusion.world_in_voxel_rot.dot(target_pcd.T).T + occlusion.world_in_voxel_tran
        target_pcd = target_pcd / occlusion.resol
        target_pcd = np.floor(target_pcd).astype(int)
        valid_filter = (target_pcd[:,0]>=0) & (target_pcd[:,0]<occlusion.voxel_x.shape[0]) & \
                        (target_pcd[:,1]>=0) & (target_pcd[:,1]<occlusion.voxel_x.shape[1]) & \
                        (target_pcd[:,2]>=0) & (target_pcd[:,2]<occlusion.voxel_x.shape[2])
        target_pcd = target_pcd[valid_filter]
        # obtain the mask for objects that are hiding in front of target
        # x <= pcd[:,0], y == pcd[:,1], z == pcd[:,2]
        blocking_mask = np.zeros(occlusion.voxel_x.shape).astype(bool)
        blocking_mask[target_pcd[:,0],target_pcd[:,1],target_pcd[:,2]] = 1
        blocking_mask = blocking_mask[::-1,:,:].cumsum(axis=0)
        blocking_mask = blocking_mask[::-1,:,:] > 0

        # remove interior of target_pcd
        blocking_mask = self.mask_pcd_xy_with_padding(blocking_mask, target_pcd, padding=1)

        # # visualize the blocking mask
        # vis_voxels = []
        # for obj_i, obj in self.perception.slam_system.objects.items():
        #     occupied_i = self.prev_occupied_dict[obj_i]
        #     vis_voxel = visualize_voxel(occlusion.voxel_x, occlusion.voxel_y, occlusion.voxel_z, occupied_i, [0,0,1])
        #     vis_voxels.append(vis_voxel)

        # vis_voxel = visualize_voxel(occlusion.voxel_x, occlusion.voxel_y, occlusion.voxel_z, blocking_mask, [1,0,0])
        # o3d.visualization.draw_geometries(vis_voxels + [vis_voxel])
                





        for i in range(len(valid_orientations)):
            # obtain robot pcd at the given joint
            rpcd = self.robot.get_pcd_at_joints(valid_joints[i])
            # robot pcd in the occlusion
            transformed_rpcd = occlusion.world_in_voxel_rot.dot(rpcd.T).T + occlusion.world_in_voxel_tran
            transformed_rpcd = transformed_rpcd / occlusion.resol
            trasnformed_rpcd_before_floor = transformed_rpcd


            transformed_rpcd = np.floor(transformed_rpcd).astype(int)
            valid_filter = (transformed_rpcd[:,0] >= 0) & (transformed_rpcd[:,0] < occlusion.voxel_x.shape[0]) & \
                            (transformed_rpcd[:,1] >= 0) & (transformed_rpcd[:,1] < occlusion.voxel_x.shape[1]) & \
                            (transformed_rpcd[:,2] >= 0) & (transformed_rpcd[:,2] < occlusion.voxel_x.shape[2])
            transformed_rpcd = transformed_rpcd[valid_filter]
            transformed_rpcds.append(transformed_rpcd)
            valid = True
            occupied = np.zeros(self.prev_occupied_label.shape).astype(bool)  # for vis
            occluded = np.zeros(self.prev_occupied_label.shape).astype(bool)
            if len(transformed_rpcd) == 0:
                blocking_objects = []
                valid = True
                for obj_i, obj in self.perception.slam_system.objects.items():
                    if obj_i == target_obj_i:
                        continue

                    occlusion_i = self.prev_occluded_dict[obj_i]
                    occupied_i = self.prev_occupied_dict[obj_i]
                    occupied = occupied | occupied_i  # for vis
                    occluded = occluded | occlusion_i # for vis


                    # voxel = visualize_voxel(occlusion.voxel_x, occlusion.voxel_y, occlusion.voxel_z, occupied_i, [0,1,0])
                    # pcd = visualize_pcd(trasnformed_rpcd_before_floor, [1,0,0])
                    # o3d.visualization.draw_geometries([voxel, pcd])    

            else:
                # check if colliding with any objects
                blocking_objects = []
                for obj_i, obj in self.perception.slam_system.objects.items():
                    if obj_i == target_obj_i:
                        continue
                    occlusion_i = self.prev_occluded_dict[obj_i]
                    occupied_i = self.prev_occupied_dict[obj_i]
                    occupied = occupied | occupied_i  # for vis
                    occluded = occluded | occlusion_i # for vis


                    # voxel = visualize_voxel(occlusion.voxel_x, occlusion.voxel_y, occlusion.voxel_z, occupied_i, [0,1,0])
                    # pcd = visualize_pcd(trasnformed_rpcd_before_floor, [1,0,0])
                    # o3d.visualization.draw_geometries([voxel, pcd])    


                    if occupied_i[transformed_rpcd[:,0],transformed_rpcd[:,1],transformed_rpcd[:,2]].sum() > 0:
                        blocking_objects.append(obj_i)
                        valid = False

            # * make sure there is no object in the straight-line path
            for obj_i, obj in self.perception.slam_system.objects.items():
                if obj_i == target_obj_i:
                    continue
                occupied_i = self.prev_occupied_dict[obj_i]
                if (occupied_i & blocking_mask).sum() > 0:
                    blocking_objects.append(obj_i)
                    valid = False
                

            if valid:
                # we found one grasp pose with no collisions. Move and Sense
                # input('pre_move found a valid object')
                # visualize the robot pcd in the occlusion

                res_pts.append(valid_pts[i])
                res_orientations.append(valid_orientations[i])
                res_joints.append(valid_joints[i])


            # if the blocking objects contain unmoved objects, then give up on this one
            blocking_objects = list(set(blocking_objects))
            print('blocking object: ', blocking_objects)
            print('moved_objects: ', moved_objects)
            # input('seeing blocking object')
            if len(set(blocking_objects) - set(moved_objects)) == 0:
                total_blocking_objects.append(blocking_objects)
                total_blocking_object_nums.append(len(blocking_objects))
                joint_indices.append(i)

        print('total_blocking_objects: ', total_blocking_objects)

        if len(res_orientations) > 0:
            return res_pts, res_orientations, res_joints


        if len(total_blocking_objects) == 0:
            # failure
            return [], [], []

        # * find the set of blocking objects with the minimum # of objects
        idx = np.argmin(total_blocking_object_nums)
        blocking_objects = total_blocking_objects[idx]
        valid_pt = valid_pts[joint_indices[idx]]
        valid_orientation = valid_orientations[joint_indices[idx]]
        valid_joint = valid_joints[joint_indices[idx]]
        transformed_rpcd = transformed_rpcds[joint_indices[idx]]

        moveable_objs = set(moved_objects) - set(blocking_objects)
        moveable_objs = list(moveable_objs)

        # * construct collision region
        collision_voxel = np.zeros(occlusion.voxel_x.shape).astype(bool)
        for obj_i, obj in self.perception.slam_system.objects.items():
            if obj_i in moved_objects:
                continue
            collision_voxel = collision_voxel | self.prev_occluded_dict[obj_i]


        transform = self.perception.slam_system.occlusion.transform
        transform = np.linalg.inv(transform)
        voxel_x, voxel_y, voxel_z = np.indices(collision_voxel.shape).astype(int)
        # remove start poses of objects in collision voxel
        for i in range(len(blocking_objects)):
            pcd = self.perception.slam_system.objects[blocking_objects[i]].sample_conservative_pcd()
            obj_start_pose = self.perception.slam_system.objects[blocking_objects[i]].transform
            transformed_pcd = obj_start_pose[:3,:3].dot(pcd.T).T + obj_start_pose[:3,3]
            transformed_pcd = transform[:3,:3].dot(transformed_pcd.T).T + transform[:3,3]
            transformed_pcd = transformed_pcd / self.perception.slam_system.occlusion.resol
            transformed_pcd = np.floor(transformed_pcd).astype(int)
            # valid_filter = (transformed_pcd[:,0] >= 0) & (transformed_pcd[:,0] < collision_voxel.shape[0]) & \
            #                 (transformed_pcd[:,1] >= 0) & (transformed_pcd[:,1] < collision_voxel.shape[1]) & \
            #                 (transformed_pcd[:,2] >= 0) & (transformed_pcd[:,2] < collision_voxel.shape[2])
            # transformed_pcd = transformed_pcd[valid_filter]
            # collision_voxel[transformed_pcd[:,0],transformed_pcd[:,1],transformed_pcd[:,2]] = 0  # mask out
            collision_voxel = self.mask_pcd_xy_with_padding(collision_voxel, transformed_pcd, padding=0)

            # add padding to the mask
            # valid_filter = (transformed_pcd[:,0] >= 0) & (transformed_pcd[:,0] < collision_voxel.shape[0]) & \
            #                 (transformed_pcd[:,1] >= 0) & (transformed_pcd[:,1] < collision_voxel.shape[1]) & \
            #                 (transformed_pcd[:,2] >= 2) & (transformed_pcd[:,2] < collision_voxel.shape[2])            


            # robot_collision_voxel[transformed_pcd[:,0],transformed_pcd[:,1],transformed_pcd[:,2]] = 0  # mask out
        for i in range(len(moveable_objs)):
            pcd = self.perception.slam_system.objects[moveable_objs[i]].sample_conservative_pcd()
            obj_start_pose = self.perception.slam_system.objects[moveable_objs[i]].transform
            transformed_pcd = obj_start_pose[:3,:3].dot(pcd.T).T + obj_start_pose[:3,3]
            transformed_pcd = transform[:3,:3].dot(transformed_pcd.T).T + transform[:3,3]
            transformed_pcd = transformed_pcd / self.perception.slam_system.occlusion.resol
            transformed_pcd = np.floor(transformed_pcd).astype(int)
            # valid_filter = (transformed_pcd[:,0] >= 0) & (transformed_pcd[:,0] < collision_voxel.shape[0]) & \
            #                 (transformed_pcd[:,1] >= 0) & (transformed_pcd[:,1] < collision_voxel.shape[1]) & \
            #                 (transformed_pcd[:,2] >= 0) & (transformed_pcd[:,2] < collision_voxel.shape[2])
            # transformed_pcd = transformed_pcd[valid_filter]
            # collision_voxel[transformed_pcd[:,0],transformed_pcd[:,1],transformed_pcd[:,2]] = 0  # mask out
            collision_voxel = self.mask_pcd_xy_with_padding(collision_voxel, transformed_pcd, padding=0)

            # robot_collision_voxel[transformed_pcd[:,0],transformed_pcd[:,1],transformed_pcd[:,2]] = 0  # mask out


        robot_collision_voxel = np.array(collision_voxel).astype(bool)
        robot_collision_voxel[transformed_rpcd[:,0],transformed_rpcd[:,1],transformed_rpcd[:,2]] = 1

        # * rearrange the blocking objects
        self.rearrange(blocking_objects, moveable_objs, collision_voxel, robot_collision_voxel)

        del collision_voxel
        del robot_collision_voxel
        del blocking_mask
        del target_pcd
        del valid_filter

        return [valid_pt], [valid_orientation], [valid_joint]


    def rearrange(self, obj_ids, moveable_obj_ids, collision_voxel, robot_collision_voxel):
        """
        rearrange the blocking objects
        """
        obj_pcds = [self.perception.slam_system.objects[obj_i].sample_conservative_pcd() for obj_i in obj_ids]
        moveable_obj_pcds = [self.perception.slam_system.objects[obj_i].sample_conservative_pcd() for obj_i in moveable_obj_ids]
        obj_start_poses = [self.perception.slam_system.objects[obj_i].transform for obj_i in obj_ids]
        moveable_obj_start_poses = [self.perception.slam_system.objects[obj_i].transform for obj_i in moveable_obj_ids]
        moved_objs = [self.perception.slam_system.objects[obj_i] for obj_i in obj_ids]
        moveable_objs = [self.perception.slam_system.objects[obj_i] for obj_i in moveable_obj_ids]

        plt.ion()
        plt.figure(figsize=(10,10))

        searched_objs, transfer_trajs, searched_trajs, reset_traj = \
            rearrangement_plan.rearrangement_plan(moved_objs, obj_pcds, obj_start_poses, moveable_objs, 
                                            moveable_obj_pcds, moveable_obj_start_poses,
                                            collision_voxel, robot_collision_voxel, 
                                            self.perception.slam_system.occlusion.transform, 
                                            self.perception.slam_system.occlusion.resol,
                                            self.robot, self.workspace, self.perception.slam_system.occlusion,
                                            self.motion_planner)
        total_obj_ids = obj_ids + moveable_obj_ids
        # execute the rearrangement plan
        for i in range(len(searched_objs)):
            move_obj_idx = total_obj_ids[searched_objs[i]]
            move_obj_pybullet_id = self.perception.data_assoc.obj_ids_reverse[move_obj_idx]
            self.execute_traj(transfer_trajs[i])     
            self.execute_traj_with_obj(searched_trajs[i], move_obj_idx, move_obj_pybullet_id)
        # reset
        self.execute_traj(reset_traj)        
        input('after rearrange...')

        del obj_pcds
        del moveable_obj_pcds
        del obj_start_poses
        del moveable_obj_start_poses
        del moved_objs
        del moveable_objs

    def run_pipeline(self, iter_n=10):
        # select object
        #TODO
        valid_objects = self.perception.obtain_unhidden_objects([self.robot.robot_id], self.workspace.component_ids)
        moved_objects = []
        orders = [5, 0, 6, 3, 2]
        iter_i = 0
        while True:
            gc.collect()            
            # planning_info = dict()
            # planning_info['objects'] = self.perception.slam_system.objects
            # planning_info['occupied_label'] = self.prev_occupied_label
            # planning_info['occlusion'] = self.perception.slam_system.occlusion
            # planning_info['occlusion_label'] = self.prev_occlusion_label
            # planning_info['occluded_dict'] = self.prev_occluded_dict
            # planning_info['workspace'] = self.workspace
            # planning_info['perception'] = self.perception
            # planning_info['robot'] = self.robot
            # planning_info['motion_planner'] = self.motion_planner
            # planning_info['moved_objs'] = moved_objects
            # move_obj_idx, tip_transforms_in_obj, move_obj_joints = pipeline_utils.snapshot_object_selection(**planning_info)

            # select object: active objects but not moved
            active_objs = []
            for obj_id, obj in self.perception.slam_system.objects.items():
                if obj.active:
                    active_objs.append(obj_id)
            valid_objects = set(active_objs) - set(moved_objects)
            valid_objects = list(valid_objects)

            move_obj_idx = np.random.choice(valid_objects)
            move_obj_idx = orders[iter_i]
            # move_obj_idx = 0
            iter_i += 1
            move_obj_pybullet_id = self.perception.data_assoc.obj_ids_reverse[move_obj_idx]
            res = self.move_and_sense(move_obj_idx, move_obj_pybullet_id, moved_objects)
            if res == True:
                moved_objects.append(move_obj_idx)
                self.pipeline_sim()
