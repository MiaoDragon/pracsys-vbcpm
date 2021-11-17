"""
the baseline of the occlusion-aware manipulation problem.
Takes the created problem instance where relavent information
is created, and use the constructed planner to solve the problem.
"""
import numpy as np
from transformations.transformations import rotation_matrix
import pybullet as p
import transformations as tf
import cv2
from visual_utilities import *
import cam_utilities
import open3d as o3d
# from evaluation import performance_eval
from perception_pipeline import PerceptionPipeline


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

        # obj_poses into a dictionary: obj_id to obj_pose
        obj_poses = {}
        for i in range(len(self.pybullet_obj_ids)):
            obj_poses[self.pybullet_obj_ids[i]] = self.pybullet_obj_poses[i]
        self.pybullet_obj_poses = obj_poses

        # obtain initial occlusion
        # rgb_img, depth_img, _, obj_poses, target_obj_pose = camera.sense(obj_pcds, target_obj_pcd, obj_ids, target_obj_id)        
        rgb_img, depth_img, _ = camera.sense()
        cv2.imwrite('start_img.jpg', rgb_img)

        workspace_low = scene_dict['workspace']['region_low']
        workspace_high = scene_dict['workspace']['region_high']

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

        perception_pipeline = PerceptionPipeline(occlusion_params, object_params)


        perception_pipeline.pipeline_sim(camera, [robot.robot_id], workspace.component_ids)

        occluded = perception_pipeline.slam_system.filtered_occluded
        
        # occluded = occlusion.scene_occlusion(depth_img, None, camera.info['extrinsics'], camera.info['intrinsics'])

        occluded_dict = perception_pipeline.slam_system.filtered_occluded_dict
        occlusion_label  = perception_pipeline.slam_system.filtered_occlusion_label
        occupied_label = perception_pipeline.slam_system.occupied_label_t
        # occlusion_label, occupied_label, occluded_list = occlusion.label_scene_occlusion(occluded, camera.info['extrinsics'], camera.info['intrinsics'],
        #                                                                     obj_poses, obj_pcds)

        self.prev_occluded = occluded
        # self.prev_occlusion_label = occlusion_label
        self.prev_occupied_label = occupied_label
        self.prev_occlusion_label = occlusion_label
        self.prev_occluded_dict = occluded_dict
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
        quat = tf.quaternion_from_matrix(pybullet_obj_pose)
        p.resetBasePositionAndOrientation(move_obj_pybullet_id, pybullet_obj_pose[:3,3], [quat[1],quat[2],quat[3],quat[0]], physicsClientId=self.pid)

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
            uncertainty = self.perception.slam_system.occlusion.obtain_object_uncertainty(obj, net_transform, camera_extrinsics, camera_intrinsics)
            if uncertainty > max_uncertainty:
                max_uncertainty = uncertainty
                max_angle = angle
                max_net_transform = net_transform
            print('sample %d uncertainty: ' % (i), uncertainty)
    

        # move the object
        prev_obj_voxel_pose = obj.transform
        transform = max_net_transform.dot(np.linalg.inv(prev_obj_voxel_pose))
        print('new transform: ', max_net_transform)
        print('max uncertainty: ', max_uncertainty)
        self.perception.slam_system.objects[obj_id].set_active()
        self.perception.slam_system.objects[obj_id].update_transform_from_relative(transform)  # set transform
        self.transform_obj_pybullet(transform, obj_pybullet_id)

        # sense the object

        # # visualize the transformed voxel
        # voxels = []
        # obj_poses = {}
        # obj_pcds = {}
        # for i, obj in self.perception.slam_system.objects.items():
        #     obj_poses[i] = obj.transform
        #     obj_pcds[i] = obj.sample_conservative_pcd()
        # occlusion = self.perception.slam_system.occlusion
        # occupied_dict = occlusion.obtain_object_occupancy(camera_extrinsics, camera_intrinsics, obj_poses, obj_pcds)

        # voxels = []
        # for i, obj in self.perception.slam_system.objects.items():
        #     if i == obj_id:
        #         pcd = obj_pcds[i]
        #         pcd = obj.transform[:3,:3].dot(pcd.T).T + obj.transform[:3,3]
        #         pcd = occlusion.world_in_voxel_rot.dot(pcd.T).T + occlusion.world_in_voxel_tran
        #         pcd = pcd / occlusion.resol
        #         pcd = visualize_pcd(pcd, [0,0,0])
        #         voxels.append(pcd)

        #         pcd = obj.sample_optimistic_pcd()
        #         pcd = obj.transform[:3,:3].dot(pcd.T).T + obj.transform[:3,3]
        #         pcd = occlusion.world_in_voxel_rot.dot(pcd.T).T + occlusion.world_in_voxel_tran
        #         pcd = pcd / occlusion.resol
        #         pcd = visualize_pcd(pcd, get_color_picks()[i])
        #         voxels.append(pcd)
        #         continue
        #     voxel1 = visualize_voxel(occlusion.voxel_x, 
        #                              occlusion.voxel_y,
        #                              occlusion.voxel_z,
        #                              occupied_dict[i], get_color_picks()[i])
        #     voxels.append(voxel1)
        # o3d.visualization.draw_geometries(voxels)            

        self.perception.sense_object(obj_id, self.camera, [self.robot.robot_id], self.workspace.component_ids)


    def solve(self, planner, iter_n=10):
        # given the planner function: planning_info -> move_object_idx, move_object_pose
        step = 0
        actions = []
        occlusions = []
        planner_samples = []
        success = 0
        for iter_i in range(iter_n):
            # until the target object becomes visible
            # set up planning info
            # planning_info = dict(self.problem_def)
            planning_info = dict()
            planning_info['objects'] = self.perception.slam_system.objects
            planning_info['occupied_label'] = self.prev_occupied_label
            planning_info['occlusion'] = self.perception.slam_system.occlusion
            planning_info['occlusion_label'] = self.prev_occlusion_label
            planning_info['occluded_dict'] = self.prev_occluded_dict
            # planning_info['depth_img'] = self.prev_depth_img
            planning_info['workspace'] = self.workspace
            # planning_info['max_sample_n'] = 1000
            move_obj_idx = planner.snapshot_object_selection(**planning_info)

            if move_obj_idx is None:
                # the plan has failed, try next iteration
                continue
            
            print('move_obj_idx: ', move_obj_idx)
            move_obj_pybullet_id = self.perception.data_assoc.obj_ids_reverse[move_obj_idx]
            # after selecting the object, move it out of the scene, and sense the scene
            intermediate_pose = np.zeros((4,4))
            intermediate_pose[3,3] = 1
            intermediate_pose[:3,:3] = np.eye(3)
            intermediate_pose[0,3] = -10.0

            prev_obj_voxel_pose = planning_info['objects'][move_obj_idx].transform
            transform = intermediate_pose.dot(np.linalg.inv(prev_obj_voxel_pose))
            self.perception.slam_system.objects[move_obj_idx].set_active()
            self.perception.slam_system.objects[move_obj_idx].update_transform_from_relative(transform)  # set transform

            self.transform_obj_pybullet(transform, move_obj_pybullet_id)

            print('after removing object %d, calling slam to update...' % (move_obj_idx))
            self.perception.pipeline_sim(self.camera, [self.robot.robot_id], self.workspace.component_ids)
            input("after planning... step...")
            # update data
            self.prev_occluded = self.perception.slam_system.filtered_occluded
            self.prev_occlusion_label = self.perception.slam_system.filtered_occlusion_label
            self.prev_occupied_label = self.perception.slam_system.occupied_label_t
            self.prev_occluded_dict = self.perception.slam_system.filtered_occluded_dict

            # sense the object
            cam_pos = self.camera.info['pos']
            cam_look_at = self.camera.info['look_at']
            world_xmin = self.workspace.region_low[0]

            vec = cam_look_at - cam_pos
            pos_x = world_xmin - 0.07
            alpha = (pos_x - cam_pos[0]) / vec[0]
            center_pos = alpha*vec + cam_pos

            for i in range(10):
                self.sense_object(move_obj_idx, move_obj_pybullet_id, center_pos)

            # input("after sensing object...")


            voxel1 = visualize_voxel(self.perception.slam_system.objects[move_obj_idx].voxel_x, 
                                    self.perception.slam_system.objects[move_obj_idx].voxel_y,
                                    self.perception.slam_system.objects[move_obj_idx].voxel_z,
                                    self.perception.slam_system.objects[move_obj_idx].get_conservative_model(), [0,0,0])
            voxel2 = visualize_voxel(self.perception.slam_system.objects[move_obj_idx].voxel_x, 
                                    self.perception.slam_system.objects[move_obj_idx].voxel_y,
                                    self.perception.slam_system.objects[move_obj_idx].voxel_z,
                                    self.perception.slam_system.objects[move_obj_idx].get_optimistic_model(), [1,0,0])

            o3d.visualization.draw_geometries([voxel1, voxel2])    


            planning_info = dict()
            planning_info['objects'] = self.perception.slam_system.objects
            planning_info['occupied_label'] = self.prev_occupied_label
            planning_info['occlusion'] = self.perception.slam_system.occlusion
            planning_info['occlusion_label'] = self.prev_occlusion_label
            planning_info['occluded_dict'] = self.prev_occluded_dict
            # planning_info['depth_img'] = self.prev_depth_img
            planning_info['workspace'] = self.workspace

            # select where to place
            planning_info['obj_i'] = move_obj_idx
            move_obj_transform = planner.snapshot_pose_generation(**planning_info)
            # voxel1 = visualize_voxel(self.perception.slam_system.occlusion.voxel_x, 
            #                         self.perception.slam_system.occlusion.voxel_y,
            #                         self.perception.slam_system.occlusion.voxel_z,
            #                         self.perception.slam_system.filtered_occluded, [1,0,0])
            # o3d.visualization.draw_geometries([voxel1])            

        

            if move_obj_transform is None:
                # reset
                transform = np.linalg.inv(transform)
                self.transform_obj_pybullet(transform, move_obj_pybullet_id)
                self.perception.slam_system.objects[move_obj_idx].update_transform_from_relative(transform)  # set transform
                input("valid pose is not found...")
                continue

            self.transform_obj_pybullet(move_obj_transform, move_obj_pybullet_id)

            self.perception.slam_system.objects[move_obj_idx].update_transform_from_relative(move_obj_transform)  # set transform

            print('move_obj_transform: ', self.pybullet_obj_poses[move_obj_pybullet_id])
            print('perception transform: ', self.perception.slam_system.objects[move_obj_idx].transform)

            input("after planning... step...")

            # sense & perceive
            self.perception.pipeline_sim(self.camera, [self.robot.robot_id], self.workspace.component_ids)


            # update data
            self.prev_occluded = self.perception.slam_system.filtered_occluded
            self.prev_occlusion_label = self.perception.slam_system.filtered_occlusion_label
            self.prev_occupied_label = self.perception.slam_system.occupied_label_t
            self.prev_occluded_dict = self.perception.slam_system.filtered_occluded_dict

            # if the target is revealed:
    
        if LOG:
            pass
            # scene_difficulty = {}
            # scene_difficulty['object_n'] = len(self.prev_obj_poses) + 1
            # fname = "experiments/evaluation.csv"
            # performance_eval(actions, occlusions, planner_samples, success, scene_difficulty, fname)