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
            input("next point...")

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
            input("next point...")
            
    def pipeline_sim(self):
        # sense & perceive
        self.perception.pipeline_sim(self.camera, [self.robot.robot_id], self.workspace.component_ids)

        # update data
        self.prev_occluded = self.perception.slam_system.filtered_occluded
        self.prev_occlusion_label = self.perception.slam_system.filtered_occlusion_label
        self.prev_occupied_label = self.perception.slam_system.occupied_label_t
        self.prev_occluded_dict = self.perception.slam_system.filtered_occluded_dict



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
        self.transform_obj_from_pose_both(max_net_transform, obj_id, obj_pybullet_id)

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


        # TODO: when sensing object, use the pixel where neighbor pixels have depth greater than it, Set depth to be max
        # for those pixels. Otherwise set depth to 0 (invalid places)
        self.perception.sense_object(obj_id, self.camera, [self.robot.robot_id], self.workspace.component_ids)
        obj.set_sensed()

    def solve(self, planner, iter_n=10):
        # given the planner function: planning_info -> move_object_idx, move_object_pose
        step = 0
        actions = []
        occlusions = []
        planner_samples = []
        success = 0

        # # sense & perceive
        # self.perception.pipeline_sim(self.camera, [self.robot.robot_id], self.workspace.component_ids)

        # # update data
        # self.prev_occluded = self.perception.slam_system.filtered_occluded
        # self.prev_occlusion_label = self.perception.slam_system.filtered_occlusion_label
        # self.prev_occupied_label = self.perception.slam_system.occupied_label_t
        # self.prev_occluded_dict = self.perception.slam_system.filtered_occluded_dict


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
            planning_info['workspace'] = self.workspace
            planning_info['perception'] = self.perception
            planning_info['robot'] = self.robot
            planning_info['motion_planner'] = self.motion_planner
            #TODO
            move_obj_idx, tip_transforms_in_obj, move_obj_joints = planner.snapshot_object_selection(**planning_info)

            if move_obj_idx is None:
                # the plan has failed, try next iteration
                continue
            
            print('move_obj_idx: ', move_obj_idx)
            move_obj_pybullet_id = self.perception.data_assoc.obj_ids_reverse[move_obj_idx]
            # after selecting the object, move it out of the scene, and sense the scene

            # TODO: sample a valid intermediate pose that won't occlude the other objects
            planning_info = dict()
            planning_info['obj_i'] = move_obj_idx
            planning_info['pybullet_obj_i'] = move_obj_pybullet_id
            planning_info['pybullet_obj_pose'] = self.pybullet_obj_poses[move_obj_pybullet_id]

            planning_info['objects'] = self.perception.slam_system.objects
            planning_info['occlusion'] = self.perception.slam_system.occlusion
            planning_info['workspace'] = self.workspace
            planning_info['gripper_tip_in_obj'] = tip_transforms_in_obj
            planning_info['move_obj_joints'] = move_obj_joints

            planning_info['robot'] = self.robot
            # planning_info['motion_planner'] = self.motion_planner
            planning_info['occluded_label'] = self.prev_occlusion_label
            planning_info['occupied_label'] = self.prev_occupied_label
            planning_info['seg_img'] = self.perception.seg_img==move_obj_pybullet_id
            planning_info['camera'] = self.camera

            intermediate_pose, selected_tip_in_obj, tip_pose, suction_joint_angles, intermediate_joint_angles = \
                planner.sample_intermediate_pose(**planning_info)
            tip_suction_pose = self.perception.slam_system.objects[move_obj_idx].transform.dot(selected_tip_in_obj)
            
            prev_obj_voxel_pose = self.perception.slam_system.objects[move_obj_idx].transform
            transform = intermediate_pose.dot(np.linalg.inv(prev_obj_voxel_pose))


            if SNAPSHOT:
                self.transform_obj_from_pose_both(intermediate_pose, move_obj_idx, move_obj_pybullet_id)
            else:
                # TODO: plan a path from start to goal

                occlusion_filter = np.zeros(self.prev_occluded.shape).astype(bool)
                # occlusion_filter = np.array(self.prev_occluded)
                for id, occlusion in self.prev_occluded_dict.items():
                    # if id == move_obj_idx:
                    #     continue
                    # should include occlusion induced by this object
                    occlusion_filter = occlusion_filter | occlusion
                # occlusion_filter[self.prev_occluded_dict[move_obj_idx]] = 0
                occlusion_filter[self.prev_occupied_label==move_obj_idx+1] = 0

                self.motion_planner.set_collision_env(self.perception.slam_system.occlusion, occlusion_filter, self.prev_occupied_label>0)
                joint_dict_list = self.motion_planner.suction_plan(self.robot.joint_dict, tip_suction_pose, suction_joint_angles, self.robot)
                #TODO: suction plan straight-line path needs to ignore collision with obj

                # given the list of [{name->val}], execute in pybullet
                self.execute_traj(joint_dict_list)
                suction_joint_dict_list = joint_dict_list
                input('after going to suction point...')

                # lift up the object
                relative_tip_pose = np.eye(4)
                relative_tip_pose[:3,3] = np.array([0,0,0.05]) # lift up by 0.05
                joint_dict_list = self.motion_planner.straight_line_motion(self.robot.joint_dict, tip_suction_pose, relative_tip_pose, self.robot)
                self.execute_traj_with_obj(joint_dict_list, move_obj_idx, move_obj_pybullet_id)
                lift_up_joint_dict_list = joint_dict_list
                input('after lifting up...')

                # TODO: check if it's the first time to move the object, if so, then directly retreat the object
                intermediate_joint_dict_list_1 = []
                if not self.perception.slam_system.objects[move_obj_idx].sensed:
                    # first time to move the object
                    current_tip_pose = self.robot.get_tip_link_pose()
                    relative_tip_pose = np.eye(4)

                    # get the distance from workspace
                    occupied_filter = self.prev_occupied_label == move_obj_idx+1
                    x_dist = self.perception.slam_system.occlusion.voxel_x[occupied_filter].max() + 1
                    x_dist = x_dist * self.perception.slam_system.occlusion.resol[0]
                    relative_tip_pose[:3,3] = -x_dist
                    # relative_tip_pose[:3,3] = tip_pose[:3,3] - current_tip_pose[:3,3]
                    relative_tip_pose[1:3,3] = 0 # only keep the x value
                    self.motion_planner.clear_octomap()
                    self.motion_planner.wait(2.0)
                    
                    joint_dict_list = self.motion_planner.straight_line_motion(self.robot.joint_dict, current_tip_pose, relative_tip_pose, self.robot,
                                                                            collision_check=True)
                    self.execute_traj_with_obj(joint_dict_list, move_obj_idx, move_obj_pybullet_id)

                    intermediate_joint_dict_list_1 = joint_dict_list
                    # then follow another straight-line path to get to the target
                    # current_tip_pose = self.robot.get_tip_link_pose()
                    # relative_tip_pose = np.eye(4)
                    # relative_tip_pose[:3,3] = tip_pose[:3,3] - current_tip_pose[:3,3]
                    # print('second movement, relative tip pose: ', relative_tip_pose)
                    # joint_dict_list = self.motion_planner.straight_line_motion(self.robot.joint_dict, 
                    #                         current_tip_pose, relative_tip_pose, self.robot, collision_check=True)
                    # self.execute_traj_with_obj(joint_dict_list, move_obj_idx, move_obj_pybullet_id)
                    # input("after setting")

                    # else:


                # reset collision env: to remove the object to be moved
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
                joint_dict_list = self.motion_planner.suction_with_obj_plan(self.robot.joint_dict, tip_pose, intermediate_joint_angles, self.robot, 
                                                        move_obj_idx, self.perception.slam_system.objects)

                # joint_dict_list = self.motion_planner.suction_plan(self.robot.joint_dict, tip_pose, joint_angles, self.robot)
                # given the list of [{name->val}], execute in pybullet
                self.execute_traj_with_obj(joint_dict_list, move_obj_idx, move_obj_pybullet_id)
                intermediate_joint_dict_list = intermediate_joint_dict_list_1 + joint_dict_list
                input("after setting")

            print('after removing object %d, calling slam to update...' % (move_obj_idx))
            self.pipeline_sim()
            input("after planning... step...")


            # * sense the object
            # TODO: use the arm to move the object in order to sense it
            cam_pos = self.camera.info['pos']
            cam_look_at = self.camera.info['look_at']
            world_xmin = self.workspace.region_low[0]

            vec = cam_look_at - cam_pos
            pos_x = world_xmin - 0.07
            alpha = (pos_x - cam_pos[0]) / vec[0]
            center_pos = alpha*vec + cam_pos
            
            #################################################
            #TODO:############################3 Test the following code!

            print('joint_dict_list: ')
            print(joint_dict_list)            
            if not self.perception.slam_system.objects[move_obj_idx].sensed:
                num_sense = 10
            else:
                num_sense = 3

            # each object we only sense once
            for i in range(num_sense):

                planning_info = dict()
                planning_info['obj_i'] = move_obj_idx
                planning_info['pybullet_obj_i'] = move_obj_pybullet_id
                planning_info['pybullet_obj_pose'] = self.pybullet_obj_poses[move_obj_pybullet_id]

                planning_info['objects'] = self.perception.slam_system.objects
                planning_info['occlusion'] = self.perception.slam_system.occlusion
                planning_info['workspace'] = self.workspace
                planning_info['selected_tip_in_obj'] = selected_tip_in_obj
                planning_info['joint_dict'] = self.robot.joint_dict

                planning_info['robot'] = self.robot
                # planning_info['motion_planner'] = self.motion_planner
                planning_info['occluded_label'] = self.prev_occlusion_label
                planning_info['occupied_label'] = self.prev_occupied_label
                planning_info['seg_img'] = self.perception.seg_img==move_obj_pybullet_id
                planning_info['camera'] = self.camera

                sense_pose, selected_tip_in_obj, tip_pose, start_joint_angles, joint_angles = \
                    planner.sample_sense_pose(**planning_info)
                tip_start_pose = self.perception.slam_system.objects[move_obj_idx].transform.dot(selected_tip_in_obj)


                occlusion_filter = np.zeros(self.prev_occluded.shape).astype(bool)
                # occlusion_filter = np.array(self.prev_occluded)
                for id, occlusion in self.prev_occluded_dict.items():
                    if id == move_obj_idx:
                        continue
                    # should include occlusion induced by this object
                    occlusion_filter = occlusion_filter | occlusion
                # occlusion_filter[self.prev_occluded_dict[move_obj_idx]] = 0
                occlusion_filter[self.prev_occupied_label==move_obj_idx+1] = 0

                # do a motion planning to current sense pose
                self.motion_planner.set_collision_env(self.perception.slam_system.occlusion, 
                                                    occlusion_filter, (self.prev_occupied_label>0)&(self.prev_occupied_label!=move_obj_idx+1))
                joint_dict_list = self.motion_planner.suction_with_obj_plan(self.robot.joint_dict, tip_pose, joint_angles, self.robot, 
                                                        move_obj_idx, self.perception.slam_system.objects)

                # joint_dict_list = self.motion_planner.suction_plan(self.robot.joint_dict, tip_pose, joint_angles, self.robot)
                # given the list of [{name->val}], execute in pybullet
                print('before executing the sense trajectory')
                self.execute_traj_with_obj(joint_dict_list, move_obj_idx, move_obj_pybullet_id)
                print('after executing the sense trajectory')
                self.perception.sense_object(move_obj_idx, self.camera, [self.robot.robot_id], self.workspace.component_ids)
                print('after sense object')
                self.perception.slam_system.objects[move_obj_idx].set_sensed()

                # sense the environment if the object is not hiding its location anymore
                self.pipeline_sim()

                input("after sensing")


                # self.sense_object(move_obj_idx, move_obj_pybullet_id, center_pos)

            # input("after sensing object...")


            # voxel1 = visualize_voxel(self.perception.slam_system.objects[move_obj_idx].voxel_x, 
            #                         self.perception.slam_system.objects[move_obj_idx].voxel_y,
            #                         self.perception.slam_system.objects[move_obj_idx].voxel_z,
            #                         self.perception.slam_system.objects[move_obj_idx].get_conservative_model(), [0,0,0])
            # voxel2 = visualize_voxel(self.perception.slam_system.objects[move_obj_idx].voxel_x, 
            #                         self.perception.slam_system.objects[move_obj_idx].voxel_y,
            #                         self.perception.slam_system.objects[move_obj_idx].voxel_z,
            #                         self.perception.slam_system.objects[move_obj_idx].get_optimistic_model(), [1,0,0])

            # o3d.visualization.draw_geometries([voxel1, voxel2])    




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
                planning_info['gripper_tip_in_obj'] = selected_tip_in_obj
    
                # select where to place
                planning_info['obj_i'] = move_obj_idx
                move_obj_transform, back_joint_angles = planner.snapshot_pose_generation(**planning_info)
                # voxel1 = visualize_voxel(self.perception.slam_system.occlusion.voxel_x, 
                #                         self.perception.slam_system.occlusion.voxel_y,
                #                         self.perception.slam_system.occlusion.voxel_z,
                #                         self.perception.slam_system.filtered_occluded, [1,0,0])
                # o3d.visualization.draw_geometries([voxel1])

            
                if move_obj_transform is None:
                    input('valid pose is not found...')
                    # reset
                    if SNAPSHOT:
                        transform = np.linalg.inv(transform)
                        self.transform_obj_pybullet(transform, move_obj_pybullet_id)
                        self.perception.slam_system.objects[move_obj_idx].update_transform_from_relative(transform)  # set transform
                        input("valid pose is not found...")
                    else:
                        # do a motion planning to put the object back                    
                        transform = np.linalg.inv(transform)  # this is the target object delta transform

                        # * step 1: plan a path to go to the intermediate pose
                        target_object_transform = intermediate_pose
                        # obtain the start tip transform
                        tip_start_pose = self.perception.slam_system.objects[move_obj_idx].transform.dot(selected_tip_in_obj)
                        target_object_pose = intermediate_pose
                        tip_pose = target_object_pose.dot(selected_tip_in_obj)
                        # do a motion planning to current sense pose

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
                        joint_dict_list = self.motion_planner.suction_with_obj_plan(self.robot.joint_dict, tip_pose, intermediate_joint_angles, self.robot, 
                                                                move_obj_idx, self.perception.slam_system.objects)

                        # joint_dict_list = self.motion_planner.suction_plan(self.robot.joint_dict, tip_pose, joint_angles, self.robot)
                        # given the list of [{name->val}], execute in pybullet
                        self.execute_traj_with_obj(joint_dict_list, move_obj_idx, move_obj_pybullet_id)

                        # * step 2: play back suction -> intermediate pose
                        self.execute_traj_with_obj(intermediate_joint_dict_list[::-1], move_obj_idx, move_obj_pybullet_id)
                        # * step 3: play back lifting
                        self.execute_traj_with_obj(lift_up_joint_dict_list[::-1], move_obj_idx, move_obj_pybullet_id)
                        # * step 4: play back suction trajectory
                        self.execute_traj(suction_joint_dict_list[::-1])
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
                    pre_transform[2,3] = pre_transform[2,3] + 0.08
                    final_tip_pose = target_transform.dot(selected_tip_in_obj)
                    tip_pose = pre_transform.dot(selected_tip_in_obj)
                    quat = tf.quaternion_from_matrix(tip_pose)
                    
                    # plan the lifting motion
                    relative_tip_pose = np.eye(4)
                    relative_tip_pose[:3,3] = np.array([0,0,0.08]) # lift up by 0.05
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
                    
                    self.execute_traj_with_obj(joint_dict_list + lifting_joint_dict_list, move_obj_idx, move_obj_pybullet_id)                
                    

                    if not SNAPSHOT:
                        # * step 2: reset the arm: reverse of suction plan
                        joint_dict_list = self.motion_planner.suction_plan(self.robot.init_joint_dict, tip_pose, self.robot.joint_vals, self.robot)                        
                        self.execute_traj(joint_dict_list[::-1])


                if object_put_back:
                    break


            input("after planning... step...")

            self.pipeline_sim()
            input("after planning... step...")



            # if the target is revealed:
            if self.perception.target_seg_img.sum() > 0:
                # if the target is reachable, then done
                depth_img = self.perception.depth_img
                seg_img = self.perception.seg_img
                assoc = self.perception.last_assoc
                seg_id = self.target_obj_id
                obj_id = assoc[seg_id]
                obj = self.perception.slam_system.objects[obj_id]
                if self.perception.slam_system.occlusion.object_hidden(depth_img, seg_img, assoc, seg_id, obj_id, obj):
                    # need to wait for next turn
                    continue
                # else, we're done since the object can now be retrieved
                print("task is finished. The target can be grasped")
                input("Enter to end...")
                return
        if LOG:
            pass
            # scene_difficulty = {}
            # scene_difficulty['object_n'] = len(self.prev_obj_poses) + 1
            # fname = "experiments/evaluation.csv"
            # performance_eval(actions, occlusions, planner_samples, success, scene_difficulty, fname)