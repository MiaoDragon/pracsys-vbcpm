"""
provide high-level action planning
integrate with other components
"""
from perception_system import PerceptionSystem
from planning_system import PlanningSystem
import pose_generation
# import rearrangement_plan
import pipeline_utils
from visual_utilities import *

import numpy as np
import transformations as tf
import matplotlib.pyplot as plt
import time
import gc
import cv2
import rospy

from sensor_msgs.msg import Image, JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from vbcpm_integrated_system.srv import ExecuteTrajectory, AttachObject

from message_filters import ApproximateTimeSynchronizer, Subscriber
from cv_bridge import CvBridge

from rearrangement_plan import Rearrangement


"""
variant of the algorithm of a random algorithm. Provide two task planning methods:
- random: randomly select an object to manipulate, and randomly select a placement pose
- random-multistep-lookahead: treat the random action as a primitive action, and sample multiple times to
    select the one with max info-gain

NOTE:
1. before moving check whether the object is moveable by using only reachability condition
2. decide if an object becomes valid using only information from the camera (same as task_planner)
3. set timeout so we don't do infinite times
4. first time of each object being manipulated, reconstruct it
"""
class TaskPlannerRandom():
    def __init__(self, scene_name, prob_name, trial_num, algo_type, timeout, num_obj):
        self.planning_system = PlanningSystem(scene_name)
        self.prob_name = prob_name
        self.trial_num = trial_num
        self.algo_type = algo_type
        self.timeout = timeout
        self.num_obj = num_obj

        workspace = self.planning_system.workspace
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
        target_params = {'target_pybullet_id': None}

        perception_pipeline = PerceptionSystem(occlusion_params, object_params, target_params)

        self.perception = perception_pipeline
        self.robot = self.planning_system.robot
        self.workspace = self.planning_system.workspace
        self.camera = self.planning_system.camera

        # rgb_sub = Subscriber('rgb_image', Image)
        # depth_sub = Subscriber('depth_image', Image)
        # seg_sub = Subscriber('seg_image', Image)
        # tss = ApproximateTimeSynchronizer([rgb_sub, depth_sub, seg_sub])
        # tss.registerCallback()
        self.bridge = CvBridge()
        self.attached_obj = None


        self.perception_time = 0.0
        self.motion_planning_time = 0.0
        self.pose_generation_time = 0.0
        self.ros_time = 0.0 # communication to execution scene
        self.rearrange_time = 0.0

        self.perception_calls = 0
        self.motion_planning_calls = 0
        self.pose_generation_calls = 0
        self.execution_calls = 0
        self.rearrange_calls = 0


        self.pipeline_sim()

        self.rearrange_planner = Rearrangement()

        self.num_executed_actions = 0

        self.num_collision = 0
        


    def transform_obj_from_pose_both(self, pose, obj_id):
        prev_obj_voxel_pose = self.perception.objects[obj_id].transform
        transform = pose.dot(np.linalg.inv(prev_obj_voxel_pose))
        self.perception.objects[obj_id].update_transform_from_relative(transform)  # set transform


    def execute_traj(self, joint_dict_list, ignored_obj_id=-1, duration=0.001):
        """
        call execution_system to execute the trajectory
        if an object has been attached, update the object model transform at the end
        """
        if len(joint_dict_list) == 0 or len(joint_dict_list) == 1:
            return
        
        start_time = time.time()

        # convert joint_dict_list to JointTrajectory
        traj = JointTrajectory()
        traj.joint_names = list(joint_dict_list[0].keys())
        points = []
        for i in range(len(joint_dict_list)):
            point = JointTrajectoryPoint()
            positions = []
            for name in traj.joint_names:
                if name in joint_dict_list[i]:
                    positions.append(joint_dict_list[i][name])
                else:
                    positions.append(joint_dict_list[i-1][name])
                    joint_dict_list[i][name] = joint_dict_list[i-1][name]
            point.positions = positions
            # point.time_from_start = i * 
            points.append(point)
        traj.points = points

        rospy.wait_for_service('execute_trajectory', timeout=10)
        try:
            execute_trajectory = rospy.ServiceProxy('execute_trajectory', ExecuteTrajectory)
            resp1 = execute_trajectory(traj, ignored_obj_id)
            self.num_collision += resp1.num_collision
            # print('number of collision: ', self.num_collision)
            # update object pose using the last joint angle if an object is attached
            if self.attached_obj is not None:
                start_pose = self.robot.get_tip_link_pose(joint_dict_list[0])
                end_pose = self.robot.get_tip_link_pose(joint_dict_list[-1])
                rel_transform = end_pose.dot(np.linalg.inv(start_pose))
                self.perception.objects[self.attached_obj].update_transform_from_relative(rel_transform)
            # update the planning scene
            for i in range(len(joint_dict_list)):
                self.robot.set_joint_from_dict(joint_dict_list[i])
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)

        self.ros_time += time.time() - start_time
        self.execution_calls += 1
    def attach_obj(self, obj_id):
        """
        call execution_system to attach the object
        """
        start_time = time.time()
        rospy.wait_for_service('attach_object', timeout=10)
        try:
            attach_object = rospy.ServiceProxy('attach_object', AttachObject)
            resp1 = attach_object(True, self.perception.data_assoc.obj_ids_reverse[obj_id])
            self.attached_obj = obj_id
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)
        self.ros_time += time.time() - start_time
        self.execution_calls += 1
    def detach_obj(self):
        """
        call execution_system to detach the object
        UPDATE April 14, 2022:
        each action is finished with a detach action. So we count
        how many detach is called, this will indicate how many actions
        are performed
        """
        start_time = time.time()

        rospy.wait_for_service('attach_object', timeout=10)
        try:
            attach_object = rospy.ServiceProxy('attach_object', AttachObject)
            resp1 = attach_object(False, -1)
            self.attached_obj = None
            self.num_executed_actions += 1

        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)
        self.ros_time += time.time() - start_time
        self.execution_calls += 1
    def get_image(self):
        print('waiting for message...')
        start_time = time.time()
        # rospy.sleep(0.2)

        color_img = rospy.wait_for_message('rgb_image', Image, timeout=10)
        depth_img = rospy.wait_for_message('depth_image', Image, timeout=10)
        seg_img = rospy.wait_for_message('seg_image', Image, timeout=10)


        color_img = self.bridge.imgmsg_to_cv2(color_img, 'passthrough')
        depth_img = self.bridge.imgmsg_to_cv2(depth_img, 'passthrough')
        seg_img = self.bridge.imgmsg_to_cv2(seg_img, 'passthrough')

        self.ros_time += time.time() - start_time
        self.execution_calls += 1

        # visualize the images
        # cv2.imshow('img', color_img)
        # print('Press space...')
        # cv2.waitKey()

        return color_img, depth_img, seg_img

    def pipeline_sim(self):
        # sense & perceive
        # wait for image to update
        color_img, depth_img, seg_img = self.get_image()
        start_time = time.time()
        self.perception.pipeline_sim(color_img, depth_img, seg_img, self.camera, 
                                    [self.robot.robot_id], self.workspace.component_ids)

        # update data
        self.prev_occluded = self.perception.filtered_occluded
        self.prev_occlusion_label = self.perception.filtered_occlusion_label
        self.prev_occupied_label = self.perception.occupied_label_t
        self.prev_occluded_dict = self.perception.filtered_occluded_dict
        self.prev_occupied_dict = self.perception.occupied_dict_t

        self.perception_time += time.time() - start_time
        self.perception_calls += 1

    def sense_object(self, obj_id, camera, robot_ids, component_ids):
        color_img, depth_img, seg_img = self.get_image()
        start_time = time.time()
        self.perception.sense_object(obj_id, color_img, depth_img, seg_img, 
                                    self.camera, [self.robot.robot_id], self.workspace.component_ids)
        self.perception_time += time.time() - start_time
        self.perception_calls += 1


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

        start_time = time.time()
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

        # mask out the ignored obj        
        if padding > 0:

            for id in ignore_occupied_list:
                pcd = self.perception.objects[id].sample_conservative_pcd()
                obj_transform = self.perception.objects[id].transform
                pcd = obj_transform[:3,:3].dot(pcd.T).T + obj_transform[:3,3]
                transform = self.perception.occlusion.transform
                transform = np.linalg.inv(transform)
                pcd = transform[:3,:3].dot(pcd.T).T + transform[:3,3]
                pcd = pcd / self.perception.occlusion.resol

                pcd = np.floor(pcd).astype(int)
                
            
                occlusion_filter = self.mask_pcd_xy_with_padding(occlusion_filter, pcd, padding)
                occupied_filter = self.mask_pcd_xy_with_padding(occupied_filter, pcd, padding)
                del pcd

        start_time = time.time()
        self.planning_system.motion_planner.set_collision_env(self.perception.occlusion, 
                                            occlusion_filter, occupied_filter)
        self.motion_planning_time += time.time() - start_time
        # self.motion_planning_calls += 1
        del occlusion_filter
        del occupied_filter

        gc.collect()

        end_time = time.time()
        print('set_collision_env takes time: ', end_time - start_time)

    
    def wrap_angle(self, angle, ll, ul):
        angle = angle % (np.pi*2)
        if angle > np.pi:
            angle = angle - np.pi*2
        if angle < ll:
            angle = ll
        if angle > ul:
            angle = ul
        return angle
    
    def generate_rot_traj(self, start_joint_dict, waypoint, dtheta=5*np.pi/180):
        start_angle = start_joint_dict[self.robot.joint_names[7]]
        change = waypoint - start_angle
        ntheta = int(np.ceil(np.abs(change) / dtheta))
        dtheta = change / ntheta

        traj = []
        joint_dict = copy.deepcopy(start_joint_dict)
        traj.append(joint_dict)
        angle = start_angle
        for i in range(ntheta):
            joint_dict = copy.deepcopy(joint_dict)
            joint_dict[self.robot.joint_names[7]] += dtheta
            traj.append(joint_dict)

        return traj



    def move_and_sense(self, move_obj_idx, action_info, collision_voxel):
        """
        move the valid object out of the workspace, sense it and the environment, and place back
        """

        start_time = time.time()
        self.planning_system.motion_planner.clear_octomap()
        self.set_collision_env_with_mask(collision_voxel, [move_obj_idx], 
                                            [self.perception.objects[move_obj_idx].transform],
                                            padding=3)
        self.motion_planning_time += time.time() - start_time

        planning_info = action_info
        suction_pose_in_obj = planning_info['tip_pose_in_obj']

        retreat_joint_dict_list = planning_info['intermediate_joint_dict_list']
        lift_joint_dict_list = planning_info['lift_up_joint_dict_list']
        pick_joint_dict_list = planning_info['suction_joint_dict_list']


        self.execute_traj(pick_joint_dict_list, self.perception.data_assoc.obj_ids_reverse[move_obj_idx], duration=0.3)
        self.attach_obj(move_obj_idx)
        self.execute_traj(lift_joint_dict_list + retreat_joint_dict_list)
            
        self.pipeline_sim()  # sense the environmnet

        for k in range(6):
            planning_info = dict()
            planning_info['obj_i'] = move_obj_idx
            planning_info['objects'] = self.perception.objects
            planning_info['occlusion'] = self.perception.occlusion
            planning_info['workspace'] = self.workspace
            planning_info['selected_tip_in_obj'] = suction_pose_in_obj
            planning_info['joint_dict'] = self.robot.joint_dict

            planning_info['robot'] = self.robot
            planning_info['occluded_label'] = self.prev_occlusion_label
            planning_info['occupied_label'] = self.prev_occupied_label
            # TODO: segmentation now does not have access to PyBullet seg id
            planning_info['camera'] = self.camera
            # self.motion_planner.clear_octomap()
            start_time = time.time()
            sense_pose, selected_tip_in_obj, tip_pose, start_joint_angles, joint_angles = \
                pipeline_utils.sample_sense_pose(**planning_info)
            self.pose_generation_time += time.time() - start_time
            self.pose_generation_calls += 1
            print('sample sense pose takes time: ', time.time() - start_time)
            start_time = time.time()
            obj_sense_joint_dict_list = self.planning_system.obj_sense_plan(self.perception.objects[move_obj_idx], joint_angles, suction_pose_in_obj)
            end_time = time.time()
            self.motion_planning_time += end_time - start_time
            self.motion_planning_calls += 1                
            print('obj_sense_plan takes time: ', end_time - start_time)

            self.execute_traj(obj_sense_joint_dict_list)
            
            start_time = time.time()
            self.sense_object(move_obj_idx, self.camera, [self.robot.robot_id], self.workspace.component_ids)
            end_time = time.time()
            print('sense_object takes time: ', end_time - start_time)

            if len(obj_sense_joint_dict_list) == 0:
                continue

            # rotate the object 360 degrees so we get a better sensing
            ul = self.robot.upper_lim[7]
            ll = self.robot.lower_lim[7]
            current_angle = self.robot.joint_vals[7]
            waypoint1 = current_angle + np.pi/2
            waypoint2 = current_angle + np.pi
            waypoint3 = current_angle - np.pi/2
            
            # make sure the waypoints are within limit
            # first put in the range of -pi to pi
            waypoint1 = self.wrap_angle(waypoint1, ll, ul)
            waypoint2 = self.wrap_angle(waypoint2, ll, ul)
            waypoint3 = self.wrap_angle(waypoint3, ll, ul)

            # generate rotation trajectory
            last_joint_dict = obj_sense_joint_dict_list[-1]
            traj1 = self.generate_rot_traj(last_joint_dict, waypoint1)
            self.execute_traj(traj1)
            # self.pipeline_sim()
            self.sense_object(move_obj_idx, self.camera, [self.robot.robot_id], self.workspace.component_ids)


            traj2 = self.generate_rot_traj(traj1[-1], waypoint2)
            self.execute_traj(traj2)
            # self.pipeline_sim()
            self.sense_object(move_obj_idx, self.camera, [self.robot.robot_id], self.workspace.component_ids)

            traj3 = self.generate_rot_traj(traj2[-1], waypoint3)
            self.execute_traj(traj3)
            # self.pipeline_sim()
            self.sense_object(move_obj_idx, self.camera, [self.robot.robot_id], self.workspace.component_ids)


            traj4 = self.generate_rot_traj(traj3[-1], current_angle)
            self.execute_traj(traj4)

            self.perception.objects[move_obj_idx].set_sensed()
            # self.pipeline_sim()

        planning_info = dict()
        planning_info['tip_pose_in_obj'] = action_info['tip_pose_in_obj']

        planning_info['intermediate_joint'] = action_info['intermediate_joint'] 
        planning_info['intermediate_joint_dict_list'] = action_info['intermediate_joint_dict_list']
        planning_info['lift_up_joint_dict_list'] = action_info['lift_up_joint_dict_list']
        planning_info['suction_joint_dict_list'] = action_info['suction_joint_dict_list']
        planning_info['obj'] = action_info['obj']

        start_time = time.time()
        placement_joint_dict_list, reset_joint_dict_list = self.planning_system.plan_to_placement_pose(**planning_info)
        end_time = time.time()
        self.motion_planning_time += end_time - start_time
        self.motion_planning_calls += 1
        print('plan_to_placement_pose takes time: ', end_time - start_time)

        self.execute_traj(placement_joint_dict_list)
        self.detach_obj()

        self.execute_traj(reset_joint_dict_list, self.perception.data_assoc.obj_ids_reverse[move_obj_idx])


    def obtain_straight_blocking_mask(self, target_obj):
        target_pcd = target_obj.sample_conservative_pcd()
        obj_transform = target_obj.transform
        transform = self.perception.occlusion.transform
        transform = np.linalg.inv(transform)
        target_pcd = obj_transform[:3,:3].dot(target_pcd.T).T + obj_transform[:3,3]
        target_pcd = transform[:3,:3].dot(target_pcd.T).T + transform[:3,3]
        target_pcd = target_pcd / self.perception.occlusion.resol
        target_pcd = np.floor(target_pcd).astype(int)

        blocking_mask = np.zeros(self.perception.occlusion.voxel_x.shape).astype(bool)
        valid_filter = (target_pcd[:,0]>=0) & (target_pcd[:,0]<blocking_mask.shape[0]) & \
                        (target_pcd[:,1]>=0) & (target_pcd[:,1]<blocking_mask.shape[1]) & \
                        (target_pcd[:,2]>=0) & (target_pcd[:,2]<blocking_mask.shape[2])                            
        target_pcd = target_pcd[valid_filter]

        blocking_mask[target_pcd[:,0],target_pcd[:,1],target_pcd[:,2]] = 1
        blocking_mask = blocking_mask[::-1,:,:].cumsum(axis=0)
        blocking_mask = blocking_mask[::-1,:,:] > 0

        # remove interior of target_pcd
        blocking_mask = self.mask_pcd_xy_with_padding(blocking_mask, target_pcd, padding=1)

        del target_pcd
        del valid_filter

        return blocking_mask

    def obtain_visibility_blocking_mask(self, target_obj):
        start_time = time.time()
        camera_extrinsics = self.camera.info['extrinsics']
        cam_transform = np.linalg.inv(camera_extrinsics)
        camera_intrinsics = self.camera.info['intrinsics']
        occlusion = self.perception.occlusion

        pcd = target_obj.sample_conservative_pcd()
        obj_transform = target_obj.transform
        pcd = obj_transform[:3,:3].dot(pcd.T).T + obj_transform[:3,3]
        
        transformed_pcd = cam_transform[:3,:3].dot(pcd.T).T + cam_transform[:3,3]
        fx = camera_intrinsics[0][0]
        fy = camera_intrinsics[1][1]
        cx = camera_intrinsics[0][2]
        cy = camera_intrinsics[1][2]
        transformed_pcd[:,0] = transformed_pcd[:,0] / transformed_pcd[:,2] * fx + cx
        transformed_pcd[:,1] = transformed_pcd[:,1] / transformed_pcd[:,2] * fy + cy
        depth = transformed_pcd[:,2]
        transformed_pcd = transformed_pcd[:,:2]
        transformed_pcd = np.floor(transformed_pcd).astype(int)
        max_j = transformed_pcd[:,0].max()+1
        max_i = transformed_pcd[:,1].max()+1
        
        vis_mask = np.zeros(self.perception.occlusion.voxel_x.shape).astype(bool)
        if max_i <= 0 or max_j <= 0:
            # not in the camera view
            self.perception_time += time.time() - start_time
            self.perception_calls += 1
            del pcd
            del transformed_pcd
            del depth
            return np.zeros(occlusion.voxel_x.shape).astype(bool)

        unique_indices = np.unique(transformed_pcd, axis=0)
        unique_valid = (unique_indices[:,0] >= 0) & (unique_indices[:,1] >= 0)
        unique_indices = unique_indices[unique_valid]
        unique_depths = np.zeros(len(unique_indices))
        for i in range(len(unique_indices)):
            unique_depths[i] = depth[(transformed_pcd[:,0]==unique_indices[i,0])&(transformed_pcd[:,1]==unique_indices[i,1])].min()
        depth_img = np.zeros((max_i, max_j)).astype(float)
        depth_img[unique_indices[:,1],unique_indices[:,0]] = unique_depths
        depth_img = cv2.medianBlur(np.float32(depth_img), 5)

        # find voxels that can project to the depth
        pt = np.array([[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1],[0.5,0.5,0.5]])
        
        for i in range(len(pt)):
            voxel_vecs = np.array([occlusion.voxel_x, occlusion.voxel_y, occlusion.voxel_z]).transpose((1,2,3,0)).reshape(-1,3)
            voxel_vecs = voxel_vecs + pt[i].reshape(1,-1)  # get the middle point
            voxel_vecs = voxel_vecs * occlusion.resol
            transformed_voxels = occlusion.transform[:3,:3].dot(voxel_vecs.T).T + occlusion.transform[:3,3]
            # get to the image space
            transformed_voxels = cam_transform[:3,:3].dot(transformed_voxels.T).T + cam_transform[:3,3]

            # cam_to_voxel_dist = np.linalg.norm(transformed_voxels, axis=1)
            cam_to_voxel_depth = np.array(transformed_voxels[:,2])
            # intrinsics
            cam_intrinsics = camera_intrinsics
            fx = cam_intrinsics[0][0]
            fy = cam_intrinsics[1][1]
            cx = cam_intrinsics[0][2]
            cy = cam_intrinsics[1][2]
            transformed_voxels[:,0] = transformed_voxels[:,0] / transformed_voxels[:,2] * fx + cx
            transformed_voxels[:,1] = transformed_voxels[:,1] / transformed_voxels[:,2] * fy + cy
            transformed_voxels = np.floor(transformed_voxels).astype(int)
            voxel_depth = np.zeros((len(transformed_voxels)))
            valid_mask = (transformed_voxels[:,0] >= 0) & (transformed_voxels[:,0] < len(depth_img[0])) & \
                            (transformed_voxels[:,1] >= 0) & (transformed_voxels[:,1] < len(depth_img))
            voxel_depth[valid_mask] = depth_img[transformed_voxels[valid_mask][:,1], transformed_voxels[valid_mask][:,0]]
            valid_mask = valid_mask.reshape(occlusion.voxel_x.shape)
            voxel_depth = voxel_depth.reshape(occlusion.voxel_x.shape)

            cam_to_voxel_depth = cam_to_voxel_depth.reshape(occlusion.voxel_x.shape)
            vis_mask = vis_mask | ((cam_to_voxel_depth - voxel_depth <= 0.) & (voxel_depth > 0.) & valid_mask)

        self.perception_calls += 1
        self.perception_time += time.time() - start_time
        # print(occluded.astype(int).sum() / valid_mask.astype(int).sum())
        del cam_to_voxel_depth
        del voxel_depth
        del valid_mask
        del transformed_voxels
        del voxel_vecs
        del pcd
        del transformed_pcd
        del depth

        return vis_mask


    def pre_move_compute_valid_joints(self, target_obj_i, moved_objects, blocking_mask, collision_voxel, collision_dict):
        target_obj = self.perception.objects[target_obj_i]
        occlusion = self.perception.occlusion
        start_time = time.time()
        valid_pts, valid_orientations, valid_joints = \
            pose_generation.grasp_pose_generation(target_obj, 
                                                  self.robot, self.workspace, 
                                                  self.perception.occlusion.transform, 
                                                  collision_voxel, self.perception.occlusion.resol, sample_n=20)
        self.pose_generation_time += time.time() - start_time
        self.pose_generation_calls += 1
        print('number of grasp poses obtained: ')
        print(len(valid_pts))
        # * check each sampled pose to see if it's colliding with any objects. Create blocking object set
        total_blocking_objects = []
        total_blocking_object_nums = []
        joint_indices = []
        transformed_rpcds = []

        res_pts = []
        res_orientations = []
        res_joints = []

        # # visualize the perception
        # v_pcds = []
        # for obj_id, obj in self.perception.objects.items():
        #     v_pcd = obj.sample_conservative_pcd()
        #     v_pcd = obj.transform[:3,:3].dot(v_pcd.T).T + obj.transform[:3,3]
        #     v_pcd = occlusion.world_in_voxel_rot.dot(v_pcd.T).T + occlusion.world_in_voxel_tran
        #     v_pcd = v_pcd / occlusion.resol
        #     v_pcds.append(visualize_pcd(v_pcd, [1,0,0]))
        # o3d.visualization.draw_geometries(v_pcds)

        # if blocking_mask.sum() > 0:
        #     v_blocking_mask = visualize_voxel(occlusion.voxel_x, occlusion.voxel_y, occlusion.voxel_z,
        #                                     blocking_mask, [0,1,0])
        # else:
        #     v_blocking_mask = None

        # v_pcd = None

        for i in range(len(valid_orientations)):
            # obtain robot pcd at the given joint
            rpcd = self.robot.get_pcd_at_joints(valid_joints[i])
            # robot pcd in the occlusion
            transformed_rpcd = occlusion.world_in_voxel_rot.dot(rpcd.T).T + occlusion.world_in_voxel_tran
            transformed_rpcd = transformed_rpcd / occlusion.resol
            # trasnformed_rpcd_before_floor = transformed_rpcd
            # v_pcd_2 = visualize_pcd(trasnformed_rpcd_before_floor, [1,1,0])


            transformed_rpcd = np.floor(transformed_rpcd).astype(int)
            valid_filter = (transformed_rpcd[:,0] >= 0) & (transformed_rpcd[:,0] < occlusion.voxel_x.shape[0]) & \
                            (transformed_rpcd[:,1] >= 0) & (transformed_rpcd[:,1] < occlusion.voxel_x.shape[1]) & \
                            (transformed_rpcd[:,2] >= 0) & (transformed_rpcd[:,2] < occlusion.voxel_x.shape[2])
            transformed_rpcd = transformed_rpcd[valid_filter]
            transformed_rpcds.append(transformed_rpcd)
            valid = True

            
            occupied = np.zeros(self.prev_occupied_label.shape).astype(bool)  # for vis
            occluded = np.zeros(self.prev_occupied_label.shape).astype(bool)
            collision = np.zeros(collision_voxel.shape).astype(bool)

            if len(transformed_rpcd) == 0:
                blocking_objects = []
                valid = True
                for obj_i, obj in self.perception.objects.items():
                    if obj_i == target_obj_i:
                        continue

                    col_i = collision_dict[obj_i]
                    collision = collision | col_i

            else:
                # check if colliding with any objects
                blocking_objects = []
                for obj_i, obj in self.perception.objects.items():
                    if obj_i == target_obj_i:
                        continue
                    col_i = collision_dict[obj_i]
                    collision = collision | col_i

                    if col_i[transformed_rpcd[:,0],transformed_rpcd[:,1],transformed_rpcd[:,2]].sum() > 0:
                        blocking_objects.append(obj_i)
                        valid = False
                        # print('blocking with object ', self.perception.data_assoc.obj_ids_reverse[obj_i])
                        # if obj_i in moved_objects:
                        #     print('blocking object is moved before')
                        # else:
                        #     print('blocking object has not been moved')

                # v_voxel = visualize_voxel(self.perception.occlusion.voxel_x, self.perception.occlusion.voxel_y, self.perception.occlusion.voxel_z,
                #                 col_i, [0,0,1])
                # v_pcd = visualize_pcd(transformed_rpcd, [1,0,0])
                # if v_blocking_mask is not None:
                #     o3d.visualization.draw_geometries([v_voxel, v_pcd, v_pcd_2, v_blocking_mask])
                # else:
                #     o3d.visualization.draw_geometries([v_voxel, v_pcd, v_pcd_2])
                    
            # v_voxel = visualize_voxel(self.perception.occlusion.voxel_x, self.perception.occlusion.voxel_y, self.perception.occlusion.voxel_z,
            #                 collision, [0,0,1])
            # if v_blocking_mask is not None:
            #     o3d.visualization.draw_geometries([v_voxel, v_blocking_mask, v_pcd_2])
            # else:
            #     o3d.visualization.draw_geometries([v_voxel, v_pcd_2])

            # * make sure there is no object in the straight-line path
            for obj_i, obj in self.perception.objects.items():
                if obj_i == target_obj_i:
                    continue
                col_i = collision_dict[obj_i]
                if (col_i & blocking_mask).sum() > 0:
                    blocking_objects.append(obj_i)
                    valid = False

            if valid:
                res_pts.append(valid_pts[i])
                res_orientations.append(valid_orientations[i])
                res_joints.append(valid_joints[i])


            # if the blocking objects contain unmoved objects, then give up on this one
            blocking_objects = list(set(blocking_objects))
            if len(set(blocking_objects) - set(moved_objects)) == 0:
                total_blocking_objects.append(blocking_objects)
                total_blocking_object_nums.append(len(blocking_objects))
                joint_indices.append(i)

        if len(res_orientations) > 0:
            return res_pts, res_orientations, res_joints, 1, \
                joint_indices, total_blocking_object_nums, total_blocking_objects, transformed_rpcds


        if len(total_blocking_objects) == 0:
            # failure since all blocking object sets include at least one unmoved objects
            return [], [], [], 0, [], [], [], []

        return valid_pts, valid_orientations, valid_joints, 0, \
            joint_indices, total_blocking_object_nums, total_blocking_objects, transformed_rpcds
    

    def pre_move(self, target_obj_i, moved_objects, collision_voxel, collision_dict):
        """
        before moving the object, check reachability constraints. Rearrange the blocking objects
        """
        # * check reachability constraints by sampling grasp poses
        target_obj = self.perception.objects[target_obj_i]
        occlusion = self.perception.occlusion

        # get the target object pcd
        # target object since hasn't been moved, is at its transform recorded by object
        target_pcd = target_obj.sample_optimistic_pcd()
        target_pcd = target_obj.transform[:3,:3].dot(target_pcd.T).T + target_obj.transform[:3,3]
        target_pcd = occlusion.world_in_voxel_rot.dot(target_pcd.T).T + occlusion.world_in_voxel_tran
        target_pcd = target_pcd / occlusion.resol

        # vis_pcd = visualize_pcd(target_pcd, [0,1,0])

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

        # * add visibility blocking constraint to the mask
        # NOTE: Update: we don't add visibility constraint in random task planner
        # blocking_mask = blocking_mask | self.obtain_visibility_blocking_mask(target_obj)


        target_pcd = target_obj.sample_conservative_pcd()
        target_pcd = target_obj.transform[:3,:3].dot(target_pcd.T).T + target_obj.transform[:3,3]
        target_pcd = occlusion.world_in_voxel_rot.dot(target_pcd.T).T + occlusion.world_in_voxel_tran
        target_pcd = target_pcd / occlusion.resol
        target_pcd = np.floor(target_pcd).astype(int)
        valid_filter = (target_pcd[:,0]>=0) & (target_pcd[:,0]<occlusion.voxel_x.shape[0]) & \
                        (target_pcd[:,1]>=0) & (target_pcd[:,1]<occlusion.voxel_x.shape[1]) & \
                        (target_pcd[:,2]>=0) & (target_pcd[:,2]<occlusion.voxel_x.shape[2])
        target_pcd = target_pcd[valid_filter]
        # remove interior of target_pcd
        blocking_mask = self.mask_pcd_xy_with_padding(blocking_mask, target_pcd, padding=1)

        # # # visualize the blocking mask
        # vis_voxels = []
        # for obj_i, obj in self.perception.objects.items():
        #     occupied_i = collision_dict[obj_i]
        #     if occupied_i.sum() == 0:
        #         continue
        #     vis_voxel = visualize_voxel(occlusion.voxel_x, occlusion.voxel_y, occlusion.voxel_z, occupied_i, [0,0,1])
        #     vis_voxels.append(vis_voxel)

        # if blocking_mask.sum() > 0:
        #     vis_voxel = visualize_voxel(occlusion.voxel_x, occlusion.voxel_y, occlusion.voxel_z, blocking_mask, [1,0,0])
        #     o3d.visualization.draw_geometries(vis_voxels + [vis_voxel,vis_pcd])
        # else:
        #     o3d.visualization.draw_geometries(vis_voxels + [vis_pcd])


        # * quick pre-check: if there are objects on the blocking mask, then fail
        for obj_i, col_i in collision_dict.items():
            if obj_i == target_obj_i:
                continue
            if (col_i & blocking_mask).sum() > 0:
                # directly fail because other objects are on the blocking mask
                # we can't safely extract the object
                return [], [], []


        valid_pts, valid_orientations, valid_joints, status, \
            joint_indices, total_blocking_object_nums, \
            total_blocking_objects, transformed_rpcds \
                = self.pre_move_compute_valid_joints(target_obj_i, moved_objects, blocking_mask, collision_voxel, collision_dict)

        if (status == 1):
            return valid_pts, valid_orientations, valid_joints
        else:
            # NOTE:
            # cases include
            # 1. no valid poses found
            # 2. some blocking objects only include objects that have been moved
            # 3. all blocking objects include at least one object that hasn't been moved
            # in all these cases, we do not go further since there must be some random action
            # that can first deal with the blocking objects and open up the path
            return [], [], []


    def set_collision_env_with_mask(self, mask, ignore_obj_list, ignore_obj_pose_list, padding=0):
        """
        given a mask of the occlusion space create collision env. Mask out the pcd in ignore_pcd_list with padding
        """
        # mask out the ignored obj        
        if padding > 0:
            for i in range(len(ignore_obj_list)):
                pcd = self.perception.objects[ignore_obj_list[i]].sample_conservative_pcd()
                obj_transform = ignore_obj_pose_list[i]
                pcd = obj_transform[:3,:3].dot(pcd.T).T + obj_transform[:3,3]
                transform = self.perception.occlusion.transform
                transform = np.linalg.inv(transform)
                pcd = transform[:3,:3].dot(pcd.T).T + transform[:3,3]
                pcd = pcd / self.perception.occlusion.resol
                pcd = np.floor(pcd).astype(int)
                mask = self.mask_pcd_xy_with_padding(mask, pcd, padding)
        start_time = time.time()
        self.planning_system.motion_planner.set_collision_env(self.perception.occlusion, 
                                            mask, np.zeros(mask.shape).astype(bool))
        self.motion_planning_time += time.time() - start_time

    def move_and_sense_precheck(self, move_obj_idx, moved_objects, collision_voxel, collision_dict):
        """
        check for reconstruction plan
        """
        start_time = time.time()
        self.planning_system.motion_planner.clear_octomap()
        self.motion_planning_time += time.time() - start_time
        # self.motion_planning_calls += 1
        print('handling object: ', self.perception.data_assoc.obj_ids_reverse[move_obj_idx])
        start_time = time.time()
        _, suction_poses_in_obj, suction_joints = self.pre_move(move_obj_idx, moved_objects, collision_voxel, collision_dict)
        end_time = time.time()
        print('pre_move takes time: ', end_time - start_time)
        print('number of generated suction poses: ', len(suction_poses_in_obj))
        if len(suction_joints) == 0:  # no valid suction joint now
            return False, None

        start_obj_pose = self.perception.objects[move_obj_idx].transform

        planning_info = dict()
        planning_info['obj_i'] = move_obj_idx

        planning_info['objects'] = self.perception.objects
        planning_info['occlusion'] = self.perception.occlusion
        planning_info['workspace'] = self.workspace
        planning_info['gripper_tip_poses_in_obj'] = suction_poses_in_obj
        planning_info['suction_joints'] = suction_joints

        planning_info['robot'] = self.robot
        # planning_info['motion_planner'] = self.motion_planner
        planning_info['occluded_label'] = self.prev_occlusion_label
        planning_info['occupied_label'] = self.prev_occupied_label
        planning_info['seg_img'] = self.perception.seg_img==move_obj_idx
        planning_info['camera'] = self.camera


        start_time = time.time()
        intermediate_pose, suction_poses_in_obj, suction_joints, intermediate_joints = \
            pipeline_utils.generate_intermediate_poses(**planning_info)
        end_time = time.time()
        self.pose_generation_time += end_time - start_time
        self.pose_generation_calls += 1
        del planning_info
        # generate intermediate pose for the obj with valid suction pose
        print('generate_intermediate_poses takes time: ', end_time - start_time)

        # set collision environment and reuse afterwards
        # (plan to suction pose, intermediate pose and sense pose do not change collision env)

        self.set_collision_env_with_mask(collision_voxel, [move_obj_idx], 
                                        [self.perception.objects[move_obj_idx].transform], padding=3)

        print('number of suction_poses_in_obj: ', len(suction_poses_in_obj))
        if len(suction_poses_in_obj) == 0:
            return False, None

        for i in range(len(suction_poses_in_obj)):
            suction_pose_in_obj = suction_poses_in_obj[i]
            suction_joint = suction_joints[i]
            intermediate_joint = intermediate_joints[i]

            start_time = time.time()
            pick_joint_dict_list, lift_joint_dict_list = \
                self.planning_system.plan_to_suction_pose(self.perception.objects[move_obj_idx],
                        suction_pose_in_obj, suction_joint, self.robot.joint_dict)  # internally, plan_to_pre_pose, pre_to_suction, lift up

            end_time = time.time()
            self.motion_planning_time += end_time - start_time
            self.motion_planning_calls += 1
            print('plan_to_suction_pose takes time: ', end_time - start_time)


            if len(pick_joint_dict_list) == 0:
                continue
            
            start_time = time.time()
            # * compute the straight-line extraction distance
            occupied_filter = self.prev_occupied_dict[move_obj_idx]
            x_dist = self.perception.occlusion.voxel_x[occupied_filter].max() + 1
            x_dist = x_dist * self.perception.occlusion.resol[0]

            retreat_joint_dict_list = self.planning_system.plan_to_intermediate_pose(move_obj_idx, self.perception.objects[move_obj_idx],
                                        suction_pose_in_obj, x_dist, intermediate_pose, intermediate_joint,
                                        lift_joint_dict_list[-1])
            end_time = time.time()
            self.motion_planning_time += end_time - start_time
            self.motion_planning_calls += 1
            print('plan_to_intermediate_pose takes time: ', end_time - start_time)

            if len(retreat_joint_dict_list) == 0:
                continue
            # found one valid plan. Record necessary information for future planning
            planning_info = dict()
            planning_info['tip_pose_in_obj'] = suction_pose_in_obj

            planning_info['intermediate_joint'] = intermediate_joint
            planning_info['intermediate_joint_dict_list'] = retreat_joint_dict_list
            planning_info['lift_up_joint_dict_list'] = lift_joint_dict_list
            planning_info['suction_joint_dict_list'] = pick_joint_dict_list
            planning_info['obj'] = self.perception.objects[move_obj_idx]

            return True, planning_info
        return False, None


    def sample_placement_poses(self, obj_idx, obj_start_pose, num_placement_sample, collision_voxel, clear_octomap=True):
        """
        use convolution method to sample placement poses
        check IK too
        """
        import scipy.signal as ss
        # TODO: debug
        # * clear the octomap each time before we sample a goal location. This makes sure that
        # IK is not causing trouble
        if clear_octomap:
            start_time = time.time()
            self.planning_system.motion_planner.clear_octomap()
            self.motion_planning_time += time.time() - start_time

        collision_voxel = np.array(collision_voxel)
        transform = self.perception.occlusion.transform
        transform = np.linalg.inv(transform)

        collision_grid = collision_voxel.sum(axis=2)>0
        voxel_resol = self.perception.occlusion.resol
        # in the z-axis, if there is at least one voxel occupied, then collision grid
        map_x, map_y = np.indices(collision_grid.shape).astype(int)
        obj = self.perception.objects[obj_idx]
        obj_pcd = self.perception.objects[obj_idx].sample_conservative_pcd()
        obj_pcd_2d = self.rearrange_planner.obj_pcd_2d_projection([obj_pcd])[0]

        voxel_transform = self.perception.occlusion.transform
        obj_start_pose_in_voxel = self.rearrange_planner.obtain_pose_in_voxel([obj_start_pose], voxel_transform)[0]

        obj_z = obj_start_pose_in_voxel[2,3]
        obj_z_in_world = obj_start_pose[2,3]

        sampled_poses = []
        total_start_valid_pts = []
        total_start_valid_poses_in_obj = []
        total_start_valid_joints = []

        total_valid_pts = []
        total_valid_poses_in_obj = []
        total_valid_joints = []

        for sample_i in range(num_placement_sample):
            mp_map = np.array(collision_grid).astype(bool)
            mp_map_3d = np.array(collision_voxel).astype(bool)

            # * sample goal locations for each included object
            color_map = np.array(mp_map).astype(int)
            # randomly rotate
            ang_x = np.random.normal(size=(2))
            ang_x = ang_x / np.linalg.norm(ang_x)
            ang = np.arcsin(ang_x[1])

            # rotate the pcd
            rot_mat = np.array([[np.cos(ang),-np.sin(ang)],[np.sin(ang),np.cos(ang)]])
            pcd = rot_mat.dot(obj_pcd_2d.T).T
            # find bounding box for pcd
            x_min = np.min(pcd[:,0])
            y_min = np.min(pcd[:,1])
            tran_vec = np.array([-x_min, -y_min])
            pcd = pcd + tran_vec
            pcd_indices = pcd / voxel_resol[:2]
            pcd_indices = np.floor(pcd_indices).astype(int)
            x_ind_max = pcd_indices[:,0].max()
            y_ind_max = pcd_indices[:,1].max()
            obj_img = np.zeros((x_ind_max+1,y_ind_max+1)).astype(bool)
            obj_img[pcd_indices[:,0],pcd_indices[:,1]] = 1



            # visualize when the shape is invalid
            # if obj_img.shape[0] > mp_map.shape[0] or obj_img.shape[1] > mp_map.shape[1]:
            #     vvoxel = visualize_voxel(self.perception.occlusion.voxel_x,
            #                     self.perception.occlusion.voxel_y,
            #                     self.perception.occlusion.voxel_z,
            #                     collision_voxel, [0,0,1])
            #     v_t_mat = np.zeros((3,3))
            #     v_t_mat[2,2] = 1
            #     v_t_mat[:2,:2] = rot_mat
            #     v_t_mat[:2,2] = 0
            #     v_obj_2d_pose = v_t_mat
            #     obj_ori_pose_in_voxel = transform.dot(obj.transform)
            #     v_obj_pose_in_voxel = self.rearrange_planner.pose_from_2d_pose(v_obj_2d_pose, obj_z)
            #     v_obj_pose_in_voxel[:3,3] = obj_ori_pose_in_voxel[:3,3]
            #     v_obj = v_obj_pose_in_voxel[:3,:3].dot(obj_pcd.T).T + v_obj_pose_in_voxel[:3,3]
            #     v_obj = v_obj / voxel_resol
            #     vpcd = visualize_pcd(v_obj, [1,0,0])
            #     o3d.visualization.draw_geometries([vvoxel, vpcd])
            #     # show the object conservative volume too
            #     v_obj_voxel = visualize_voxel(obj.voxel_x, obj.voxel_y, obj.voxel_z,
            #                                   obj.get_conservative_model(), [0,0,1])
            #     v_bbox = visualize_bbox(obj.voxel_x, obj.voxel_y, obj.voxel_z)
            #     o3d.visualization.draw_geometries([v_obj_voxel, v_bbox])
      



            # compute convolution of object with environment
            conv_img = ss.correlate(mp_map.astype(int), obj_img.astype(int), mode='valid')


            # find the indices in the convoluted image where value is 0
            xs, ys = (conv_img==0).nonzero()
            if len(xs) == 0:
                valid = False
                break
            # randomly select one as sampled goal
            selected_i = np.random.choice(len(xs))
            # check if it collides with the environment
            transformed_pcd_indices = pcd_indices + np.array([xs[selected_i],ys[selected_i]])
            assert mp_map[transformed_pcd_indices[:,0],transformed_pcd_indices[:,1]].sum() == 0

            # put the pcd in the map as collision
            # mp_map_i[transformed_pcd_indices[:,0],transformed_pcd_indices[:,1]] = 1
            color_map[transformed_pcd_indices[:,0],transformed_pcd_indices[:,1]] = 2
            
            # #####################################
            # NOTE: visualization of rearrangement
            # plt.clf()
            # plt.pcolor(map_x*voxel_resol[0], map_y*voxel_resol[1], color_map)
            # plt.pause(0.0001)
            # #####################################

            # * record the sampled pose
            x_in_voxel = xs[selected_i] * voxel_resol[0]
            y_in_voxel = ys[selected_i] * voxel_resol[1]
            t_mat = np.zeros((3,3))
            t_mat[2,2] = 1
            t_mat[:2,:2] = rot_mat
            t_mat[:2,2] = tran_vec + np.array([x_in_voxel, y_in_voxel])
            obj_2d_pose = t_mat

            # * check IK
            # make sure it's not too tight
            pcd = obj_start_pose_in_voxel[:3,:3].dot(obj_pcd.T).T + \
                    obj_start_pose_in_voxel[:3,3]
            pcd = pcd / voxel_resol
            pcd = np.floor(pcd).astype(int)
            valid_filter = (pcd[:,0] >= 0) & (pcd[:,0] < mp_map_3d.shape[0]) & \
                            (pcd[:,1] >= 0) & (pcd[:,1] < mp_map_3d.shape[1]) & \
                            (pcd[:,2] >= 0) & (pcd[:,2] < mp_map_3d.shape[2])
            pcd = pcd[valid_filter]
            mp_map_3d = self.mask_pcd_xy_with_padding(mp_map_3d, pcd, padding=1)

            # total_obj_poses is 2D pose in the voxel
            # obtain the pose in voxel from the 2D pose
            obj_pose_in_voxel = self.rearrange_planner.pose_from_2d_pose(obj_2d_pose, obj_z)
            obj_pose = self.rearrange_planner.pose_to_pose_in_world([obj_pose_in_voxel], [obj_z_in_world], voxel_transform)[0]
            pcd = obj_pose_in_voxel[:3,:3].dot(obj_pcd.T).T + \
                    obj_pose_in_voxel[:3,3]

            pcd = pcd / voxel_resol

            pcd = np.floor(pcd).astype(int)
            valid_filter = (pcd[:,0] >= 0) & (pcd[:,0] < mp_map_3d.shape[0]) & \
                            (pcd[:,1] >= 0) & (pcd[:,1] < mp_map_3d.shape[1]) & \
                            (pcd[:,2] >= 0) & (pcd[:,2] < mp_map_3d.shape[2])
            pcd = pcd[valid_filter]
            mp_map_3d = self.mask_pcd_xy_with_padding(mp_map_3d, pcd, padding=1)


            valid_pts, valid_poses_in_obj, valid_joints, target_valid_pts, target_valid_poses_in_obj, target_valid_joints \
                = self.rearrange_planner.ik_check_start_target_pose(obj, obj_pose, obj_start_pose, self.robot, self.workspace, mp_map_3d, voxel_transform, voxel_resol)
            if valid_pts is None:
                continue

            sampled_poses.append(obj_pose)
            total_start_valid_pts.append(valid_pts)
            total_start_valid_poses_in_obj.append(valid_poses_in_obj)
            total_start_valid_joints.append(valid_joints)

            total_valid_pts.append(target_valid_pts)
            total_valid_poses_in_obj.append(target_valid_poses_in_obj)
            total_valid_joints.append(target_valid_joints)

        # TODO check if IK is valid
        del mp_map
        del collision_grid
        del map_x
        del map_y
        del pcd

        return sampled_poses, total_start_valid_pts, total_start_valid_poses_in_obj, total_start_valid_joints, \
            total_valid_pts, total_valid_poses_in_obj, total_valid_joints

    def move_precheck(self, obj_idx, obj_start_pose, sampled_poses, total_start_valid_poses_in_obj, total_start_valid_joints, 
                        total_valid_poses_in_obj, total_valid_joints, collision_voxel):
        """
        motion planning check whether obj can be moved from start pose to target pose
        Return:
        [(sampled_pose, trajectory)]
        """
        start_time = time.time()

        tip_poses_in_obj = total_start_valid_poses_in_obj
        start_joint_vals = total_start_valid_joints
        target_joint_vals = total_valid_joints
        robot = self.robot

        results = []

        for i in range(len(sampled_poses)):
            suction_mp = np.array(collision_voxel)  # suction map consider object at start as collision
            transfer_mp = np.array(collision_voxel) # transfer map removes objects at start and goal
            reset_mp = np.array(collision_voxel)

            pcd = self.perception.objects[obj_idx].sample_conservative_pcd()
            obj_transform = obj_start_pose
            pcd = obj_transform[:3,:3].dot(pcd.T).T + obj_transform[:3,3]
            transform = self.perception.occlusion.transform
            transform = np.linalg.inv(transform)
            pcd = transform[:3,:3].dot(pcd.T).T + transform[:3,3]
            pcd = pcd / self.perception.occlusion.resol
            pcd = np.floor(pcd).astype(int)
            transfer_mp = self.mask_pcd_xy_with_padding(transfer_mp, pcd, 1)
            reset_mp = self.mask_pcd_xy_with_padding(reset_mp, pcd, 1)


            pcd = self.perception.objects[obj_idx].sample_conservative_pcd()
            obj_transform = sampled_poses[i]
            pcd = obj_transform[:3,:3].dot(pcd.T).T + obj_transform[:3,3]
            transform = self.perception.occlusion.transform
            transform = np.linalg.inv(transform)
            pcd = transform[:3,:3].dot(pcd.T).T + transform[:3,3]
            pcd = pcd / self.perception.occlusion.resol
            pcd = np.floor(pcd).astype(int)
            transfer_mp = self.mask_pcd_xy_with_padding(transfer_mp, pcd, 1)
            suction_mp = self.mask_pcd_xy_with_padding(suction_mp, pcd, 1)

            # set up reset collision voxel
            pcd = self.perception.objects[obj_idx].sample_conservative_pcd()
            obj_transform = sampled_poses[i]
            pcd = obj_transform[:3,:3].dot(pcd.T).T + obj_transform[:3,3]
            transform = self.perception.occlusion.transform
            transform = np.linalg.inv(transform)
            pcd = transform[:3,:3].dot(pcd.T).T + transform[:3,3]
            pcd = pcd / self.perception.occlusion.resol
            pcd = np.floor(pcd).astype(int)
            valid_filter = (pcd[:,0] >= 0) & (pcd[:,0] < reset_mp.shape[0]) & \
                            (pcd[:,1] >= 0) & (pcd[:,1] < reset_mp.shape[1]) & \
                            (pcd[:,2] >= 0) & (pcd[:,2] < reset_mp.shape[2])
            pcd = pcd[valid_filter]
            reset_mp[pcd[:,0],pcd[:,1],pcd[:,2]] = 1

            for j in range(len(start_joint_vals[i])):
                start_joint_dict = robot.joint_vals_to_dict(start_joint_vals[i][j])
                target_joint_dict = robot.joint_vals_to_dict(target_joint_vals[i][j])
                suction_pose =  obj_start_pose.dot(tip_poses_in_obj[i][j])
                self.planning_system.motion_planner.clear_octomap()

                self.set_collision_env_with_mask(suction_mp, [], [], 0)

                suction_traj = self.planning_system.motion_planner.suction_plan(self.robot.joint_dict, 
                                suction_pose, start_joint_vals[i][j], robot, workspace=self.workspace, display=False)
                self.motion_planning_calls += 1
                # transfer_traj = motion_planner.joint_dict_motion_plan(previous_joint_dict, start_joint_dict, robot)
                if len(suction_traj) == 0:
                    continue
                self.planning_system.motion_planner.clear_octomap()
                self.set_collision_env_with_mask(transfer_mp, [], [], 0)
            
                # lift up to avoid collision with bottom
                relative_tip_pose = np.eye(4)
                relative_tip_pose[:3,3] = np.array([0,0,0.05]) # lift up by 0.05

                lift_traj = self.planning_system.motion_planner.straight_line_motion(start_joint_dict, 
                                    suction_pose, relative_tip_pose, self.robot, workspace=self.workspace)        
                if len(lift_traj) == 0:
                    continue

                relative_tip_pose = np.eye(4)
                relative_tip_pose[:3,3] = np.array([0,0,0.05]) # lift up by 0.05
                tip_suction_pose = sampled_poses[i].dot(tip_poses_in_obj[i][j])
                drop_traj = self.planning_system.motion_planner.straight_line_motion(target_joint_dict, 
                                    tip_suction_pose, relative_tip_pose, self.robot, workspace=self.workspace)   
                if len(drop_traj) == 0:
                    continue
                drop_traj = drop_traj[::-1]
                joint_vals = robot.joint_dict_to_vals(drop_traj[0])
                traj = self.planning_system.motion_planner.suction_with_obj_plan(lift_traj[-1], 
                                tip_poses_in_obj[i][j], joint_vals, robot, self.perception.objects[obj_idx])
                self.motion_planning_calls += 1
                if len(traj) == 0:
                    continue

                # * reset
                self.planning_system.motion_planner.clear_octomap()
                self.set_collision_env_with_mask(reset_mp, [], [], 0)
                # TODO: transfer plan should include first going to pre-grasp pose, then going staright-line to grasp the object
                start_joint_dict = drop_traj[-1]
                goal_joint_dict = robot.init_joint_dict
                reset_traj = self.planning_system.motion_planner.joint_dict_motion_plan(start_joint_dict, goal_joint_dict, robot)
                self.motion_planning_calls += 1
                if len(reset_traj) == 0:
                    continue

                if len(suction_traj) > 0 and len(lift_traj) > 0 and len(traj) > 0 and len(drop_traj) > 0 \
                    and len(reset_traj) > 0:
                    # add trajectory to list
                    results.append((sampled_poses[i], suction_traj,lift_traj,traj,drop_traj,reset_traj))
                    break
        # motion planning from start joint to target joint. Set the collision using the collision_voxel
        self.motion_planning_time += time.time() - start_time

        if len(sampled_poses) > 0:
            del suction_mp
            del transfer_mp
            del reset_mp
            del pcd
            del valid_filter
        return results

    
    def move(self, move_obj_idx, pose, suction_traj, lift_traj, transfer_traj, drop_traj, reset_traj):        
        self.execute_traj(suction_traj, self.perception.data_assoc.obj_ids_reverse[move_obj_idx], duration=0.3)
        self.attach_obj(move_obj_idx)
        self.execute_traj(lift_traj+transfer_traj+drop_traj)
        self.detach_obj()
        # reset
        self.execute_traj(reset_traj, self.perception.data_assoc.obj_ids_reverse[move_obj_idx])

    def compute_info_gain(self, obj_idx, obj_start_pose_dict, pose_list, occlusion_voxel):
        """
        compute the new occlusion when obj_idx is at new pose
        compute the info gain based on this
        TODO: debug
        """
        obj_poses = copy.deepcopy(obj_start_pose_dict)
        obj_pcds = {}
        info_gains = []
        for obj_id, _ in obj_start_pose_dict.items():
            # NOTE: we are using optimistic volume because this reflects the uncertain space for
            # objects that haven't been reconstructed
            pcd = self.perception.objects[obj_id].sample_optimistic_pcd()
            obj_pcds[obj_id] = pcd
        for i in range(len(pose_list)):
            pose = pose_list[i]
            obj_poses[obj_idx] = pose
            start_time = time.time()
            occluded = self.perception.occlusion.occlusion_from_pcd(self.camera.info['extrinsics'], self.camera.info['intrinsics'],
                                                                 self.camera.info['img_shape'], obj_poses, obj_pcds)
            self.perception_time += time.time() - start_time
            # occluded is generated by pcd. previous occlusion filter is also generated by pcd because we
            # were using occlusion_labeled to compute that

            # print('previous occlusion before...')
            # voxel1 = visualize_voxel(self.perception.occlusion.voxel_x,
            #                         self.perception.occlusion.voxel_y,
            #                         self.perception.occlusion.voxel_z,
            #                         occlusion_voxel, [0,0,1])
            # o3d.visualization.draw_geometries([voxel1])

            # print('new occlusion after move...')
            # voxel2 = visualize_voxel(self.perception.occlusion.voxel_x,
            #                         self.perception.occlusion.voxel_y,
            #                         self.perception.occlusion.voxel_z,
            #                         occluded, [0,0,1])
            # o3d.visualization.draw_geometries([voxel2])

            # print('revealed occlusion after move...')

            # if (occluded & occlusion_voxel).sum() > 0 and (occlusion_voxel & (~(occluded & occlusion_voxel))).sum() > 0:
            #     voxel4 = visualize_voxel(self.perception.occlusion.voxel_x,
            #                             self.perception.occlusion.voxel_y,
            #                             self.perception.occlusion.voxel_z,
            #                             (occluded & occlusion_voxel), [0,0,1])
            #     voxel3 = visualize_voxel(self.perception.occlusion.voxel_x,
            #                             self.perception.occlusion.voxel_y,
            #                             self.perception.occlusion.voxel_z,
            #                             occlusion_voxel & (~(occluded & occlusion_voxel)), [1,0,0])
            #     o3d.visualization.draw_geometries([voxel3, voxel4])
            
            info_gain = occlusion_voxel.sum() - (occluded & occlusion_voxel).sum()
            print('info_gain: ', info_gain)
            info_gains.append(info_gain)
        return info_gains

    def random_action(self, obj_list, moved_objects, obj_start_pose_dict, num_obj_sample, num_placement_sample, collision_voxel, collision_dict, occlusion_voxel):
        # generate samples for a random action: sample N random objects and K placement samples for each
        # verify if these are valid by motion planning
        # if the sampled object has not been reconstructed, then do not need to sample placement poses
        # we just plan reconstruction motions for it. The action will have max info gain because we are
        # reconstructing
        # NOTE: N and K determine the number of samples tried. They may be invalid, and only the valid
        # subset is going to be added to returned info
        # NOTE: we can stack multiple actions, which can take into account actions like moving others to reveal objects
        # then collision check should only use the collision voxel propagated
        # NOTE: collision_voxel: the last collision voxel of ALL objects where all objects are at their start poses
        # NOTE: collision_dict: object id -> collision voxel
        # NOTE: occlusion_voxel: net occlusion observed so far (intersection)

        # generate N random samples for object
        sampled_actions = {}  # [object -> [(pose, traj)]]
        sampled_objs = np.random.choice(obj_list, size=num_obj_sample)

        # NOTE: we allow repeating of objects to increase the changes of successful planning
        # for objects that haven't been reconstructed, if there exists a successful planning
        # then we don't try another time

        for i in range(len(sampled_objs)):
            obj_id = sampled_objs[i]
            if obj_id not in moved_objects:
                # then this action is a reconstruction action
                if obj_id in sampled_actions:
                    # for successful plan found, we don't need to do another plan
                    continue
                # otherwise verify if the plan is atainable
                status, planning_info = self.move_and_sense_precheck(obj_id, moved_objects, collision_voxel, collision_dict)
                if status:
                    sampled_actions[obj_id] = planning_info
                    # assume the info gain is the space occluded by the object
                    # sampled_actions[obj_id]['info_gains'] = self.prev_occluded_dict[obj_id].sum()
                    # TODO
                    # info-gain: the occlusion space directly occluded by the object
            else:
                # obj has been moved already. Sample placement poses
                # verify if the plan can be found
                # * sample placement poses and compute grasp poses
                sampled_poses, total_start_valid_pts, total_start_valid_poses_in_obj, total_start_valid_joints, \
                            total_valid_pts, total_valid_poses_in_obj, total_valid_joints \
                                = self.sample_placement_poses(obj_id, obj_start_pose_dict[obj_id], num_placement_sample, collision_voxel)
                if len(sampled_poses) == 0:
                    continue
                # *  motion planning check using the sampled poses
                pose_and_traj_list = self.move_precheck(obj_id, obj_start_pose_dict[obj_id], sampled_poses, total_start_valid_poses_in_obj, total_start_valid_joints, 
                                   total_valid_poses_in_obj, total_valid_joints, collision_voxel)
                if len(pose_and_traj_list) == 0:
                    continue
                # random action do not need to use info-gain

                if obj_id not in sampled_actions:
                    sampled_actions[obj_id] = {'pose_and_traj_list': []}
                sampled_actions[obj_id]['pose_and_traj_list'] += pose_and_traj_list
                # sampled_actions[obj_id]['info_gains'] += info_gains
        return sampled_actions


    def random_greedy_action(self, obj_list, moved_objects, obj_start_pose_dict, num_obj_sample, num_placement_sample, collision_voxel, collision_dict, occlusion_voxel):
        # generate samples for a random action: sample N random objects and K placement samples for each
        # verify if these are valid by motion planning
        # if the sampled object has not been reconstructed, then do not need to sample placement poses
        # we just plan reconstruction motions for it. The action will have max info gain because we are
        # reconstructing
        # NOTE: N and K determine the number of samples tried. They may be invalid, and only the valid
        # subset is going to be added to returned info
        # NOTE: we can stack multiple actions, which can take into account actions like moving others to reveal objects
        # then collision check should only use the collision voxel propagated
        # NOTE: collision_voxel: the last collision voxel of ALL objects where all objects are at their start poses
        # NOTE: collision_dict: object id -> collision voxel
        # NOTE: occlusion_voxel: net occlusion observed so far (intersection)

        # generate N random samples for object
        sampled_actions = {}  # [object -> [(pose, traj)]]

        # * first try if we can reconstruct objects *
        unmoved_objects = list(set(obj_list) - set(moved_objects))
        # try out each object to see if we can find one to reconstruct
        for i in range(len(unmoved_objects)):
            obj_id = unmoved_objects[i]
            status, planning_info = self.move_and_sense_precheck(obj_id, moved_objects, collision_voxel, collision_dict)
            if status:
                sampled_actions[obj_id] = planning_info
                # assume the info gain is the space occluded by the object
                sampled_actions[obj_id]['info_gains'] = self.prev_occluded_dict[obj_id].sum()
                # successfully found one reconstruction action. stop
                return sampled_actions


        # * if there are no more objects to be reconstructed, randomly sample objects to rearrange *
        sampled_objs = np.random.choice(moved_objects, size=num_obj_sample)
        max_info_gain = -1

        total_sampled_objs = []  # (obj_id, others, info_gains)
        total_info_gains = []
        total_sampled_poses = []
        total_start_valid_poses_in_obj = []
        total_start_valid_joints = []
        total_valid_poses_in_obj = []
        total_valid_joints = []

        start_time = time.time()
        self.planning_system.motion_planner.clear_octomap()
        self.motion_planning_time += time.time() - start_time

        # NOTE: we allow repeating of objects to increase the changes of successful planning
        for i in range(len(sampled_objs)):
            obj_id = sampled_objs[i]
            # obj has been moved already. Sample placement poses
            # verify if the plan can be found
            # * sample placement poses and compute grasp poses
            sampled_poses, _, start_valid_poses_in_obj, start_valid_joints, \
                _, valid_poses_in_obj, valid_joints \
                            = self.sample_placement_poses(obj_id, obj_start_pose_dict[obj_id], num_placement_sample, collision_voxel, clear_octomap=False)
            if len(sampled_poses) == 0:
                continue

            # * compute info gain to see if new occlusion space is going to be revealed
            # do this before motion planning so we can prune
            
            total_info_gains += self.compute_info_gain(obj_id, obj_start_pose_dict, sampled_poses, occlusion_voxel)
            total_sampled_objs += [obj_id for j in range(len(sampled_poses))]
            total_sampled_poses += sampled_poses
            total_start_valid_poses_in_obj += start_valid_poses_in_obj
            total_start_valid_joints += start_valid_joints
            total_valid_poses_in_obj += valid_poses_in_obj
            total_valid_joints += valid_joints

        
        total_info_gains = np.array(total_info_gains)
        sorted_i = np.argsort(-total_info_gains)  # large to small. We stop at the first available one
        print('sorted info-gain: ', total_info_gains[sorted_i])
        for i in range(len(sorted_i)):
            idx = sorted_i[i]
            print('trying for info-gain: ', total_info_gains[idx])
            obj_id = total_sampled_objs[idx]
            # *  motion planning check using the sampled poses
            pose_and_traj_list = self.move_precheck(obj_id, obj_start_pose_dict[obj_id], [total_sampled_poses[idx]], 
                                                    [total_start_valid_poses_in_obj[idx]], [total_start_valid_joints[idx]],
                                                    [total_valid_poses_in_obj[idx]], [total_valid_joints[idx]], collision_voxel)
            if len(pose_and_traj_list) == 0:
                continue
            # otherwise update the max info gain. Since we found the max info gain, terminate
            if total_info_gains[idx] > max_info_gain:
                max_info_gain = total_info_gains[idx]
                print('adding info-gain: ', max_info_gain)
                if obj_id not in sampled_actions:
                    sampled_actions[obj_id] = {'pose_and_traj_list': [], 'info_gains': []}
                sampled_actions[obj_id]['pose_and_traj_list'] += pose_and_traj_list
                sampled_actions[obj_id]['info_gains'] += [total_info_gains[idx]]
                # break because we have found the max info-gain possible
                break
        return sampled_actions

    def run_pipeline_random(self):

        # select object
        #TODO
        moved_objects = []
        iter_i = 0

        valid_objects = []
        start_time = time.time()
        while True:
            print('iteration: ', iter_i)
            gc.collect()
            # select object: active objects but not moved
            active_objs = []
            for obj_id, obj in self.perception.objects.items():
                # check if the object becomes active when the hiding object have all been moved
                obj_hide_list = list(obj.obj_hide_set)
                print('object ', self.perception.data_assoc.obj_ids_reverse[obj_id], ' hiding list: ')
                for k in range(len(obj_hide_list)):
                    print(self.perception.data_assoc.obj_ids_reverse[obj_hide_list[k]])
                if obj.active:
                    active_objs.append(obj_id)
                    if (obj_id not in valid_objects):
                        valid_objects.append(obj_id)  # a new object that becomes valid

            print('moved object list: ')
            for k in moved_objects:
                print(self.perception.data_assoc.obj_ids_reverse[k])
            print('valid but unmoved object list:')
            for k in list(set(valid_objects)-set(moved_objects)):
                print(self.perception.data_assoc.obj_ids_reverse[k])
            # Terminate when the entire space has been observed???
            # TODO: need to visualize to make sure workspace does not cause problems
            # TODO: also visualize in our planner if the space becomes empty at convergence
            if (self.prev_occluded).sum() == 0 or len(moved_objects) == self.num_obj or time.time()-start_time > self.timeout:
                print('self.prev_occluded.sum: ', self.prev_occluded.sum())
                print('moved_objects: ', len(moved_objects))
                print('num_obj: ', self.num_obj)
                print('time: ', time.time() - start_time)
                print('timeout: ', self.timeout)

                running_time = time.time() - start_time
                print('#############Finished##############')
                print('number of reconstructed objects: ', len(moved_objects))
                print('number of executed actions: ', self.num_executed_actions)
                print('running time: ', time.time() - start_time, 's')

                import pickle
                f = open('random-' + self.prob_name + '-trial-' + str(self.trial_num) + '-result.pkl', 'wb')
                res_dict = {}
                res_dict['num_reconstructed_objs'] = len(moved_objects)
                res_dict['num_collision'] = self.num_collision
                res_dict['running_time'] = running_time
                res_dict['num_executed_actions'] = self.num_executed_actions
                res_dict['perception_time'] = self.perception_time
                res_dict['motion_planning_time'] = self.motion_planning_time + self.rearrange_planner.motion_planning_time
                res_dict['pose_generation_time'] = self.pose_generation_time + self.rearrange_planner.pose_generation_time
                res_dict['rearrange_time'] = self.rearrange_time
                res_dict['ros_time'] = self.ros_time
                res_dict['perception_calls'] = self.perception_calls
                res_dict['motion_planning_calls'] = self.motion_planning_calls + self.rearrange_planner.motion_planning_calls
                res_dict['pose_generation_calls'] = self.pose_generation_calls + self.rearrange_planner.pose_generation_calls
                res_dict['rearrange_calls'] = self.rearrange_calls
                res_dict['execution_calls'] = self.execution_calls
                res_dict['final_occluded_volume'] = self.prev_occluded.sum()
                pickle.dump(res_dict, f)
                f.close()
                return

            obj_start_pose_dict = {}
            for obj_i, obj in self.perception.objects.items():
                obj_start_pose_dict[obj_i] = obj.transform
            collision_voxel = self.prev_occluded | (self.prev_occupied_label>0)
            collision_dict = copy.deepcopy(self.prev_occluded_dict)
            for obj_i, occupied_i in self.prev_occupied_dict.items():
                collision_dict[obj_i] |= occupied_i
            occlusion_voxel = self.prev_occluded
            sampled_actions = self.random_action(valid_objects, moved_objects, obj_start_pose_dict,
                                        num_obj_sample=1, num_placement_sample=1,
                                        collision_voxel=collision_voxel,
                                        collision_dict=collision_dict, 
                                        occlusion_voxel=occlusion_voxel)
            
            if len(sampled_actions) == 0:
                continue
            move_obj_idx = list(sampled_actions.keys())[0]
            action_dict = sampled_actions[move_obj_idx]
            if 'pose_and_traj_list' not in action_dict:
                # * move and sense
                self.move_and_sense(move_obj_idx, action_dict, collision_voxel)
                moved_objects.append(move_obj_idx)  # has been moved
            else:
                # * rearrange
                # randomly select one object
                pose_and_traj_list = action_dict['pose_and_traj_list']
                if len(pose_and_traj_list) == 0:
                    continue
                sample_i = np.random.choice(len(pose_and_traj_list))
                pose, suction_traj, lift_traj, transfer_traj, drop_traj, reset_traj = pose_and_traj_list[sample_i]
                self.move(move_obj_idx, pose, suction_traj, lift_traj, transfer_traj, drop_traj, reset_traj)
            iter_i += 1
            self.pipeline_sim()

    def run_pipeline_multistep_lookahead(self, iter_n=10):
        print('using multi-step method...')
        # select object
        #TODO
        moved_objects = []
        iter_i = 0

        valid_objects = []
        start_time = time.time()
        while True:
            print('iteration: ', iter_i)
            gc.collect()
            # select object: active objects but not moved
            active_objs = []
            for obj_id, obj in self.perception.objects.items():
                # check if the object becomes active when the hiding object have all been moved
                obj_hide_list = list(obj.obj_hide_set)
                print('object ', self.perception.data_assoc.obj_ids_reverse[obj_id], ' hiding list: ')
                for k in range(len(obj_hide_list)):
                    print(self.perception.data_assoc.obj_ids_reverse[obj_hide_list[k]])
                if obj.active:
                    active_objs.append(obj_id)
                    if (obj_id not in valid_objects):
                        valid_objects.append(obj_id)  # a new object that becomes valid

            # move_obj_idx = np.random.choice(valid_objects)
            print('moved object list: ')
            for k in moved_objects:
                print(self.perception.data_assoc.obj_ids_reverse[k])
            print('valid but unmoved object list:')
            for k in list(set(valid_objects)-set(moved_objects)):
                print(self.perception.data_assoc.obj_ids_reverse[k])
            # Terminate when the entire space has been observed???
            # TODO: need to visualize to make sure workspace does not cause problems
            # TODO: also visualize in our planner if the space becomes empty at convergence
            if (self.prev_occluded).sum() == 0 or len(moved_objects) == self.num_obj or time.time() - start_time > self.timeout:
                print('self.prev_occluded.sum: ', self.prev_occluded.sum())
                print('moved_objects: ', len(moved_objects))
                print('num_obj: ', self.num_obj)
                print('time: ', time.time() - start_time)
                print('timeout: ', self.timeout)


                running_time = time.time() - start_time
                print('#############Finished##############')
                print('number of reconstructed objects: ', len(moved_objects))
                print('number of executed actions: ', self.num_executed_actions)
                print('running time: ', time.time() - start_time, 's')

                import pickle
                f = open('multistep-lookahead-' + self.prob_name + '-trial-' + str(self.trial_num) + '-result.pkl', 'wb')
                res_dict = {}
                res_dict['num_reconstructed_objs'] = len(moved_objects)
                res_dict['num_collision'] = self.num_collision
                res_dict['running_time'] = running_time
                res_dict['num_executed_actions'] = self.num_executed_actions
                res_dict['perception_time'] = self.perception_time
                res_dict['motion_planning_time'] = self.motion_planning_time + self.rearrange_planner.motion_planning_time
                res_dict['pose_generation_time'] = self.pose_generation_time + self.rearrange_planner.pose_generation_time
                res_dict['rearrange_time'] = self.rearrange_time
                res_dict['ros_time'] = self.ros_time
                res_dict['perception_calls'] = self.perception_calls
                res_dict['motion_planning_calls'] = self.motion_planning_calls + self.rearrange_planner.motion_planning_calls
                res_dict['pose_generation_calls'] = self.pose_generation_calls + self.rearrange_planner.pose_generation_calls
                res_dict['rearrange_calls'] = self.rearrange_calls
                res_dict['execution_calls'] = self.execution_calls
                res_dict['final_occluded_volume'] = self.prev_occluded.sum()
                pickle.dump(res_dict, f)
                f.close()
                return

            obj_start_pose_dict = {}
            for obj_i, obj in self.perception.objects.items():
                obj_start_pose_dict[obj_i] = obj.transform
            collision_voxel = self.prev_occluded | (self.prev_occupied_label>0)
            collision_dict = copy.deepcopy(self.prev_occluded_dict)
            for obj_i, occupied_i in self.prev_occupied_dict.items():
                collision_dict[obj_i] |= occupied_i
            occlusion_voxel = self.prev_occluded

            sampled_actions = self.random_greedy_action(valid_objects, moved_objects, obj_start_pose_dict,
                                        num_obj_sample=4, num_placement_sample=2,
                                        collision_voxel=collision_voxel,
                                        collision_dict=collision_dict, 
                                        occlusion_voxel=occlusion_voxel)
            
            if len(sampled_actions) == 0:
                continue
            # select the action with the max info-gain
            max_obj_idx = -1
            max_info_gain = -1
            max_pose_i = -1
            for obj_idx, action in sampled_actions.items():
                if 'pose_and_traj_list' not in action:
                    # reconstruction action
                    info_gain = action['info_gains']
                    if info_gain > max_info_gain:
                        max_info_gain = info_gain
                        max_obj_idx = obj_idx
                else:
                    if len(action['info_gains']) == 0:
                        continue
                    info_gain_i = np.argmax(action['info_gains'])
                    info_gain = action['info_gains'][info_gain_i]
                    if info_gain > max_info_gain:
                        max_info_gain = info_gain
                        max_obj_idx = obj_idx
                        max_pose_i = info_gain_i
            if max_obj_idx == -1:
                continue

            move_obj_idx = max_obj_idx
            action_dict = sampled_actions[move_obj_idx]
            # input('max_info_gain: %f ' % (max_info_gain))
            # input('object id: : %d ' % (self.perception.data_assoc.obj_ids_reverse[max_obj_idx]))
            # if 'pose_and_traj_list' not in action_dict:
            #     input('action type: reconstruction')
            # else:
            #     input('action type: move')

            if 'pose_and_traj_list' not in action_dict:
                # * move and sense
                self.move_and_sense(move_obj_idx, action_dict, collision_voxel)
                moved_objects.append(move_obj_idx)  # has been moved
            else:
                # * rearrange
                # randomly select one object
                pose_and_traj_list = action_dict['pose_and_traj_list']
                if len(pose_and_traj_list) == 0:
                    continue
                sample_i = max_pose_i
                pose, suction_traj, lift_traj, transfer_traj, drop_traj, reset_traj = pose_and_traj_list[sample_i]
                self.move(move_obj_idx, pose, suction_traj, lift_traj, transfer_traj, drop_traj, reset_traj)
            iter_i += 1
            self.pipeline_sim()


    def run_pipeline(self):
        if self.algo_type == 0:
            self.run_pipeline_random()
        elif self.algo_type == 1:
            self.run_pipeline_multistep_lookahead()
        else:
            raise RuntimeError("only support algorithm of type 0 (random) or 1 (multi-step-lookahead). Caught an unknown input %d" %(self.algo_type))            



import sys
def main():
    rospy.init_node("task_planner_random")
    rospy.sleep(1.0)
    scene_name = 'scene1'
    prob_name = sys.argv[1]
    trial_num = int(sys.argv[2])
    algo_type = int(sys.argv[3])  # 0: random  1: multi-step-lookahead
    timeout = int(sys.argv[4])
    num_obj = int(sys.argv[5])
    task_planner = TaskPlannerRandom(scene_name, prob_name, trial_num, algo_type, timeout, num_obj)
    # input('ENTER to start planning...')
    task_planner.run_pipeline()

if __name__ == "__main__":
    main()