"""
provide high-level action planning
integrate with other components
"""
from re import S
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
from camera import Camera

class TaskPlanner():
    def __init__(self, scene_name, prob_name, trial_num):
        self.planning_system = PlanningSystem(scene_name)
        self.prob_name = prob_name
        self.trial_num = trial_num

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

        perception_pipeline = PerceptionSystem(occlusion_params, object_params, target_params, tsdf_color_flag=True)

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
        
        cam_pos =np.array([-0.4, 0, 2])
        look_at =np.array([1.37, 0, 1.0])

        # cam_pos=[0.39,0.0,1.3]
        # look_at=[1.44,0,0.7]        

        # cam_pos = (look_at - cam_pos1) * 0.14 + cam_pos1

        self.vis_cam = Camera(cam_pos=cam_pos, look_at=look_at, fov=60, far=4, img_size=1280)
        self.saved_joint_dict_list = []

        self.vis_cam_imgs = []
        self.main_cam_imgs = []
        self.conserv_vol_list = []
        self.opt_vol_list = []
        self.occlude_vol_list = []
        self.num_execute = -1
        self.move_obj_id = -1
        # self.vis = o3d.visualization.Visualizer()
        # self.vis.create_window(visible = False)

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

        # * save the joint list
        self.saved_joint_dict_list.append(joint_dict_list)

        # * save the conservative volume of the object
        target_obj_id = self.move_obj_id
        obj = self.perception.objects[target_obj_id]

        def visualize_obj(model):
            # vis = o3d.visualization.Visualizer()
            # vis.create_window()
            pcd = obj.sample_pcd(model) / obj.resol
            pcd_ind = np.floor(pcd).astype(int)
            # Get vertex colors
            rgb_vals = obj.color_tsdf[pcd_ind[:, 0], pcd_ind[:, 1], pcd_ind[:, 2]] / 255        
            pcd = visualize_pcd(pcd, rgb_vals)
            bbox = visualize_bbox(obj.voxel_x, obj.voxel_y, obj.voxel_z)
            # voxel = visualize_voxel(obj.voxel_x, obj.voxel_y, obj.voxel_z, model, [1,0,0])

            center = [obj.voxel_y.shape[0]/2, obj.voxel_y.shape[1]/2, obj.voxel_y.shape[2]/2]  # look_at target
            eye = [-obj.voxel_y.shape[0]*1, obj.voxel_y.shape[1]/2, obj.voxel_y.shape[2]*2]  # camera position
            up = [0, 0, 1]  # camera orientation

            render = setup_render(center, eye, up)
            # Show the original coordinate axes for comparison.
            # X is red, Y is green and Z is blue.
            render.scene.show_axes(True)
            # Define a simple unlit Material.
            # (The base color does not replace the arrows' own colors.)
            # mtl = create_material([1.0, 1.0, 1.0, 0.01], 'defaultUnlit')
            mtl2 = create_material([1.0, 1.0, 1.0, 1], 'defaultUnlit')
            # render.scene.add_geometry("voxel", voxel, mtl)
            render.scene.add_geometry("pcd", pcd, mtl2)
            render.scene.add_geometry("bbox", bbox, mtl2)
            # Read the image into a variable
            img_o3d = render.render_to_image()
            # Display the image in a separate window
            # (Note: OpenCV expects the color in BGR format, so swop red and blue.)
            # img_cv2 = cv2.cvtColor(np.array(img_o3d), cv2.COLOR_RGBA2BGRA)
            # cv2.imshow("Preview window", img_cv2)
            # cv2.waitKey()
            return img_o3d

        consv_img = visualize_obj(obj.get_conservative_model())
        opt_img = visualize_obj(obj.get_optimistic_model())
        # self.conserv_vol_list.append(consv_img)
        # self.opt_vol_list.append(opt_img)


        # * visualize occlusion space
        # Get vertex colors
        occlusion = self.perception.occlusion
        bbox = visualize_bbox(occlusion.voxel_x, occlusion.voxel_y, occlusion.voxel_z)
        # voxel = visualize_voxel(occlusion.voxel_x, occlusion.voxel_y, occlusion.voxel_z, self.prev_occluded, [1,0,0])
        pcd = occlusion.sample_pcd(self.prev_occluded) / occlusion.resol



        # o3d.visualization.draw_geometries([voxel])

        center = [occlusion.voxel_y.shape[0]/2, occlusion.voxel_y.shape[1]/2, occlusion.voxel_y.shape[2]/2]  # look_at target
        eye = [-occlusion.voxel_y.shape[0]*0.9, occlusion.voxel_y.shape[1]/2, occlusion.voxel_y.shape[2]*3]  # camera position
        up = [0, 0, 1]  # camera orientation


        render = setup_render(center, eye, up)
        # Show the original coordinate axes for comparison.
        # X is red, Y is green and Z is blue.
        render.scene.show_axes(True)
        # Define a simple unlit Material.
        # (The base color does not replace the arrows' own colors.)
        mtl = create_material([1.0, 0.0, 0.0, 1], 'defaultUnlit')
        mtl2 = create_material([1.0, 1.0, 1.0, 1], 'defaultUnlit')
        render.scene.add_geometry("bbox", bbox, mtl2)

        if len(pcd) > 0:
            pcd = visualize_pcd(pcd, np.array([[1,0,0] for i in range(len(pcd))]))
            render.scene.add_geometry("pcd", pcd, mtl)

        # Read the image into a variable
        img_o3d = render.render_to_image()
        # Display the image in a separate window
        # (Note: OpenCV expects the color in BGR format, so swop red and blue.)
        # img_cv2 = cv2.cvtColor(np.array(img_o3d), cv2.COLOR_RGBA2BGRA)
        # cv2.imshow("Preview window", img_cv2)
        # cv2.waitKey()
        # self.occlude_vol_list.append(img_o3d)

        # capture image and save to file
        self.num_execute += 1
        import os
        os.makedirs('demo-videos/%s/conrv-img/' % (self.prob_name), exist_ok=True)
        os.makedirs('demo-videos/%s/opt-img/' % (self.prob_name), exist_ok=True)
        os.makedirs('demo-videos/%s/occlude/' % (self.prob_name), exist_ok=True)

        fname = 'demo-videos/%s/conrv-img/%d.png' % (self.prob_name, self.num_execute)
        # vis cam
        img_cv2 = cv2.cvtColor(np.array(consv_img), cv2.COLOR_RGBA2BGRA)
        cv2.imwrite(fname, img_cv2)

        fname = 'demo-videos/%s/opt-img/%d.png' % (self.prob_name, self.num_execute)
        img_cv2 = cv2.cvtColor(np.array(opt_img), cv2.COLOR_RGBA2BGRA)
        cv2.imwrite(fname, img_cv2)


        fname = 'demo-videos/%s/occlude/%d.png' % (self.prob_name, self.num_execute)
        img_cv2 = cv2.cvtColor(np.array(img_o3d), cv2.COLOR_RGBA2BGRA)
        cv2.imwrite(fname, img_cv2)

        # plt.imshow()
        # save the occlusion of the scene

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
        start_time = time.time()
        color_img, depth_img, seg_img = self.get_image()
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
    
    def generate_rot_traj(self, start_joint_dict, waypoint, dtheta=15*np.pi/180):
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



    def move_and_sense(self, move_obj_idx, moved_objects):
        """
        move the valid object out of the workspace, sense it and the environment, and place back
        """
        start_time = time.time()
        self.planning_system.motion_planner.clear_octomap()
        self.motion_planning_time += time.time() - start_time
        # self.motion_planning_calls += 1
        print('handling object: ', self.perception.data_assoc.obj_ids_reverse[move_obj_idx])
        start_time = time.time()
        _, suction_poses_in_obj, suction_joints = self.pre_move(move_obj_idx, moved_objects)
        end_time = time.time()
        print('pre_move takes time: ', end_time - start_time)
        print('number of generated suction poses: ', len(suction_poses_in_obj))
        if len(suction_joints) == 0:  # no valid suction joint now
            return False

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
        self.set_collision_env(list(self.prev_occluded_dict.keys()), [move_obj_idx], [move_obj_idx], padding=3)

        self.move_obj_id = move_obj_idx

        print('number of suction_poses_in_obj: ', len(suction_poses_in_obj))
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

            self.execute_traj(pick_joint_dict_list, self.perception.data_assoc.obj_ids_reverse[move_obj_idx], duration=0.3)
            self.attach_obj(move_obj_idx)
            self.execute_traj(lift_joint_dict_list + retreat_joint_dict_list)
            
            self.pipeline_sim()  # sense the environmnet

            for k in range(6):
                start_time = time.time()

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

                # input("after sensing")
            planning_info = dict()
            planning_info['tip_pose_in_obj'] = suction_pose_in_obj

            # planning_info['motion_planner'] = self.motion_planner
            planning_info['intermediate_joint'] = intermediate_joint
            planning_info['intermediate_joint_dict_list'] = retreat_joint_dict_list
            planning_info['lift_up_joint_dict_list'] = lift_joint_dict_list
            planning_info['suction_joint_dict_list'] = pick_joint_dict_list
            planning_info['obj'] = self.perception.objects[move_obj_idx]


            occlusion_filter = np.zeros(self.prev_occluded.shape).astype(bool)
            # occlusion_filter = np.array(self.prev_occluded)
            for id, occlusion in self.prev_occluded_dict.items():
                if id == move_obj_idx:
                    continue
                # should include occlusion induced by this object
                occlusion_filter = occlusion_filter | occlusion
            # occlusion_filter[self.prev_occluded_dict[move_obj_idx]] = 0
            occlusion_filter[self.prev_occupied_dict[move_obj_idx]] = 0

            self.planning_system.motion_planner.set_collision_env(self.perception.occlusion, 
                                                                occlusion_filter, (self.prev_occupied_label>0)&(self.prev_occupied_label!=move_obj_idx+1))
            start_time = time.time()        
            placement_joint_dict_list, reset_joint_dict_list = self.planning_system.plan_to_placement_pose(**planning_info)
            end_time = time.time()
            self.motion_planning_time += end_time - start_time
            self.motion_planning_calls += 1
            del planning_info
            print('plan_to_placement_pose takes time: ', end_time - start_time)

            self.execute_traj(placement_joint_dict_list)
            self.detach_obj()

            self.execute_traj(reset_joint_dict_list, self.perception.data_assoc.obj_ids_reverse[move_obj_idx])
            return True
        return False

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


    def pre_move_compute_valid_joints(self, target_obj_i, moved_objects, blocking_mask):
        target_obj = self.perception.objects[target_obj_i]
        occlusion = self.perception.occlusion
        start_time = time.time()
        valid_pts, valid_orientations, valid_joints = \
            pose_generation.grasp_pose_generation(target_obj, 
                                                  self.robot, self.workspace, 
                                                  self.perception.occlusion.transform, 
                                                  self.prev_occlusion_label>0, self.perception.occlusion.resol, sample_n=20)
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



        for i in range(len(valid_orientations)):
            # obtain robot pcd at the given joint
            rpcd = self.robot.get_pcd_at_joints(valid_joints[i])
            # robot pcd in the occlusion
            transformed_rpcd = occlusion.world_in_voxel_rot.dot(rpcd.T).T + occlusion.world_in_voxel_tran
            transformed_rpcd = transformed_rpcd / occlusion.resol
            # trasnformed_rpcd_before_floor = transformed_rpcd


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
                for obj_i, obj in self.perception.objects.items():
                    if obj_i == target_obj_i:
                        continue

                    occlusion_i = self.prev_occluded_dict[obj_i]
                    occupied_i = self.prev_occupied_dict[obj_i]
                    occupied = occupied | occupied_i  # for vis
                    occluded = occluded | occlusion_i # for vis

            else:
                # check if colliding with any objects
                blocking_objects = []
                for obj_i, obj in self.perception.objects.items():
                    if obj_i == target_obj_i:
                        continue
                    occlusion_i = self.prev_occluded_dict[obj_i]
                    occupied_i = self.prev_occupied_dict[obj_i]
                    occupied = occupied | occupied_i  # for vis
                    occluded = occluded | occlusion_i # for vis

                    if occupied_i[transformed_rpcd[:,0],transformed_rpcd[:,1],transformed_rpcd[:,2]].sum() > 0:
                        blocking_objects.append(obj_i)
                        valid = False
                        # print('blocking with object ', self.perception.data_assoc.obj_ids_reverse[obj_i])
                        # if obj_i in moved_objects:
                        #     print('blocking object is moved before')
                        # else:
                        #     print('blocking object has not been moved')

                # v_voxel = visualize_voxel(self.perception.occlusion.voxel_x, self.perception.occlusion.voxel_y, self.perception.occlusion.voxel_z,
                #                 occupied, [0,0,1])
                # v_pcd = visualize_pcd(transformed_rpcd, [1,0,0])
                # o3d.visualization.draw_geometries([v_voxel, v_pcd])
                


            # * make sure there is no object in the straight-line path
            for obj_i, obj in self.perception.objects.items():
                if obj_i == target_obj_i:
                    continue
                occupied_i = self.prev_occupied_dict[obj_i]
                if (occupied_i & blocking_mask).sum() > 0:
                    blocking_objects.append(obj_i)
                    valid = False
            # * also make sure there is no object in the visibility constraint
                            
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
    

    def pre_move(self, target_obj_i, moved_objects):
        """
        before moving the object, check reachability constraints. Rearrange the blocking objects
        """
        # * check reachability constraints by sampling grasp poses
        target_obj = self.perception.objects[target_obj_i]
        occlusion = self.perception.occlusion

        # get the target object pcd
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
        # NOTE: Update: we need to compute the visibility blocking mask no matter what, because we need
        # to ensure other objects won't be rearranged and block the target object again

        blocking_mask = blocking_mask | self.obtain_visibility_blocking_mask(target_obj)


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


        # # # # visualize the blocking mask
        # vis_voxels = []
        # for obj_i, obj in self.perception.objects.items():
        #     occupied_i = self.prev_occupied_dict[obj_i]
        #     if occupied_i.sum() == 0:
        #         continue
        #     vis_voxel = visualize_voxel(occlusion.voxel_x, occlusion.voxel_y, occlusion.voxel_z, occupied_i, [0,0,1])
        #     vis_voxels.append(vis_voxel)

        # if blocking_mask.sum() > 0:
        #     vis_voxel = visualize_voxel(occlusion.voxel_x, occlusion.voxel_y, occlusion.voxel_z, blocking_mask, [1,0,0])
        #     o3d.visualization.draw_geometries(vis_voxels + [vis_voxel,vis_pcd])
        # else:
        #     o3d.visualization.draw_geometries(vis_voxels + [vis_pcd])


        valid_pts, valid_orientations, valid_joints, status, \
            joint_indices, total_blocking_object_nums, \
            total_blocking_objects, transformed_rpcds \
                = self.pre_move_compute_valid_joints(target_obj_i, moved_objects, blocking_mask)
        
        if (status == 1) or (len(valid_pts) == 0):
            return valid_pts, valid_orientations, valid_joints

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
        for obj_i, obj in self.perception.objects.items():
            if obj_i in moved_objects:
                continue
            collision_voxel = collision_voxel | self.prev_occluded_dict[obj_i]


        transform = self.perception.occlusion.transform
        transform = np.linalg.inv(transform)
        voxel_x, voxel_y, voxel_z = np.indices(collision_voxel.shape).astype(int)
        # remove start poses of objects in collision voxel
        for i in range(len(blocking_objects)):
            pcd = self.perception.objects[blocking_objects[i]].sample_conservative_pcd()
            obj_start_pose = self.perception.objects[blocking_objects[i]].transform
            transformed_pcd = obj_start_pose[:3,:3].dot(pcd.T).T + obj_start_pose[:3,3]
            transformed_pcd = transform[:3,:3].dot(transformed_pcd.T).T + transform[:3,3]
            transformed_pcd = transformed_pcd / self.perception.occlusion.resol
            transformed_pcd = np.floor(transformed_pcd).astype(int)
            collision_voxel = self.mask_pcd_xy_with_padding(collision_voxel, transformed_pcd, padding=0)


            # robot_collision_voxel[transformed_pcd[:,0],transformed_pcd[:,1],transformed_pcd[:,2]] = 0  # mask out
        for i in range(len(moveable_objs)):
            pcd = self.perception.objects[moveable_objs[i]].sample_conservative_pcd()
            obj_start_pose = self.perception.objects[moveable_objs[i]].transform
            transformed_pcd = obj_start_pose[:3,:3].dot(pcd.T).T + obj_start_pose[:3,3]
            transformed_pcd = transform[:3,:3].dot(transformed_pcd.T).T + transform[:3,3]
            transformed_pcd = transformed_pcd / self.perception.occlusion.resol
            transformed_pcd = np.floor(transformed_pcd).astype(int)
            collision_voxel = self.mask_pcd_xy_with_padding(collision_voxel, transformed_pcd, padding=0)


        robot_collision_voxel = np.array(collision_voxel).astype(bool)
        robot_collision_voxel[transformed_rpcd[:,0],transformed_rpcd[:,1],transformed_rpcd[:,2]] = 1

        # add collision to make sure we can extract the object in straight-line
        blocking_mask = self.obtain_straight_blocking_mask(target_obj)
        robot_collision_voxel = robot_collision_voxel | blocking_mask

        # add visibility "collision" to make sure the goal does not hide potential object to be revealed
        blocking_mask = self.obtain_visibility_blocking_mask(target_obj)
        robot_collision_voxel = robot_collision_voxel | blocking_mask


        # * rearrange the blocking objects
        res = self.rearrange(blocking_objects, moveable_objs, collision_voxel, robot_collision_voxel)
        del collision_voxel
        del robot_collision_voxel
        del target_pcd
        del valid_filter

        if res:
            # update the occlusion and occupied space so motion planning won't be messed up
            self.pipeline_sim()
            # do another check for blocking objects since we may reveal more grasp poses
            valid_pts, valid_orientations, valid_joints, status, \
                joint_indices, total_blocking_object_nums, \
                total_blocking_objects, transformed_rpcds \
                    = self.pre_move_compute_valid_joints(target_obj_i, moved_objects, blocking_mask)
            del blocking_mask
            if status == 1:
                return [valid_pt] + valid_pts, [valid_orientation] + valid_orientations, [valid_joint] + valid_joints
            else:
                return [valid_pt], [valid_orientation], [valid_joint]
        else:
            del blocking_mask
            return [], [], []


    def rearrange(self, obj_ids, moveable_obj_ids, collision_voxel, robot_collision_voxel):
        """
        rearrange the blocking objects
        """
        obj_pcds = [self.perception.objects[obj_i].sample_conservative_pcd() for obj_i in obj_ids]
        moveable_obj_pcds = [self.perception.objects[obj_i].sample_conservative_pcd() for obj_i in moveable_obj_ids]
        obj_start_poses = [self.perception.objects[obj_i].transform for obj_i in obj_ids]
        moveable_obj_start_poses = [self.perception.objects[obj_i].transform for obj_i in moveable_obj_ids]
        moved_objs = [self.perception.objects[obj_i] for obj_i in obj_ids]
        moveable_objs = [self.perception.objects[obj_i] for obj_i in moveable_obj_ids]

        plt.ion()
        # plt.figure(figsize=(10,10))
        start_time = time.time()
        searched_objs, transfer_trajs, searched_trajs, reset_traj = \
            self.rearrange_planner.rearrangement_plan(moved_objs, obj_pcds, obj_start_poses, moveable_objs, 
                                            moveable_obj_pcds, moveable_obj_start_poses,
                                            collision_voxel, robot_collision_voxel, 
                                            self.perception.occlusion.transform, 
                                            self.perception.occlusion.resol,
                                            self.robot, self.workspace, self.perception.occlusion,
                                            self.planning_system.motion_planner,
                                            n_iter=10)
        self.rearrange_time += time.time() - start_time
        self.rearrange_calls += 1

        success = False
        if searched_objs is not None:
            total_obj_ids = obj_ids + moveable_obj_ids
            # execute the rearrangement plan
            for i in range(len(searched_objs)):
                move_obj_idx = total_obj_ids[searched_objs[i]]
                self.move_obj_id = move_obj_idx                
                self.execute_traj(transfer_trajs[i], self.perception.data_assoc.obj_ids_reverse[move_obj_idx], duration=0.3)
                self.attach_obj(move_obj_idx)
                self.execute_traj(searched_trajs[i])
                self.detach_obj()
            # reset
            self.execute_traj(reset_traj, self.perception.data_assoc.obj_ids_reverse[move_obj_idx])
            success = True
        else:
            success = False
        # input('after rearrange...')

        del obj_pcds
        del moveable_obj_pcds
        del obj_start_poses
        del moveable_obj_start_poses
        del moved_objs
        del moveable_objs

        return success

    def run_pipeline(self, iter_n=10):
        # select object
        #TODO
        # * demo pipeline
        # moved_objects = []
        # iter_i = 0

        # valid_objects = []
        # start_time = time.time()

        # sequence = [7,8,10,6,9]  # only for 5-1-1
        # while True:
        #     print('iteration: ', iter_i)
        #     gc.collect()
        #     # select object: active objects but not moved
        #     active_objs = []
        #     for obj_id, obj in self.perception.objects.items():
        #         # check if the object becomes active when the hiding object have all been moved
        #         # if len(obj.obj_hide_set - set(moved_objects)) == 0:
        #         #     obj.set_active()
        #         # UPDATE: we are now using image to decide if an object is active or not

        #         obj_hide_list = list(obj.obj_hide_set)
        #         print('object ', self.perception.data_assoc.obj_ids_reverse[obj_id], ' hiding list: ')
        #         for k in range(len(obj_hide_list)):
        #             print(self.perception.data_assoc.obj_ids_reverse[obj_hide_list[k]])
        #         if obj.active:
        #             active_objs.append(obj_id)

        #             if (obj_id not in valid_objects) and (obj_id not in moved_objects):
        #                 valid_objects.append(obj_id)  # a new object that becomes valid
        #     # valid_objects = set(active_objs) - set(moved_objects)
        #     # valid_objects = list(valid_objects)

        #     # move_obj_idx = np.random.choice(valid_objects)
        #     print('valid object list: ')
        #     # for k in range(len(valid_objects)):
        #     #     print(self.perception.data_assoc.obj_ids_reverse[valid_objects[k]])
        #     # # Terminated when there are no valid_objects left
        #     #     return

        #     # move_obj_idx = valid_objects[0]
        #     move_obj_idx = self.perception.data_assoc.obj_ids[sequence[0]]
        #     # if iter_i < len(orders):
        #     #     move_obj_idx = orders[iter_i]

        #     # move_obj_idx = 0
        #     iter_i += 1
        #     res = self.move_and_sense(move_obj_idx, moved_objects)
        #     if res == True:
        #         moved_objects.append(move_obj_idx)
        #         # valid_objects.pop(0)
        #         sequence.pop(0)
        #         # # visualize the reconstructed object
        #         # if self.perception.data_assoc.obj_ids[move_obj_idx] == 9:
        #         #     save_flag = 1
        #         # else:
        #         #     save_flag = 0
        #         # self.perception.objects[move_obj_idx].get_conservative_surface(save_flag)

        #     else:
        #         pass
        #         # put the first element at the back so we can try again
        #         # valid_objects.pop(0)
        #         # valid_objects.append(move_obj_idx)
        #     self.pipeline_sim()
        #     if len(sequence) == 0:
        #         # save the stuff
        #         return


        # select object
        # #TODO
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
                # if len(obj.obj_hide_set - set(moved_objects)) == 0:
                #     obj.set_active()
                # UPDATE: we are now using image to decide if an object is active or not

                obj_hide_list = list(obj.obj_hide_set)
                print('object ', self.perception.data_assoc.obj_ids_reverse[obj_id], ' hiding list: ')
                for k in range(len(obj_hide_list)):
                    print(self.perception.data_assoc.obj_ids_reverse[obj_hide_list[k]])
                if obj.active:
                    active_objs.append(obj_id)

                    if (obj_id not in valid_objects) and (obj_id not in moved_objects):
                        valid_objects.append(obj_id)  # a new object that becomes valid
            # valid_objects = set(active_objs) - set(moved_objects)
            # valid_objects = list(valid_objects)

            # move_obj_idx = np.random.choice(valid_objects)
            print('valid object list: ')
            for k in range(len(valid_objects)):
                print(self.perception.data_assoc.obj_ids_reverse[valid_objects[k]])
            # Terminated when there are no valid_objects left
            if len(valid_objects) == 0:
                running_time = time.time() - start_time
                print('#############Finished##############')
                print('number of reconstructed objects: ', len(moved_objects))
                print('number of executed actions: ', self.num_executed_actions)
                print('running time: ', time.time() - start_time, 's')
                return

            move_obj_idx = valid_objects[0]

            # if iter_i < len(orders):
            #     move_obj_idx = orders[iter_i]

            # move_obj_idx = 0
            iter_i += 1
            res = self.move_and_sense(move_obj_idx, moved_objects)
            if res == True:
                moved_objects.append(move_obj_idx)
                valid_objects.pop(0)
            else:
                # put the first element at the back so we can try again
                valid_objects.pop(0)
                valid_objects.append(move_obj_idx)
            self.pipeline_sim()


import sys
def main():
    rospy.init_node("task_planner")
    rospy.sleep(1.0)
    scene_name = 'scene1'
    prob_name = sys.argv[1]
    trial_num = int(sys.argv[2])
    
    task_planner = TaskPlanner(scene_name, prob_name, trial_num)
    # input('ENTER to start planning...')
    task_planner.run_pipeline()

if __name__ == "__main__":
    main()