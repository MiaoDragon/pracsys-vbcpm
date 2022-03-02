"""
rearrangement functions to be used in the pipeline to move objects
so that they remain valid.
Combine with robot motion planning and IK checking so we can transfer the solutions
easily
"""
import numpy as np
import matplotlib.pyplot as plt
import heapq
import transformations as tf
import pose_generation
import collision_utils
from visual_utilities import *
import open3d as o3d
import rospy

import gc

def projection_rot_matrix(obj_poses):
    obj_2d_poses = []
    for i in range(len(obj_poses)):
        obj_pose = obj_poses[i]
        # projection from 3D rotation to 2D
        vec = np.array([1.0, 0, 0])
        rot_vec = obj_pose[:3,:3].dot(vec)
        rot_vec[2] = 0
        rot_vec = rot_vec / np.linalg.norm(rot_vec)

        # print('obj pose: ')
        # print(obj_pose)
        # print('rot_vec: ')
        # print(rot_vec)

        angle = np.arctan2(rot_vec[1], rot_vec[0])
        obj_2d_pose = np.zeros((3,3))
        obj_2d_pose[2,2] = 1
        obj_2d_pose[0,0] = np.cos(angle)
        obj_2d_pose[0,1] = -np.sin(angle)
        obj_2d_pose[1,0] = np.sin(angle)
        obj_2d_pose[1,1] = np.cos(angle)
        obj_2d_pose[:2,2] = obj_pose[:2,3]
        obj_2d_poses.append(obj_2d_pose)
    return obj_2d_poses

def projection_2d_state(obj_poses):
    obj_2d_states = []
    for i in range(len(obj_poses)):
        obj_pose = obj_poses[i]
        # projection from 3D rotation to 2D
        vec = np.array([1.0, 0, 0])
        rot_vec = obj_pose[:3,:3].dot(vec)
        rot_vec[2] = 0
        rot_vec = rot_vec / np.linalg.norm(rot_vec)
        angle = np.arctan2(rot_vec[1], rot_vec[0])
        obj_2d_states.append([obj_pose[0,3], obj_pose[1,3], angle])
    return obj_2d_states


def states_from_2d_pose(obj_poses):
    # TODO: debug

    obj_2d_states = []
    for i in range(len(obj_poses)):
        obj_pose = obj_poses[i]
        x = obj_pose[0,2]
        y = obj_pose[1,2]
        theta = np.arctan2(obj_pose[1,0], obj_pose[0,0])
        obj_2d_states.append([x,y,theta])
    return obj_2d_states

def poses_from_2d_state(obj_states):
    # TODO: debug

    obj_2d_poses = []
    for i in range(len(obj_states)):
        obj_2d_poses.append(pose_from_2d_state(obj_states[i]))
    return obj_2d_poses

def state_from_2d_pose(obj_pose):
    x = obj_pose[0,2]
    y = obj_pose[1,2]
    theta = np.arctan2(obj_pose[1,0], obj_pose[0,0])
    return [x,y,theta]

def pose_from_2d_state(obj_state):
    obj_2d_pose = np.zeros((3,3))
    obj_2d_pose[2,2] = 1
    obj_2d_pose[0,0] = np.cos(obj_state[2])
    obj_2d_pose[0,1] = -np.sin(obj_state[2])
    obj_2d_pose[1,0] = np.sin(obj_state[2])
    obj_2d_pose[1,1] = np.cos(obj_state[2])
    obj_2d_pose[:2,2] = obj_state[:2]
    return obj_2d_pose

def pose_from_2d_pose(obj_pose_2d, z):
    # from 2D pose to 3D pose. given the original z value 
    # NOTE: z is in voxel transform
    obj_pose = np.zeros((4,4))
    obj_pose[3,3] = 1
    theta = np.arctan2(obj_pose_2d[1,0], obj_pose_2d[0,0])

    R = tf.rotation_matrix(theta, [0,0,1])
    obj_pose[:3,:3] = R[:3,:3]
    obj_pose[:2,3] = obj_pose_2d[:2,2]
    obj_pose[2,3] = z
    return obj_pose


def obtain_pose_in_voxel(obj_poses, voxel_transform):
    # TODO: debug

    # trasnform the 3D object pose in world to 3D pose in voxel
    transformed_obj_poses = []
    for i in range(len(obj_poses)):
        # transformed_obj_pose = obj_poses[i].dot(np.linalg.inv(voxel_transform))
        transformed_obj_pose = np.linalg.inv(voxel_transform).dot(obj_poses[i])
        transformed_obj_poses.append(transformed_obj_pose)
    return transformed_obj_poses

def pose_to_pose_in_world(obj_poses, zs_in_world, voxel_transform):
    # TODO: debug

    # transform the 3D object pose in voxel to 3D pose in world
    transformed_obj_poses = []
    for i in range(len(obj_poses)):
        transformed_obj_pose = voxel_transform.dot(obj_poses[i])
        transformed_obj_pose[2,3] = zs_in_world[i]
        transformed_obj_poses.append(transformed_obj_pose)
    return transformed_obj_poses

def state_to_pose_in_world(obj_states, voxel_transform, zs, zs_in_world):
    # TODO: debug

    # transform the 3D object pose in voxel to 3D pose in world
    transformed_obj_poses = []
    for i in range(len(obj_states)):
        obj_2d_pose = pose_from_2d_state(obj_states[i])
        obj_pose = pose_from_2d_pose(obj_2d_pose, zs[i])
        transformed_obj_pose = voxel_transform.dot(obj_pose)
        transformed_obj_pose[2,3] = zs_in_world[i]
        transformed_obj_poses.append(transformed_obj_pose)
    return transformed_obj_poses



def ik_check_on_state(obj, obj_state, z, z_in_world, robot, workspace, collision_voxel, collision_transform, voxel_resol):
    # generate grasp pose based on object pose
    # TODO: implement a new grasp_pose_generation code that checks collision with the workspace and a selected
    # set of objects
    obj_pose = state_to_pose_in_world([obj_state], collision_transform, [z], [z_in_world])[0]
    return ik_check_on_pose(obj, obj_pose, robot, workspace, collision_voxel, collision_transform, voxel_resol)



def ik_check_on_2d_pose(obj, obj_2d_pose, z, z_in_world, robot, workspace, collision_voxel, collision_transform, voxel_resol):
    # generate grasp pose based on object pose
    # TODO: implement a new grasp_pose_generation code that checks collision with the workspace and a selected
    # set of objects
    obj_pose_in_voxel = pose_from_2d_pose(obj_2d_pose, z) # pose in voxel
    obj_pose = pose_to_pose_in_world([obj_pose_in_voxel], [z_in_world], collision_transform)[0]
    return ik_check_on_pose(obj, obj_pose, robot, workspace, collision_voxel, collision_transform, voxel_resol)


def ik_check_on_pose(obj, obj_pose, robot, workspace, collision_voxel, collision_transform, voxel_resol, sample_n=10):
    # generate grasp pose based on object pose
    # TODO: implement a new grasp_pose_generation code that checks collision with the workspace and a selected
    # set of objects

    # voxel_x, voxel_y, voxel_z = np.indices(collision_voxel.shape).astype(int)
    # voxel = visualize_voxel(voxel_x, voxel_y, voxel_z, collision_voxel, [0,1,0])
    # obj_pcd = obj.sample_conservative_pcd()
    # obj_pcd = obj_pose[:3,:3].dot(obj_pcd.T).T + obj_pose[:3,3]
    # transform = np.linalg.inv(collision_transform)
    # obj_pcd = transform[:3,:3].dot(obj_pcd.T).T + transform[:3,3]
    # obj_pcd = obj_pcd / voxel_resol
    # obj_pcd = visualize_pcd(obj_pcd, [1,0,0])
    # # print('ik checking on pose...')
    # o3d.visualization.draw_geometries([voxel, obj_pcd])
    

    pts, poses_in_obj, joints = pose_generation.grasp_pose_generation(None, obj, robot, workspace, None, None, None, sample_n=20)

    valid_pts = []
    valid_poses_in_obj = []
    valid_joints = []

    for i in range(len(poses_in_obj)):
        # * get transform of the tip link
        transformed_tip = obj_pose.dot(poses_in_obj[i])
        # TODO: see if the object pose is correct

        quat = tf.quaternion_from_matrix(transformed_tip)
        # * check ik
        start_suction_joint = robot.joint_vals
        valid, dof_joint_vals = robot.get_ik(robot.tip_link_name, transformed_tip[:3,3], [quat[1],quat[2],quat[3],quat[0]], 
                                            start_suction_joint, collision_check=True)
        if not valid:
            continue
        # * check collision with collision map (entire robot)
        collision = collision_utils.robot_collision_with_voxel_env(dof_joint_vals, robot, collision_transform, collision_voxel, voxel_resol)
        if collision:
            # print('robot colliding with environment...')
            continue
        valid_pts.append(pts[i])
        valid_poses_in_obj.append(poses_in_obj[i])
        valid_joints.append(dof_joint_vals)

    # select 10 of the poses
    if sample_n < len(valid_pts):
        valid_indices = np.random.choice(len(valid_pts), size=sample_n, replace=False)
        valid_pts = [valid_pts[i] for i in valid_indices]
        valid_poses_in_obj = [valid_poses_in_obj[i] for i in valid_indices]
        valid_joints = [valid_joints[i] for i in valid_indices]

    del pts
    del poses_in_obj
    del joints
    # del obj_pcd
    # del voxel_x
    # del voxel_y
    # del voxel_z
    # del voxel

    return valid_pts, valid_poses_in_obj, valid_joints


def ik_check_on_pose_with_grasp_pose(obj, obj_pose, poses_in_obj, robot, workspace, collision_voxel, collision_transform, voxel_resol):
    # generate grasp pose based on object pose
    # TODO: implement a new grasp_pose_generation code that checks collision with the workspace and a selected
    # set of objects

    # voxel_x, voxel_y, voxel_z = np.indices(collision_voxel.shape).astype(int)
    # voxel = visualize_voxel(voxel_x, voxel_y, voxel_z, collision_voxel, [0,1,0])
    # obj_pcd = obj.sample_conservative_pcd()
    # obj_pcd = obj_pose[:3,:3].dot(obj_pcd.T).T + obj_pose[:3,3]
    # transform = np.linalg.inv(collision_transform)
    # obj_pcd = transform[:3,:3].dot(obj_pcd.T).T + transform[:3,3]
    # obj_pcd = obj_pcd / voxel_resol
    # obj_pcd = visualize_pcd(obj_pcd, [1,0,0])
    # # print('ik checking on pose...')
    # o3d.visualization.draw_geometries([voxel, obj_pcd])
    

    # pts, poses_in_obj, joints = pose_generation.grasp_pose_generation(None, obj, robot, workspace, None, None, None)

    valid_pts = []
    valid_poses_in_obj = []
    valid_joints = []

    for i in range(len(poses_in_obj)):
        # * get transform of the tip link
        transformed_tip = obj_pose.dot(poses_in_obj[i])
        # TODO: see if the object pose is correct

        quat = tf.quaternion_from_matrix(transformed_tip)
        # * check ik
        start_suction_joint = robot.joint_vals
        valid, dof_joint_vals = robot.get_ik(robot.tip_link_name, transformed_tip[:3,3], [quat[1],quat[2],quat[3],quat[0]], 
                                            start_suction_joint, collision_check=True)
        if not valid:
            valid_pts.append(None)
            valid_poses_in_obj.append(None)
            valid_joints.append(None)
            continue
        # * check collision with collision map (entire robot)
        collision = collision_utils.robot_collision_with_voxel_env(dof_joint_vals, robot, collision_transform, collision_voxel, voxel_resol)
        if collision:
            # print('robot colliding with environment...')
            valid_pts.append(None)
            valid_poses_in_obj.append(None)
            valid_joints.append(None)
            continue
        valid_pts.append([])
        valid_poses_in_obj.append(poses_in_obj[i])
        valid_joints.append(dof_joint_vals)


    # del obj_pcd
    # del voxel_x
    # del voxel_y
    # del voxel_z
    # del voxel

    return valid_pts, valid_poses_in_obj, valid_joints


def ik_check_start_target_pose(obj, obj_pose, obj_start_pose, robot, workspace, collision_voxel, voxel_transform, voxel_resol):

    
    # * IK to generate joint_vals and suction poses since we have a new collision map
    target_valid_pts, target_valid_poses_in_obj, target_valid_joints = \
        ik_check_on_pose(obj, obj_pose, robot, workspace, collision_voxel, voxel_transform, voxel_resol)
    if len(target_valid_pts) == 0:
        return None, None, None, None, None, None
    

    valid_pts, valid_poses_in_obj, valid_joints = \
        ik_check_on_pose_with_grasp_pose(obj, obj_start_pose, 
                                        target_valid_poses_in_obj, 
                                        robot, workspace,
                                        collision_voxel, voxel_transform, voxel_resol)

    # cross check
    filtered_target_valid_pts = []
    filtered_target_valid_poses_in_obj = []
    filtered_target_valid_joints = []
    filtered_valid_pts = []
    filtered_valid_poses_in_obj = []
    filtered_valid_joints = []
    for pose_i in range(len(valid_poses_in_obj)):
        if valid_poses_in_obj[pose_i] is not None:
            filtered_target_valid_pts.append(target_valid_pts[pose_i])
            filtered_target_valid_poses_in_obj.append(target_valid_poses_in_obj[pose_i])
            filtered_target_valid_joints.append(target_valid_joints[pose_i])
            filtered_valid_pts.append(valid_pts[pose_i])
            filtered_valid_poses_in_obj.append(valid_poses_in_obj[pose_i])
            filtered_valid_joints.append(valid_joints[pose_i])               

    target_valid_pts = filtered_target_valid_pts
    target_valid_poses_in_obj = filtered_target_valid_poses_in_obj
    target_valid_joints = filtered_target_valid_joints
    valid_pts = filtered_valid_pts
    valid_poses_in_obj = filtered_valid_poses_in_obj
    valid_joints = filtered_valid_joints

    del filtered_target_valid_joints
    del filtered_target_valid_poses_in_obj
    del filtered_target_valid_pts
    del filtered_valid_pts
    del filtered_valid_poses_in_obj
    del filtered_valid_joints

    if len(target_valid_pts) == 0:
        return None, None, None, None, None, None
    
    return valid_pts, valid_poses_in_obj, valid_joints, target_valid_pts, target_valid_poses_in_obj, target_valid_joints



def rearrangement_plan(objs, obj_pcds, obj_start_poses, moveable_objs, moveable_obj_pcds, moveable_obj_start_poses, 
                        collision_voxel, robot_collision_voxel, voxel_transform, voxel_resol, 
                        robot, workspace, occlusion, motion_planner,
                        n_iter=25):
    # 3d voxel to 2d grid
    # TODO: debug
    print('rearranging...')
    collision_grid = collision_voxel.sum(axis=2)>0
    # in the z-axis, if there is at least one voxel occupied, then collision grid
    grid_resol = voxel_resol[:2]
    robot_collision_grid = robot_collision_voxel.sum(axis=2)>0

    # convert the start poses to 2D     # TODO: debug
    obj_start_poses_in_voxel = obtain_pose_in_voxel(obj_start_poses, voxel_transform)
    obj_2d_start_poses = projection_rot_matrix(obj_start_poses_in_voxel)
    moveable_obj_start_poses_in_voxel = obtain_pose_in_voxel(moveable_obj_start_poses, voxel_transform)
    moveable_obj_2d_start_poses = projection_rot_matrix(moveable_obj_start_poses_in_voxel)

    obj_2d_start_states = states_from_2d_pose(obj_2d_start_poses)
    moveable_obj_2d_start_states = states_from_2d_pose(moveable_obj_2d_start_poses)

    obj_pcd_2ds = obj_pcd_2d_projection(obj_pcds)
    moveable_obj_pcd_2ds = obj_pcd_2d_projection(moveable_obj_pcds)



    obj_zs = np.zeros(len(obj_pcds)).tolist()
    for i in range(len(obj_zs)):
        obj_zs[i] = obj_start_poses_in_voxel[i][2,3]
    moveable_obj_zs = np.zeros(len(moveable_obj_pcds)).tolist()
    for i in range(len(moveable_obj_zs)):
        moveable_obj_zs[i] = moveable_obj_start_poses_in_voxel[i][2,3]

    obj_zs_in_world = np.zeros(len(obj_pcds)).tolist()
    moveable_obj_zs_in_world = np.zeros(len(moveable_obj_pcds)).tolist()
    for i in range(len(obj_zs_in_world)):
        obj_zs_in_world[i] = obj_start_poses[i][2,3]
    for i in range(len(moveable_obj_zs)):
        moveable_obj_zs_in_world[i] = moveable_obj_start_poses[i][2,3]

 
    # we use the collision voxel with robot since we don't want the object to collide with the robot at grasp pose
    obj_included_list, total_obj_poses, total_start_valid_pts, total_start_valid_poses_in_obj, total_start_valid_joints, \
        total_valid_pts, total_valid_poses_in_obj, total_valid_joints = \
            sample_goal_locations(objs, obj_pcds, obj_start_poses, obj_zs, obj_zs_in_world,
                                moveable_objs, moveable_obj_pcds, moveable_obj_start_poses, moveable_obj_zs, moveable_obj_zs_in_world,
                                collision_voxel, robot_collision_voxel, voxel_resol, 
                                voxel_transform, robot, workspace, n_iter)

    # input('after sampling goal...')

    # obj_poses: for all objects (obj_pcds and moveable_objs)

    # sampled pose: relative to voxel frame. 2D

    # * concatenate with moveable     # TODO: debug

    total_obj_pcds = obj_pcds + moveable_obj_pcds
    total_obj_pcd_2ds = obj_pcd_2ds + moveable_obj_pcd_2ds
    total_obj_2d_start_poses = obj_2d_start_poses + moveable_obj_2d_start_poses
    total_obj_2d_start_states = obj_2d_start_states + moveable_obj_2d_start_states
    total_obj_start_poses = obj_start_poses + moveable_obj_start_poses
    total_obj_start_poses_in_voxel = obj_start_poses_in_voxel + moveable_obj_start_poses_in_voxel

    total_objs = objs + moveable_objs
    total_obj_zs = obj_zs + moveable_obj_zs
    total_obj_zs_in_world = obj_zs_in_world + moveable_obj_zs_in_world


    total_obj_states = []
    total_obj_target_poses = []
    total_obj_poses_in_voxel = []

    if total_obj_poses is None:
        return None, None, None, None

    for i in range(len(total_obj_poses)):
        total_obj_states.append(state_from_2d_pose(total_obj_poses[i]))
        total_obj_target_pose = pose_from_2d_pose(total_obj_poses[i], total_obj_zs[i])
        total_obj_poses_in_voxel.append(total_obj_target_pose)
        total_obj_target_pose = pose_to_pose_in_world([total_obj_target_pose], [total_obj_zs_in_world[i]], voxel_transform)[0]
        total_obj_target_poses.append(total_obj_target_pose)

    # obj_poses = obj_target_poses  # pose in world frame

    # * make sure collision space includes objects that are not moved but moveable
    # * and add safety padding
    # TODO: debug and clean up code
    collision_voxel = np.array(collision_voxel)
    for i in range(len(total_objs)):
        if not obj_included_list[i]:
            pcd = total_obj_start_poses_in_voxel[i][:3,:3].dot(total_obj_pcds[i].T).T + \
                    total_obj_start_poses_in_voxel[i][:3,3]
            pcd = pcd / voxel_resol
            pcd = np.floor(pcd).astype(int)
            valid_filter = (pcd[:,0] >= 0) & (pcd[:,0] < collision_voxel.shape[0]) & \
                            (pcd[:,1] >= 0) & (pcd[:,1] < collision_voxel.shape[1]) & \
                            (pcd[:,2] >= 0) & (pcd[:,2] < collision_voxel.shape[2])
            pcd = pcd[valid_filter]
            collision_voxel[pcd[:,0], pcd[:,1], pcd[:,2]] = 1  # include objects that are not moved into collision

    # make sure it's not too tight

    for i in range(len(total_objs)):
        if obj_included_list[i]:
            pcd = total_obj_start_poses_in_voxel[i][:3,:3].dot(total_obj_pcds[i].T).T + \
                    total_obj_start_poses_in_voxel[i][:3,3]
            pcd = pcd / voxel_resol
            pcd = np.floor(pcd).astype(int)
            valid_filter = (pcd[:,0] >= 0) & (pcd[:,0] < collision_voxel.shape[0]) & \
                            (pcd[:,1] >= 0) & (pcd[:,1] < collision_voxel.shape[1]) & \
                            (pcd[:,2] >= 0) & (pcd[:,2] < collision_voxel.shape[2])
            pcd = pcd[valid_filter]
            collision_voxel = mask_pcd_xy_with_padding(collision_voxel, pcd, padding=1)

            # total_obj_poses is 2D pose in the voxel
            # obtain the pose in voxel from the 2D pose
            obj_pose_in_voxel = total_obj_poses_in_voxel[i]

            pcd = obj_pose_in_voxel[:3,:3].dot(total_obj_pcds[i].T).T + \
                    obj_pose_in_voxel[:3,3]
            pcd = pcd / voxel_resol
            pcd = np.floor(pcd).astype(int)
            valid_filter = (pcd[:,0] >= 0) & (pcd[:,0] < collision_voxel.shape[0]) & \
                            (pcd[:,1] >= 0) & (pcd[:,1] < collision_voxel.shape[1]) & \
                            (pcd[:,2] >= 0) & (pcd[:,2] < collision_voxel.shape[2])
            pcd = pcd[valid_filter]
            collision_voxel = mask_pcd_xy_with_padding(collision_voxel, pcd, padding=1)


    
    # * given the target poses, rearrange the objects to those locations
    # DFS for searching
    # TODO: make sure that the trajectory is executable by the robot
    # preprocessed_data = preprocess(obj_pcds, objs)  # preprocess to generate Minkowski sum
    searched_objs = []  # record what object id at each step
    searched_objs_set = set()
    searched_trajs = []
    transfer_trajs = []

    search_start = 0  # record where to start search for the current depth
    valid = False

    num_move_objs = np.sum(obj_included_list)

    while len(searched_objs) < num_move_objs:
        valid = False
        for i in range(search_start, len(total_objs)):
            if i in searched_objs_set:
                continue
            if not obj_included_list[i]:
                continue
        
            if len(searched_objs) == 0:
                # use current robot joint dict
                previous_joint_dict = robot.joint_dict
            else:
                previous_joint_dict = searched_trajs[len(searched_objs)-1][-1]
            transfer_joint_dict_list, joint_dict_list = \
                find_trajectory_mp(i, total_objs[i], total_obj_pcds[i], total_obj_start_poses[i], total_obj_start_poses_in_voxel[i],
                                        total_obj_target_poses[i], total_obj_poses_in_voxel[i], total_obj_pcds, 
                                        total_obj_start_poses, total_obj_start_poses_in_voxel,
                                        total_obj_target_poses, total_obj_poses_in_voxel, searched_objs, obj_included_list,
                                        collision_voxel, voxel_resol, voxel_transform, previous_joint_dict,
                                        total_start_valid_joints[i], total_valid_joints[i],
                                        total_valid_poses_in_obj[i],
                                        occlusion, robot, motion_planner, workspace)
            valid_i = (len(transfer_joint_dict_list) > 0) and (len(joint_dict_list) > 0)
            if valid_i:
                valid = True
                break
        if valid:
            # add the new object to rearrangement list
            searched_objs.append(i)
            searched_objs_set.add(i)
            searched_trajs.append(joint_dict_list)
            transfer_trajs.append(transfer_joint_dict_list)
            search_start = 0  # reset
        else:
            # all possible choices fail at this depth. back up and change search_start
            if len(searched_objs) == 0:
                # if empty, then failed
                return None, None, None, None
            idx = searched_objs.pop()
            searched_objs_set.remove(idx)
            searched_trajs.pop()
            transfer_trajs.pop()
            search_start = idx + 1  # start from the next one

    # * now the order of arrangement is indicated by the list search_objs


    # TODO: we need to generate a transfer trajectory from the previous joint angle to the next joint angle
    # include a transfer_trajs that will happen in the middle of trajs

    # searched objs: indicate which object the trajectory is associated with
    # searched trajs: joint_dict_list indicating the trajectory of the robot

    # * reset to robot init pose
    # set up collision space
    mp_map = np.array(collision_voxel).astype(bool)
    for i in range(len(total_obj_poses_in_voxel)):
        if obj_included_list[i]:
            obj_pose = total_obj_poses_in_voxel[i]
        else:
            obj_pose = total_obj_start_poses_in_voxel[i]
        transformed_obj_pcd = obj_pose[:3,:3].dot(total_obj_pcds[i].T).T + obj_pose[:3,3]
        transformed_obj_pcd = transformed_obj_pcd / voxel_resol
        transformed_obj_pcd = np.floor(transformed_obj_pcd).astype(int)
        valid_filter = (transformed_obj_pcd[:,0] >= 0) & (transformed_obj_pcd[:,0] < mp_map.shape[0]) & \
                        (transformed_obj_pcd[:,1] >= 0) & (transformed_obj_pcd[:,1] < mp_map.shape[1]) & \
                        (transformed_obj_pcd[:,2] >= 0) & (transformed_obj_pcd[:,2] < mp_map.shape[2])
        transformed_obj_pcd = transformed_obj_pcd[valid_filter]
        mp_map[transformed_obj_pcd[:,0],transformed_obj_pcd[:,1],transformed_obj_pcd[:,2]] = 1
        
    motion_planner.clear_octomap()
    motion_planner.set_collision_env_with_filter(occlusion, mp_map)
    # TODO: transfer plan should include first going to pre-grasp pose, then going staright-line to grasp the object

    relative_tip_pose = np.eye(4)
    relative_tip_pose[:3,3] = np.array([-0.05,0,0.0]) # retreat by 0.05

    start_joint_dict = searched_trajs[-1][-1]
    tip_suction_pose = robot.get_tip_link_pose(start_joint_dict)
    rest_traj_1 = motion_planner.straight_line_motion(searched_trajs[-1][-1], tip_suction_pose, relative_tip_pose, robot, workspace=workspace)   


    start_joint_dict = rest_traj_1[-1]
    goal_joint_dict = robot.init_joint_dict
    reset_traj = motion_planner.joint_dict_motion_plan(start_joint_dict, goal_joint_dict, robot)


    del collision_grid
    del robot_collision_grid
    del obj_pcd_2ds
    del moveable_obj_pcd_2ds
    del mp_map
    del total_obj_pcds
    del total_obj_pcd_2ds
    del total_objs
    gc.collect()


    return searched_objs, transfer_trajs, searched_trajs, reset_traj


def minkowski_diff(obj_indices_i, obj_indices_j):
    # obj_indices_i = np.array(obj_grid_i.nonzero()).astype(int).T
    # obj_indices_j = np.array(obj_grid_j.nonzero()).astype(int).T
    minkowski_diff = obj_indices_i.reshape((-1,1,2)) - obj_indices_j.reshape((1,-1,2))
    minkowski_diff = minkowski_diff.reshape((-1,2))
    
    return minkowski_diff

def preprocess(obj_pcds, objs):
    # Minkowski diff between pairs of objects
    obj_grids = [objs[i].get_conservative_model().sum(axis=2).astype(bool) for i in range(len(objs))]
    obj_indices = [obj_grids[i].nonzero().astype(int).T for i in range(len(objs))]  # N X 2
    
    minkowski_diffs = [[None for i in range(len(objs))] for j in range(len(objs))]
    for i in range(len(objs)):
        for j in range(i+1, len(objs)):
            # i - j
            minkowski_diff = obj_indices[i].reshape((-1,1,2)) - obj_indices[j].reshape((1,-1,2))
            minkowski_diffs[i][j] = minkowski_diff.reshape((-1,2))
    return minkowski_diffs


# (i, objs[i], obj_pcds[i], obj_start_poses[i], obj_start_poses_in_voxel[i],
#                                         obj_target_poses[i], obj_poses[i], obj_pcds, 
#                                         obj_start_poses, obj_start_poses_in_voxel,
#                                         obj_target_poses, obj_poses, searched_objs,
#                                         collision_voxel, voxel_resol,
#                                         total_start_valid_joints[i], total_valid_joints[i],
#                                         total_valid_poses_in_obj[i],
#                                         occlusion, robot, motion_planner



def mask_pcd_xy_with_padding(occ_filter, pcd_indices, padding=1):
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


def find_trajectory_mp(obj_i, obj, obj_pcd, obj_start_pose, obj_start_pose_in_voxel,
                    obj_pose, obj_pose_in_voxel,
                    obj_pcds, obj_start_poses, obj_start_poses_in_voxel,
                    obj_poses, obj_poses_in_voxel, searched_objs, obj_included_list,
                    collision_voxel, voxel_resol, voxel_transform, previous_joint_dict,
                    start_joint_vals, target_joint_vals, tip_poses_in_obj,                     
                    occlusion, robot, motion_planner, workspace):
    # for objs in the saerched_objs, they use the final obj_poses
    # for other objects, they use the obj_start_poses
    """
    try using DFS resolution-complete search
    """
    # put object collision areas into the grid
    total_objs = set(list(range(len(obj_pcds))))
    serached_obj_set = set(searched_objs)
    unsearched_obj_set = total_objs - serached_obj_set
    unsearched_objs = list(unsearched_obj_set)

    # print('searched objects: ', searched_objs)
    # print('obj_i: ', obj_i)
    # print('obj pcd: ')
    # print(obj_pcds)
    mp_map = np.array(collision_voxel)

    x_map, y_map, z_map = np.indices(mp_map.shape).astype(int)

    # TODO: debug here why this gives us mp_map.sum() > 0?
    for obj_idx in searched_objs:
        transformed_pcd = obj_poses_in_voxel[obj_idx][:3,:3].dot(obj_pcds[obj_idx].T).T + obj_poses_in_voxel[obj_idx][:3,3]
        transformed_pcd = transformed_pcd / voxel_resol
        transformed_pcd = np.floor(transformed_pcd).astype(int)
        valid_filter = (transformed_pcd[:,0] >= 0) & (transformed_pcd[:,0] < mp_map.shape[0]) & \
                        (transformed_pcd[:,1] >= 0) & (transformed_pcd[:,1] < mp_map.shape[1]) & \
                        (transformed_pcd[:,2] >= 0) & (transformed_pcd[:,2] < mp_map.shape[2])        
        transformed_pcd = transformed_pcd[valid_filter]

        mp_map[transformed_pcd[:,0],transformed_pcd[:,1], transformed_pcd[:,2]] = 1

    for obj_idx in unsearched_objs:
        if obj_idx == obj_i:
            continue
        # if it is not in the included list (moveable but didn't get selected to be moved), then
        # we already have generated the collision space.
        if not obj_included_list[obj_idx]:
            continue
        # TODO: debug


        transformed_pcd = obj_start_poses_in_voxel[obj_idx][:3,:3].dot(obj_pcds[obj_idx].T).T + obj_start_poses_in_voxel[obj_idx][:3,3]
        transformed_pcd = transformed_pcd / voxel_resol
        transformed_pcd = np.floor(transformed_pcd).astype(int)
        valid_filter = (transformed_pcd[:,0] >= 0) & (transformed_pcd[:,0] < mp_map.shape[0]) & \
                        (transformed_pcd[:,1] >= 0) & (transformed_pcd[:,1] < mp_map.shape[1]) & \
                        (transformed_pcd[:,2] >= 0) & (transformed_pcd[:,2] < mp_map.shape[2])
        transformed_pcd = transformed_pcd[valid_filter]

        mp_map[transformed_pcd[:,0],transformed_pcd[:,1],transformed_pcd[:,2]] = 1        

    # check collision for start and goal. If they're in collision, then automatically return failure
    transformed_pcd = obj_start_pose_in_voxel[:3,:3].dot(obj_pcd.T).T + obj_start_pose_in_voxel[:3,3]

    transformed_pcd = transformed_pcd / voxel_resol
    transformed_pcd = np.floor(transformed_pcd).astype(int)
    valid_filter = (transformed_pcd[:,0] >= 0) & (transformed_pcd[:,0] < mp_map.shape[0]) & \
                    (transformed_pcd[:,1] >= 0) & (transformed_pcd[:,1] < mp_map.shape[1]) & \
                    (transformed_pcd[:,2] >= 0) & (transformed_pcd[:,2] < mp_map.shape[2])
    transformed_pcd = transformed_pcd[valid_filter]


    # input('start pose check')
    # voxel = visualize_voxel(x_map, y_map, z_map, mp_map, [0,0,1])
    # start_vis_pcd = visualize_pcd(transformed_pcd, [0,1,0])
    # o3d.visualization.draw_geometries([voxel, start_vis_pcd])



    if mp_map[transformed_pcd[:,0], transformed_pcd[:,1], transformed_pcd[:,2]].sum() > 0:
        print('start in collision')
        return [], []
    start_transformed_pcd = transformed_pcd

    transformed_pcd = obj_pose_in_voxel[:3,:3].dot(obj_pcd.T).T + obj_pose_in_voxel[:3,3]

    transformed_pcd = transformed_pcd / voxel_resol
    transformed_pcd = np.floor(transformed_pcd).astype(int)
    valid_filter = (transformed_pcd[:,0] >= 0) & (transformed_pcd[:,0] < mp_map.shape[0]) & \
                    (transformed_pcd[:,1] >= 0) & (transformed_pcd[:,1] < mp_map.shape[1]) & \
                    (transformed_pcd[:,2] >= 0) & (transformed_pcd[:,2] < mp_map.shape[2])
    transformed_pcd = transformed_pcd[valid_filter]


    # input('goal pose check')
    # voxel = visualize_voxel(x_map, y_map, z_map, mp_map, [0,0,1])
    # vis_pcd = visualize_pcd(transformed_pcd, [1,0,0])
    # o3d.visualization.draw_geometries([voxel, vis_pcd, start_vis_pcd])


    if mp_map[transformed_pcd[:,0], transformed_pcd[:,1], transformed_pcd[:,2]].sum() > 0:
        print('goal in collision')
        return [], []
    goal_transformed_pcd = transformed_pcd


    # * after making sure collision does not happen at start and goal, create a threshold filter

    mp_map = mask_pcd_xy_with_padding(mp_map, start_transformed_pcd, padding=1)

    mp_map = mask_pcd_xy_with_padding(mp_map, goal_transformed_pcd, padding=1)


    # * obtain IK for start and goal since we have a new collision map
    _, tip_poses_in_obj, start_joint_vals, _, _, target_joint_vals = \
        ik_check_start_target_pose(obj, obj_pose, obj_start_pose, robot, workspace, mp_map, voxel_transform, voxel_resol)
    if tip_poses_in_obj is None:
        return [], []  # IK failed

    # * transfer plan from previous robot joint to start pose

    transfer_mp_map = np.array(mp_map).astype(bool)
    transformed_pcd = obj_start_poses_in_voxel[obj_i][:3,:3].dot(obj_pcds[obj_i].T).T + obj_start_poses_in_voxel[obj_i][:3,3]
    transformed_pcd = transformed_pcd / voxel_resol
    transformed_pcd = np.floor(transformed_pcd).astype(int)
    
    valid_filter = (transformed_pcd[:,0] >= 0) & (transformed_pcd[:,0] < transfer_mp_map.shape[0]) & \
                    (transformed_pcd[:,1] >= 0) & (transformed_pcd[:,1] < transfer_mp_map.shape[1]) & \
                    (transformed_pcd[:,2] >= 0) & (transformed_pcd[:,2] < transfer_mp_map.shape[2])
    transformed_pcd = transformed_pcd[valid_filter]

    transfer_mp_map[transformed_pcd[:,0],transformed_pcd[:,1],transformed_pcd[:,2]] = 1        


    for i in range(len(start_joint_vals)):
        start_joint_dict = robot.joint_vals_to_dict(start_joint_vals[i])
        target_joint_dict = robot.joint_vals_to_dict(target_joint_vals[i])
        suction_pose =  obj_start_pose.dot(tip_poses_in_obj[i])
        motion_planner.clear_octomap()
        motion_planner.set_collision_env_with_filter(occlusion, transfer_mp_map)
        
        transfer_traj = motion_planner.suction_plan(previous_joint_dict, suction_pose, start_joint_vals[i], robot, workspace=workspace, display=False)
        # transfer_traj = motion_planner.joint_dict_motion_plan(previous_joint_dict, start_joint_dict, robot)
        if len(transfer_traj) == 0:
            continue
        motion_planner.clear_octomap()
        motion_planner.set_collision_env_with_filter(occlusion, mp_map)
        # rospy.sleep(1.0)

        # v_voxel = visualize_voxel(x_map, y_map, z_map, mp_map, [1,0,0])
        # o3d.visualization.draw_geometries([v_voxel])
    

        # lift up to avoid collision with bottom
        relative_tip_pose = np.eye(4)
        relative_tip_pose[:3,3] = np.array([0,0,0.05]) # lift up by 0.05

        # transfer_end_pose = robot.get_tip_link_pose(transfer_traj[-1])

        # tip_suction_pose = obj_start_pose.dot(tip_poses_in_obj[i])

        # show the tip_suction_pose

        lift_traj = motion_planner.straight_line_motion(start_joint_dict, suction_pose, relative_tip_pose, robot, workspace=workspace)        
        if len(lift_traj) == 0:
            continue

        relative_tip_pose = np.eye(4)
        relative_tip_pose[:3,3] = np.array([0,0,0.05]) # lift up by 0.05
        tip_suction_pose = obj_pose.dot(tip_poses_in_obj[i])
        drop_traj = motion_planner.straight_line_motion(target_joint_dict, tip_suction_pose, relative_tip_pose, robot, workspace=workspace)   
        if len(drop_traj) == 0:
            continue
        drop_traj = drop_traj[::-1]
        joint_vals = robot.joint_dict_to_vals(drop_traj[0])
        traj = motion_planner.suction_with_obj_plan(lift_traj[-1], tip_poses_in_obj[i], joint_vals, robot, obj)
        # input('after planning... planning is success? %d' % (len(traj)>0))
        if len(transfer_traj) > 0 and len(lift_traj) > 0 and len(traj) > 0 and len(drop_traj) > 0:
            break
    # motion planning from start joint to target joint. Set the collision using the collision_voxel

    del mp_map
    del x_map
    del y_map
    del z_map
    del transformed_pcd
    # del voxel
    # del vis_pcd
    del valid_filter
    del start_transformed_pcd
    del goal_transformed_pcd
    del transfer_mp_map

    if len(transfer_traj) > 0 and len(lift_traj) > 0 and len(traj) > 0 and len(drop_traj) > 0:
        return transfer_traj, lift_traj+traj+drop_traj
    else:
        return [], []




def find_trajectory(obj_i, obj, obj_start_pose, obj_start_state, obj_pose, obj_state, 
                    objs, obj_start_poses, obj_start_states, obj_poses, obj_states, searched_objs, 
                    collision_grid, grid_resol):
    # for objs in the saerched_objs, they use the final obj_poses
    # for other objects, they use the obj_start_poses
    """
    try using DFS resolution-complete search
    """
    # put object collision areas into the grid
    total_objs = set(list(range(len(objs))))
    serached_obj_set = set(searched_objs)
    unsearched_obj_set = total_objs - serached_obj_set
    unsearched_objs = list(unsearched_obj_set)

    print('searched objects: ', searched_objs)
    print('obj_i: ', obj_i)
    print('obj pcd: ')
    print(objs)
    mp_map = np.array(collision_grid)

    plt.clf()
    x_map, y_map = np.indices(mp_map.shape).astype(int)
    plt.pcolor(x_map * grid_resol[0], y_map * grid_resol[1], mp_map)        

    plt.scatter(objs[0][:,0], objs[0][:,1])
    # input('object 0')

    plt.clf()


    for obj_idx in searched_objs:
        print('searched object idx: ', obj_idx)
        plt.scatter(objs[obj_idx][:,0], objs[obj_idx][:,1])
        # input('object see')
        plt.clf()
        transformed_pcd = obj_poses[obj_idx][:2,:2].dot(objs[obj_idx].T).T + obj_poses[obj_idx][:2,2]
        transformed_pcd = transformed_pcd / grid_resol
        plt.scatter(objs[obj_idx][:,0], objs[obj_idx][:,1])
        # input('object see')
        plt.clf()

        transformed_pcd = np.floor(transformed_pcd).astype(int)
        mp_map[transformed_pcd[:,0],transformed_pcd[:,1]] = 1
        plt.scatter(transformed_pcd[:,0], transformed_pcd[:,1])
        input('plotted searched_obj')
    for obj_idx in unsearched_objs:
        if obj_idx == obj_i:
            continue
        transformed_pcd = obj_start_poses[obj_idx][:2,:2].dot(objs[obj_idx].T).T + obj_start_poses[obj_idx][:2,2]
        transformed_pcd = transformed_pcd / grid_resol
        transformed_pcd = np.floor(transformed_pcd).astype(int)
        mp_map[transformed_pcd[:,0],transformed_pcd[:,1]] = 1        
        plt.scatter(transformed_pcd[:,0], transformed_pcd[:,1])
        input("plotted unsearched_obj")

    # A* search to find the path
    dlinear = grid_resol[0]
    dtheta = 0.01

    plt.clf()
    x_map, y_map = np.indices(mp_map.shape).astype(int)
    plt.pcolor(x_map * grid_resol[0], y_map * grid_resol[1], mp_map)        
    input("before a_star")


    # check collision for start and goal. If they're in collision, then automatically return failure
    transformed_pcd = obj_start_pose[:2,:2].dot(obj.T).T + obj_start_pose[:2,2]
    # plot to see if the start is in collision
    plt.clf()
    plt.pcolor(x_map * grid_resol[0], y_map * grid_resol[1], mp_map)        
    plt.scatter(transformed_pcd[:,0], transformed_pcd[:,1])
    input('after visualizing start pose of object')

    transformed_pcd = transformed_pcd / grid_resol
    transformed_pcd = np.floor(transformed_pcd).astype(int)
    valid_filter = (transformed_pcd[:,0] >= 0) & (transformed_pcd[:,0] < mp_map.shape[0]) & \
                    (transformed_pcd[:,1] >= 0) & (transformed_pcd[:,1] < mp_map.shape[1])
    transformed_pcd = transformed_pcd[valid_filter]
    if mp_map[transformed_pcd[:,0], transformed_pcd[:,1]].sum() > 0:
        return []


    transformed_pcd = obj_pose[:2,:2].dot(obj.T).T + obj_pose[:2,2]
    # plot to see if the start is in collision
    plt.clf()
    plt.pcolor(x_map * grid_resol[0], y_map * grid_resol[1], mp_map)        
    plt.scatter(transformed_pcd[:,0], transformed_pcd[:,1])
    input('after visualizing goal pose of object')


    transformed_pcd = transformed_pcd / grid_resol
    transformed_pcd = np.floor(transformed_pcd).astype(int)
    valid_filter = (transformed_pcd[:,0] >= 0) & (transformed_pcd[:,0] < mp_map.shape[0]) & \
                    (transformed_pcd[:,1] >= 0) & (transformed_pcd[:,1] < mp_map.shape[1])
    transformed_pcd = transformed_pcd[valid_filter]
    if mp_map[transformed_pcd[:,0], transformed_pcd[:,1]].sum() > 0:
        return []
    

    traj = a_star_se2(obj, obj_start_state, obj_start_pose, obj_state, obj_pose, mp_map, grid_resol, dlinear, dtheta)    

    print('found trajectory: ', traj)
    input("after found trajectory")
    # plot the trajectory by putting the object at each of the state
    x_grid, y_grid = np.indices(collision_grid.shape).astype(int)
    return traj


def se2_distance(state1, state2):
    dist = (state1[0] - state2[0])**2 + (state1[1] - state2[1])**2
    dist = np.sqrt(dist)
    ang_dist = state1[2] - state2[2]
    ang_dist = ang_dist % (np.pi*2)
    if ang_dist > np.pi:
        ang_dist = ang_dist - np.pi * 2
    # dist += ang_dist ** 2
    # dist = np.sqrt(dist)
    ang_dist = np.abs(ang_dist)
    return dist, ang_dist  # we return a tuple


def a_star_se2(obj_pcd, start_state, start_pose, goal_state, goal_pose, collision_map, map_resol, dlinear, dtheta):
    # expansion number: we use L1-distance equal to 1
    plt.clf()
    
    # expansion: move 6 directions
    corner = np.array([1.0, 1.0, 0.0])
    corner[:2] = corner[:2] * map_resol
    # corner = corner / np.linalg.norm(corner)
    expansions = [[dlinear, 0, 0], [-dlinear, 0, 0], [0, dlinear, 0], [0,-dlinear, 0]]
    # expansions += [[corner[0],corner[1],0],[corner[0],-corner[1],0],[-corner[0],corner[1],0],[-corner[0],-corner[1],0]]
    # expansions +=[[0,0,dtheta], [0,0,-dtheta]]
    # make sure we don't repeat states, we round the floating point number to .00
    # and also wrap the angle to [0, 2pi)
    heap = []  # for a star
    # initialization
    # item in the heap: (g_linear+h_linear, g_angle+h_angle, g_linear, g_angle, parent, item)
    start_state = np.array(start_state)
    start_state[2] = start_state[2] % (np.pi*2)
    goal_state = np.array(goal_state)
    goal_state[2] = goal_state[2] % (np.pi*2)

    rounded_start = np.round(start_state, 2)
    rounded_goal = np.round(goal_state, 2)
    rounded_start_pose = pose_from_2d_state(rounded_start)
    rounded_goal_pose = pose_from_2d_state(rounded_goal)

    rounded_start_pcd = rounded_start_pose[:2,:2].dot(obj_pcd.T).T + rounded_start_pose[:2,2]
    rounded_goal_pcd = rounded_goal_pose[:2,:2].dot(obj_pcd.T).T + rounded_goal_pose[:2,2]

    
    dist, ang_dist = se2_distance(rounded_start, rounded_goal)
    heapq.heappush(heap, (dist+ang_dist, 0, -1, None, rounded_start))
    explored_states = np.zeros(collision_map.shape).astype(bool)
    
    parent_of_states = dict()
    parent_of_states_x = np.zeros(collision_map.shape)
    parent_of_states_y = np.zeros(collision_map.shape)

    start_indices = rounded_start[:2] / map_resol
    start_indices = np.floor(start_indices).astype(int)

    explored_states[start_indices[0],start_indices[1]] = 1


    done = False
    iter_i = 0

    map_x, map_y = np.indices(collision_map.shape).astype(int)

    linear_threshold = 1e-3
    ang_threshold = 1e-3

    while len(heap)>0:
        # print('iteration: ', iter_i)
        iter_i += 1
        # expansion
        data = heapq.heappop(heap)
        g_linear = data[1]
        # g_angle = data[2]
        # g = data[1]
        state = data[4]

        state_indices = np.round(state[:2]/map_resol).astype(int)
        explored_states[state_indices[0],state_indices[1]] = 1

        plt.clf()
        plt.pcolor(map_x*map_resol[0], map_y*map_resol[1], collision_map)


        plt.scatter(rounded_start_pcd[:,0], rounded_start_pcd[:,1], c='g')
        plt.scatter(rounded_goal_pcd[:,0], rounded_goal_pcd[:,1], c='r')

        pose = pose_from_2d_state(state)
        pcd = pose[:2,:2].dot(obj_pcd.T).T + pose[:2,2]
        plt.scatter(pcd[:,0], pcd[:,1], c='b')
        plt.pause(0.1)

        # if the state is close to goal, then stop
        d_linear, d_angle = se2_distance(state, rounded_goal)

        if d_linear <= linear_threshold:# and d_angle <= ang_threshold:
            # when the position is right, we directly change the orientation of the object
            done = True
            print('done.')
            break

        for i in range(len(expansions)):
            # when the object is at the target location, not using linear moves anymore
            ds = expansions[i]
            if d_linear <= linear_threshold:
                if ds[2] == 0:
                    continue
                

            new_state = np.array([state[0]+ds[0], state[1]+ds[1], state[2]+ds[2]])
            new_state = np.round(new_state, 2)
            new_state[2] = new_state[2] % (np.pi*2)

            # check whether the new state has been explored before
            new_state_indices = np.round(new_state[:2] / map_resol).astype(int)
            print('new_state_indices: ', new_state_indices)
            if explored_states[new_state_indices[0],new_state_indices[1]]:
                print('explored...')
                continue
            # check whether the new state is in collision
            new_pose = pose_from_2d_state(new_state)
            transformed_pcd = new_pose[:2,:2].dot(obj_pcd.T).T + new_pose[:2,2]
            transformed_pcd = transformed_pcd / map_resol
            transformed_pcd = np.floor(transformed_pcd).astype(int)
            valid_filter = (transformed_pcd[:,0]>=0) & (transformed_pcd[:,0]<collision_map.shape[0]) & \
                            (transformed_pcd[:,1]>=0) & (transformed_pcd[:,1]<collision_map.shape[1])
            if valid_filter.sum() != len(transformed_pcd):
                print('outiside of workspace...')
                continue
            transformed_pcd = transformed_pcd[valid_filter]
            if collision_map[transformed_pcd[:,0], transformed_pcd[:,1]].sum() > 0:
                # collision
                print('collision with environment...')
                continue
            # add the new states
            # TODO: now hard to converge
            h_linear, h_angle = se2_distance(new_state, rounded_goal)
            
            parent_of_states_x[new_state_indices[0],new_state_indices[1]] = state[0]
            parent_of_states_y[new_state_indices[0],new_state_indices[1]] = state[1]

            c_linear = np.linalg.norm(new_state[:2]-state[:2])
            c_theta = np.abs(new_state[2]-state[2])

            heapq.heappush(heap, (g_linear+c_linear+5*h_linear, 
                                  g_linear+c_linear, iter_i*len(expansions)+i, state, new_state))
            explored_states[new_state_indices[0],new_state_indices[1]] = 1


        # TODO: correction: update the route to previous nodes if there is a shorter one
        # since we explore every time the shortest distance possible, the correction is not necessary

    if done:
        # retreive the path from start to goal
        trajs = [tuple(rounded_goal.tolist())]
        state = tuple(state.tolist())
        while state != tuple(rounded_start.tolist()):
            trajs.append(state)
            state_indices = np.array(state)[:2] / map_resol
            state_indices = np.round(state_indices).astype(int)
            state_x = parent_of_states_x[state_indices[0],state_indices[1]]
            state_y = parent_of_states_y[state_indices[0],state_indices[1]]
            state = tuple([state_x,state_y,state[2]])

        return trajs[::-1]
    return []



def sample_goal_locations(objs, obj_pcds, start_obj_poses, zs, zs_in_world,
                            moveable_objs, moveable_obj_pcds, moveable_start_obj_poses, moveable_zs, moveable_zs_in_world,
                            collision_voxel, robot_collision_voxel, voxel_resol, 
                            voxel_transform, robot, workspace, n_iter=20):
    """
    return:
    list indicating whether object is included in the returned pose
    """

    # TODO: debug
    robot_collision_grid = robot_collision_voxel.sum(axis=2)>0
    # in the z-axis, if there is at least one voxel occupied, then collision grid
    grid_resol = voxel_resol[:2]
    # collision_grid, grid_resol,
    map_x, map_y = np.indices(robot_collision_grid.shape).astype(int)

    plt.clf()
    plt.pcolor(map_x*grid_resol[0], map_y*grid_resol[1], robot_collision_grid)
    plt.pause(0.0001)

    obj_pcd_2ds = obj_pcd_2d_projection(obj_pcds)
    moveable_obj_pcd_2ds = obj_pcd_2d_projection(moveable_obj_pcds)

    # TODO get 2d moveable object poses
    moveable_obj_poses = []
    # moveable objs: use its initial pose first. If in collision then move them

    # NOTE: now we put all as obj
    # obj_pcd_2ds = obj_pcd_2ds + moveable_obj_pcd_2ds
    # objs = objs + moveable_objs
    # zs = zs + moveable_zs
    # zs_in_world = zs_in_world + moveable_zs_in_world

    # start_obj_poses = start_obj_poses + moveable_start_obj_poses

    obj_start_poses_in_voxel = obtain_pose_in_voxel(start_obj_poses, voxel_transform)
    obj_2d_start_poses = projection_rot_matrix(obj_start_poses_in_voxel)
    obj_start_states = states_from_2d_pose(obj_2d_start_poses)
    obj_start_states = np.array(obj_start_states)
    obj_start_thetas = obj_start_states[:,2]
    moveable_obj_start_poses_in_voxel = obtain_pose_in_voxel(moveable_start_obj_poses, voxel_transform)
    moveable_obj_2d_start_poses = projection_rot_matrix(moveable_obj_start_poses_in_voxel)
    # moveable_obj_start_states = states_from_2d_pose(moveable_obj_2d_start_poses)
    # moveable_obj_start_states = np.array(moveable_obj_start_states)
    # moveable_obj_start_thetas = moveable_obj_start_states[:,2]


    total_obj_pcd_2ds = obj_pcd_2ds + moveable_obj_pcd_2ds
    total_objs = objs + moveable_objs
    total_obj_pcds = obj_pcds + moveable_obj_pcds

    total_zs = zs + moveable_zs
    total_zs_in_world = zs_in_world + moveable_zs_in_world

    total_start_obj_poses = start_obj_poses + moveable_start_obj_poses
    total_obj_2d_start_poses = obj_2d_start_poses + moveable_obj_2d_start_poses
    total_start_poses_in_voxel = obj_start_poses_in_voxel + moveable_obj_start_poses_in_voxel

    total_obj_start_states = states_from_2d_pose(total_obj_2d_start_poses)
    total_obj_start_states = np.array(total_obj_start_states)
    total_obj_start_thetas = total_obj_start_states[:,2]


    for i_iter in range(n_iter):
        # * first sample blocking objects
        if i_iter <= int(0.9 * n_iter):
            obj_included_list = [1 for i in range(len(objs))] + [0 for i in range(len(moveable_objs))]
        else:
            obj_included_list = [1 for i in range(len(objs))] + [1 for i in range(len(moveable_objs))]

        total_obj_poses = initialize_poses(total_obj_pcd_2ds, obj_included_list, robot_collision_grid, grid_resol, total_obj_start_thetas)
        for k in range(len(total_obj_poses)):
            if total_obj_poses[k] is None:
                total_obj_poses[k] = total_obj_2d_start_poses[k]

        # obj pose is 2D
        trial_n = 40
        for trial_i in range(trial_n):
            valid = True
            forces = np.zeros((len(total_objs),2))

            for i in range(len(total_objs)):
                for j in range(i+1, len(total_objs)):
                    # * if i and j are both moveable but unsampled objects, then check collision
                    # only when iteration has passed
                    if not ((i_iter >= int(0.5*n_iter)) and (trial_i >= int(0.8 * trial_n))):
                        if (obj_included_list[i] !=1) and (obj_included_list[j] != 1):
                            continue

                    if obj_collision(total_obj_pcd_2ds[i], total_obj_poses[i], total_obj_pcd_2ds[j], total_obj_poses[j], robot_collision_grid, grid_resol):
                        # if many iterations have passed and still in collision, then add that to included_list
                        if (i_iter >= int(0.5*n_iter)) and (trial_i >= int(0.8 * trial_n)):
                            obj_included_list[i] = 1
                            obj_included_list[j] = 1

                        if obj_included_list[i]:
                            forces[i] += obj_obj_force(total_obj_pcd_2ds[i], total_obj_poses[i], total_obj_pcd_2ds[j], total_obj_poses[j])
                        if obj_included_list[j]:
                            # if the collision object is a blocking objects
                            forces[j] += (-forces[i])
                        valid = False
            for i in range(len(total_objs)):
                if obj_included_list[i]:
                    if in_collision(total_obj_pcd_2ds[i], total_obj_poses[i], robot_collision_grid, grid_resol):
                        forces[i] += obj_collision_force(total_obj_pcd_2ds[i], total_obj_poses[i], robot_collision_grid, grid_resol)
                        valid = False

                    if outside_boundary(total_obj_pcd_2ds[i], total_obj_poses[i], robot_collision_grid, grid_resol):
                        forces[i] += obj_boundary_force(total_obj_pcd_2ds[i], total_obj_poses[i], robot_collision_grid, grid_resol)
                        valid = False
            # print('forces: ')
            # print(forces)
            # update
            for i in range(len(total_objs)):
                if obj_included_list[i]:
                    total_obj_poses[i] = update(total_obj_poses[i], forces[i])

            # after update, plot
            plt.clf()
            plt.pcolor(map_x*grid_resol[0], map_y*grid_resol[1], robot_collision_grid)
            for i in range(len(total_objs)):
                pcd = total_obj_2d_start_poses[i][:2,:2].dot(total_obj_pcd_2ds[i].T).T + total_obj_2d_start_poses[i][:2,2]
                plt.scatter(pcd[:,0], pcd[:,1], c='g')
                pcd = total_obj_poses[i][:2,:2].dot(total_obj_pcd_2ds[i].T).T + total_obj_poses[i][:2,2]
                plt.scatter(pcd[:,0], pcd[:,1], c='r')

            plt.pause(0.0001)

            # input('next...')

            # validate obj poses  
            if valid:
                # * valid. Test IK
                ik_valid = True
                total_valid_pts = []
                total_valid_poses_in_obj = []
                total_valid_joints = []
                total_start_valid_pts = []
                total_start_valid_poses_in_obj = []
                total_start_valid_joints = []

                # * put moveable objects that won't be moved into collision space
                mp_map = np.array(collision_voxel).astype(bool)


                for i in range(len(total_objs)):
                    if not obj_included_list[i]:
                        pcd = total_start_poses_in_voxel[i][:3,:3].dot(total_obj_pcds[i].T).T + \
                                total_start_poses_in_voxel[i][:3,3]
                        pcd = pcd / voxel_resol
                        pcd = np.floor(pcd).astype(int)
                        valid_filter = (pcd[:,0] >= 0) & (pcd[:,0] < mp_map.shape[0]) & \
                                        (pcd[:,1] >= 0) & (pcd[:,1] < mp_map.shape[1]) & \
                                        (pcd[:,2] >= 0) & (pcd[:,2] < mp_map.shape[2])
                        pcd = pcd[valid_filter]
                        mp_map[pcd[:,0], pcd[:,1], pcd[:,2]] = 1  # include objects that are not moved into collision

                # make sure it's not too tight

                for i in range(len(total_objs)):
                    if obj_included_list[i]:
                        pcd = total_start_poses_in_voxel[i][:3,:3].dot(total_obj_pcds[i].T).T + \
                                total_start_poses_in_voxel[i][:3,3]
                        pcd = pcd / voxel_resol
                        pcd = np.floor(pcd).astype(int)
                        valid_filter = (pcd[:,0] >= 0) & (pcd[:,0] < mp_map.shape[0]) & \
                                        (pcd[:,1] >= 0) & (pcd[:,1] < mp_map.shape[1]) & \
                                        (pcd[:,2] >= 0) & (pcd[:,2] < mp_map.shape[2])
                        pcd = pcd[valid_filter]
                        mp_map = mask_pcd_xy_with_padding(mp_map, pcd, padding=1)

                        # total_obj_poses is 2D pose in the voxel
                        # obtain the pose in voxel from the 2D pose
                        obj_pose_in_voxel = pose_from_2d_pose(total_obj_poses[i], total_zs[i])

                        pcd = obj_pose_in_voxel[:3,:3].dot(total_obj_pcds[i].T).T + \
                                obj_pose_in_voxel[:3,3]
                        pcd = pcd / voxel_resol
                        pcd = np.floor(pcd).astype(int)
                        valid_filter = (pcd[:,0] >= 0) & (pcd[:,0] < mp_map.shape[0]) & \
                                        (pcd[:,1] >= 0) & (pcd[:,1] < mp_map.shape[1]) & \
                                        (pcd[:,2] >= 0) & (pcd[:,2] < mp_map.shape[2])
                        pcd = pcd[valid_filter]
                        mp_map = mask_pcd_xy_with_padding(mp_map, pcd, padding=1)


                for i in range(len(total_objs)):
                    if not obj_included_list[i]:
                        total_start_valid_pts.append([])
                        total_start_valid_poses_in_obj.append([])
                        total_start_valid_joints.append([])

                        total_valid_pts.append([])
                        total_valid_poses_in_obj.append([])
                        total_valid_joints.append([])
                        continue
                    # input('ik check on target pose...')
                    target_valid_pts, target_valid_poses_in_obj, target_valid_joints = \
                        ik_check_on_2d_pose(total_objs[i], total_obj_poses[i], total_zs[i], total_zs_in_world[i], 
                                            robot, workspace, mp_map, voxel_transform, voxel_resol)
                    if len(target_valid_pts) == 0:
                        ik_valid = False
                        break
                    

                    # input('ik check on start pose...')
                    valid_pts, valid_poses_in_obj, valid_joints = \
                        ik_check_on_pose_with_grasp_pose(total_objs[i], total_start_obj_poses[i], 
                                                        target_valid_poses_in_obj, 
                                                        robot, workspace,
                                                        mp_map, voxel_transform, voxel_resol)

                    # cross check
                    filtered_target_valid_pts = []
                    filtered_target_valid_poses_in_obj = []
                    filtered_target_valid_joints = []
                    filtered_valid_pts = []
                    filtered_valid_poses_in_obj = []
                    filtered_valid_joints = []
                    for pose_i in range(len(valid_poses_in_obj)):
                        if valid_poses_in_obj[pose_i] is not None:
                            filtered_target_valid_pts.append(target_valid_pts[pose_i])
                            filtered_target_valid_poses_in_obj.append(target_valid_poses_in_obj[pose_i])
                            filtered_target_valid_joints.append(target_valid_joints[pose_i])
                            filtered_valid_pts.append(valid_pts[pose_i])
                            filtered_valid_poses_in_obj.append(valid_poses_in_obj[pose_i])
                            filtered_valid_joints.append(valid_joints[pose_i])               

                    target_valid_pts = filtered_target_valid_pts
                    target_valid_poses_in_obj = filtered_target_valid_poses_in_obj
                    target_valid_joints = filtered_target_valid_joints
                    valid_pts = filtered_valid_pts
                    valid_poses_in_obj = filtered_valid_poses_in_obj
                    valid_joints = filtered_valid_joints                    

                    if len(valid_pts) == 0:
                        ik_valid = False
                        break

                    total_start_valid_pts.append(valid_pts)
                    total_start_valid_poses_in_obj.append(valid_poses_in_obj)
                    total_start_valid_joints.append(valid_joints)

                    total_valid_pts.append(target_valid_pts)
                    total_valid_poses_in_obj.append(target_valid_poses_in_obj)
                    total_valid_joints.append(target_valid_joints)

                
                
                del mp_map
                # TODO: now we concatenate obj_poses and moveable_obj_poses. In the future distinguish them
                # for i in range(len(moveable_objs)):
                #     valid_pts, valid_poses_in_obj, valid_joints = \
                #         ik_check_on_2d_pose(moveable_objs[i], moveable_obj_poses[i], moveable_zs[i], 
                #                         robot, workspace,
                #                         collision_voxel, voxel_transform, voxel_resol)
                #     if len(valid_pts) == 0:
                #         ik_valid = False
                #         break                
                if not ik_valid:
                    break
                
                del robot_collision_grid
                del map_x
                del map_y
                del obj_pcd_2ds
                del moveable_obj_pcd_2ds
                del total_obj_pcd_2ds
                del total_objs
                del pcd
                return obj_included_list, total_obj_poses, total_start_valid_pts, total_start_valid_poses_in_obj, \
                    total_start_valid_joints, total_valid_pts, total_valid_poses_in_obj, total_valid_joints
            if np.abs(forces).sum() == 0:
                # converged and not valid. try next time
                # print('try another initialization...')
                break

    del robot_collision_grid
    del map_x
    del map_y
    del obj_pcd_2ds
    del moveable_obj_pcd_2ds
    del total_obj_pcd_2ds
    del total_objs
    del pcd
    del total_obj_pcds

    return None, None, None, None, None, None, None, None  # no solution


def update(obj_pose, force):
    res = np.array(obj_pose)
    res[:2,2] += force
    return res


def obj_pcd_2d_projection(objs):
    obj_pcds = []
    for i in range(len(objs)):
        pcd = objs[i]
        obj_pcds.append(pcd[:,:2])
    return obj_pcds

def filter_indices_map_size(indices, map):
    valid_i = (indices[:,0] >= 0) & (indices[:,0]<map.shape[0]) & \
                (indices[:,1] >= 0) & (indices[:,1]<map.shape[1])
    return indices[valid_i]


def initialize_poses(objs, obj_included_list, collision_grid, grid_resol, start_thetas):
    # initialize 2d pose: x, y, theta
    # x, y are at the valid grids
    valid_indices = np.nonzero(~collision_grid)
    valid_indices = np.array(valid_indices).T
    valid_pos = valid_indices * grid_resol
    
    pos_indices = np.random.choice(len(valid_pos), size=len(objs))
    sampled_pos = valid_pos[pos_indices]
    thetas = np.random.uniform(low=start_thetas-np.pi/2, high=start_thetas+np.pi/2, size=len(objs))

    obj_poses = []
    for i in range(len(objs)):
        if obj_included_list[i]:
            obj_pose_i = np.eye(3)
            obj_pose_i[0,0] = np.cos(thetas[i])
            obj_pose_i[0,1] = -np.sin(thetas[i])
            obj_pose_i[1,0] = np.sin(thetas[i])
            obj_pose_i[1,1] = np.cos(thetas[i])
            obj_pose_i[0,2] = sampled_pos[i,0]
            obj_pose_i[1,2] = sampled_pos[i,1]
            obj_poses.append(obj_pose_i)
        else:
            obj_poses.append(None)

    return obj_poses


def obj_collision(obj_i, obj_pose_i, obj_j, obj_pose_j, collision_grid, grid_resol):
    grid_i = np.zeros(collision_grid.shape).astype(bool)
    transformed_pcd_i = obj_pose_i[:2,:2].dot(obj_i.T).T + obj_pose_i[:2,2]
    transformed_pcd_i = transformed_pcd_i / grid_resol
    transformed_pcd_i = np.floor(transformed_pcd_i).astype(int)
    valid_i = (transformed_pcd_i[:,0] >= 0) & (transformed_pcd_i[:,0]<collision_grid.shape[0]) & \
                (transformed_pcd_i[:,1] >= 0) & (transformed_pcd_i[:,1]<collision_grid.shape[1])
    transformed_pcd_i = transformed_pcd_i[valid_i]
    grid_i[transformed_pcd_i[:,0],transformed_pcd_i[:,1]] = 1

    # valid_filter = (transformed_pcd_i[:,0]>0) & (transformed_pcd_i[:,0]<collision_grid.shape[0]-1) &\
    #             (transformed_pcd_i[:,1]>0) & (transformed_pcd_i[:,1]<collision_grid.shape[1]-1)
    # transformed_pcd_i = transformed_pcd_i[valid_filter]
    # grid_i[transformed_pcd_i[:,0]-1, transformed_pcd_i[:,1]] = 1
    # grid_i[transformed_pcd_i[:,0]+1, transformed_pcd_i[:,1]] = 1
    # grid_i[transformed_pcd_i[:,0], transformed_pcd_i[:,1]-1] = 1
    # grid_i[transformed_pcd_i[:,0], transformed_pcd_i[:,1]+1] = 1
    # grid_i[transformed_pcd_i[:,0]-1, transformed_pcd_i[:,1]-1] = 1
    # grid_i[transformed_pcd_i[:,0]+1, transformed_pcd_i[:,1]+1] = 1
    # grid_i[transformed_pcd_i[:,0]-1, transformed_pcd_i[:,1]+1] = 1
    # grid_i[transformed_pcd_i[:,0]+1, transformed_pcd_i[:,1]-1] = 1




    grid_j = np.zeros(collision_grid.shape).astype(bool)
    transformed_pcd_j = obj_pose_j[:2,:2].dot(obj_j.T).T + obj_pose_j[:2,2]
    transformed_pcd_j = transformed_pcd_j / grid_resol
    transformed_pcd_j = np.floor(transformed_pcd_j).astype(int)
    valid_j = (transformed_pcd_j[:,0] >= 0) & (transformed_pcd_j[:,0]<collision_grid.shape[0]) & \
                (transformed_pcd_j[:,1] >= 0) & (transformed_pcd_j[:,1]<collision_grid.shape[1])
    transformed_pcd_j = transformed_pcd_j[valid_j]
    grid_j[transformed_pcd_j[:,0],transformed_pcd_j[:,1]] = 1


    # valid_filter = (transformed_pcd_j[:,0]>0) & (transformed_pcd_j[:,0]<collision_grid.shape[0]-1) &\
    #             (transformed_pcd_j[:,1]>0) & (transformed_pcd_j[:,1]<collision_grid.shape[1]-1)
    # transformed_pcd_j = transformed_pcd_j[valid_filter]
    # grid_j[transformed_pcd_j[:,0]-1, transformed_pcd_j[:,1]] = 1
    # grid_j[transformed_pcd_j[:,0]+1, transformed_pcd_j[:,1]] = 1
    # grid_j[transformed_pcd_j[:,0], transformed_pcd_j[:,1]-1] = 1
    # grid_j[transformed_pcd_j[:,0], transformed_pcd_j[:,1]+1] = 1
    # grid_j[transformed_pcd_j[:,0]-1, transformed_pcd_j[:,1]-1] = 1
    # grid_j[transformed_pcd_j[:,0]+1, transformed_pcd_j[:,1]+1] = 1
    # grid_j[transformed_pcd_j[:,0]-1, transformed_pcd_j[:,1]+1] = 1
    # grid_j[transformed_pcd_j[:,0]+1, transformed_pcd_j[:,1]-1] = 1

    # del transformed_pcd_i
    # del transformed_pcd_j

    return (grid_i & grid_j).sum() > 0


def obj_obj_force(obj_i, pose_i, obj_j, pose_j):
    obj_i_transformed = pose_i[:2,:2].dot(obj_i.T).T + pose_i[:2,2]
    center_i = obj_i_transformed.mean(axis=0)    

    obj_j_transformed = pose_j[:2,:2].dot(obj_j.T).T + pose_j[:2,2]
    center_j = obj_j_transformed.mean(axis=0)    

    direction = center_i - center_j
    distance = np.linalg.norm(direction)
    direction = direction / np.linalg.norm(direction)

    # assuming bounding circle, obtain the radius
    r1 = obj_i - center_i.reshape(-1,2)
    r1 = np.linalg.norm(r1, axis=1)
    r1 = np.max(r1)

    r2 = obj_j - center_j.reshape(-1,2)
    r2 = np.linalg.norm(r2, axis=1)
    r2 = np.max(r2)

    dis = (r1 + r2 - distance) / 2
    if dis > 0:
        dis = 1
    force_factor = 0.01
    force = dis * direction * force_factor
    return force

def in_collision(obj_i, pose_i, collision_grid, grid_resol):
    obj_i_transformed = pose_i[:2,:2].dot(obj_i.T).T + pose_i[:2,2]
    obj_i_transformed = obj_i_transformed / grid_resol
    obj_i_transformed = np.floor(obj_i_transformed).astype(int)
    obj_i_transformed = filter_indices_map_size(obj_i_transformed, collision_grid)

    occupied_map_i = np.zeros(collision_grid.shape).astype(bool)
    occupied_map_i[obj_i_transformed[:,0], obj_i_transformed[:,1]] = 1
    # increase the range to +-1
    # valid_filter = (obj_i_transformed[:,0]>0) & (obj_i_transformed[:,0]<collision_grid.shape[0]-1) &\
    #             (obj_i_transformed[:,1]>0) & (obj_i_transformed[:,1]<collision_grid.shape[1]-1)
    # obj_i_transformed = obj_i_transformed[valid_filter]
    # occupied_map_i[obj_i_transformed[:,0]-1, obj_i_transformed[:,1]] = 1
    # occupied_map_i[obj_i_transformed[:,0]+1, obj_i_transformed[:,1]] = 1
    # occupied_map_i[obj_i_transformed[:,0], obj_i_transformed[:,1]-1] = 1
    # occupied_map_i[obj_i_transformed[:,0], obj_i_transformed[:,1]+1] = 1
    # occupied_map_i[obj_i_transformed[:,0]-1, obj_i_transformed[:,1]-1] = 1
    # occupied_map_i[obj_i_transformed[:,0]+1, obj_i_transformed[:,1]+1] = 1
    # occupied_map_i[obj_i_transformed[:,0]-1, obj_i_transformed[:,1]+1] = 1
    # occupied_map_i[obj_i_transformed[:,0]+1, obj_i_transformed[:,1]-1] = 1

    intersection = occupied_map_i & collision_grid
    return intersection.sum() > 0

def obj_collision_force(obj_i, pose_i, collision_grid, grid_resol):
    # get occupied space of obj_i
    obj_i_transformed = pose_i[:2,:2].dot(obj_i.T).T + pose_i[:2,2]
    obj_i_transformed = obj_i_transformed / grid_resol
    obj_i_transformed = np.floor(obj_i_transformed).astype(int)
    obj_i_transformed = filter_indices_map_size(obj_i_transformed, collision_grid)

    occupied_map_i = np.zeros(collision_grid.shape).astype(bool)
    occupied_map_i[obj_i_transformed[:,0], obj_i_transformed[:,1]] = 1
    # increase the range to +-1
    # valid_filter = (obj_i_transformed[:,0]>0) & (obj_i_transformed[:,0]<collision_grid.shape[0]-1) &\
    #             (obj_i_transformed[:,1]>0) & (obj_i_transformed[:,1]<collision_grid.shape[1]-1)
    # obj_i_transformed = obj_i_transformed[valid_filter]
    # occupied_map_i[obj_i_transformed[:,0]-1, obj_i_transformed[:,1]] = 1
    # occupied_map_i[obj_i_transformed[:,0]+1, obj_i_transformed[:,1]] = 1
    # occupied_map_i[obj_i_transformed[:,0], obj_i_transformed[:,1]-1] = 1
    # occupied_map_i[obj_i_transformed[:,0], obj_i_transformed[:,1]+1] = 1
    # occupied_map_i[obj_i_transformed[:,0]-1, obj_i_transformed[:,1]-1] = 1
    # occupied_map_i[obj_i_transformed[:,0]+1, obj_i_transformed[:,1]+1] = 1
    # occupied_map_i[obj_i_transformed[:,0]-1, obj_i_transformed[:,1]+1] = 1
    # occupied_map_i[obj_i_transformed[:,0]+1, obj_i_transformed[:,1]-1] = 1


    intersection = occupied_map_i & collision_grid
    inter_pts = np.nonzero(intersection)
    inter_pts = np.array(inter_pts).T
    inter_x = inter_pts[:,0] + 0.5
    inter_y = inter_pts[:,1] + 0.5

    # get the force from each intersection grid
    inter_x = inter_x * grid_resol[0]
    inter_y = inter_y * grid_resol[1]

    center = pose_i[:2,:2].dot(obj_i.T).T + pose_i[:2,2]
    center = center.mean(axis=0)
    # print("center mean: ", center)

    direction = center.reshape(-1,2) - np.array([inter_x, inter_y]).T
    direction = direction / np.linalg.norm(direction, axis=1).reshape(-1,1)

    # direction = direction / np.linalg.norm(direction)
    # force_factor = np.linalg.norm(grid_resol)
    # forces = direction * force_factor
    force = direction.mean(axis=0)
    force = force / np.linalg.norm(force)
    force = force * grid_resol
    # force = force * force_factor

    # TODO: maybe not a good idea to sum all the forces
    return force


def outside_boundary(obj_i, pose_i, collision_grid, grid_resol):
    # see if the current object is outside of the map
    obj_i_transformed = pose_i[:2,:2].dot(obj_i.T).T + pose_i[:2,2]
    obj_i_transformed = obj_i_transformed / grid_resol
    obj_i_transformed = np.floor(obj_i_transformed).astype(int)
    outside_map_filter = (obj_i_transformed[:,0] < 0) | (obj_i_transformed[:,0] >= collision_grid.shape[0]) | \
                         (obj_i_transformed[:,1] < 0) | (obj_i_transformed[:,1] >= collision_grid.shape[1])
    return outside_map_filter.sum()>0

def obj_boundary_force(obj_i, pose_i, collision_grid, grid_resol):
    # see if the current object is outside of the map
    obj_i_transformed = pose_i[:2,:2].dot(obj_i.T).T + pose_i[:2,2]
    obj_i_transformed = obj_i_transformed / grid_resol
    obj_i_transformed = np.floor(obj_i_transformed).astype(int)
    outside_map_filter = (obj_i_transformed[:,0] < 0) | (obj_i_transformed[:,0] >= collision_grid.shape[0]) | \
                         (obj_i_transformed[:,1] < 0) | (obj_i_transformed[:,1] >= collision_grid.shape[1])
    # print("outside map filter: ", outside_map_filter.sum() == 0)
    if outside_map_filter.sum() == 0:
        return np.zeros(2)
        
    x_force = np.zeros(2)
    y_force = np.zeros(2)
    if (obj_i_transformed[:,0] < 0).sum() > 0:
        x_force = np.array([-obj_i_transformed[:,0].min(),0.0])
    if (obj_i_transformed[:,0] >= collision_grid.shape[0]).sum() > 0:
        x_force = np.array([collision_grid.shape[0]-1-obj_i_transformed[:,0].max(), 0.0])
    if (obj_i_transformed[:,1] < 0).sum() > 0:
        y_force = np.array([0.0,-obj_i_transformed[:,1].min()])
    if (obj_i_transformed[:,1] >= collision_grid.shape[1]).sum() > 0:
        y_force = np.array([0.0, collision_grid.shape[1]-1-obj_i_transformed[:,1].max()])
        
    force = x_force + y_force
    force = force * grid_resol
    # force = force / np.linalg.norm(force)
    # force_factor = 0.1
    # force = force * force_factor
    # print('move back to ws force: ', force)
    return force




def sample_circle(center, radius, n_samples=100):
    # sample a circle with center and radius
    # pcd_cylinder_r = np.random.uniform(low=0, high=radius, size=n_samples)
    circle_r = np.random.triangular(left=0., mode=radius, right=radius, size=n_samples)
    circle_xy = np.random.normal(loc=[0.,0.], scale=[1.,1.], size=(n_samples,2))
    circle_xy = circle_xy / np.linalg.norm(circle_xy, axis=1).reshape(-1,1)
    circle_xy = circle_xy * circle_r.reshape(-1,1)
    circle_xy = circle_xy + center.reshape(1,2)
    return circle_xy

def sample_triangle(p1, p2, p3, n_samples=100):
    # sample alpha, beta
    alpha = np.random.uniform(low=0, high=1, size=n_samples).reshape((-1,1))
    beta = np.random.uniform(low=0, high=1, size=n_samples).reshape((-1,1))
    # u = min(alpha, beta)
    u = np.minimum(alpha, beta)
    v = np.maximum(alpha, beta) - u
    w = 1 - u - v
    p1 = p1.reshape((1,2))
    p2 = p2.reshape((1,2))
    p3 = p3.reshape((1,2))
    return p1 * u + p2 * v + p3 * w
    


def sample_rectangle(x_size, y_size, n_samples=100):
    pcd = np.random.uniform(low=[-0.5,-0.5],high=[0.5,0.5], size=(n_samples,2))
    pcd = pcd * np.array([x_size, y_size])
    return pcd

    