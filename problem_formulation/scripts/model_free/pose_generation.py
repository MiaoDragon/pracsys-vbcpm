"""
generate grasp/suction poses given object geometry input
"""
import numpy as np
from visual_utilities import *
import open3d as o3d
import transformations as tf
import pybullet as p
from tqdm import trange

def grasp_pose_generation(obj_id, obj, robot, workspace, occlusion, occlusion_label, occupied_label, sample_n=10):
    """
    given the object pose and voxels, generate the grasp pose
    reference:
    https://arxiv.org/pdf/1706.09911.pdf

    suction gripper might be easier since there are large objects that we may not 
    be able to grasp from front, and it may require pushing the object

    suction disc is the discritization the suction gripper with discretization that satisfies
    completeness. The local coordinate center is at one tip of the suction (the suction tip),
    and the orientation is defined such that the z-axis will point to the other tip
    suction disc: a vector of length N
    """
    # suction_disc = np.arange(start=0.0,stop=0.06, step=0.005)[1:]

    suction_disc = np.arange(start=0.0,stop=robot.suction_length, step=0.005)[1:]

    # * sample points from the object voxels
    suction_pts, suction_normal = obj.get_surface_normal()  # normal is relative to the object frame

    # * filter out by checking angle with x-y plane
    angle_with_xy_plane = np.arcsin(np.abs(suction_normal[:,2]))
    angle_filter = angle_with_xy_plane < (15 * np.pi / 180)
    suction_pts = suction_pts[angle_filter]
    suction_normal = suction_normal[angle_filter]

    # * filter out by checking collision with the object model
    # transform the suction pcd at each of the sampled points
    normals = suction_normal.reshape((len(suction_normal),1,3))
    suction_pcd = -suction_disc.reshape((1,len(suction_disc),1)) * normals + suction_pts.reshape((len(suction_pts),1,3))
    # print('suction_pcd: ', suction_pcd)
    # check if the pcd is in voxel
    suction_pcd_combined = suction_pcd.reshape(-1,3)
    obj_cons = obj.get_conservative_model()
    suction_pcd_combined_int = np.floor(suction_pcd_combined/obj.resol).astype(int)
    
    total_filter = np.zeros(suction_pcd_combined_int.shape[0]).astype(bool)
    valid_filter = (suction_pcd_combined_int[:,0] >= 0) & (suction_pcd_combined_int[:,0] < obj_cons.shape[0]) & \
                    (suction_pcd_combined_int[:,1] >= 0) & (suction_pcd_combined_int[:,1] < obj_cons.shape[1]) & \
                    (suction_pcd_combined_int[:,2] >= 0) & (suction_pcd_combined_int[:,2] < obj_cons.shape[2])
    # total_filter[~valid_filter] = 0
    collision_filter = obj_cons[suction_pcd_combined_int[valid_filter][:,0],
                                suction_pcd_combined_int[valid_filter][:,1],
                                suction_pcd_combined_int[valid_filter][:,2]]
    total_filter[valid_filter][collision_filter] = 1
    total_filter = total_filter.reshape(len(suction_pts),len(suction_disc))
    collision_filter = total_filter.sum(axis=1).astype(bool)

    filtered_suction_pts = suction_pts[~collision_filter]
    filtered_suction_normal = suction_normal[~collision_filter]

    other_suction_pts = suction_pts[collision_filter]
    other_suction_normal = suction_normal[collision_filter]


    # * transform the local grasp pose to global one using the object transform
    transformed_suction_pts = obj.transform[:3,:3].dot(filtered_suction_pts.T).T + obj.transform[:3,3]
    transformed_suction_normal = obj.transform[:3,:3].dot(filtered_suction_normal.T).T# + obj.transform[:3,3]
    transformed_suction_y = np.array(transformed_suction_normal)
    transformed_random = np.random.normal(loc=0., scale=1.0, size=transformed_suction_normal.shape)
    transformed_random = transformed_random / np.linalg.norm(transformed_random, axis=1).reshape(-1,1)

    transformed_suction_y = np.cross(transformed_random, transformed_suction_normal)    
    transformed_suction_y = transformed_suction_y / np.linalg.norm(transformed_suction_y, axis=1).reshape(-1,1)

    transformed_suction_x = np.cross(transformed_suction_y, transformed_suction_normal)


    # generate point cloud of the occluded and occupied space
    # occluded_pcd = np.array([occlusion.voxel_x[occlusion_label>0],occlusion.voxel_y[occlusion_label>0],occlusion.voxel_z[occlusion_label>0]]).T+0.5
    # occupied_pcd = np.array([occlusion.voxel_x[(occupied_label>0)&(occupied_label!=obj_id)],
    #                          occlusion.voxel_y[(occupied_label>0)&(occupied_label!=obj_id)],
    #                          occlusion.voxel_z[(occupied_label>0)&(occupied_label!=obj_id)]]).T+0.5
    # env_pcd = np.concatenate([occluded_pcd, occupied_pcd], axis=0)
    # # to world coordinate frame
    # env_pcd = occlusion.resol * env_pcd
    # env_pcd = occlusion.transform[:3,:3].dot(env_pcd.T).T + occlusion.transform[:3,3]

    valid_pts = []
    valid_orientations = []
    valid_joints = []
    # * filter out unreachable grasp poses by the robot IK

    frames = []

    # select 10 suction pts
    selected_suction_pts = np.random.choice(len(transformed_suction_pts), sample_n)
    transformed_suction_pts = transformed_suction_pts[selected_suction_pts]
    transformed_suction_normal = transformed_suction_normal[selected_suction_pts]
    transformed_suction_y = transformed_suction_y[selected_suction_pts]

    # check collision using PyBullet: we need to open a new pybullet session to do so. Instead use MoveIt
    for i in trange(len(transformed_suction_pts)):
        # rotating the y axis to get several potential samples
        n_theta = 8
        d_theta = np.pi * 2 / n_theta
        for j in range(n_theta):
            suction_y = tf.rotation_matrix(d_theta*j, transformed_suction_normal[i])[:3,:3]
            suction_y = suction_y.dot(transformed_suction_y[i])
            suction_x = np.cross(suction_y, transformed_suction_normal[i])
            rot_mat = np.eye(4)
            rot_mat[:3,:3] = np.array([suction_x, suction_y, transformed_suction_normal[i]]).T
            quat = tf.quaternion_from_matrix(rot_mat)  # w x y z
            valid, dof_joint_vals = robot.get_ik(robot.tip_link_name, transformed_suction_pts[i], [quat[1],quat[2],quat[3],quat[0]], 
                                                robot.joint_vals, collision_check=True)
            if not valid:
                # ik failed. next
                print('ik not valid, i=%d/%d, j=%d/%d' % (i, len(transformed_suction_pts), j, n_theta))
                continue
            # reset the robot joint angles, and check collision with the environment
            prev_joint_vals = robot.joint_vals
            robot.set_joints(dof_joint_vals)
            # input("before checking collision...")
            
            # * filter out grasp poses that cause collisions with the environment

            # check collision with environment
            collision = False
            
            for comp_name, comp_id in workspace.component_id_dict.items():
                contacts = p.getClosestPoints(robot.robot_id, comp_id, distance=0.,physicsClientId=robot.pybullet_id)
                if len(contacts):
                    collision = True
                    break   
            """
            for k in range(0, len(env_pcd), p.MAX_RAY_INTERSECTION_BATCH_SIZE):
                raytests = p.rayTestBatch(rayFromPositions=env_pcd[k:k+p.MAX_RAY_INTERSECTION_BATCH_SIZE], 
                                          rayToPositions=env_pcd[k:k+p.MAX_RAY_INTERSECTION_BATCH_SIZE]+0.5, 
                                          physicsClientId=robot.pybullet_id)
                # collision = False
                # check if the ray hits the robot
                for ri in range(len(raytests)):
                    print('%d-th raytest: ' % (ri))
                    print(raytests[ri])
                    if raytests[ri][0] == robot.robot_id:
                        collision = True
                        break

                if collision:
                    break
            """        
            robot.set_joints(prev_joint_vals)

            if collision:
                print('collision, i=%d/%d, j=%d/%d' % (i, len(transformed_suction_pts), j, n_theta))
                continue

            rot_mat = np.array(rot_mat)
            rot_mat[:3,3] = transformed_suction_pts[i]
            rot_mat = np.linalg.inv(obj.transform).dot(rot_mat)  # O T {tip}
            # rot_mat[:3,3] = valid_pts[-1]#/obj.resol
            # rot_mat = obj.world_in_voxel.dot(rot_mat)
            # orientation: relative to object transform
            valid_pts.append(transformed_suction_pts[i])
            valid_orientations.append(rot_mat)
            valid_joints.append(dof_joint_vals)
            print('found valid, i=%d/%d, j=%d/%d' % (i, len(transformed_suction_pts), j, n_theta))
    
    valid_poses_in_obj = valid_orientations

    #TODO: test this code
    return valid_pts, valid_poses_in_obj, valid_joints







# def suction_pose_generation(obj_id, obj, robot):
#     """
#     given the object pose and voxels, generate the grasp pose
#     reference:
#     https://arxiv.org/pdf/1706.09911.pdf

#     suction gripper might be easier since there are large objects that we may not 
#     be able to grasp from front, and it may require pushing the object

#     suction disc is the discritization the suction gripper with discretization that satisfies
#     completeness. The local coordinate center is at one tip of the suction (the suction tip),
#     and the orientation is defined such that the z-axis will point to the other tip
#     suction disc: a vector of length N
#     """
#     # suction_disc = np.arange(start=0.0,stop=0.06, step=0.005)[1:]

#     suction_disc = np.arange(start=0.0,stop=robot.suction_length, step=0.005)[1:]

#     # * sample points from the object voxels
#     suction_pts, suction_normal = obj.get_surface_normal()  # normal is relative to the object frame

#     # * filter out by checking angle with x-y plane
#     angle_with_xy_plane = np.arcsin(np.abs(suction_normal[:,2]))
#     angle_filter = angle_with_xy_plane < (15 * np.pi / 180)
#     suction_pts = suction_pts[angle_filter]
#     suction_normal = suction_normal[angle_filter]

#     # * filter out by checking collision with the object model
#     # transform the suction pcd at each of the sampled points
#     normals = suction_normal.reshape((len(suction_normal),1,3))
#     suction_pcd = -suction_disc.reshape((1,len(suction_disc),1)) * normals + suction_pts.reshape((len(suction_pts),1,3))
#     # print('suction_pcd: ', suction_pcd)
#     # check if the pcd is in voxel
#     suction_pcd_combined = suction_pcd.reshape(-1,3)
#     obj_cons = obj.get_conservative_model()
#     suction_pcd_combined_int = np.floor(suction_pcd_combined/obj.resol).astype(int)
    
#     total_filter = np.zeros(suction_pcd_combined_int.shape[0]).astype(bool)
#     valid_filter = (suction_pcd_combined_int[:,0] >= 0) & (suction_pcd_combined_int[:,0] < obj_cons.shape[0]) & \
#                     (suction_pcd_combined_int[:,1] >= 0) & (suction_pcd_combined_int[:,1] < obj_cons.shape[1]) & \
#                     (suction_pcd_combined_int[:,2] >= 0) & (suction_pcd_combined_int[:,2] < obj_cons.shape[2])
#     # total_filter[~valid_filter] = 0
#     collision_filter = obj_cons[suction_pcd_combined_int[valid_filter][:,0],
#                                 suction_pcd_combined_int[valid_filter][:,1],
#                                 suction_pcd_combined_int[valid_filter][:,2]]
#     total_filter[valid_filter][collision_filter] = 1
#     total_filter = total_filter.reshape(len(suction_pts),len(suction_disc))
#     collision_filter = total_filter.sum(axis=1).astype(bool)

#     filtered_suction_pts = suction_pts[~collision_filter]
#     filtered_suction_normal = suction_normal[~collision_filter]

#     filtered_random = np.random.normal(loc=0., scale=1.0, size=filtered_suction_normal.shape)
#     filtered_random = filtered_random / np.linalg.norm(filtered_random, axis=1).reshape(-1,1)

#     filtered_suction_y = np.cross(filtered_random, filtered_suction_normal)    
#     filtered_suction_y = filtered_suction_y / np.linalg.norm(filtered_suction_y, axis=1).reshape(-1,1)

#     valid_pts = []
#     valid_orientations = []
#     valid_joints = []
#     # * filter out unreachable grasp poses by the robot IK

#     frames = []

#     # select 10 suction pts
#     selected_suction_pts = np.random.choice(len(filtered_suction_pts), 10)
#     filtered_suction_pts = filtered_suction_pts[selected_suction_pts]
#     filtered_suction_normal = filtered_suction_normal[selected_suction_pts]
#     filtered_suction_y = filtered_suction_y[selected_suction_pts]

#     # check collision using PyBullet: we need to open a new pybullet session to do so. Instead use MoveIt
#     for i in trange(len(filtered_suction_pts)):
#         # rotating the y axis to get several potential samples
#         n_theta = 8
#         d_theta = np.pi * 2 / n_theta
#         for j in range(n_theta):
#             suction_y = tf.rotation_matrix(d_theta*j, filtered_suction_normal[i])[:3,:3]
#             suction_y = suction_y.dot(filtered_suction_y[i])
#             suction_x = np.cross(suction_y, filtered_suction_normal[i])
#             rot_mat = np.eye(4)
#             rot_mat[:3,:3] = np.array([suction_x, suction_y, filtered_suction_normal[i]]).T
#             quat = tf.quaternion_from_matrix(rot_mat)  # w x y z

#             # reset the robot joint angles, and check collision with the environment
#             prev_joint_vals = robot.joint_vals
#             robot.set_joints(dof_joint_vals)
#             # input("before checking collision...")
            
#             # * filter out grasp poses that cause collisions with the environment

#             # check collision with environment
#             collision = False
            
#             for comp_name, comp_id in workspace.component_id_dict.items():
#                 contacts = p.getClosestPoints(robot.robot_id, comp_id, distance=0.,physicsClientId=robot.pybullet_id)
#                 if len(contacts):
#                     collision = True
#                     break
#             """
#             for k in range(0, len(env_pcd), p.MAX_RAY_INTERSECTION_BATCH_SIZE):
#                 raytests = p.rayTestBatch(rayFromPositions=env_pcd[k:k+p.MAX_RAY_INTERSECTION_BATCH_SIZE], 
#                                           rayToPositions=env_pcd[k:k+p.MAX_RAY_INTERSECTION_BATCH_SIZE]+0.5, 
#                                           physicsClientId=robot.pybullet_id)
#                 # collision = False
#                 # check if the ray hits the robot
#                 for ri in range(len(raytests)):
#                     print('%d-th raytest: ' % (ri))
#                     print(raytests[ri])
#                     if raytests[ri][0] == robot.robot_id:
#                         collision = True
#                         break

#                 if collision:
#                     break
#             """        
#             robot.set_joints(prev_joint_vals)

#             if collision:
#                 print('collision')
#                 continue

#             rot_mat = np.array(rot_mat)
#             rot_mat[:3,3] = transformed_suction_pts[i]
#             rot_mat = np.linalg.inv(obj.transform).dot(rot_mat)  # O T {tip}
#             # rot_mat[:3,3] = valid_pts[-1]#/obj.resol
#             # rot_mat = obj.world_in_voxel.dot(rot_mat)
#             # orientation: relative to object transform
#             valid_pts.append(transformed_suction_pts[i])
#             valid_orientations.append(rot_mat)
#             valid_joints.append(dof_joint_vals)
    

#     #TODO: test this code
#     return valid_pts, valid_orientations, valid_joints
