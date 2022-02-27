"""
provide utility functions for the pipeline
"""
import pose_generation
import numpy as np
import transformations as tf
import collision_utils
import pybullet as p
from visual_utilities import *
import open3d as o3d

def snapshot_object_selection(objects, moved_objs, occlusion, occlusion_label, occupied_label, occluded_dict, 
                                robot, workspace, perception, motion_planner):
    """
    1. generate feasible object sets: from occlusion info and reachability info
    2. randomly select object from the feasible object sets
    """
    # valid_objects = perception.obtain_unhidden_objects([robot.robot_id], workspace.component_ids)

    # # for other objects that are unseen but active, put them in valid objects as well
    # assoc = perception.last_assoc
    # seen_objs = []
    # for seg_id, obj_id in assoc.items():
    #     seen_objs.append(obj_id)

    # seen_objs = set(seen_objs)
    # total_obj_ids = set(list(perception.slam_system.objects.keys()))
    # unseen_objs = total_obj_ids - seen_objs

    # for obj_id in unseen_objs:
    #     if (obj_id in perception.slam_system.objects) and perception.slam_system.objects[obj_id].active:
    #         valid_objects.append(obj_id)

    valid_objects = []
    for obj_id, obj in objects.items():
        if obj.active:
            valid_objects.append(obj_id)


    # remove objects that have been moved
    valid_object_set = set(valid_objects) - set(moved_objs)
    valid_objects = list(valid_object_set)


    # TODO: check reachability constraints
    reachable_objects = []
    reachable_joints = []
    reachable_oris = []
    for i in range(len(valid_objects)):
        obj = objects[valid_objects[i]]
        valid_pts, valid_orientations, valid_joints = pose_generation.grasp_pose_generation(valid_objects[i], obj, robot, workspace, occlusion, occlusion_label, occupied_label)
        #TODO

        if len(valid_joints)>0:
            reachable_objects.append(valid_objects[i])
            reachable_joints.append(valid_joints)
            reachable_oris.append(valid_orientations)

    i = np.random.choice(len(reachable_objects))
    obj_i = reachable_objects[i]
    joints = reachable_joints[i]
    orientations = reachable_oris[i]

    obj = objects[obj_i]
    return obj_i, orientations, joints

def generate_start_poses(obj_id, obj, robot, workspace, occlusion, occlusion_label, occupied_label, sample_n=10):
    """
    generate suction poses using the start pose of the object
    """
    return pose_generation.grasp_pose_generation(obj_id, obj, robot, workspace, occlusion, occlusion_label, occupied_label, sample_n)


def generate_intermediate_poses(obj_i, pybullet_obj_i, pybullet_obj_pose, objects, 
                                seg_img, occlusion, occluded_label, occupied_label, 
                                gripper_tip_poses_in_obj, suction_joints, camera, robot, workspace):
    """
    sample reachable intermediate pose that is outside of the camera view
    UPDATE: we want the object to not hide the parts where it is located before
    """
    x_max = workspace.region_low[0]-0.05
    x_min = workspace.region_low[0]-0.4  #robot.transform[0,3] + 0.6
    y_min = workspace.region_low[1]*0.9
    y_max = workspace.region_high[1]*1.1
    z_min = workspace.region_low[2]*0.8
    z_max = workspace.region_high[2]*1.1


    extrinsics = camera.info['extrinsics']
    intrinsics = camera.info['intrinsics']
    img_size = camera.info['img_size']
    cam_transform = np.linalg.inv(extrinsics)
    fx = intrinsics[0][0]
    fy = intrinsics[1][1]
    cx = intrinsics[0][2]
    cy = intrinsics[1][2]

    obj = objects[obj_i]
    obj_pcd = obj.sample_conservative_pcd()

    safety_padding = np.array([0.05, 0.02, 0.02])
    # make sure the object pcd is not in view
    while True:
        dx_min = -0.4
        dy_min = -0.3
        dz_min = 0.02
        dx_max = 0
        dy_max = 0.3
        dz_max = 0.2
        pos = np.random.uniform(low=[x_min, y_min, z_min], high=[x_max, y_max, z_max])

        angle = np.random.normal(loc=0,scale=1,size=[2])
        angle = angle / np.linalg.norm(angle)
        angle = np.arcsin(angle[1])
        # rotate around z axis
        ori = tf.rotation_matrix(angle, [0,0,1])
        transform = np.eye(4)
        transform[:3,:3] = ori[:3,:3]
        transform[:3,3] = pos
        net_transform = obj.get_net_transform_from_center_frame(obj.sample_conservative_pcd(), transform)
        transform = net_transform

        # * first check whether the object is in camera view
        transformed_pcd = transform[:3,:3].dot(obj_pcd.T).T + transform[:3,3]



        # visualize pybullet
        # delta_transform = transform.dot(np.linalg.inv(obj.transform))
        # new_pybullet_pose = delta_transform.dot(pybullet_obj_pose)
        # quat = tf.quaternion_from_matrix(new_pybullet_pose) # w x y z

        # p.resetBasePositionAndOrientation(pybullet_obj_i, new_pybullet_pose[:3,3], [quat[1],quat[2],quat[3],quat[0]], physicsClientId=robot.pybullet_id)
        # input('after sampling.')



        inside_workspace = (transformed_pcd[:,0] >= workspace.region_low[0]-0.05-safety_padding[0]) & \
                                    (transformed_pcd[:,0] < workspace.region_high[0]) & \
                                    (transformed_pcd[:,1] > workspace.region_low[1]-safety_padding[1]) & \
                                    (transformed_pcd[:,1] < workspace.region_high[1]+safety_padding[1])            
        if inside_workspace.sum() > 0:
            # input('next...')
            continue


        if ((transformed_pcd[:,1] < workspace.region_low[1]).sum()>0) or (((transformed_pcd[:,1] > workspace.region_high[1]).sum()>0)):
            # input("next...")
            continue
        # collision with upper ceiling: when the object is within workspace region and too high
        collide_with_upper_filter = (transformed_pcd[:,0] > workspace.region_low[0]-safety_padding[0]) & \
                                    (transformed_pcd[:,0] < workspace.region_high[0]+safety_padding[0]) & \
                                    (transformed_pcd[:,1] > workspace.region_low[1]-safety_padding[1]) & \
                                    (transformed_pcd[:,1] < workspace.region_high[1]+safety_padding[1]) & \
                                    (transformed_pcd[:,2] > workspace.region_high[2]-safety_padding[2])
        if collide_with_upper_filter.sum() > 0:
            # input("next...")
            continue

        if collision_utils.obj_pose_collision_with_obj(obj_i, obj, transform, occlusion, occluded_label, occupied_label, workspace):
            # input("next...")
            continue

        transformed_pcd = cam_transform[:3,:3].dot(transformed_pcd.T).T + cam_transform[:3,3]
        transformed_pcd[:,0] = transformed_pcd[:,0] / transformed_pcd[:,2] * fx + cx
        transformed_pcd[:,1] = transformed_pcd[:,1] / transformed_pcd[:,2] * fy + cy

        pcd_pixel_idx = np.floor(transformed_pcd[:,:2]).astype(int)
        valid_filter = (pcd_pixel_idx[:,0]>=0) & (pcd_pixel_idx[:,0]<img_size) & \
                        (pcd_pixel_idx[:,1]>=0) & (pcd_pixel_idx[:,1]<img_size)
        pcd_pixel_idx = pcd_pixel_idx[valid_filter]
        if seg_img[pcd_pixel_idx[:,1], pcd_pixel_idx[:,0]].sum() > 0:
            # within camera view
            # input("next...")
            continue
        

        # * then check whether it's reachable
        filtered_tip_poses_in_obj = []
        filtered_start_suction_joints = []
        filtered_intermediate_joints = []
        for i in range(len(gripper_tip_poses_in_obj)):
            selected_tip_in_obj = gripper_tip_poses_in_obj[i]
            transformed_tip = transform.dot(gripper_tip_poses_in_obj[i])
            start_suction_joint = suction_joints[i]
            quat = tf.quaternion_from_matrix(transformed_tip)  # w x y z

            # visualize the target pose             
            valid, dof_joint_vals = robot.get_ik(robot.tip_link_name, transformed_tip[:3,3], [quat[1],quat[2],quat[3],quat[0]], suction_joints[i], 
                                                collision_check=True, workspace=workspace)

            if not valid:
                continue

            prev_joint_vals = robot.joint_vals
            robot.set_joints(dof_joint_vals)

            # for each of the suction gripper transform, check whether ik is reachable and without collision
            collision = False
            for comp_name, comp_id in workspace.component_id_dict.items():
                contacts = p.getClosestPoints(robot.robot_id, comp_id, distance=0.,physicsClientId=robot.pybullet_id)
                if len(contacts):
                    collision = True
                    break
            # input('current robot pose')
            robot.set_joints(prev_joint_vals)
            if collision:
                continue

            # * robot shouldn't be within camera view
            transformed_pcd = robot.get_pcd_at_joints(dof_joint_vals)
            transformed_pcd = cam_transform[:3,:3].dot(transformed_pcd.T).T + cam_transform[:3,3]
            transformed_pcd[:,0] = transformed_pcd[:,0] / transformed_pcd[:,2] * fx + cx
            transformed_pcd[:,1] = transformed_pcd[:,1] / transformed_pcd[:,2] * fy + cy

            pcd_pixel_idx = np.floor(transformed_pcd[:,:2]).astype(int)
            valid_filter = (pcd_pixel_idx[:,0]>=0) & (pcd_pixel_idx[:,0]<img_size) & \
                            (pcd_pixel_idx[:,1]>=0) & (pcd_pixel_idx[:,1]<img_size)
            pcd_pixel_idx = pcd_pixel_idx[valid_filter]
            if (len(pcd_pixel_idx) > 0) and (seg_img[pcd_pixel_idx[:,1], pcd_pixel_idx[:,0]].sum() > 0):
                # within camera view
                continue


            filtered_tip_poses_in_obj.append(selected_tip_in_obj)
            filtered_start_suction_joints.append(start_suction_joint)
            filtered_intermediate_joints.append(dof_joint_vals)


        # quat = tf.quaternion_from_matrix(pybullet_obj_pose) # w x y z
        # p.resetBasePositionAndOrientation(pybullet_obj_i, pybullet_obj_pose[:3,3], [quat[1],quat[2],quat[3],quat[0]], physicsClientId=robot.pybullet_id)
        # input('after resetting.')


        if len(filtered_tip_poses_in_obj) > 0:
            break
    intermediate_pose = transform
    return intermediate_pose, filtered_tip_poses_in_obj, filtered_start_suction_joints, filtered_intermediate_joints


def sample_sense_pose(obj_i, pybullet_obj_i, pybullet_obj_pose, objects, 
                        seg_img, occlusion, occluded_label, occupied_label, 
                        selected_tip_in_obj, joint_dict, camera, robot, workspace, sample_n=5):
    """
    sample reachable intermediate pose that is outside of the camera view
    UPDATE: we want the object to not hide the parts where it is located before
    """
    x_max = workspace.region_low[0]
    x_min = robot.transform[0,3] + 0.6
    y_min = workspace.region_low[1]*0.9
    y_max = workspace.region_high[1]*1.1
    z_min = workspace.region_low[2]*0.8
    z_max = workspace.region_high[2]*1.1

    # x_max = workspace.region_low[0]-0.05
    # x_min = workspace.region_low[0]-0.4  #robot.transform[0,3] + 0.6
    # y_min = workspace.region_low[1]*0.9
    # y_max = workspace.region_high[1]*1.1
    # z_min = workspace.region_low[2]*0.8
    # z_max = workspace.region_high[2]*1.1


    extrinsics = camera.info['extrinsics']
    intrinsics = camera.info['intrinsics']
    img_size = camera.info['img_size']
    cam_transform = np.linalg.inv(extrinsics)
    fx = intrinsics[0][0]
    fy = intrinsics[1][1]
    cx = intrinsics[0][2]
    cy = intrinsics[1][2]

    obj = objects[obj_i]
    obj_pcd = obj.sample_conservative_pcd()

    # make sure the object pcd is not in view
    max_uncertainty = 0
    max_net_transform = objects[obj_i].transform
    for sample_i in range(sample_n):
        # get 10 sample position and find the one with the maximum uncertainty
        while True:
            pos = np.random.uniform(low=[x_min,y_min,z_min], high=[x_max,y_max,z_max])

            angle = np.random.normal(loc=0,scale=1,size=[2])
            angle = angle / np.linalg.norm(angle)
            angle = np.arcsin(angle[1])
            # rotate around z axis
            rot_mat = tf.rotation_matrix(angle, [0,0,1])

            transform = rot_mat
            # transform[:3,:3] = rot_mat
            transform[:3,3] = pos
            net_transform = obj.get_net_transform_from_center_frame(obj.sample_conservative_pcd(), transform)
            transform = net_transform

            # * first check whether the object is in camera view
            transformed_pcd = transform[:3,:3].dot(obj_pcd.T).T + transform[:3,3]



            # visualize pybullet
            # delta_transform = transform.dot(np.linalg.inv(obj.transform))
            # new_pybullet_pose = delta_transform.dot(pybullet_obj_pose)
            # quat = tf.quaternion_from_matrix(new_pybullet_pose) # w x y z
            # p.resetBasePositionAndOrientation(pybullet_obj_i, new_pybullet_pose[:3,3], [quat[1],quat[2],quat[3],quat[0]], physicsClientId=robot.pybullet_id)

            # input('after sampling.')

            if ((transformed_pcd[:,1] < workspace.region_low[1]).sum()>0) or (((transformed_pcd[:,1] > workspace.region_high[1]).sum()>0)):
                # input("next...")
                continue
            # collision with upper ceiling: when the object is within workspace region and too high
            collide_with_upper_filter = (transformed_pcd[:,0] > workspace.region_low[0]) & \
                                        (transformed_pcd[:,0] < workspace.region_high[0]) & \
                                        (transformed_pcd[:,1] > workspace.region_low[1]) & \
                                        (transformed_pcd[:,1] < workspace.region_high[1]) & \
                                        (transformed_pcd[:,2] > workspace.region_high[2])
            if collide_with_upper_filter.sum() > 0:
                # input("next...")
                continue

            # collision with lower floor: when the object is within workspace region and too low
            collide_with_lower_filter = (transformed_pcd[:,0] > workspace.region_low[0]) & \
                                        (transformed_pcd[:,0] < workspace.region_high[0]) & \
                                        (transformed_pcd[:,1] > workspace.region_low[1]) & \
                                        (transformed_pcd[:,1] < workspace.region_high[1]) & \
                                        (transformed_pcd[:,2] < workspace.region_low[2])
            if collide_with_lower_filter.sum() > 0:
                # input("next...")
                continue


            if collision_utils.obj_pose_collision_with_obj(obj_i, obj, transform, occlusion, occluded_label, occupied_label, workspace):
                # input("next...")
                continue



            transformed_pcd = cam_transform[:3,:3].dot(transformed_pcd.T).T + cam_transform[:3,3]
            transformed_pcd[:,0] = transformed_pcd[:,0] / transformed_pcd[:,2] * fx + cx
            transformed_pcd[:,1] = transformed_pcd[:,1] / transformed_pcd[:,2] * fy + cy

            pcd_pixel_idx = np.floor(transformed_pcd[:,:2]).astype(int)
            valid_filter = (pcd_pixel_idx[:,0]>=0) & (pcd_pixel_idx[:,0]<img_size) & \
                            (pcd_pixel_idx[:,1]>=0) & (pcd_pixel_idx[:,1]<img_size)
            pcd_pixel_idx = pcd_pixel_idx[valid_filter]
            
            if len(pcd_pixel_idx) == 0:
                # outside of camera view
                continue


            # * then check whether it's reachable
            transformed_tip = transform.dot(selected_tip_in_obj)
            # start_suction_joint = joint_dict
            # joint dict to joint list
            start_suction_joint = []
            for i in range(len(robot.joint_names)):
                start_suction_joint.append(joint_dict[robot.joint_names[i]])

            quat = tf.quaternion_from_matrix(transformed_tip)  # w x y z

            # visualize the target pose             
            valid, dof_joint_vals = robot.get_ik(robot.tip_link_name, transformed_tip[:3,3], [quat[1],quat[2],quat[3],quat[0]], start_suction_joint, 
                                                collision_check=True, workspace=workspace)

            if not valid:
                continue


            # prev_joint_vals = robot.joint_vals
            # robot.set_joints(dof_joint_vals)

            # # for each of the suction gripper transform, check whether ik is reachable and without collision
            # collision = False
            # for comp_name, comp_id in workspace.component_id_dict.items():
            #     contacts = p.getClosestPoints(robot.robot_id, comp_id, distance=0.,physicsClientId=robot.pybullet_id)
            #     if len(contacts):
            #         collision = True
            #         break
            # # input('current robot pose')
            # robot.set_joints(prev_joint_vals)


            # quat = tf.quaternion_from_matrix(pybullet_obj_pose) # w x y z
            # p.resetBasePositionAndOrientation(pybullet_obj_i, pybullet_obj_pose[:3,3], [quat[1],quat[2],quat[3],quat[0]], physicsClientId=robot.pybullet_id)
            # input('after resetting.')


            break
        camera_extrinsics = camera.info['extrinsics']
        camera_intrinsics = camera.info['intrinsics']
        uncertainty = occlusion.obtain_object_uncertainty(obj, net_transform, camera_extrinsics, camera_intrinsics, camera.info['img_shape'])
        if uncertainty > max_uncertainty:
            max_uncertainty = uncertainty
            max_net_transform = net_transform
            max_selected_tip_in_obj = selected_tip_in_obj
            max_transformed_tip = transformed_tip
            max_dof_joint_vals = dof_joint_vals
            max_start_suction_joint = start_suction_joint

    sense_pose = max_net_transform
    del transformed_pcd
    # sense_pose = transform
    return sense_pose, max_selected_tip_in_obj, max_transformed_tip, max_start_suction_joint, max_dof_joint_vals



def placement_pose_generation(obj_i, objects, occlusion, occlusion_label, occupied_label, occluded_dict, 
                            workspace, robot, gripper_tip_in_obj):
    obj = objects[obj_i]
    # obj = np.random.choice(objects)
    # try a number of times for randomly sample a location q, until it's feasible

    voxel_x = occlusion.voxel_x
    voxel_y = occlusion.voxel_y
    voxel_z = occlusion.voxel_z

    free_x = voxel_x[(occlusion_label<=0) & (occupied_label==0)].astype(int)
    free_y = voxel_y[(occlusion_label<=0) & (occupied_label==0)].astype(int)
    free_z = voxel_z[(occlusion_label<=0) & (occupied_label==0)].astype(int)

    # extract the bottom level of the voxel
    free_x = free_x[free_z==0]
    free_y = free_y[free_z==0]
    free_z = free_z[free_z==0]

    # transform the free space to world
    free_pts = np.array([free_x, free_y, free_z]).T.astype(float)

    # TODO
    # make sure the sampled points are within the world boundary:
    # the upper bound is within the lower boundary
    # the lower bound is within the upper boundary
    # free_pts_in_world = (free_pts+1) * occlusion.resol
    # free_pts_in_world = occlusion.transform[:3,:3].dot(free_pts_in_world.T).T + occlusion.transform[:3,3]
    # free_pts_in_world 


    color_pick = np.zeros((6,3))
    color_pick[0] = np.array([1., 0., 0.])
    color_pick[1] = np.array([0., 1.0, 0.])
    color_pick[2] = np.array([0., 0., 1.])
    color_pick[3] = np.array([252/255, 169/255, 3/255])
    color_pick[4] = np.array([252/255, 3/255, 252/255])
    color_pick[5] = np.array([20/255, 73/255, 82/255])


    sampled_poses = []
    sampled_occlusion_volumes = []
    max_sample_n = 20

    sample_i = 0

    success = False
    while True:
        sample_i += 1
        if sample_i > max_sample_n:
            break
        # randomly select one index
        cell_idx = np.random.choice(len(free_pts))
        # randomly pick one point in the cell
        pt = free_pts[cell_idx]
        pt[:2] = np.random.uniform(low=free_pts[cell_idx][:2], high=free_pts[cell_idx][:2]+1)

        # # try uniform sampling in the entire space
        # pt[:2] = np.random.uniform(low=[0,0], high=[occlusion.voxel_x.shape[0],occlusion.voxel_x.shape[1]])

        free_pt_padded = np.zeros(4)
        free_pt_padded[:3] = pt * occlusion.resol
        free_pt_padded[3] = 1

        free_pt_world = occlusion.transform.dot(free_pt_padded)
        free_pt_world = free_pt_world[:3]

        # validate the sample
        # TODO: orientation sample
        angle = np.random.normal(loc=0,scale=1,size=[2])
        angle = angle / np.linalg.norm(angle)
        angle = np.arcsin(angle[1])
        # rotate around z axis
        rot_mat = tf.rotation_matrix(angle, [0,0,1])
        # center_transform: the transform of the center of the pcd
        center_transform = np.array(rot_mat)
        center_transform[:2,3] = free_pt_world[:2]
        center_transform[2,3] = obj.zmin  # set it on the table


        # transform the point cloud and check if there is intersection with the occlusion space
        pcd = obj.sample_conservative_pcd()

        new_obj_pose = obj.get_net_transform_from_center_frame(pcd, center_transform)

        transformed_pcd = new_obj_pose[:3,:3].dot(pcd.T).T + new_obj_pose[:3,3]
        transformed_pcd_in_voxel = occlusion.world_in_voxel_rot.dot(transformed_pcd.T).T + occlusion.world_in_voxel_tran
        transformed_pcd_in_voxel = transformed_pcd_in_voxel / occlusion.resol

        transformed_pcd_indices = np.floor(transformed_pcd_in_voxel).astype(int)
        valid_filter = (transformed_pcd_indices[:,0] >= 0) & (transformed_pcd_indices[:,0] < occlusion_label.shape[0]) & \
                        (transformed_pcd_indices[:,1] >= 0) & (transformed_pcd_indices[:,1] < occlusion_label.shape[1]) & \
                        (transformed_pcd_indices[:,2] >= 0) & (transformed_pcd_indices[:,2] < occlusion_label.shape[2])
        valid_transformed_pcd_indices = transformed_pcd_indices[valid_filter]
        extracted_occlusion = occlusion_label[valid_transformed_pcd_indices[:,0],valid_transformed_pcd_indices[:,1],valid_transformed_pcd_indices[:,2]]

        sampled_pose = new_obj_pose

        voxels = []
        vis_obj_pcds = []

        # input("press enter to see next sample...")

        # NOTE: we can't be in occlusion induced by any objects, including the current object
        #  since we don't know what's inside the occlusion region
        #  but for occupied space we need to exclude the ones occupied by the current object

        # visualize
        # voxel1 = visualize_voxel(occlusion.voxel_x, occlusion.voxel_y, occlusion.voxel_z, occlusion_label>0, [0,0,0])
        # pcd1 = visualize_pcd(transformed_pcd_in_voxel, [1,0,0])
        # voxel2 = visualize_voxel(occlusion.voxel_x, occlusion.voxel_y, occlusion.voxel_z, 
        #                         (occupied_label>0)&(occupied_label!=obj_i+1), [0,0,1])

        # o3d.visualization.draw_geometries([voxel1, pcd1, voxel2])            

        if (extracted_occlusion>0).sum() > 0:
            # invalid
            continue

        # check collision with workspace using pcd: should be within the workspace boundary
        valid_filter = (transformed_pcd[:,0] >= workspace.region_low[0]) & (transformed_pcd[:,0] <= workspace.region_high[0]) & \
                        (transformed_pcd[:,1] >= workspace.region_low[1]) & (transformed_pcd[:,1] <= workspace.region_high[1])
        if (transformed_pcd[~valid_filter].sum() > 0):
            # invalid due to collision with workspace
            continue

        # check collision with other obstacle using occupied region
        extracted_occupied = occupied_label[valid_transformed_pcd_indices[:,0],valid_transformed_pcd_indices[:,1],valid_transformed_pcd_indices[:,2]]
        if ((extracted_occupied>0) & (extracted_occupied!=obj_i+1)).sum() > 0:
            # Note: label of occupied space is object_id+1
            # invalid
            continue
        
        transform = new_obj_pose
        # * check whether the IK is valid
        transformed_tip = transform.dot(gripper_tip_in_obj)
        # start_suction_joint = joint_dict
        # joint dict to joint list
        start_suction_joint = robot.joint_vals

        quat = tf.quaternion_from_matrix(transformed_tip)  # w x y z

        # visualize the target pose             
        valid, dof_joint_vals = robot.get_ik(robot.tip_link_name, transformed_tip[:3,3], [quat[1],quat[2],quat[3],quat[0]], start_suction_joint, 
                                            collision_check=True, workspace=workspace)

        if not valid:
            continue
        prev_joint_vals = robot.joint_vals
        robot.set_joints(dof_joint_vals)

        # for each of the suction gripper transform, check whether ik is reachable and without collision
        collision = False
        for comp_name, comp_id in workspace.component_id_dict.items():
            contacts = p.getClosestPoints(robot.robot_id, comp_id, distance=0.,physicsClientId=robot.pybullet_id)
            if len(contacts):
                collision = True
                break
        # input('current robot pose')
        robot.set_joints(prev_joint_vals)

        # quat = tf.quaternion_from_matrix(pybullet_obj_pose) # w x y z
        # p.resetBasePositionAndOrientation(pybullet_obj_i, pybullet_obj_pose[:3,3], [quat[1],quat[2],quat[3],quat[0]], physicsClientId=robot.pybullet_id)
        # input('after resetting.')

        success = True
        break
        # sampled_poses.append(sampled_pose)

    # * select the sample that minimizes the target occlusion
    if success:
        # obtain the relative transform of the object
        delta_transform = sampled_pose.dot(np.linalg.inv(obj.transform))
        return delta_transform, dof_joint_vals

    else:
        return None, None