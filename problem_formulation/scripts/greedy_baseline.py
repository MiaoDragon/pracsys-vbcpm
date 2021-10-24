"""
a greedy baseline that moves the object with the maximum volume of shadow
region each time
"""
from cv2 import transform
import numpy as np
import pybullet as p
from visual_utilities import *
import cam_utilities
LOG = 1  # denote whether we log information or not

def greedy_baseline_snapshot(planning_info):
    """
    provide a snapshot (sequence of objs to move) based on greedy approach:
    each time we move the object with the max volume of shadow, and selects
    the place to put it.
    If the occlusion region before moving is:
        occ_still  U  occ_moving
    then after the movement, the occlusion region is:
        occ_still U ((occ_still U occ_moving) \cap occ_moving_2)
    OR
        occlusion_1 \cap occlusion_2
    """

    """
    planning_info:
    pid, scene_dict, robot, workspace, camera, occlusion, obj_poses (None if unseen), obj_pcds, target_obj_pcd
    obj_ids, target_obj_id

    occlusion_label, occupied_label, occluded_list, depth_img
    """

    # * compute the volume of the occluded region for each object
    occluded_list = planning_info['occluded_list']  # occluded place for each object
    # occluded list is only for non-target objects

    obj_poses = planning_info['obj_poses']  # obj_poses are for non-target objects
    obj_pcds = planning_info['obj_pcds']  # obj_pcds are for non-target objects
    obj_ids = planning_info['obj_ids']  # obj_ids are for non-target objects
    max_sample_n = planning_info['max_sample_n']
    # occlusion label: object idx + 1 (start from 1)

    vol_max = 0.
    vol_max_i = -1
    for i in range(len(occluded_list)):
        if occluded_list[i] is not None:
            # known and seen
            vol_i = occluded_list[i].astype(int).sum()
            if vol_i > vol_max:
                vol_max = vol_i
                vol_max_i = i
    print('max volume corresponds to object: ', vol_max_i)

    # * sample target places that satisfy the following constraints:
    #   - not in the occluded region
    #   - not in collision with other objects
    
    # obtain the free space: not occlusion or occupied
    occlusion_label = planning_info['occlusion_label']
    occupied_label = planning_info['occupied_label']
    occlusion = planning_info['occlusion']
    workspace = planning_info['workspace']
    camera = planning_info['camera']
    depth_img = planning_info['depth_img']
    pid = planning_info['pid']
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

    env_pcd, _ = cam_utilities.pcd_from_depth(camera.info['intrinsics'], camera.info['extrinsics'], depth_img, None)

    env_pcd = np.concatenate([env_pcd, np.ones(env_pcd.shape[0]).reshape((-1,1))],axis=1)
    # transform the ray vector into the voxel_grid space
    # notice that we have to divide by the resolution vector
    env_pcd = np.linalg.inv(occlusion.transform).dot(env_pcd.T).T
    env_pcd = env_pcd[:,:3] / occlusion.resol

    env_pcd = visualize_pcd(env_pcd, [0.,1.,1.])

    sampled_poses = []
    sampled_occlusion_volumes = []
    sampled_n = 10

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

        free_pt_padded = np.zeros(4)
        free_pt_padded[:3] = pt * occlusion.resol
        free_pt_padded[3] = 1

        # free_pts_padded = np.zeros((len(free_pts),4))
        # free_pts_padded[:,:3] = free_pts
        # free_pts_padded[:,3] = 1.0

        free_pt_world = occlusion.transform.dot(free_pt_padded)
        free_pt_world = free_pt_world[:3]

        # validate the sample
        # TODO: orientation sample
        new_obj_pose = np.array(obj_poses[vol_max_i])
        new_obj_pose[:2,3] = free_pt_world[:2]

        # transform the point cloud and check if there is intersection with the occlusion space
        pcd = obj_pcds[vol_max_i]
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

        # input("press enter to see sample...")
        # p.resetBasePositionAndOrientation(obj_ids[vol_max_i], obj_poses[vol_max_i][:3,3], [0,0,0,1],physicsClientId=pid)

        p.resetBasePositionAndOrientation(obj_ids[vol_max_i], sampled_pose[:3,3], [0,0,0,1], physicsClientId=pid)

        voxels = []
        vis_obj_pcds = []
        print('occluded_list number: ', len(occluded_list))
        for i in range(len(occluded_list)):
            if occluded_list[i] is None:
                print('occluded_list is None')
                continue
            if i == vol_max_i:
                continue
            voxel = visualize_voxel(occlusion.voxel_x, occlusion.voxel_y, occlusion.voxel_z, occluded_list[i], color_pick[i%len(color_pick)])
            voxels.append(voxel)
            obj_voxel = visualize_voxel(occlusion.voxel_x, occlusion.voxel_y, occlusion.voxel_z, occupied_label==i+1, color_pick[i%len(color_pick)])
            voxels.append(obj_voxel)

        pcd = visualize_pcd(transformed_pcd_in_voxel, [0.,0.,0.])


        # input("press enter to see next sample...")

        # TODO: better occlusion handle should use the occluded_list, since
        # the label might overlap and miss some regions
        if (extracted_occlusion>0).sum() > 0:
            # invalid
            print('part of the object lies in the occluded region')
            continue

        # check collision with workspace using pcd: should be within the workspace boundary
        valid_filter = (transformed_pcd[:,0] >= workspace.region_low[0]) & (transformed_pcd[:,0] <= workspace.region_high[0]) & \
                        (transformed_pcd[:,1] >= workspace.region_low[1]) & (transformed_pcd[:,1] <= workspace.region_high[1])
        if (transformed_pcd[~valid_filter].sum() > 0):
            # invalid due to collision with workspace
            print('part of the object lies out of the boundary')
            continue

        # check collision with other obstacle using occupied region
        extracted_occupied = occupied_label[valid_transformed_pcd_indices[:,0],valid_transformed_pcd_indices[:,1],valid_transformed_pcd_indices[:,2]]
        if ((extracted_occupied>0) & (extracted_occupied!=vol_max_i+1)).sum() > 0:
            # Note: label of occupied space is object_id+1
            # invalid
            print('object colliding with environment')
            continue
        
        print("valid pose")
        sampled_poses.append(sampled_pose)

        # compute the occluded region: the intersection of previous occlusion with the current occlusion
        new_occupied, new_occluded = occlusion.single_object_occlusion(camera.info['extrinsics'], camera.info['intrinsics'], sampled_pose, obj_pcds[vol_max_i])

        sampled_occlusion = (occlusion_label>0) & (new_occluded)
        for i in range(len(occluded_list)):
            if i == vol_max_i:
                continue
            if obj_poses[i] is None:
                continue
            sampled_occlusion = sampled_occlusion | occluded_list[i]

        new_occ_voxel = visualize_voxel(occlusion.voxel_x, occlusion.voxel_y, occlusion.voxel_z, new_occluded>0, [0,0,1])
        prev_occ_voxel = visualize_voxel(occlusion.voxel_x, occlusion.voxel_y, occlusion.voxel_z, occlusion_label>0, [0,1,0])
        intersection_voxel = visualize_voxel(occlusion.voxel_x, occlusion.voxel_y, occlusion.voxel_z, sampled_occlusion, [1,0,0])
        # o3d.visualization.draw_geometries([env_pcd] + [new_occ_voxel,intersection_voxel])

        sampled_occlusion_volume = sampled_occlusion.astype(int).sum()
        sampled_occlusion_volumes.append(sampled_occlusion_volume)
        print('number of sample: ', len(sampled_occlusion_volumes))
        print('volume: ', sampled_occlusion_volume)
        if len(sampled_occlusion_volumes) == sampled_n:
            success = True
            break
    p.resetBasePositionAndOrientation(obj_ids[vol_max_i], obj_poses[vol_max_i][:3,3], [0,0,0,1],physicsClientId=pid)

    # * select the sample that minimizes the target occlusion
    if success:
        min_i = np.argmin(sampled_occlusion_volumes)
        min_pose = sampled_poses[min_i]
    else:
        min_pose = None

    if LOG:
        return vol_max_i, min_pose, sample_i
    else:
        return vol_max_i, min_pose
