"""
provide functions for checking collisions
"""
from cv2 import transform
import numpy as np
from visual_utilities import *
import open3d as o3d
def obj_pose_collision(obj_id, obj, transform, occlusion, occluded_label, occupied_label, workspace):
    obj_pcd = obj.sample_conservative_pcd()
    obj_pcd = transform[:3,:3].dot(obj_pcd.T).T + transform[:3,3]
    # * check collision with workspace
    ws_transforms = workspace.transforms
    ws_lls = workspace.bbox_lls
    ws_uls = workspace.bbox_uls
    
    # for visualization
    obj_pcd_in_occ = occlusion.world_in_voxel_rot.dot(obj_pcd.T).T + occlusion.world_in_voxel_tran
    obj_pcd_in_occ = obj_pcd_in_occ / occlusion.resol

    for comp_name, ws_transform in ws_transforms.items():
        world_in_comp = np.linalg.inv(ws_transform)
        pcd_in_ws = world_in_comp[:3,:3].dot(obj_pcd.T).T + world_in_comp[:3,3]
        ws_ll = ws_lls[comp_name]
        ws_ul = ws_uls[comp_name]
        valid_filter = (pcd_in_ws >= ws_ll) & (pcd_in_ws <= ws_ul)
        if (~valid_filter).sum() > 0:

            # draw the scene
            voxel = visualize_voxel(occlusion.voxel_x, occlusion.voxel_y, occlusion.voxel_z, occluded_label>0, [0,0,0])
            
            pcd = visualize_pcd(obj_pcd_in_occ, [1,0,0])

            o3d.visualization.draw_geometries([voxel, pcd])            

            return True  # collision

    # * check collision with other objects
    pcd_in_occ = occlusion.world_in_voxel_rot.dot(obj_pcd.T).T + occlusion.world_in_voxel_tran
    pcd_in_occ_ind = pcd_in_occ / occlusion.resol
    pcd_in_occ_ind = np.floor(pcd_in_occ_ind).astype(int)
    
    valid_filter = (pcd_in_occ_ind[:,0] >= 0) & (pcd_in_occ_ind[:,0] < occupied_label.shape[0]) & \
                   (pcd_in_occ_ind[:,1] >= 0) & (pcd_in_occ_ind[:,1] < occupied_label.shape[1]) & \
                   (pcd_in_occ_ind[:,2] >= 0) & (pcd_in_occ_ind[:,2] < occupied_label.shape[2])

    pcd_in_occ_ind = pcd_in_occ_ind[valid_filter]
    extracted_occupied = occupied_label[pcd_in_occ_ind[:,0], pcd_in_occ_ind[:,1], pcd_in_occ_ind[:,2]]
    if ((extracted_occupied>0) & (extracted_occupied!=obj_id+1)).sum() >0:
        return True

    # * check collision with occlusion
    extracted_occluded = occluded_label[pcd_in_occ_ind[:,0], pcd_in_occ_ind[:,1], pcd_in_occ_ind[:,2]]
    if ((extracted_occluded>0) & (extracted_occluded!=obj_id+1)).sum() >0:
        return True
    return False

def obj_pose_collision_with_obj(obj_id, obj, transform, occlusion, occluded_label, occupied_label, workspace):
    obj_pcd = obj.sample_conservative_pcd()
    obj_pcd = transform[:3,:3].dot(obj_pcd.T).T + transform[:3,3]
    # * check collision with workspace
    ws_transforms = workspace.transforms
    ws_lls = workspace.bbox_lls
    ws_uls = workspace.bbox_uls
    

    # collision with other objects
    pcd_in_occ = occlusion.world_in_voxel_rot.dot(obj_pcd.T).T + occlusion.world_in_voxel_tran
    pcd_in_occ_ind = pcd_in_occ / occlusion.resol
    pcd_in_occ_ind = np.floor(pcd_in_occ_ind).astype(int)
    
    valid_filter = (pcd_in_occ_ind[:,0] >= 0) & (pcd_in_occ_ind[:,0] < occupied_label.shape[0]) & \
                   (pcd_in_occ_ind[:,1] >= 0) & (pcd_in_occ_ind[:,1] < occupied_label.shape[1]) & \
                   (pcd_in_occ_ind[:,2] >= 0) & (pcd_in_occ_ind[:,2] < occupied_label.shape[2])

    pcd_in_occ_ind = pcd_in_occ_ind[valid_filter]
    extracted_occupied = occupied_label[pcd_in_occ_ind[:,0], pcd_in_occ_ind[:,1], pcd_in_occ_ind[:,2]]
    if ((extracted_occupied>0) & (extracted_occupied!=obj_id+1)).sum() >0:
        return True

    # * check collision with occlusion
    extracted_occluded = occluded_label[pcd_in_occ_ind[:,0], pcd_in_occ_ind[:,1], pcd_in_occ_ind[:,2]]
    if ((extracted_occluded>0) & (extracted_occluded!=obj_id+1)).sum() >0:
        return True
    return False


def robot_collision_with_voxel_env(joint_vals, robot, collision_transform, collision_voxel, voxel_resol):
    # * obtain robot pcd    
    rpcd = robot.get_pcd_at_joints(joint_vals)    

    # collision_transform: the transform of the collision voxel
    world_in_collision = np.linalg.inv(collision_transform)
    
    # * trasnform robot pcd to voxel frame
    transformed_rpcd = world_in_collision[:3,:3].dot(rpcd.T).T + world_in_collision[:3,3]

    

    transformed_rpcd = transformed_rpcd / voxel_resol
    vis_transformed_rpcd = transformed_rpcd


    transformed_rpcd = np.floor(transformed_rpcd).astype(int)
    valid_filter = (transformed_rpcd[:,0] >= 0) & (transformed_rpcd[:,0] < collision_voxel.shape[0]) & \
                    (transformed_rpcd[:,1] >= 0) & (transformed_rpcd[:,1] < collision_voxel.shape[1]) & \
                    (transformed_rpcd[:,2] >= 0) & (transformed_rpcd[:,2] < collision_voxel.shape[2])
    transformed_rpcd = transformed_rpcd[valid_filter]


    # * check against collision
    # collision: return True
    return collision_voxel[transformed_rpcd[:,0], transformed_rpcd[:,1], transformed_rpcd[:,2]].sum() > 0


