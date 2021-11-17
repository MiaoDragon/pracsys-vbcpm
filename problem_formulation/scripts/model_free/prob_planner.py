"""
This implements the planner which is probablistic complete.
It involves two actions:
- select an object to grasp, sense the scene without the object, then put the object at q
- select an object to grasp, sense the object, then put the object at q

if target object has been revealed, sample the target object with some probability > 0

It is probablistic complete since at each time we pick an object with probability >0, and
sample a pose q with probability >0
"""
import numpy as np
import cam_utilities
from visual_utilities import *
import transformations as tf

class ProbPlanner():
    def __init__(self):
        pass
    def snapshot_object_selection(self, objects, occlusion, occlusion_label, occupied_label, occluded_dict, workspace):
        obj_i = np.random.choice(list(objects.keys()))
        obj = objects[obj_i]
        return obj_i

    def snapshot_pose_generation(self, obj_i, objects, occlusion, occlusion_label, occupied_label, occluded_dict, workspace):
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
            voxel1 = visualize_voxel(occlusion.voxel_x, occlusion.voxel_y, occlusion.voxel_z, occlusion_label>0, [0,0,0])
            pcd1 = visualize_pcd(transformed_pcd_in_voxel, [1,0,0])
            voxel2 = visualize_voxel(occlusion.voxel_x, occlusion.voxel_y, occlusion.voxel_z, 
                                    (occupied_label>0)&(occupied_label!=obj_i+1), [0,0,1])

            o3d.visualization.draw_geometries([voxel1, pcd1, voxel2])            

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
            if ((extracted_occupied>0) & (extracted_occupied!=obj_i+1)).sum() > 0:
                # Note: label of occupied space is object_id+1
                # invalid
                print('object colliding with environment')
                continue
            
            print("valid pose")
            success = True
            break
            # sampled_poses.append(sampled_pose)

        # * select the sample that minimizes the target occlusion
        if success:
            # obtain the relative transform of the object
            delta_transform = sampled_pose.dot(np.linalg.inv(obj.transform))
            return delta_transform

        else:
            return None

