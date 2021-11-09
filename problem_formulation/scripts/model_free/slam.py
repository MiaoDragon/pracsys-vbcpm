"""
SLAM system to reconstruct objects and update occlusion volumes
use this whenever a new picture is obtained
"""
from object import ObjectModel
from occlusion import Occlusion

import numpy as np
import cv2
from visual_utilities import *
import open3d as o3d
class SLAMPerception():
    def __init__(self, occlusion_params, object_params):
        occlusion = Occlusion(**occlusion_params)
        self.object_params = object_params  # resol and scale
        self.occlusion = occlusion
        self.objects = {}
        self.obj_initial_poses = {}  # useful to backpropagate
        self.sensed_imgs = []
        self.sensed_poses = []
    
    def perceive(self, depth_img, color_img, seg_img, assoc, camera_extrinsics, camera_intrinsics):
        """
        given depth img and color img from camera, and segmented and labeld images from 
        Segmentation and Data Association, update occlusion and object model
        
        Object model: 
        for sensed ones, use the segmented depth image to update model
        for new ones, create a new entry in the object list

        Occlusion:
        use the new reconstructed object model, get new occlusion of the scene
        """
        total_occluded = self.occlusion.scene_occlusion(depth_img, color_img, camera_extrinsics, camera_intrinsics)
        for seg_id, obj_id in assoc.items():
            print((seg_img==seg_id).astype(int).sum())
            seg_depth_img = np.array(depth_img)
            seg_depth_img[seg_img!=seg_id] = 0
            seg_color_img = np.array(color_img)
            seg_color_img[seg_img!=seg_id,:] = 0
            occluded = self.occlusion.scene_occlusion(seg_depth_img, seg_color_img, camera_extrinsics, camera_intrinsics)
            cv2.imshow('segmented depth', seg_depth_img)
            cv2.waitKey(0)

            print('transform: ', self.occlusion.transform)

            if (not (obj_id in self.objects)) or (not self.objects[obj_id].active):
                # compute bounding box for the conservative region
                voxel_x_min = self.occlusion.voxel_x[occluded].reshape(-1)-np.ceil(self.object_params['resol'][0]*self.object_params['scale'])-1
                voxel_y_min = self.occlusion.voxel_y[occluded].reshape(-1)-np.ceil(self.object_params['resol'][1]*self.object_params['scale'])-1
                voxel_z_min = self.occlusion.voxel_z.reshape(-1)-np.ceil(self.object_params['resol'][2]*self.object_params['scale'])-1
                voxel_z_min = voxel_z_min[:len(voxel_x_min)]
                voxel_z_min[:] = voxel_z_min.min()  # we want the min to be the lowest in the workspace

                voxel_x_min = voxel_x_min * self.occlusion.resol[0]
                voxel_y_min = voxel_y_min * self.occlusion.resol[1]
                voxel_z_min = voxel_z_min * self.occlusion.resol[2]
                voxel_min = np.array([voxel_x_min,voxel_y_min,voxel_z_min])
                voxel_min = self.occlusion.transform[:3,:3].dot(voxel_min).T + self.occlusion.transform[:3,3]
                

                voxel_x_max = self.occlusion.voxel_x[occluded].reshape(-1)+np.ceil(self.object_params['resol'][0]*self.object_params['scale'])+1
                voxel_y_max = self.occlusion.voxel_y[occluded].reshape(-1)+np.ceil(self.object_params['resol'][1]*self.object_params['scale'])+1
                voxel_z_max = self.occlusion.voxel_z[occluded].reshape(-1)+np.ceil(self.object_params['resol'][2]*self.object_params['scale'])+1
                voxel_x_max = voxel_x_max * self.occlusion.resol[0]
                voxel_y_max = voxel_y_max * self.occlusion.resol[1]
                voxel_z_max = voxel_z_max * self.occlusion.resol[2]
                voxel_max = np.array([voxel_x_max,voxel_y_max,voxel_z_max])
                voxel_max = self.occlusion.transform[:3,:3].dot(voxel_max).T + self.occlusion.transform[:3,3]

                xmin = voxel_min[:,0].min()
                ymin = voxel_min[:,1].min()
                zmin = voxel_min[:,2].min()
                xmax = voxel_max[:,0].max()
                ymax = voxel_max[:,1].max()
                zmax = voxel_max[:,2].max()


            if (not (obj_id in self.objects)):
                # create new object
                new_object = ObjectModel(xmin, ymin, zmin, xmax, ymax, zmax, self.object_params['resol'], self.object_params['scale'])
                self.objects[obj_id] = new_object
                self.obj_initial_poses[obj_id] = new_object.transform
            
            # expand the model if inactive
            if not self.objects[obj_id].active:
                self.objects[obj_id].expand_model(xmin, ymin, zmin, xmax, ymax, zmax)
                self.obj_initial_poses[obj_id] = self.objects[obj_id].transform  # pose is updated

            # update TSDF of the object model
            self.objects[obj_id].update_tsdf(seg_depth_img, seg_color_img, camera_extrinsics, camera_intrinsics)

        # * Occlusion
        # get raw occlusion
        occluded = self.occlusion.scene_occlusion(depth_img, color_img, camera_extrinsics, camera_intrinsics)

        # generate point cloud for each of the object using conservative volumes
        obj_pcds = {}
        obj_poses = {}
        for obj_id, obj in self.objects.items():
            pcd = obj.sample_optimistic_pcd(n_sample=10)
            obj_poses[obj_id] = obj.transform
            obj_pcds[obj_id] = pcd
        # label the occlusion
        occlusion_label, occupied_label, occluded_dict = self.occlusion.label_scene_occlusion(occluded, camera_extrinsics, camera_intrinsics, obj_poses, obj_pcds)

        # record new depth image
        self.sensed_imgs.append(depth_img)
        self.sensed_poses.append(obj_poses)

        self.occluded_t = occluded
        self.occlusion_label_t = occlusion_label
        self.occupied_label_t = occupied_label
        self.occluded_dict_t = occluded_dict

        filtered_occluded, filtered_occluded_dict = \
            self.filtering(camera_extrinsics, camera_intrinsics)
        
        self.filtered_occluded = filtered_occluded
        # self.filtered_occlusion_label = filtered_occlusion_label
        # self.filtered_occupied_label = filtered_occupied_label
        self.filtered_occluded_dict = filtered_occluded_dict


    def filtering(self, camera_extrinsics, camera_intrinsics):
        """
        use the current reconstructed object models (obj_pcds) to filter the occlusion space of previous time
        and obtain the new occlusion
        """
        obj_poses = {}
        obj_pcds = {}
        for obj_id, obj in self.objects.items():
            obj_poses[obj_id] = self.obj_initial_poses[obj_id]
            pcd = self.objects[obj_id].sample_optimistic_pcd(n_sample=10)
            obj_pcds[obj_id] = pcd

        net_occluded = None
        for i in range(len(self.sensed_imgs)):
            depth_img = self.sensed_imgs[i]
            occluded = self.occlusion.scene_occlusion(depth_img, None, camera_extrinsics, camera_intrinsics)

            for obj_id, obj_pose in self.sensed_poses[i].items():
                obj_poses[obj_id] = obj_pose
            occlusion_label, occupied_label, occluded_dict = self.occlusion.label_scene_occlusion(occluded, camera_extrinsics, camera_intrinsics, obj_poses, obj_pcds)
            if net_occluded is None:
                net_occluded = (occlusion_label>0)
            else:
                net_occluded = net_occluded & (occlusion_label>0)
        
        # obtain occlusion for each of the object
        for obj_id, obj_occlusion in occluded_dict.items():
            occluded_dict[obj_id] = occluded_dict[obj_id] & net_occluded
        return net_occluded, occluded_dict

    def update_obj_pose(self, obj_id, pose):
        pass