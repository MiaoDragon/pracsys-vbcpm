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
        self.filtered_occluded = None
        self.filtered_occlusion_label = None

        # self.table_z = self.occlusion.z_base
        # self.filtered_occupied_label = filtered_occupied_label
        self.filtered_occluded_dict = None
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
            # print((seg_img==seg_id).astype(int).sum())
            seg_depth_img = np.array(depth_img)
            seg_depth_img[seg_img!=seg_id] = 0
            seg_color_img = np.array(color_img)
            seg_color_img[seg_img!=seg_id,:] = 0
            occluded = self.occlusion.scene_occlusion(seg_depth_img, seg_color_img, camera_extrinsics, camera_intrinsics)
            # cv2.imshow('segmented depth', seg_depth_img)
            # cv2.waitKey(0)

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
        obj_opt_pcds = {}
        obj_poses = {}
        for obj_id, obj in self.objects.items():
            pcd = obj.sample_conservative_pcd(n_sample=10)  # for checking occupied space, use conservative volume
            opt_pcd = obj.sample_optimistic_pcd(n_sample=10)
            obj_poses[obj_id] = obj.transform
            obj_pcds[obj_id] = pcd
            obj_opt_pcds[obj_id] = opt_pcd
        # label the occlusion
        occlusion_label, occupied_label, occluded_dict = self.occlusion.label_scene_occlusion(occluded, camera_extrinsics, camera_intrinsics, obj_poses, obj_pcds, obj_opt_pcds)



        # record new depth image
        self.sensed_imgs.append(depth_img)
        self.sensed_poses.append(obj_poses)

        self.occluded_t = occluded
        self.occlusion_label_t = occlusion_label
        self.occupied_label_t = occupied_label
        self.occluded_dict_t = occluded_dict

        if self.filtered_occluded is not None:
            voxel_env = visualize_voxel(self.occlusion.voxel_x,self.occlusion.voxel_y,self.occlusion.voxel_z,self.filtered_occluded,[0,0,0])
        color_pick = np.zeros((8,3))
        color_pick[0] = np.array([1., 0., 0.])
        color_pick[1] = np.array([0., 1.0, 0.])
        color_pick[2] = np.array([0., 0., 1.])
        color_pick[3] = np.array([252/255, 169/255, 3/255])
        color_pick[4] = np.array([252/255, 3/255, 252/255])
        color_pick[5] = np.array([20/255, 73/255, 82/255])
        color_pick[6] = np.array([22/255, 20/255, 82/255])
        color_pick[7] = np.array([60/255, 73/255, 10/255])


        opt_occupied_dict = self.occlusion.obtain_object_occupancy(camera_extrinsics, camera_intrinsics, obj_poses, obj_opt_pcds)

        # occupied_label
        voxel_occlusions = []
        voxel_occupied = []
        voxel_opts = []
        for obj_id, obj in self.objects.items():
            if opt_occupied_dict[obj_id].sum() != 0:
                voxel1 = visualize_voxel(self.occlusion.voxel_x, 
                                        self.occlusion.voxel_y,
                                        self.occlusion.voxel_z,
                                        opt_occupied_dict[obj_id], color_pick[obj_id])
                voxel_opts.append(voxel1)
            if (occluded_dict[obj_id]).sum() != 0:
                voxel1 = visualize_voxel(self.occlusion.voxel_x, 
                                        self.occlusion.voxel_y,
                                        self.occlusion.voxel_z,
                                        occluded_dict[obj_id], color_pick[obj_id])
                voxel_occlusions.append(voxel1)
            if (occupied_label==obj_id+1).sum() != 0:
                voxel1 = visualize_voxel(self.occlusion.voxel_x, 
                                        self.occlusion.voxel_y,
                                        self.occlusion.voxel_z,
                                        occupied_label==obj_id+1, color_pick[obj_id])
                voxel_occupied.append(voxel1)
        voxel_occluded = visualize_voxel(self.occlusion.voxel_x, 
                                        self.occlusion.voxel_y,
                                        self.occlusion.voxel_z,
                                        occluded, color_pick[obj_id])

        # print('occlusion:')
        # o3d.visualization.draw_geometries([voxel_occluded])      

        # print('optimistic model:')
        # o3d.visualization.draw_geometries(voxel_opts)      
        # print('label occlusions:')
        # o3d.visualization.draw_geometries(voxel_occlusions)      
        # print('label occupied:')
        # o3d.visualization.draw_geometries(voxel_occupied)      
        # print('occlusion with label occupied:')
        # o3d.visualization.draw_geometries([voxel_occluded] + voxel_occupied)      


        pcd = self.objects[obj_id].sample_optimistic_pcd(n_sample=10)



        filtered_occluded, filtered_occlusion_label, filtered_occluded_dict = \
            self.filtering(camera_extrinsics, camera_intrinsics)
        
        self.filtered_occluded = filtered_occluded
        self.filtered_occlusion_label = filtered_occlusion_label
        # self.filtered_occupied_label = filtered_occupied_label
        self.filtered_occluded_dict = filtered_occluded_dict


    def filtering(self, camera_extrinsics, camera_intrinsics):
        """
        use the current reconstructed object models (obj_pcds) to filter the occlusion space of previous time
        and obtain the new occlusion
        """
        obj_poses = {}
        obj_opt_pcds = {}
        obj_conserv_pcds = {}
        for obj_id, obj in self.objects.items():
            obj_poses[obj_id] = self.obj_initial_poses[obj_id]
            pcd = self.objects[obj_id].sample_optimistic_pcd(n_sample=10)
            obj_opt_pcds[obj_id] = pcd
            obj_conserv_pcds[obj_id] = self.objects[obj_id].sample_conservative_pcd(n_sample=10)

        net_occluded = None
        for i in range(len(self.sensed_imgs)):
            depth_img = self.sensed_imgs[i]
            occluded = self.occlusion.scene_occlusion(depth_img, None, camera_extrinsics, camera_intrinsics)

            for obj_id, obj_pose in self.sensed_poses[i].items():
                obj_poses[obj_id] = obj_pose
            occlusion_label, occupied_label, occluded_dict = self.occlusion.label_scene_occlusion(occluded, camera_extrinsics, camera_intrinsics, obj_poses, obj_conserv_pcds, obj_opt_pcds)

            # we can use occupied space to refine the reconstruction
            # if the object pcd lies in the optimistic occupied space, then set it to be free
            # TODO: debug this part


            color_pick = np.zeros((8,3))
            color_pick[0] = np.array([1., 0., 0.])
            color_pick[1] = np.array([0., 1.0, 0.])
            color_pick[2] = np.array([0., 0., 1.])
            color_pick[3] = np.array([252/255, 169/255, 3/255])
            color_pick[4] = np.array([252/255, 3/255, 252/255])
            color_pick[5] = np.array([20/255, 73/255, 82/255])
            color_pick[6] = np.array([22/255, 20/255, 82/255])
            color_pick[7] = np.array([60/255, 73/255, 10/255])
            voxels = []
            opt_voxels = []
            occupied_dict = self.occlusion.obtain_object_occupancy(camera_extrinsics, camera_intrinsics, obj_poses, obj_conserv_pcds)
            opt_occupied_dict = self.occlusion.obtain_object_occupancy(camera_extrinsics, camera_intrinsics, obj_poses, obj_opt_pcds)
            for obj_id, obj in self.objects.items():
                if opt_occupied_dict[obj_id].sum() != 0:
                    # voxel_i = visualize_voxel(self.occlusion.voxel_x,self.occlusion.voxel_y,self.occlusion.voxel_z,occupied_dict[obj_id],color_pick[obj_id])
                    # voxels.append(voxel_i)
                    voxel_i = visualize_voxel(self.occlusion.voxel_x,self.occlusion.voxel_y,self.occlusion.voxel_z,opt_occupied_dict[obj_id],color_pick[obj_id])
                    voxels.append(voxel_i)
                if (occupied_label==obj_id+1).sum() != 0:
                    voxel_i = visualize_voxel(self.occlusion.voxel_x,self.occlusion.voxel_y,self.occlusion.voxel_z,occupied_label==obj_id+1,[0,0,0])
                    voxels.append(voxel_i)

            # o3d.visualization.draw_geometries(voxels)

            opt_occupied_label = np.zeros(occupied_label.shape).astype(int)
            for obj_id, opt_occupied in opt_occupied_dict.items():
                opt_occupied_label[opt_occupied] = obj_id+1

            after_occupied_pcds = {}

            for obj_id, obj in self.objects.items():
                pcd = np.array([obj.voxel_x, obj.voxel_y, obj.voxel_z]).transpose([1,2,3,0])
                pcd = np.array([pcd,pcd,pcd,pcd,pcd,pcd,pcd,pcd,pcd,pcd]).transpose(1,2,3,0,4)
                ori_pcd = np.array(pcd).astype(int).reshape(-1,3)
                rand_pcd = np.random.uniform(low=0.0,high=1.0,size=(1,1,1,10,3))
                pcd = pcd + rand_pcd
                # pcd = pcd + 0.5
                pcd = pcd * obj.resol
                pcd = pcd.reshape(-1,3)
                pcd = obj_poses[obj_id][:3,:3].dot(pcd.T).T + obj_poses[obj_id][:3,3]
                pcd = self.occlusion.world_in_voxel_rot.dot(pcd.T).T + self.occlusion.world_in_voxel_tran
                pcd = pcd / self.occlusion.resol
                pcd = np.floor(pcd).astype(int)
                valid_mask = (pcd[:,0] >= 0) & (pcd[:,0] < self.occlusion.voxel_x.shape[0]) & \
                             (pcd[:,1] >= 0) & (pcd[:,1] < self.occlusion.voxel_x.shape[1]) & \
                             (pcd[:,2] >= 0) & (pcd[:,2] < self.occlusion.voxel_x.shape[2])
                occupied_mask = (opt_occupied_label[pcd[valid_mask][:,0],pcd[valid_mask][:,1],pcd[valid_mask][:,2]]>0) & \
                                ((opt_occupied_label[pcd[valid_mask][:,0],pcd[valid_mask][:,1],pcd[valid_mask][:,2]]!=obj_id+1)).reshape(-1)
                
                prev_conserv_model = obj.get_conservative_model()                
                prev_optimistic_model = obj.get_optimistic_model()

                obj.tsdf[ori_pcd[valid_mask][occupied_mask][:,0],ori_pcd[valid_mask][occupied_mask][:,1],ori_pcd[valid_mask][occupied_mask][:,2]] = obj.max_v * 1.1
                obj.tsdf_count[ori_pcd[valid_mask][occupied_mask][:,0],ori_pcd[valid_mask][occupied_mask][:,1],ori_pcd[valid_mask][occupied_mask][:,2]] += 10

                after_conserv_model = obj.get_conservative_model()
                after_optimistic_model = obj.get_optimistic_model()

                after_occupied_pcds[obj_id] = obj.sample_conservative_pcd(n_sample=10)

                if ori_pcd[valid_mask][occupied_mask].shape[0] > 0:
                    print("tsdf update using object occupancy info")
                    print(ori_pcd[valid_mask][occupied_mask].shape[0])
                    print('object id: ', obj_id)
                    voxel1 = visualize_voxel(obj.voxel_x, obj.voxel_y, obj.voxel_z, prev_conserv_model, [0,0,0])
                    voxel2 = visualize_voxel(obj.voxel_x, obj.voxel_y, obj.voxel_z, after_conserv_model, [1,0,0])
                    # o3d.visualization.draw_geometries([voxel1, voxel2])           
 

            after_occupied_dict = self.occlusion.obtain_object_occupancy(camera_extrinsics, camera_intrinsics, obj_poses, after_occupied_pcds)
            voxels = []
            for obj_id, obj in self.objects.items():
                print('obj_id: ', obj_id)
                if obj_poses[obj_id][0,3] < -5:
                    continue  # out of boundary
                if occupied_dict[obj_id].sum() != 0:
                    transformed_pcd = after_occupied_pcds[obj_id]
                    transformed_pcd = obj_poses[obj_id][:3,:3].dot(transformed_pcd.T).T + obj_poses[obj_id][:3,3]
                    transformed_pcd = self.occlusion.world_in_voxel_rot.dot(transformed_pcd.T).T + self.occlusion.world_in_voxel_tran
                    transformed_pcd = transformed_pcd / self.occlusion.resol
                    voxel_i = visualize_pcd(transformed_pcd, [0,0,0])
                    # voxel_i = visualize_pcd(self.occlusion.voxel_x,self.occlusion.voxel_y,self.occlusion.voxel_z,after_occupied_pcds[obj_id],[0,0,0])
                    # voxels.append(voxel_i)
                    voxels.append(voxel_i)

                    transformed_pcd = obj_opt_pcds[obj_id]
                    transformed_pcd = obj_poses[obj_id][:3,:3].dot(transformed_pcd.T).T + obj_poses[obj_id][:3,3]
                    transformed_pcd = self.occlusion.world_in_voxel_rot.dot(transformed_pcd.T).T + self.occlusion.world_in_voxel_tran
                    transformed_pcd = transformed_pcd / self.occlusion.resol
                    voxel_i = visualize_pcd(transformed_pcd, [1,1,0])
                    # voxel_i = visualize_pcd(self.occlusion.voxel_x,self.occlusion.voxel_y,self.occlusion.voxel_z,after_occupied_pcds[obj_id],[0,0,0])
                    # voxels.append(voxel_i)
                    voxels.append(voxel_i)


                    if (occupied_dict[obj_id]&(~after_occupied_dict[obj_id])).sum() != 0:
                        print("non-empty")
                        voxel_i = visualize_voxel(self.occlusion.voxel_x,self.occlusion.voxel_y,self.occlusion.voxel_z,occupied_dict[obj_id]&(~after_occupied_dict[obj_id]),color_pick[obj_id])
                        voxels.append(voxel_i)
            o3d.visualization.draw_geometries(voxels)           



            if net_occluded is None:
                net_occluded = (occlusion_label>0)
            else:
                net_occluded = net_occluded & (occlusion_label>0)
        
        # obtain occlusion for each of the object
        new_occlusion_label = np.zeros(occlusion_label.shape).astype(int)
        for obj_id, obj_occlusion in occluded_dict.items():
            occluded_dict[obj_id] = occluded_dict[obj_id] & net_occluded
            new_occlusion_label[(occlusion_label==obj_id+1) & net_occluded] = obj_id+1
        return net_occluded, new_occlusion_label, occluded_dict

    def update_obj_pose(self, obj_id, pose):
        pass

    def update_obj_model(self, obj_id, depth_img, color_img, seg_img, reverse_assoc, camera_extrinsics, camera_intrinsics, camera_far):
        seg_id = reverse_assoc[obj_id]
        seg_depth_img = np.array(depth_img)
        seg_depth_img[seg_img!=seg_id] = camera_far  #0  # UPDATE: other regions should correspond to max depth value since it's free space

        seg_color_img = np.array(color_img)
        seg_color_img[seg_img!=seg_id,:] = 0
        # cv2.imshow('segmented depth', seg_depth_img)
        # cv2.waitKey(0)

        self.objects[obj_id].update_tsdf_unhidden(seg_depth_img, seg_color_img, camera_extrinsics, camera_intrinsics)
