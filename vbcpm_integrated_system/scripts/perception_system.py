"""
integrated in the task planner for faster response
- occlusion
- objects
"""
from occlusion import Occlusion
from object import ObjectModel
from data_association import GroundTruthDataAssociation
from segmentation import GroundTruthSegmentation

import numpy as np
import gc

from visual_utilities import *

import cv2

LOG = 0

class PerceptionSystem():
    def __init__(self, occlusion_params, object_params, target_params, tsdf_color_flag=False):
        occlusion = Occlusion(**occlusion_params)
        self.object_params = object_params  # resol and scale
        self.target_params = target_params  # for recognizing target object
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

        self.data_assoc = GroundTruthDataAssociation()
        self.segmentation = GroundTruthSegmentation()
        self.tsdf_color_flag = tsdf_color_flag
        
    def perceive(self, depth_img, color_img, seg_img, sensed_obj_ids, obj_hide_sets, camera_extrinsics, camera_intrinsics, camera_far, robot_ids, workspace_ids,
                 visualize=False):
        """
        given depth img and color img from camera, and segmented and labeld images from 
        Segmentation and Data Association, update occlusion and object model
        
        Object model: 
        for sensed ones, use the segmented depth image to update model
        for new ones, create a new entry in the object list

        Occlusion:
        use the new reconstructed object model, get new occlusion of the scene
        """
        depth_img = np.array(depth_img)

        for cid in workspace_ids:
            depth_img[seg_img==cid] = 0  # workspace
        depth_img[seg_img==-1] = 0  # background

        # generate valid objects: objects that are not hidden by others
        valid_objects = []
        for obj_id, obj_hide_set in obj_hide_sets.items():
            if len(obj_hide_set) == 0:
                valid_objects.append(obj_id)


        for obj_id in sensed_obj_ids:
            # UPDATE: the background needs to be set with depth value FAR
            seg_depth_img = np.array(depth_img)
            # seg_depth_img[seg_img!=seg_id] = 0  # including other objects and robot
            # # handle background, workspace
            seg_depth_img[seg_depth_img==0] = camera_far  # TODO: might use the actual depth?
            for cid in workspace_ids:
                seg_depth_img[seg_img==cid] = camera_far  # workspace
            seg_depth_img[seg_img==-1] = camera_far  # background
            # # handle robot
            # # handle other object (objects that are hiding this one will be 0, and the reset will be camera_far)
            # seg_depth_img[seg_img!=seg_id] = camera_far
            for rid in robot_ids:
                seg_depth_img[seg_img==rid] = 0

            for obj_id_ in sensed_obj_ids:
                if obj_id_ == obj_id:
                    continue
                seg_depth_img[seg_img==obj_id_] = camera_far            

            # NOTE: problem. When the object that is being grasped is hiding the object behind,
            # this will cause the object to be seen-through
            # Added condition check to see whether the object is hidden by others
            obj_hide_set = obj_hide_sets[obj_id]
            for i in range(len(obj_hide_set)):
                seg_depth_img[seg_img==obj_hide_set[i]] = 0

            seg_color_img = np.array(color_img)
            seg_color_img[seg_img!=obj_id,:] = 0


            occluded = self.occlusion.scene_occlusion(seg_depth_img, seg_color_img, camera_extrinsics, camera_intrinsics)


            if (not (obj_id in self.objects)) or (not self.objects[obj_id].active):
                # NOTE: the object can be valid (fully seen currently) and we still compute the new bounding box. This is because
                # the object might be the first time to be fully seen, and we still need to update its bounding box to consider previously
                # hidden parts. Only when an object becomes active we will not update it anymore
                # compute bounding box for the conservative region
                voxel_x_min = self.occlusion.voxel_x[occluded].reshape(-1)-np.ceil(self.object_params['scale']/self.object_params['resol'][0])-1
                voxel_y_min = self.occlusion.voxel_y[occluded].reshape(-1)-np.ceil(self.object_params['scale']/self.object_params['resol'][1])-1
                voxel_z_min = np.zeros(len(voxel_x_min)) + self.occlusion.voxel_z.min()

                if len(voxel_z_min) == 0:
                    continue

                voxel_z_min[:] = voxel_z_min.min()  # we want the min to be the lowest in the workspace

                voxel_x_min = voxel_x_min * self.occlusion.resol[0]
                voxel_y_min = voxel_y_min * self.occlusion.resol[1]
                voxel_z_min = voxel_z_min * self.occlusion.resol[2]
                voxel_min = np.array([voxel_x_min,voxel_y_min,voxel_z_min])
                voxel_min = self.occlusion.transform[:3,:3].dot(voxel_min).T + self.occlusion.transform[:3,3]
                

                voxel_x_max = self.occlusion.voxel_x[occluded].reshape(-1)+np.ceil(self.object_params['scale']/self.object_params['resol'][0])+1
                voxel_y_max = self.occlusion.voxel_y[occluded].reshape(-1)+np.ceil(self.object_params['scale']/self.object_params['resol'][1])+1
                voxel_z_max = self.occlusion.voxel_z[occluded].reshape(-1)+np.ceil(self.object_params['scale']/self.object_params['resol'][2])+1
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

                del voxel_x_min
                del voxel_y_min
                del voxel_z_min
                del voxel_x_max
                del voxel_y_max
                del voxel_z_max
                del voxel_min
                del voxel_max


            if (not (obj_id in self.objects)):
                # create new object
                new_object = ObjectModel(obj_id, self.data_assoc.obj_ids_reverse[obj_id], xmin, ymin, zmin, xmax, ymax, zmax, self.object_params['resol'], self.object_params['scale'], use_color=self.tsdf_color_flag)
                self.objects[obj_id] = new_object
                self.obj_initial_poses[obj_id] = new_object.transform
                            
            # expand the model if inactive
            if not self.objects[obj_id].active:
                self.objects[obj_id].expand_model(xmin, ymin, zmin, xmax, ymax, zmax)
                self.obj_initial_poses[obj_id] = self.objects[obj_id].transform  # pose is updated
                self.objects[obj_id].update_depth_belief(depth_img, seg_img, workspace_ids)
                # this update the depth_img and seg_img for the object, and compute the boundary for the obj

            if obj_id in valid_objects:
                # NOTE: when it's the first time the object becomes fully revealed, we need to
                # expand the model again to take into account space that was hidden
                self.objects[obj_id].set_active()


            # self.objects[obj_id].update_tsdf(seg_depth_img, seg_color_img, camera_extrinsics, camera_intrinsics)
            # show updated conservative volume
            if visualize:
                ovoxel = visualize_voxel(self.occlusion.voxel_x, self.occlusion.voxel_y, self.occlusion.voxel_z,
                                         occluded, [1,0,0])
                obox = visualize_bbox(self.occlusion.voxel_x, self.occlusion.voxel_y, self.occlusion.voxel_z)
                opcd = self.objects[obj_id].sample_conservative_pcd()
                opcd = self.objects[obj_id].transform[:3,:3].dot(opcd.T).T + self.objects[obj_id].transform[:3,3]
                opcd = self.occlusion.world_in_voxel_rot[:3,:3].dot(opcd.T).T + self.occlusion.world_in_voxel_tran
                opcd = opcd / self.occlusion.resol
                opcd = visualize_pcd(opcd, [0,0,1])
                frame = visualize_coordinate_frame_centered()
                o3d.visualization.draw_geometries([ovoxel, obox, opcd, frame])

                self.objects[obj_id].update_tsdf(seg_depth_img, seg_color_img, camera_extrinsics, camera_intrinsics, True)
                vvoxel = visualize_voxel(self.objects[obj_id].voxel_x, self.objects[obj_id].voxel_y, self.objects[obj_id].voxel_z,
                                self.objects[obj_id].get_optimistic_model(), [0,0,1])
                o3d.visualization.draw_geometries([vvoxel])
            else:
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
        occlusion_label, occupied_label, occluded_dict, occupied_dict = self.occlusion.label_scene_occlusion(occluded, camera_extrinsics, camera_intrinsics, obj_poses, obj_pcds, obj_opt_pcds)

        if len(self.sensed_imgs) == 0:
            self.sensed_imgs.append(depth_img)
            self.sensed_poses.append(obj_poses)
        else:
            self.sensed_imgs[0] = depth_img
            self.sensed_poses[0] = obj_poses

        self.occluded_t = occluded
        self.occlusion_label_t = occlusion_label
        self.occupied_label_t = occupied_label
        self.occupied_dict_t = occupied_dict
        self.occluded_dict_t = occluded_dict


        if LOG:
            color_pick = np.zeros((8,3))
            color_pick[0] = np.array([1., 0., 0.])
            color_pick[1] = np.array([0., 1.0, 0.])
            color_pick[2] = np.array([0., 0., 1.])
            color_pick[3] = np.array([252/255, 169/255, 3/255])
            color_pick[4] = np.array([252/255, 3/255, 252/255])
            color_pick[5] = np.array([20/255, 73/255, 82/255])
            color_pick[6] = np.array([22/255, 20/255, 82/255])
            color_pick[7] = np.array([60/255, 73/255, 10/255])


        filtered_occluded, filtered_occlusion_label, filtered_occluded_dict = \
            self.filtering(camera_extrinsics, camera_intrinsics)
        
        self.filtered_occluded = filtered_occluded
        self.filtered_occlusion_label = filtered_occlusion_label
        # self.filtered_occupied_label = filtered_occupied_label
        self.filtered_occluded_dict = filtered_occluded_dict
        

        del occluded
        del occlusion_label
        del occupied_label
        del occupied_dict
        del occluded_dict
        del filtered_occluded
        del filtered_occlusion_label
        del filtered_occluded_dict
        del obj_pcds
        del obj_opt_pcds

        if len(sensed_obj_ids) > 0:
            del seg_depth_img
        



    def filtering(self, camera_extrinsics, camera_intrinsics):
        """
        since we remove each object and sense at each time, recording the list of past sensed depth images
        is not necessary. We just need to keep track of the intersection of occlusion to represent it

        A complete approach is to keep track of the list of past sensed depth images and get occlusion for each
        of them and then obtain the intersection
        """
        obj_poses = {}
        obj_opt_pcds = {}
        obj_conserv_pcds = {}
        for obj_id, obj in self.objects.items():
            obj_poses[obj_id] = self.obj_initial_poses[obj_id]
            pcd = self.objects[obj_id].sample_optimistic_pcd(n_sample=10)
            obj_opt_pcds[obj_id] = pcd
            obj_conserv_pcds[obj_id] = self.objects[obj_id].sample_conservative_pcd(n_sample=10)

        net_occluded = self.filtered_occluded
        
        for i in range(len(self.sensed_imgs)):
            depth_img = self.sensed_imgs[i]
            occluded = self.occlusion.scene_occlusion(depth_img, None, camera_extrinsics, camera_intrinsics)
    
            for obj_id, obj_pose in self.sensed_poses[i].items():
                obj_poses[obj_id] = obj_pose
            occlusion_label, occupied_label, occluded_dict, _ = self.occlusion.label_scene_occlusion(occluded, camera_extrinsics, camera_intrinsics, obj_poses, obj_conserv_pcds, obj_opt_pcds)
        

            opt_occupied_dict = self.occlusion.obtain_object_occupancy(camera_extrinsics, camera_intrinsics, obj_poses, obj_opt_pcds)

            opt_occupied_label = np.zeros(occupied_label.shape).astype(int)
            for obj_id, opt_occupied in opt_occupied_dict.items():
                opt_occupied_label[opt_occupied] = obj_id+1


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
                

                obj.tsdf[ori_pcd[valid_mask][occupied_mask][:,0],ori_pcd[valid_mask][occupied_mask][:,1],ori_pcd[valid_mask][occupied_mask][:,2]] = obj.max_v * 1.1
                obj.tsdf_count[ori_pcd[valid_mask][occupied_mask][:,0],ori_pcd[valid_mask][occupied_mask][:,1],ori_pcd[valid_mask][occupied_mask][:,2]] += 10
            
            if net_occluded is None:
                net_occluded = occlusion_label != 0 #occlusion_label > 0
            else:
                net_occluded = net_occluded & (occlusion_label!=0)#(occlusion_label>0)
        
        # obtain occlusion for each of the object
        new_occlusion_label = np.zeros(occlusion_label.shape).astype(int)
        for obj_id, obj_occlusion in occluded_dict.items():
            occluded_dict[obj_id] = occluded_dict[obj_id] & net_occluded
            new_occlusion_label[(occlusion_label==obj_id+1) & net_occluded] = obj_id+1

        del obj_conserv_pcds
        del obj_opt_pcds
        del obj_poses
        del pcd
        del occluded
        del ori_pcd
        del occupied_mask
        del occlusion_label
        gc.collect()

        return net_occluded, new_occlusion_label, occluded_dict


    def update_obj_model(self, obj_id, depth_img, color_img, seg_img, sensed_obj_ids, 
                        camera_extrinsics, camera_intrinsics, camera_far, robot_ids, workspace_ids):
        seg_depth_img = np.array(depth_img)
        seg_depth_img[seg_img!=obj_id] = camera_far  #0  # UPDATE: other regions should correspond to max depth value since it's free space
        for rid in robot_ids:
            seg_depth_img[seg_img==rid] = 0  # robot as potential hiding object

        seg_color_img = np.array(color_img)
        seg_color_img[seg_img!=obj_id,:] = 0

        self.objects[obj_id].update_tsdf(seg_depth_img, seg_color_img, camera_extrinsics, camera_intrinsics)




    def pipeline_sim(self, color_img, depth_img, seg_img, camera, robot_ids, workspace_ids):
        """
        given the camera input, segment the image, and data association
        """    
        # color_img, depth_img, seg_img = camera.sense()

        # visualzie segmentation image

        self.segmentation.set_ground_truth_seg_img(seg_img)
        seg_img = self.segmentation.segment_img(color_img, depth_img)

        # self.target_recognition.set_ground_truth_seg_img(seg_img)
        # target_seg_img = self.target_recognition.recognize(color_img, depth_img)

        self.depth_img = depth_img
        self.color_img = color_img
        # self.target_seg_img = target_seg_img

        assoc, seg_img, sensed_obj_ids = self.data_assoc.data_association(seg_img, robot_ids, workspace_ids)
        # sensed_obj_ids: currently seen objects in the scene
        """
        in reality the association can change in time, but the object
        label shouldn't change. So we should only remember the result
        after applying data association
        """
        self.last_assoc = assoc
        self.seg_img = seg_img

        # objects that have been revealed will stay revealed

        valid_objects = self.obtain_unhidden_objects(robot_ids, workspace_ids)
        
        object_hide_set = self.obtain_object_hide_set(robot_ids, workspace_ids)

        self.current_hide_set = object_hide_set
        self.perceive(depth_img, color_img, seg_img, 
                    sensed_obj_ids, object_hide_set, 
                    camera.info['extrinsics'], camera.info['intrinsics'], camera.info['far'], 
                    robot_ids, workspace_ids)

        # update each object's hide set
        for obj_i, obj_hide_list in object_hide_set.items():
            self.objects[obj_i].update_obj_hide_set(obj_hide_list)


        for obj_id in valid_objects:
            self.objects[obj_id].set_active()

    def sense_object(self, obj_id, color_img, depth_img, seg_img, camera, robot_ids, workspace_ids):
        self.segmentation.set_ground_truth_seg_img(seg_img)
        seg_img = self.segmentation.segment_img(color_img, depth_img)
        assoc, seg_img, sensed_obj_ids = self.data_assoc.data_association(seg_img, robot_ids, workspace_ids)
        
        self.update_obj_model(obj_id, depth_img, color_img, seg_img, sensed_obj_ids, 
                                          camera.info['extrinsics'], camera.info['intrinsics'], camera.info['far'], robot_ids, workspace_ids)
    

    def obtain_object_hide_set(self, robot_ids, workspace_ids):
        depth_img = self.depth_img
        seg_img = self.seg_img
        assoc = self.last_assoc
        # determine hiding relation: the target object shouldn't be hidden and inactive
        # hidden: at least one depth value is larger than a neighboring object depth value

        # determine where there are objects in the segmented img

        # UPDATE: we want to consider robot hiding as well
        obj_seg_filter = np.ones(seg_img.shape).astype(bool)
        for wid in workspace_ids:
            obj_seg_filter[seg_img==wid] = 0
        obj_seg_filter[seg_img==-1] = 0
        for rid in robot_ids:
            obj_seg_filter[seg_img==rid] = 0


        valid_objects = []  # only return unhidden objects.

        seen_objs = []
        hiding_objs = {}  # obj_id -> objects that are hiding it
        for _, obj_id in assoc.items():
            seen_objs.append(obj_id)
            hiding_set = set()

            seged_depth_img = np.zeros(depth_img.shape)
            seged_depth_img[seg_img==obj_id] = depth_img[seg_img==obj_id]
            # cv2.imshow("seen_obj", seged_depth_img)
            # cv2.waitKey(0)
            # obtain indices of the segmented object
            img_i, img_j = np.indices(seg_img.shape)
            # check if the neighbor object is hiding this object
            valid_1 = (img_i-1>=0) & (seg_img==obj_id)
            # the neighbor object should be 
            # 1. an object (can be robot)
            # 2. not the current object
            filter1 = obj_seg_filter[img_i[valid_1]-1,img_j[valid_1]]
            filter2 = (seg_img[img_i[valid_1]-1,img_j[valid_1]] != obj_id)


            depth_filtered = depth_img[img_i[valid_1]-1,img_j[valid_1]][filter1&filter2]
            seg_obj_depth_filtered = depth_img[img_i[valid_1],img_j[valid_1]][filter1&filter2]
            seg_obj_filtered = seg_img[img_i[valid_1]-1,img_j[valid_1]][filter1&filter2]

            hiding_seg_obj_filtered = seg_obj_filtered[depth_filtered<seg_obj_depth_filtered]
            hiding_set = hiding_set.union(set(hiding_seg_obj_filtered.tolist()))


            valid_1 = (img_i+1<seg_img.shape[0]) & (seg_img==obj_id)
            filter1 = obj_seg_filter[img_i[valid_1]+1,img_j[valid_1]]
            filter2 = (seg_img[img_i[valid_1]+1,img_j[valid_1]] != obj_id)
            depth_filtered = depth_img[img_i[valid_1]+1,img_j[valid_1]][filter1&filter2]
            seg_obj_depth_filtered = depth_img[img_i[valid_1],img_j[valid_1]][filter1&filter2]
            seg_obj_filtered = seg_img[img_i[valid_1]+1,img_j[valid_1]][filter1&filter2]

            hiding_seg_obj_filtered = seg_obj_filtered[depth_filtered<seg_obj_depth_filtered]
            hiding_set = hiding_set.union(set(hiding_seg_obj_filtered.tolist()))




            valid_1 = (img_j-1>=0) & (seg_img==obj_id)
            filter1 = obj_seg_filter[img_i[valid_1],img_j[valid_1]-1]
            filter2 = (seg_img[img_i[valid_1],img_j[valid_1]-1] != obj_id)
            depth_filtered = depth_img[img_i[valid_1],img_j[valid_1]-1][filter1&filter2]
            seg_obj_depth_filtered = depth_img[img_i[valid_1],img_j[valid_1]][filter1&filter2]
            seg_obj_filtered = seg_img[img_i[valid_1],img_j[valid_1]-1][filter1&filter2]

            hiding_seg_obj_filtered = seg_obj_filtered[depth_filtered<seg_obj_depth_filtered]
            hiding_set = hiding_set.union(set(hiding_seg_obj_filtered.tolist()))




            valid_1 = (img_j+1<seg_img.shape[1]) & (seg_img==obj_id)
            filter1 = obj_seg_filter[img_i[valid_1],img_j[valid_1]+1]
            filter2 = (seg_img[img_i[valid_1],img_j[valid_1]+1] != obj_id)
            depth_filtered = depth_img[img_i[valid_1],img_j[valid_1]+1][filter1&filter2]
            seg_obj_depth_filtered = depth_img[img_i[valid_1],img_j[valid_1]][filter1&filter2]
            seg_obj_filtered = seg_img[img_i[valid_1],img_j[valid_1]+1][filter1&filter2]

            hiding_seg_obj_filtered = seg_obj_filtered[depth_filtered<seg_obj_depth_filtered]
            hiding_set = hiding_set.union(set(hiding_seg_obj_filtered.tolist()))


            # NOTE: hiding_set stores seg_ids, which are pybullet ids instead of obj_id
            # we need to convert them
            hiding_set = list(hiding_set)
            hiding_set = [oid for oid in hiding_set]
            hiding_objs[obj_id] = hiding_set

        return hiding_objs
            

    def obtain_unhidden_objects(self, robot_ids, workspace_ids):

        depth_img = self.depth_img
        seg_img = self.seg_img
        assoc = self.last_assoc
        # determine hiding relation: the target object shouldn't be hidden and inactive
        # hidden: at least one depth value is larger than a neighboring object depth value

        # determine where there are objects in the segmented img

        # UPDATE: we want to consider robot hiding as well
        obj_seg_filter = np.ones(seg_img.shape).astype(bool)
        for wid in workspace_ids:
            obj_seg_filter[seg_img==wid] = 0
        obj_seg_filter[seg_img==-1] = 0
        # for seg_id, obj_id in assoc.items():
        #     obj_seg_filter[seg_img==seg_id] = 1

        valid_objects = []  # only return unhidden objects.

        seen_objs = []
        for _, obj_id in assoc.items():
            seen_objs.append(obj_id)

            seged_depth_img = np.zeros(depth_img.shape)
            seged_depth_img[seg_img==obj_id] = depth_img[seg_img==obj_id]
            # cv2.imshow("seen_obj", seged_depth_img)
            # cv2.waitKey(0)
            # obtain indices of the segmented object
            img_i, img_j = np.indices(seg_img.shape)
            # check if the neighbor object is hiding this object
            valid_1 = (img_i-1>=0) & (seg_img==obj_id)
            # the neighbor object should be 
            # 1. an object (can be robot)
            # 2. not the current object
            filter1 = obj_seg_filter[img_i[valid_1]-1,img_j[valid_1]]
            filter2 = (seg_img[img_i[valid_1]-1,img_j[valid_1]] != obj_id)


            depth_filtered = depth_img[img_i[valid_1]-1,img_j[valid_1]][filter1&filter2]
            seg_obj_depth_filtered = depth_img[img_i[valid_1],img_j[valid_1]][filter1&filter2]
            if (depth_filtered < seg_obj_depth_filtered).sum() > 0:
                # this object is hidden
                continue


            valid_1 = (img_i+1<seg_img.shape[0]) & (seg_img==obj_id)
            filter1 = obj_seg_filter[img_i[valid_1]+1,img_j[valid_1]]
            filter2 = (seg_img[img_i[valid_1]+1,img_j[valid_1]] != obj_id)
            depth_filtered = depth_img[img_i[valid_1]+1,img_j[valid_1]][filter1&filter2]
            seg_obj_depth_filtered = depth_img[img_i[valid_1],img_j[valid_1]][filter1&filter2]
            if (depth_filtered < seg_obj_depth_filtered).sum() > 0:
                # this object is hidden
                continue
            valid_1 = (img_j-1>=0) & (seg_img==obj_id)
            filter1 = obj_seg_filter[img_i[valid_1],img_j[valid_1]-1]
            filter2 = (seg_img[img_i[valid_1],img_j[valid_1]-1] != obj_id)
            depth_filtered = depth_img[img_i[valid_1],img_j[valid_1]-1][filter1&filter2]
            seg_obj_depth_filtered = depth_img[img_i[valid_1],img_j[valid_1]][filter1&filter2]
            if (depth_filtered < seg_obj_depth_filtered).sum() > 0:
                # this object is hidden
                continue


            valid_1 = (img_j+1<seg_img.shape[1]) & (seg_img==obj_id)
            filter1 = obj_seg_filter[img_i[valid_1],img_j[valid_1]+1]
            filter2 = (seg_img[img_i[valid_1],img_j[valid_1]+1] != obj_id)
            depth_filtered = depth_img[img_i[valid_1],img_j[valid_1]+1][filter1&filter2]
            seg_obj_depth_filtered = depth_img[img_i[valid_1],img_j[valid_1]][filter1&filter2]
            if (depth_filtered < seg_obj_depth_filtered).sum() > 0:
                # this object is hidden
                continue
            valid_objects.append(obj_id)


        return valid_objects
    
    def label_obj_seg_img(self):
        """
        label the object segmented image so that 
        """
        pass

    def sample_pcd(self, mask, n_sample=10):
        # sample voxels in te mask
        # obtain sample in one voxel cell
        grid_sample = np.random.uniform(low=[0,0,0], high=[1,1,1], size=(n_sample, 3))
        voxel_x = self.voxel_x[mask]
        voxel_y = self.voxel_y[mask]
        voxel_z = self.voxel_z[mask]

        total_sample = np.zeros((len(voxel_x), n_sample, 3))
        total_sample = total_sample + grid_sample
        total_sample = total_sample + np.array([voxel_x, voxel_y, voxel_z]).T.reshape(len(voxel_x),1,3)

        total_sample = total_sample.reshape(-1, 3) * np.array(self.resol)

        del voxel_x
        del voxel_y
        del voxel_z

        return total_sample