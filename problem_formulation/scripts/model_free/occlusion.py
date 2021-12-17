"""
implement the occlusion state
This should achieve the following functionalities:
- obtain occlusion from depth image (O(t))
- label occlusion given current object shape (O_i(t))
  label object occupancy given O_i(t)
- filtering to obtain the net occlusion:
  should keep track of depth images from t=0 to now,
  and the pose of each object shape
  Then O(t') = O(t') - U_i X_i(t)
  (remove sensed parts of objects)

Here we use conservative estimate for X_i(t) since eventually it will shrink to the 
object model (including the interior)
TODO: if we use conservative estimate, then the occlusion region is much smaller, and we can't infer
where the target object might locate. Should we just denote those areas as Unknown?
    Maybe a better approach is to check when the reconstructed object surface becomes "closed"
    *Easy Check for closeness*: no intersection with boundaries

Occlusion is represented in the global voxel grid
"""
from enum import unique
import numpy as np
import cv2
import open3d as o3d
from visual_utilities import *

class Occlusion():
    def __init__(self, world_x, world_y, world_z, resol, x_base, y_base, z_base, x_vec, y_vec, z_vec):
        self.world_x = world_x
        self.world_y = world_y
        self.world_z = world_z
        self.x_base = x_base
        self.y_base = y_base
        self.z_base = z_base
        self.resol = resol
        self.occupied = np.zeros((int(self.world_x / self.resol[0]), \
                                  int(self.world_y / self.resol[1]), \
                                  int(self.world_z / self.resol[2]))).astype(bool)
        self.occlusion = np.zeros((int(self.world_x / self.resol[0]), \
                                  int(self.world_y / self.resol[1]), \
                                  int(self.world_z / self.resol[2]))).astype(bool)

        self.voxel_x, self.voxel_y, self.voxel_z = np.indices(self.occlusion.shape).astype(float)

        self.visible_grid = np.array(self.occlusion)

        # self.vis_objs = np.zeros(len(self.objects)).astype(bool)

        # self.move_times = 10. + np.zeros(len(self.objects))
        self.transform = np.zeros((4,4))
        self.transform[:3,0] = x_vec
        self.transform[:3,1] = y_vec
        self.transform[:3,2] = z_vec
        self.transform[:3,3] = np.array([self.x_base, self.y_base, self.z_base])
        self.transform[3,3] = 1.

        # self.transform = transform  # the transform of the voxel grid cooridnate system in the world as {world}T{voxel}
        self.world_in_voxel = np.linalg.inv(self.transform)
        self.world_in_voxel_rot = self.world_in_voxel[:3,:3]
        self.world_in_voxel_tran = self.world_in_voxel[:3,3]



    def scene_occlusion(self, depth_img, color_img, camera_extrinsics, camera_intrinsics):
        # generate the occlusion for the entire scene
        # occlusion includes: object occupied space, occlusion due to known object, occlusion due to 
        #                     unknown object
        voxel_vecs = np.array([self.voxel_x, self.voxel_y, self.voxel_z]).transpose((1,2,3,0))
        # voxel_vecs = np.concatenate([self.voxel_x, self.voxel_y, self.voxel_z], axis=3)
        voxel_vecs = voxel_vecs.reshape(-1,3) * self.resol
        transformed_voxels = self.transform[:3,:3].dot(voxel_vecs.T).T + self.transform[:3,3]
        # get to the image space
        cam_transform = np.linalg.inv(camera_extrinsics)
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
        valid_mask = valid_mask.reshape(self.voxel_x.shape)
        voxel_depth = voxel_depth.reshape(self.voxel_x.shape)

        cam_to_voxel_depth = cam_to_voxel_depth.reshape(self.voxel_x.shape)
        occluded = (cam_to_voxel_depth - voxel_depth >= 0.) & (voxel_depth > 0.) & valid_mask
        # print(occluded.astype(int).sum() / valid_mask.astype(int).sum())
        return occluded

    def update_occlusion(self, occlusion_label1, occupied_label1, occluded_list1, \
                               occlusion_label2, occupied_label2, occluded_list2):
        """
        given the previous occlusion (occluded1) and the new occlusion (occluded2), update the
        occlusion to be the intersection of the two. (excluding occupied space)
        Note: we can't do intersection on occupied space with occluded space. Hence intersection
        on raw occlusion space will cause trouble. We should do the intersection only for the 
        occlusion space. This is because we know for sure that the previous occupied space is free.

        occluded: occupied + occlusion (raw, unlabeled)

        For each object, the new occlusion space is occlusion2 intersected with occlusion1
        (which means the previous free space remains known)

        TODO: 
        currently we're ignoring unknown occlusion (occluded by unknown entities) due to the noise
        of occlusion computation. In the future we should have a better computation and consider that.
        """
        
        new_occlusion_label = np.zeros(occlusion_label2.shape).astype(int)
        # new_occlusion_label[(occlusion_label2<0)&(occlusion_label1!=0)] = -1
        new_occluded_list = []
        for i in range(len(occluded_list1)):
            # TODO here
            new_occlusion_label[(occlusion_label2==i+1)&(occlusion_label1>0)] = i+1
            # since we assume if an object becomes revealed, the pose will be known, if
            # the occluded_list2[i] is None, then it means the object hasn't been revealed
            # yet. Thus we just need to set the new value to be None
            if occluded_list2[i] is None:
                new_occluded = None
            else:
                new_occluded = occluded_list2[i] & (occlusion_label1>0)
            new_occluded_list.append(new_occluded)
        return new_occlusion_label, occupied_label2, new_occluded_list


    def single_object_occlusion(self, camera_extrinsics, camera_intrinsics, obj_pose, obj_pcd):
        occupied = np.zeros(self.voxel_x.shape).astype(bool)
        occluded = np.zeros(self.voxel_x.shape).astype(bool)
        R = obj_pose[:3,:3]
        T = obj_pose[:3,3]

        pcd = R.dot(obj_pcd.T).T + T

        # ** filter out the voxels that correspond to object occupied space
        # map the pcd to voxel space
        pcd_in_voxel = self.world_in_voxel_rot.dot(pcd.T).T + self.world_in_voxel_tran
        pcd_in_voxel = pcd_in_voxel / self.resol
        # the floor of each axis will give us the index in the voxel
        indices = np.floor(pcd_in_voxel).astype(int)
        # extract the ones that are within the limit
        indices = indices[indices[:,0]>=0]
        indices = indices[indices[:,0]<self.voxel_x.shape[0]]
        indices = indices[indices[:,1]>=0]
        indices = indices[indices[:,1]<self.voxel_x.shape[1]]
        indices = indices[indices[:,2]>=0]
        indices = indices[indices[:,2]<self.voxel_x.shape[2]]

        occupied[indices[:,0],indices[:,1],indices[:,2]] = 1

        # ** extract the occlusion by object id
        cam_transform = np.linalg.inv(camera_extrinsics)

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
        depth_img = np.zeros((max_i, max_j)).astype(float)
        depth_img[transformed_pcd[:,1],transformed_pcd[:,0]] = depth
        
        # depth_img = cv2.resize(depth_img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        # depth_img = cv2.resize(depth_img, ori_shape, interpolation=cv2.INTER_LINEAR)
        # depth_img = cv2.resize(depth_img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        # depth_img = cv2.medianBlur(np.float32(depth_img), 5)
        depth_img = cv2.boxFilter(np.float32(depth_img), -1, (5,5))

        occluded_i = self.scene_occlusion(depth_img, None, camera_extrinsics, camera_intrinsics)

        occluded = (~occupied) & occluded_i

        return occupied, occluded


    def label_scene_occlusion(self, occluded, camera_extrinsics, camera_intrinsics, obj_poses, obj_pcds, obj_opt_pcds, depth_nn=1):
        """
        depth_nn: maximum distance in the depth image to count for the object
        """
        # given the scene occlusion, label each of the occluded region based on the object known info
        # object -> render depth image -> mask
        # * intersection of mask and scene occlusion corresponds to the parts that belong to object
        # * UPDATE: we don't use the intersection anymore. We use the object point cloud to determine the occupied space

        # * object occupied space can be determined by point cloud voxelization

        # * the remaining space is unknown parts

        occluded = np.array(occluded).astype(bool)  # make a copy
        occlusion_label = np.zeros(occluded.shape).astype(int)  # 0: free space, id: occluded by id, -1: unknown
        occupied_label = np.zeros(occluded.shape).astype(int)  # id: occupied by id
        occluded_dict = {}
        depth_nn_steps = []
        # for i in range(depth_nn):
        #     depth_nn_steps.append([i,0,0])
        #     depth_nn_steps.append([-i,0,0])
        #     depth_nn_steps.append([0,i,0])
        #     depth_nn_steps.append([0,-i,0])
        #     depth_nn_steps.append([0,0,i])
        #     depth_nn_steps.append([0,0,-i])
        # depth_nn_steps = np.array(depth_nn_steps
        # )
        for obj_id, obj_pose in obj_poses.items():
            # * Use optimistic pcd to filter out occupied space from occlusion
            obj_pcd = obj_opt_pcds[obj_id]
            R = obj_pose[:3,:3]
            T = obj_pose[:3,3]

            pcd = R.dot(obj_pcd.T).T + T
            # ** filter out the voxels that correspond to object occupied space
            # map the pcd to voxel space
            transformed_pcds = self.world_in_voxel_rot.dot(pcd.T).T + self.world_in_voxel_tran
            transformed_pcds = transformed_pcds / self.resol
            # the floor of each axis will give us the index in the voxel
            indices = np.floor(transformed_pcds).astype(int)
            # extract the ones that are within the limit
            indices = indices[indices[:,0]>=0]
            indices = indices[indices[:,0]<self.voxel_x.shape[0]]
            indices = indices[indices[:,1]>=0]
            indices = indices[indices[:,1]<self.voxel_x.shape[1]]
            indices = indices[indices[:,2]>=0]
            indices = indices[indices[:,2]<self.voxel_x.shape[2]]
            
            occluded[indices[:,0],indices[:,1],indices[:,2]] = 0

            # * Use conservative pcd to determine occupied space
            obj_pcd = obj_pcds[obj_id]
            R = obj_pose[:3,:3]
            T = obj_pose[:3,3]
            pcd = R.dot(obj_pcd.T).T + T
            # ** filter out the voxels that correspond to object occupied space
            # map the pcd to voxel space
            transformed_pcds = self.world_in_voxel_rot.dot(pcd.T).T + self.world_in_voxel_tran
            transformed_pcds = transformed_pcds / self.resol
            # the floor of each axis will give us the index in the voxel
            indices = np.floor(transformed_pcds).astype(int)
            # extract the ones that are within the limit
            indices = indices[indices[:,0]>=0]
            indices = indices[indices[:,0]<self.voxel_x.shape[0]]
            indices = indices[indices[:,1]>=0]
            indices = indices[indices[:,1]<self.voxel_x.shape[1]]
            indices = indices[indices[:,2]>=0]
            indices = indices[indices[:,2]<self.voxel_x.shape[2]]
            
            occupied = np.zeros(occluded.shape).astype(bool)
            occupied[indices[:,0],indices[:,1],indices[:,2]] = 1
            # occupied = occluded & occupied  # occupied shouldn't concern occlusion
            occupied_label[occupied==1] = obj_id+1


        # Step 2: determine occlusion label: using pcd for depth image (TODO: try opt or conservative)
        for obj_id, obj_pose in obj_poses.items():
            obj_pcd = obj_opt_pcds[obj_id]
            R = obj_pose[:3,:3]
            T = obj_pose[:3,3]

            pcd = R.dot(obj_pcd.T).T + T

            # ** extract the occlusion by object id
            cam_transform = np.linalg.inv(camera_extrinsics)

            # NOTE: multiple pcds can map to the same depth. We need to use the min value of the depth if this happens
            if len(pcd) == 0:
                continue
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

            if max_i <= 0 or max_j <= 0:
                # not in the camera view
                continue

            unique_indices = np.unique(transformed_pcd, axis=0)
            unique_valid = (unique_indices[:,0] >= 0) & (unique_indices[:,1] >= 0)
            unique_indices = unique_indices[unique_valid]
            unique_depths = np.zeros(len(unique_indices))
            for i in range(len(unique_indices)):
                unique_depths[i] = depth[(transformed_pcd[:,0]==unique_indices[i,0])&(transformed_pcd[:,1]==unique_indices[i,1])].min()
            depth_img = np.zeros((max_i, max_j)).astype(float)
            depth_img[unique_indices[:,1],unique_indices[:,0]] = unique_depths
            # depth_img[transformed_pcd[:,1],transformed_pcd[:,0]] = depth
            
            ori_shape = depth_img.shape
            # depth_img = cv2.resize(depth_img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
            # depth_img = cv2.resize(depth_img, ori_shape, interpolation=cv2.INTER_LINEAR)
            # depth_img = cv2.resize(depth_img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
            depth_img = cv2.medianBlur(np.float32(depth_img), 5)
            # depth_img
            # depth_img = cv2.boxFilter(np.float32(depth_img), -1, (2,2))

            # cv2.imshow('depth', depth_img)
            # cv2.waitKey(0)
            occluded_i = self.scene_occlusion(depth_img, None, camera_extrinsics, camera_intrinsics)

            occluded_i = occluded_i & occluded
            occluded_dict[obj_id] = occluded_i
            occlusion_label[occluded_i==1] = obj_id+1

        # the rest of the space is unknown space
        occlusion_label[(occlusion_label==0)&(occluded==1)] = -1
        return occlusion_label, occupied_label, occluded_dict

    def obtain_object_occupancy(self, camera_extrinsics, camera_intrinsics, obj_poses, obj_pcds, depth_nn=1):
        occupied_dict = {}
        for obj_id, obj_pose in obj_poses.items():
            obj_pcd = obj_pcds[obj_id]
            R = obj_pose[:3,:3]
            T = obj_pose[:3,3]

            pcd = R.dot(obj_pcd.T).T + T
            # ** filter out the voxels that correspond to object occupied space
            # map the pcd to voxel space
            transformed_pcds = self.world_in_voxel_rot.dot(pcd.T).T + self.world_in_voxel_tran
            transformed_pcds = transformed_pcds / self.resol
            # the floor of each axis will give us the index in the voxel
            indices = np.floor(transformed_pcds).astype(int)
            # extract the ones that are within the limit
            indices = indices[indices[:,0]>=0]
            indices = indices[indices[:,0]<self.voxel_x.shape[0]]
            indices = indices[indices[:,1]>=0]
            indices = indices[indices[:,1]<self.voxel_x.shape[1]]
            indices = indices[indices[:,2]>=0]
            indices = indices[indices[:,2]<self.voxel_x.shape[2]]
            occupied = np.zeros(self.voxel_x.shape).astype(bool)
            occupied[indices[:,0],indices[:,1],indices[:,2]] = 1
            # occupied = occluded & occupied  # occupied shouldn't concern occlusion
            occupied_dict[obj_id] = occupied
        return occupied_dict

    def obtain_object_uncertainty(self, obj, obj_pose, camera_extrinsics, camera_intrinsics):
        """
        given the object info and camera info, obtain the uncertainty of the object
        """
        # map object voxel indices to camera pixel
        voxel_indices = np.array([obj.voxel_x, obj.voxel_y, obj.voxel_z]).transpose([1,2,3,0]).reshape(-1,3).astype(int)
        transformed_voxel_indices = (voxel_indices+0.5) * obj.resol
        transformed_voxel_indices = obj_pose[:3,:3].dot(transformed_voxel_indices.T).T + obj_pose[:3,3]
        cam_transform = np.linalg.inv(camera_extrinsics)
        transformed_voxel_indices = cam_transform[:3,:3].dot(transformed_voxel_indices.T).T + cam_transform[:3,3]
        fx = camera_intrinsics[0][0]
        fy = camera_intrinsics[1][1]
        cx = camera_intrinsics[0][2]
        cy = camera_intrinsics[1][2]

        # # visualize the transformed voxel
        # frame = visualize_coordinate_frame_centered()
        # conservative_filter = obj.get_conservative_model().reshape(-1)
        # pcds = visualize_pcd(transformed_voxel_indices[conservative_filter,:], [0,0,0])


        # o3d.visualization.draw_geometries([frame, pcds])           


        transformed_voxel_indices[:,0] = transformed_voxel_indices[:,0] / transformed_voxel_indices[:,2] * fx + cx
        transformed_voxel_indices[:,1] = transformed_voxel_indices[:,1] / transformed_voxel_indices[:,2] * fy + cy
        depth = transformed_voxel_indices[:,2]



        transformed_voxel_indices = np.floor(transformed_voxel_indices).astype(int)
        transformed_voxel_indices = transformed_voxel_indices[:,:2]
        # find unique values in the voxel indices
        unique_indices = np.unique(transformed_voxel_indices, axis=0)

        

        max_j = transformed_voxel_indices[:,0].max()+1
        max_i = transformed_voxel_indices[:,1].max()+1
        uncertain_img = np.zeros((max_i, max_j)).astype(float)

        # give the number of pixels that corespond to uncertainty
        sum_uncertainty = 0
        # print('number of unique indices: ')
        # print(len(unique_indices))

        # print('number of origin indices: ')
        # print(len(transformed_voxel_indices))

        for i in range(len(unique_indices)):
            if (unique_indices[i,0]<0 or unique_indices[i,1] < 0):
                continue

            mask = (transformed_voxel_indices[:,0] == unique_indices[i,0]) & (transformed_voxel_indices[:,1] == unique_indices[i,1])
            # find the first value (lowest depth)
            mask_min_is = depth[mask].argsort()
            masked_indices = voxel_indices[mask][mask_min_is,:]
            masked_tsdfs = obj.tsdf[masked_indices[:,0],masked_indices[:,1],masked_indices[:,2]]
            masked_tsdf_counts = obj.tsdf_count[masked_indices[:,0],masked_indices[:,1],masked_indices[:,2]]
            # if len(mask_min_is)>1:
            #     print('masked_tsdfs: ', masked_tsdfs)
            #     print('masked_tasf_counts: ', masked_tsdf_counts)
            threshold = 1
            # find the first tsdf that is min_v
            min_v_mask = ((masked_tsdfs <= obj.min_v) | (masked_tsdf_counts < threshold))
            min_v_id = min_v_mask.argmax()
            if min_v_mask.sum() == 0:
                # if len(mask_min_is)>1:
                #     print('not uncertain')
                continue
            if min_v_id == 0:
                # if len(mask_min_is)>1:
                #     print('uncertain since the first hit is uncertain')
                sum_uncertainty += 1
                uncertain_img[unique_indices[i,1],unique_indices[i,0]] = 1
                continue
            # if not the first one then check if all previous tsdf values >= max_v
            if (masked_tsdfs[:min_v_id]>=obj.max_v).astype(int).sum() == min_v_id:
                # if len(mask_min_is)>1:
                #     print('uncertain since transit from freespace to unseen')
                sum_uncertainty += 1
                uncertain_img[unique_indices[i,1],unique_indices[i,0]] = 1
                continue
    
        # cv2.imshow('depth', uncertain_img)
        # cv2.waitKey(0)        
        return sum_uncertainty

    def object_hidden(self, depth_img, seg_img, assoc, seg_id, obj_id, obj):
        """
        check if the object is hidden (not reconstructed yet) by others
        """
        # determine where there are objects in the segmented img
        obj_seg_filter = np.zeros(seg_img.shape).astype(bool)
        for seg_id_, _ in assoc.items():
            obj_seg_filter[seg_img==seg_id_] = 1


        seged_depth_img = np.zeros(depth_img.shape)
        seged_depth_img[seg_img==seg_id] = depth_img[seg_img==seg_id]
        # cv2.imshow("seen_obj", seged_depth_img)
        # cv2.waitKey(0)
        # obtain indices of the segmented object
        img_i, img_j = np.indices(seg_img.shape)
        # check if the neighbor object is hiding this object
        valid_1 = (img_i-1>=0) & (seg_img==seg_id)
        # the neighbor object should be 
        # 1. an object
        # 2. not the current object
        filter1 = obj_seg_filter[img_i[valid_1]-1,img_j[valid_1]]
        filter2 = (seg_img[img_i[valid_1]-1,img_j[valid_1]] != seg_id)

        filter1_img = np.zeros(depth_img.shape)
        filter1_img[img_i[valid_1]-1, img_j[valid_1]] = (filter1&filter2)
        # cv2.imshow('filter obj', filter1_img)
        # cv2.waitKey(0)


        depth_filtered = depth_img[img_i[valid_1]-1,img_j[valid_1]][filter1&filter2]
        seg_obj_depth_filtered = depth_img[img_i[valid_1],img_j[valid_1]][filter1&filter2]
        if (depth_filtered < seg_obj_depth_filtered).sum() > 0:
            # this object is hidden
            if not obj.active:
                return True

        valid_1 = (img_i+1<seg_img.shape[0]) & (seg_img==seg_id)
        filter1 = obj_seg_filter[img_i[valid_1]+1,img_j[valid_1]]
        filter2 = (seg_img[img_i[valid_1]+1,img_j[valid_1]] != seg_id)
        depth_filtered = depth_img[img_i[valid_1]+1,img_j[valid_1]][filter1&filter2]
        seg_obj_depth_filtered = depth_img[img_i[valid_1],img_j[valid_1]][filter1&filter2]
        if (depth_filtered < seg_obj_depth_filtered).sum() > 0:
            # this object is hidden
            if not obj.active:
                return True

        valid_1 = (img_j-1>=0) & (seg_img==seg_id)
        filter1 = obj_seg_filter[img_i[valid_1],img_j[valid_1]-1]
        filter2 = (seg_img[img_i[valid_1],img_j[valid_1]-1] != seg_id)
        depth_filtered = depth_img[img_i[valid_1],img_j[valid_1]-1][filter1&filter2]
        seg_obj_depth_filtered = depth_img[img_i[valid_1],img_j[valid_1]][filter1&filter2]
        if (depth_filtered < seg_obj_depth_filtered).sum() > 0:
            # this object is hidden
            if not obj.active:
                return True

        valid_1 = (img_j+1<seg_img.shape[1]) & (seg_img==seg_id)
        filter1 = obj_seg_filter[img_i[valid_1],img_j[valid_1]+1]
        filter2 = (seg_img[img_i[valid_1],img_j[valid_1]+1] != seg_id)
        depth_filtered = depth_img[img_i[valid_1],img_j[valid_1]+1][filter1&filter2]
        seg_obj_depth_filtered = depth_img[img_i[valid_1],img_j[valid_1]][filter1&filter2]
        if (depth_filtered < seg_obj_depth_filtered).sum() > 0:
            # this object is hidden
            if not obj.active:
                return True
        return False    


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

        return total_sample


    def shadow_occupancy_single_obj(self, occluded, camera_extrinsics, camera_intrinsics, obj_pcd):
        """
        obtain the shadow occupancy of the hidden object, which corresponds to the subset of the occlusion region
        where the object is possible to occupy. This should shrink the occlusion space.
        Method to generate this:
        Shrink:
            [i,j]=1   if  [i',j'] =1 in the object voxel, and [i+i',j+j']=1  in the world voxel, for all object voxels

        Minkowski sum:
            for each [i,j]=1, set [i+i',j+j']=1 for [i',j'] in the object voxel
            OR
            [i,j]=1   if  [i',j'] in the shrinked voxel, and [i-i',j-j'] in the object voxel

        """
        # obtain the voxelized object pcd
        obj_pcd = self.world_in_voxel_rot.dot(obj_pcd.T).T + self.world_in_voxel_tran
        obj_pcd = obj_pcd / self.resol
        obj_voxel_pts = np.floor(obj_pcd).astype(int)
        obj_voxel_pts_min = obj_voxel_pts.min(axis=0)
        obj_voxel_pts_max = obj_voxel_pts.max(axis=0)
        obj_voxel = np.zeros(obj_voxel_pts_max-obj_voxel_pts_min+1).astype(bool)
        obj_voxel_pts = obj_voxel_pts - obj_voxel_pts_min
        obj_voxel[obj_voxel_pts[:,0],obj_voxel_pts[:,1],obj_voxel_pts[:,2]] = 1
        obj_voxel_x, obj_voxel_y, obj_voxel_z = np.indices(obj_voxel.shape).astype(int)
        obj_voxel_x = obj_voxel_x[obj_voxel==1]
        obj_voxel_y = obj_voxel_y[obj_voxel==1]
        obj_voxel_z = obj_voxel_z[obj_voxel==1]

        # obtain the shrinked voxels

        intersected = np.zeros((occluded.shape[0]-obj_voxel.shape[0],
                                occluded.shape[1]-obj_voxel.shape[1],
                                occluded.shape[2]-obj_voxel.shape[2])).astype(bool)
        obj_voxel_nonzero = obj_voxel.astype(int).sum()

        for i in range(len(intersected)):
            for j in range(intersected.shape[1]):
                for k in range(intersected.shape[2]):
                    world_voxel_masked = occluded[i:i+obj_voxel.shape[0],j:j+obj_voxel.shape[1],k:k+obj_voxel.shape[2]]
                    val = (obj_voxel & world_voxel_masked).astype(int).sum()
                    intersected[i,j,k] = (val == obj_voxel_nonzero)

        shadow_occupancy = np.zeros(occluded.shape).astype(bool)
        # minkowski sum                        
        for i in range(len(intersected)):
            for j in range(intersected.shape[1]):
                for k in range(intersected.shape[2]):
                    if intersected[i,j,k]:
                        shadow_occupancy[obj_voxel_x+i, obj_voxel_y+j, obj_voxel_z+k] = 1
        # print('invalid number: ', (shadow_occupancy & (~occluded)).astype(int).sum())
        return intersected, shadow_occupancy

