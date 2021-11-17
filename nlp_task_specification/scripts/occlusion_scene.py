"""
a representation of the occlusion of the entire scene.
Use depth image to capture the current occlusion state,
and then associate each occlusion voxels with the corresponding
object based on object's pose and geometry
"""
import numpy as np
import cv2
import scipy.signal as scipy_signal

class OcclusionScene():
    def __init__(self, world_x, world_y, world_z, resol, x_base, y_base, z_base, x_vec, y_vec, z_vec):
        # specify parameters to represent the occlusion space
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
        # print('cam_to_voxel_depth: ')
        # print(cam_to_voxel_depth)
        # print('depth images: ')
        # print(depth_img)
        # print('cam_to_voxel_depth between 0 and 0.2: ', ((cam_to_voxel_depth > 0) & (cam_to_voxel_depth < 0.2)).astype(int).sum())
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
        # print('valid_mask: ')
        # print(valid_mask.astype(int).sum())
        # print('voxel_depth: ')
        # print(cam_to_voxel_depth - voxel_depth)
        occluded = (cam_to_voxel_depth - voxel_depth >= 0.) & (voxel_depth > 0.) & valid_mask
        print(occluded.astype(int).sum() / valid_mask.astype(int).sum())
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
        new_occlusion_label[(occlusion_label2<0)&(occlusion_label1!=0)] = -1
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


    def label_scene_occlusion(self, occluded, camera_extrinsics, camera_intrinsics, obj_poses, obj_pcds, depth_nn=1):
        """
        depth_nn: maximum distance in the depth image to count for the object
        """
        # given the scene occlusion, label each of the occluded region based on the object known info
        # object -> render depth image -> mask
        # * intersection of mask and scene occlusion corresponds to the parts that belong to object

        # * object occupied space can be determined by point cloud voxelization

        # * the remaining space is unknown parts

        occluded = np.array(occluded).astype(bool)  # make a copy
        occlusion_label = np.zeros(occluded.shape).astype(int)  # 0: free space, id: occluded by id, -1: unknown
        occupied_label = np.zeros(occluded.shape).astype(int)  # id: occupied by id
        occluded_list = []
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
        for i in range(len(obj_poses)):
            if obj_poses[i] is None:
                # unseen object
                continue
            obj_pose = obj_poses[i]
            obj_pcd = obj_pcds[i]
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
            occupied = occluded & occupied

            occupied_label[occupied==1] = i+1
            occluded[indices[:,0],indices[:,1],indices[:,2]] = 0



        for i in range(len(obj_poses)):
            if obj_poses[i] is None:
                # unseen object
                occluded_list.append(None)
                continue
            obj_pose = obj_poses[i]
            obj_pcd = obj_pcds[i]
            R = obj_pose[:3,:3]
            T = obj_pose[:3,3]

            pcd = R.dot(obj_pcd.T).T + T

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
            
            ori_shape = depth_img.shape
            # depth_img = cv2.resize(depth_img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
            # depth_img = cv2.resize(depth_img, ori_shape, interpolation=cv2.INTER_LINEAR)
            # depth_img = cv2.resize(depth_img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
            # depth_img = cv2.medianBlur(np.float32(depth_img), 5)
            depth_img = cv2.boxFilter(np.float32(depth_img), -1, (5,5))

            # cv2.imshow('depth', depth_img)
            # cv2.waitKey(0)
            occluded_i = self.scene_occlusion(depth_img, None, camera_extrinsics, camera_intrinsics)

            occluded_i = occluded_i & occluded
            occluded_list.append(occluded_i)
            occlusion_label[occluded_i==1] = i+1

        # the rest of the space is unknown space
        occlusion_label[(occlusion_label==0)&(occluded==1)] = -1
        return occlusion_label, occupied_label, occluded_list


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
        print('invalid number: ', (shadow_occupancy & (~occluded)).astype(int).sum())
        return intersected, shadow_occupancy