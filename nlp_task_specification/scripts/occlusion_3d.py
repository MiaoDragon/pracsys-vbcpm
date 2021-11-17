"""
This represents the occlusion in the scene, given a depth image,
and known object models.
The occlusion region is given by the formula:
    hidden area - known occupied area
"""
import numpy as np
from cam_utilities import *
from tqdm import trange
import time

import cv2

class Occlusion():
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
    # def get_occlusions(self, objects, camera):
    #     pass
    def get_total_occlusions(self, depth_img, color_img, camera):
        """
        given the depth image and the camera information, obtain the total occlusions by
        1. construct pcd from the depth image
        2. mark pcds in the voxel grid as occupied
        3. pcds define casted rays. For the ray behind the pcd, mark the voxels as hidden
        4. filter out irrelevant voxels, such as places which are out of the boundary
        """
        pcds, depth_vals = pcd_from_depth(camera['intrinsics'], camera['extrinsics'], depth_img, color_img)
        # transform the camera into the voxel space
        cam_in_voxel = self.world_in_voxel @ camera['extrinsics']
        # cam_in_voxel = self.world_in_voxel_rot @ camera['extrinsics'] + self.world_in_voxel_tran
        # cam_in_voxel = cam_in_voxel / self.resol
        cam_in_voxel = cam_in_voxel[:3,3] / self.resol
        # this obtains pcd in the world frame. We need to transform them into the voxel frame
        pcd_in_voxel = (self.world_in_voxel_rot @ pcds.T).T + self.world_in_voxel_tran
        pcd_in_voxel = pcd_in_voxel / self.resol  # obtain the indices
        # obtain the ray
        rays = pcd_in_voxel - cam_in_voxel

        cam_to_voxel_dist = np.sqrt((self.voxel_x - cam_in_voxel[0]) ** 2 + \
                                    (self.voxel_y - cam_in_voxel[1]) ** 2 + \
                                    (self.voxel_z - cam_in_voxel[2]) ** 2)

        # compute intersections with the voxel grid
        occluded = np.zeros((int(self.world_x / self.resol[0]), \
                            int(self.world_y / self.resol[1]), \
                            int(self.world_z / self.resol[2]))).astype(bool)
        for i in trange(len(rays)):
            start_time = time.time()
            mask, _, _ = ray_intersect(cam_in_voxel, rays[i], self.voxel_x, self.voxel_y, self.voxel_z)
            print('ray casting takes time: ', time.time() - start_time)
            start_time = time.time()

            mask = mask & (cam_to_voxel_dist >= np.linalg.norm(rays[i]))
            occluded[mask] = 1
            print('the rest takes time: ', time.time() - start_time)
            # occluded = occluded | mask

        return occluded

    def get_total_occlusions_v2(self, depth_img, color_img, camera_extrinsics, camera_intrinsics):
        """
        convert voxel indices to depth image pixels, and extract the depth values
        compute the difference of the distance. Get voxels which have negative values
        """
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

    def get_occlusions(self, depth_img, color_img, camera_extrinsics, camera_intrinsics, obj_poses, obj_pcds):
        """
        given the depth image and the camera information, obtain the occlusions by
        1. obtain full occlusion (occupied space + hidden areas)
        2. subtract the seen object occupation space
        """
        occluded = self.get_total_occlusions_v2(depth_img, color_img, camera_extrinsics, camera_intrinsics)
        for i in range(len(obj_poses)):
            if obj_poses[i] is None:
                # unable to obtain the pose
                continue
            pose = obj_poses[i]
            R = pose[:3,:3]
            T = pose[:3,3]
            pcd = obj_pcds[i]
            # pcd = pcd * 1.1  # add some padding
            transformed_pcds = R.dot(pcd.T).T + T
            # map the pcd to voxel space
            transformed_pcds = self.world_in_voxel_rot.dot(transformed_pcds.T).T + self.world_in_voxel_tran
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
        return occluded, transformed_pcds
    def get_occlusion_single_obj(self, camera_extrinsics, obj_pose, obj_pcd):
        """
        for a given object at the known pose, we can generate a depth image for it and compute the
        voxels that represent the occlusion region.
        """
        # convert the obj_pcd to image space
        R = obj_pose[:3,:3]
        T = obj_pose[:3,3]
        pcd = R.dot(obj_pcd.T).T + T
        cam_transform = np.linalg.inv(camera_extrinsics)
        transformed_pcd = cam_transform[:3,:3].dot(pcd.T).T + cam_transform[:3,3]
        cam_intrinsics = np.zeros((3,3))
        cam_intrinsics[0,0] = 1/self.resol[0] / 2
        cam_intrinsics[1,1] = 1/self.resol[1] / 2
        fx = cam_intrinsics[0][0]
        fy = cam_intrinsics[1][1]

        cam_intrinsics[0,2] = np.abs((transformed_pcd[:,0] / transformed_pcd[:,2] * fx).min())
        cam_intrinsics[1,2] = np.abs((transformed_pcd[:,1] / transformed_pcd[:,2] * fy).min())

        cx = cam_intrinsics[0,2]
        cy = cam_intrinsics[1,2]
        transformed_pcd[:,0] = transformed_pcd[:,0] / transformed_pcd[:,2] * fx + cx
        transformed_pcd[:,1] = transformed_pcd[:,1] / transformed_pcd[:,2] * fy + cy
        depth = transformed_pcd[:,2]
        transformed_pcd = transformed_pcd[:,:2]
        transformed_pcd = np.floor(transformed_pcd).astype(int)
        max_j = transformed_pcd[:,0].max()+1
        max_i = transformed_pcd[:,1].max()+1
        depth_img = np.zeros((max_i, max_j)).astype(float)
        depth_img[transformed_pcd[:,1],transformed_pcd[:,0]] = depth
        print(depth_img)
        cv2.imshow('depth', depth_img)
        cv2.waitKey(0) 

        occluded, _ = self.get_occlusions(depth_img, None, camera_extrinsics, cam_intrinsics, [obj_pose], [obj_pcd])
        # occluded = self.get_total_occlusions_v2(depth_img, None, camera)
        return occluded










