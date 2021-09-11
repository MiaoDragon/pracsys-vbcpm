"""
voxel grid representation of occlusion
"""
# from cam_utilities import *
import cam_utilities
import numpy as np
import tqdm
class Occlusion():
    def __init__(self, x_base, y_base, z_base, x_vec, y_vec, z_vec, resolution_vec, x_num, y_num, z_num):
        # given the origin and axes, construct the voxel grid
        self.x_base = x_base
        self.y_base = y_base
        self.z_base = z_base
        self.x_vec = x_vec
        self.y_vec = y_vec
        self.z_vec = z_vec
        self.resol = resolution_vec
        self.x_num = x_num
        self.y_num = y_num
        self.z_num = z_num
        # the x,y and z vectors define a rotation matrix, and the base defines a translation matrix
        self.T = np.array([x_vec, y_vec, z_vec])
        self.T = np.zeros((4,4))
        self.T[:3,0] = x_vec
        self.T[:3,1] = y_vec
        self.T[:3,2] = z_vec
        self.T[:3,3] = np.array([self.x_base, self.y_base, self.z_base])
        self.T[3,3] = 1.
        self.occupied_voxel = np.zeros((self.x_num, self.y_num, self.z_num)).astype(bool)
        self.occluded_voxel = np.zeros((self.x_num, self.y_num, self.z_num)).astype(bool)

        self.voxel_x, self.voxel_y, self.voxel_z = np.indices(self.occupied_voxel.shape).astype(float)
        self.voxel_dist = np.sqrt(self.voxel_x ** 2 + self.voxel_y ** 2 + self.voxel_z ** 2)

    def update(self, prev_occ_voxel, prev_vis_voxel, occ_voxel, vis_voxel):
        # given two voxels, update the map
        self.occupied_voxel = prev_occ_voxel | occ_voxel
        self.occluded_voxel = prev_vis_voxel | vis_voxel
    
    def update_map(self, occ_voxel, vis_voxel):
        # using the new observation to update the map
        self.occupied_voxel = self.occupied_voxel | occ_voxel
        self.occluded_voxel = self.occluded_voxel | vis_voxel

    def observe_model_based(self, cam_center, cam_intrinsics, cam_extrinsics, depth_img, color_img, obj_pcds=None):
        # given a depth image, obtain the observed voxels
        """
        assume we have knowledge of the object mapping in the scene (which index in the pcd id it corresponds to)
        put the object pcd into the voxel grid by checking each point, and set the voxel grid as occupied
        """
        # obtain occupancy grid
        occluded_voxel = self.ray_cast(cam_center, cam_intrinsics, cam_extrinsics, depth_img, color_img)
        handled_objs = []
        occupied_voxel = None
        if obj_pcds is not None:
            for i in range(len(obj_pcds)):
                # place the pcd into the voxel
                occupied_voxel = self.integrate_pcd(occluded_voxel, obj_pcds[i])
        return occupied_voxel, occluded_voxel
    
    def ray_cast(self, cam_center, cam_intrinsics, cam_extrinsics, depth_img, color_img):
        """
        cast rays in the field of view to obtain the voxels
        """
        img_pts, depth_vals = cam_utilities.pcd_from_depth(cam_intrinsics, cam_extrinsics, depth_img, color_img)  # N x 3
        ray_vector = img_pts - cam_center
        ray_vector = ray_vector.T / np.linalg.norm(ray_vector, axis=1)  # normalize
        ray_vector = ray_vector.T
        # transform the ray vector into the voxel_grid space
        # notice that we have to divide by the resolution vector
        ray_vector = np.linalg.inv(self.T[:3,:3]).dot(ray_vector.T).T / self.resol
        # ray_vector = ray_vector[:,:3]

        transformed_cam_center = np.array([cam_center[0], cam_center[1], cam_center[2], 1.])
        transformed_cam_center = np.linalg.inv(self.T).dot(transformed_cam_center.T).T
        transformed_cam_center = transformed_cam_center[:3] / self.resol
        print('transformed camera center: ', transformed_cam_center)
        # obtain intersection with the voxel grid
        threshold = 0.01  # 1cm
        print('ray vector: ')
        print(ray_vector)
        occluded_voxel = np.zeros(self.occupied_voxel.shape)
        print('depth values: ')
        print(depth_vals.shape)
        print('depth min: ')
        print(depth_vals.min())
        print('depth max: ')
        print(depth_vals.max())
        for i in tqdm.trange(len(ray_vector)):
            intersect_mask, tmin, tmax = cam_utilities.ray_intersect(transformed_cam_center, ray_vector[i], self.voxel_x, self.voxel_y, self.voxel_z)
            print('intersection number: ')
            print(intersect_mask.sum() / len(intersect_mask.reshape(-1)))
            # assume tmin always > 0 (in the positive direction)
            # find the intersections where:
            # mark -threshold <= distance - depth_img <= threshold as occupied
            # mark distance - depth_img > threshold as invisible
            # occupied_mask = (tmin - depth_vals[i] >= -threshold) & (tmin - depth_vals[i] <= threshold)
            voxel_dist = np.sqrt((self.voxel_x*self.resol[0] - transformed_cam_center[0]*self.resol[0]) ** 2 + \
                        (self.voxel_y*self.resol[1] - transformed_cam_center[1]*self.resol[1]) ** 2 + \
                        (self.voxel_z*self.resol[2] - transformed_cam_center[2]*self.resol[2]) ** 2)
            print('depth_vals: shape')
            print(depth_vals.shape)
            print('depth_val: ')
            print(depth_vals[i])
            print('voxel distance minn: ')
            print((voxel_dist - depth_vals[i]).min())
            print('voxel distance max: ')
            print((voxel_dist - depth_vals[i]).max())
            print('occlusion number: ')
            occluded_mask = intersect_mask & (voxel_dist - depth_vals[i] >= 0.02)    
            print(np.sum(occluded_mask) / len(occluded_mask.reshape(-1)))
            print((voxel_dist - depth_vals[i])[occluded_mask])
            occluded_voxel[occluded_mask] = 1
            print('total occlusion number: ')
            print(np.sum(occluded_voxel) / len(occluded_voxel.reshape(-1)))
        return occluded_voxel

    def integrate_pcd(self, occupied_voxel, obj_pcd):
        obj_pcd_voxel = np.zeros(occupied_voxel.shape).astype(bool)
        obj_pcd = np.concatenate([obj_pcd, np.ones(obj_pcd.shape[0]).reshape((-1,1))], axis=1)
        # transform to voxel space
        transformed_obj_pcd = np.linalg.inv(self.T).dot(obj_pcd.T).T / self.resol
        transformed_obj_pcd = np.floor(transformed_obj_pcd).astype(int)
        obj_pcd_voxel[transformed_obj_pcd[:,0],transformed_obj_pcd[:,1],transformed_obj_pcd[:,2]] = 1
        occupied_voxel = occupied_voxel & obj_pcd_voxel
        return occupied_voxel

if __name__ == '__main__':
    # vol_box = o3d.geometry.OrientedBoundingBox()
    # vol_box.center = vol_bnds.mean(1)
    # vol_box.extent = vol_bnds[:, 1] - vol_bnds[:, 0]
    # o3d.visualization.draw_geometries([vol_box, transformed_pcd])

    # observe_model_based(self, cam_center, cam_intrinsics, cam_extrinsics, depth_img, color_img, obj_pcds)
    pass