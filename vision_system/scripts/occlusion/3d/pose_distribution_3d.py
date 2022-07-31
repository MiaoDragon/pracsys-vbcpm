"""
This implements the pose distribution of the object that is occluded.
Potential Constraints:
- occlusion (+)
- collision (-)
"""
import numpy as np
import transformations as tf
from tqdm import trange
import open3d as o3d

class DiscreteDistribution():
    """
    For the discrete case, we discretize the world into voxels [x,y,theta],
    representing the possible poses of the object on the shelf.
    Our assumption here is that the object is "stable", and the pose only rotates
    around the z-axis. A more generalized representation should consider All Possible
    Stable Poses in the scene.
    """
    def __init__(self, world_sz, base, axes, resol, occlusion_resol, obj_id, obj_height, obj_pcd):
        self.base = base
        self.axes = axes
        self.transform = np.zeros((4,4))
        self.transform[3,3] = 1
        self.transform[:3,0] = axes[0]
        self.transform[:3,1] = axes[1]
        self.transform[:3,2] = axes[2]
        self.transform[:3,3] = base
        self.occlusion_resol = occlusion_resol
        self.resol = resol
        self.world_sz = world_sz

        self.voxel = np.zeros((int(self.world_sz[0] / self.resol[0]), \
                                int(self.world_sz[1] / self.resol[1]), \
                                int(np.pi*2 / self.resol[2]))).astype(bool)
        voxel_x, voxel_y, voxel_z = np.indices(self.voxel.shape).astype(float)
        voxel_z = voxel_z * 0 + obj_height/2/self.resol[2]
        self.voxel_x = voxel_x
        self.voxel_y = voxel_y
        self.voxel_z = voxel_z

        self.obj_height = obj_height
        print('object height: ', self.obj_height)
        self.obj_z_padding = obj_height/2 + base[2]
        print('obj_z_padding: ', self.obj_z_padding)

        self.obj_pcd = obj_pcd

        self.world_in_voxel = np.linalg.inv(self.transform)
        self.world_in_voxel_rot = self.world_in_voxel[:3,:3]
        self.world_in_voxel_tran = self.world_in_voxel[:3,3]
    def compute_occupancy_batch(self):
        """
        for each of the pose candidate, compute the occupancy grid and store
        Approximation for speed:
        we compute the occupancy at origin, and then directly shift the voxels
        to get the value for each other voxel place
        at origin, we obtain the indices of the voxel that are occupied
        then we just need to add this to the voxel
        """
        occupancy = np.zeros((int(self.world_sz[0] / self.resol[0]), \
                                int(self.world_sz[1] / self.resol[1]), \
                                int(np.pi*2 / self.resol[2]), \
                                int(self.world_sz[0] / self.resol[0]),
                                int(self.world_sz[1] / self.resol[1]),
                                int(self.world_sz[2] / self.resol[2]))).astype(bool)
        occupancy_base = np.zeros((int(self.world_sz[0] / self.resol[0]), \
                                int(self.world_sz[1] / self.resol[1]), \
                                int(np.pi*2 / self.resol[2]), \
                                1,3))
        occupancy_base = np.array([self.voxel_x, self.voxel_y, self.voxel_z]).transpose((1,2,3,0))
        occupancy_base = occupancy_base.reshape(occupancy_base.shape[0],occupancy_base.shape[1],occupancy_base.shape[2],1,3)
        occupancy_vec = np.zeros((int(self.world_sz[0] / self.resol[0]), \
                                int(self.world_sz[1] / self.resol[1]), \
                                int(np.pi*2 / self.resol[2]), \
                                len(self.obj_pcd),3))
        # add one more axis so we can later do element-wise operations
        for k in range(len(self.voxel[0,0])):
            z = self.obj_z_padding
            ori = tf.rotation_matrix(k*self.resol[2]-np.pi, np.array([0.,0.,1.]))
            pcd_transform_rot = ori
            pcd_transform_tran = np.array([self.base[0], self.base[1], z])
            new_pcd = pcd_transform_rot.dot(self.obj_pcd.T).T + pcd_transform_tran
            # obtain pcd in the voxel for occupancy grid at origin
            new_pcd = self.world_in_voxel_rot.dot(new_pcd.T).T + self.world_in_voxel_tran
            new_pcd = new_pcd / self.occlusion_resol
            # new_pcd = np.floor(new_pcd).astype(int)  # new_pcd defines the voxel index for occupancy
            # shift the occupancy at origin to each grid
            occupancy_vec[:,:,k] = occupancy_base[:,:,k] + new_pcd.reshape(1,1,1,-1,3)
        # now we have the grid position for each of the voxel. filter and set to occupancy
        occupancy_vec = np.floor(occupancy_vec).astype(int)
        valid_filter = (occupancy_vec[:,:,:,:,0] >= 0) & (occupancy_vec[:,:,:,:,0] < occupancy.shape[0]) & \
                        (occupancy_vec[:,:,:,:,1] >= 0) & (occupancy_vec[:,:,:,:,1] < occupancy.shape[1]) & \
                        (occupancy_vec[:,:,:,:,2] >= 0) & (occupancy_vec[:,:,:,:,2] < occupancy.shape[2])
        # since applying the filter maps the array to 1D. We need to know hte corresponding x,y,z axes of the filtered entries
        occupancy_vec_x, occupancy_vec_y, occupancy_vec_z, _, _ = np.indices(valid_filter.shape)
        x_indices = occupancy_vec_x[valid_filter]
        y_indices = occupancy_vec_y[valid_filter]
        z_indices = occupancy_vec_z[valid_filter]
        voxel_x_indices = occupancy_vec[valid_filter][:,0]
        voxel_y_indices = occupancy_vec[valid_filter][:,1]
        voxel_z_indices = occupancy_vec[valid_filter][:,2]

        occupancy[x_indices,y_indices,z_indices,voxel_x_indices,voxel_y_indices,voxel_z_indices] = 1
        self.occupancy = occupancy

    def compute_occupancy(self):
        pass                        
    def compute_distribution(self, occlusion, threshold):
        """
        given the occlusion, compute the potential poses of the target object.
        Assume we use the same x-y resolution in the occlusion
        For each potential pose, we can transform the object pcd to check if 
        it has intersections with the occlusion space.
        TODO: speed up the for loop
        TODO: The pose -> occupied space mapping can also be precomputed
        """
        occupancy = np.zeros(self.voxel.shape).astype(bool)
        
        new_pcds = []
        for k in range(self.voxel.shape[2]):
            # transform the pcd into the voxel space
            z = self.obj_z_padding
            ori = tf.rotation_matrix(k*self.resol[2]-np.pi, np.array([0.,0.,1.]))
            pcd_transform_rot = ori[:3,:3]
            pcd_transform_tran = np.array([self.base[0],self.base[1],z])
            new_pcd = pcd_transform_rot.dot(self.obj_pcd.T).T + pcd_transform_tran
            # obtain pcd in the voxel for occupancy grid
            new_pcd = self.world_in_voxel_rot.dot(new_pcd.T).T + self.world_in_voxel_tran
            new_pcd = new_pcd / self.occlusion_resol
            new_pcd = np.floor(new_pcd).astype(int)
            new_pcds.append(new_pcd)
        new_pcds = np.array(new_pcds)

        for looper in trange(self.voxel.shape[0]*self.voxel.shape[1]):
            i = looper // self.voxel.shape[1]
            j = looper % self.voxel.shape[1]
            pcd_transform_tran = np.array([i,j,0])
            new_pcds_i = new_pcds + pcd_transform_tran
            new_pcds_i = new_pcds_i
            new_pcds_i = np.floor(new_pcds_i).astype(int)
            if looper == 0:
                return_pcds = new_pcds_i[0]
            for k in range(self.voxel.shape[2]):
                # transform the pcd into the voxel space
                new_pcd = new_pcds_i[k]

                valid_filter = (new_pcd[:,0] >= 0) & (new_pcd[:,0]<occlusion.shape[0]) & \
                                (new_pcd[:,1] >= 0) & (new_pcd[:,1]<occlusion.shape[1]) & \
                                (new_pcd[:,2] >= 0) & (new_pcd[:,2]<occlusion.shape[2])
                # check in the occlusion
                valid_x = new_pcd[valid_filter][:,0]
                valid_y = new_pcd[valid_filter][:,1]
                valid_z = new_pcd[valid_filter][:,2]

                # check intersection: how much is in occlusion?
                occluded_sum = occlusion[valid_x, valid_y, valid_z].astype(int).sum()
                ratio = occluded_sum / len(valid_x)
                # if ratio > 0:
                #     print('ratio: ', ratio)
                if ratio > threshold:
                    # print('ratio: ', ratio)
                    occupancy[i,j,k] = 1
        return occupancy, return_pcds