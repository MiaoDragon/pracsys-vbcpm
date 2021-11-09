"""
the implementation for the object reconstruction.
Maintain TSDF individually for each of the object. Volumes can
be extracted by two types:
- optimistic volume: including the parts that we have seen. Since
    interior of the object may not be observed ever, this represents
    a shell surface around the object.
- conservative volume: initially include only the unseen parts. This
    will shrink as more observations are made.

Objects that have been occluded may have parts not represented in the TSDF.
To make sure collision is not caused, we need to sense them before moving them.
As for occupancy representation, we need to expand their volume until they're not
occluded anymore

To represent the voxel we need:
- origin_x, origin_y, origin_z
- axis_x, axis_y, axis_z
- size_x, size_y, size_z
- resol_x, resol_y, resol_z

TODO: a better representation of the object:
when the reconstructed surface is OPEN, the unseen parts may belong to the object
when the reconstructed surface is CLOSED, the unseen parts belong to the object (interior)

we can probably differentiate them by checking whether the unseen parts of the object has a MAX_V neighbor, or at boundary

This requires one region to belong to object region and occlusion region at the same time
at the beginning

TODO: TSDF has some problems
"""
from cv2 import threshold
import numpy as np
class ObjectModel():
    def __init__(self, xmin, ymin, zmin, xmax, ymax, zmax, resol, scale=0.03):
        """
        initialize the object model to be the bounding box containing the conservative volume of the object
        """
        self.origin_x = xmin
        self.origin_y = ymin
        self.origin_z = zmin
        self.xmin = xmin
        self.ymin = ymin
        self.zmin = zmin
        self.xmax = xmax
        self.ymax = ymax
        self.zmax = zmax

        self.resol = resol
        size_x = int(np.ceil((xmax - xmin) / resol[0]))
        size_y = int(np.ceil((ymax - ymin) / resol[1]))
        size_z = int(np.ceil((zmax - zmin) / resol[2]))
        
        self.xmax = xmin + size_x * resol[0]
        self.ymax = ymin + size_y * resol[1]
        self.zmax = zmin + size_z * resol[2]

        self.axis_x = np.array([1.,0,0])
        self.axis_y = np.array([0.,1,0])
        self.axis_z = np.array([0.,0,1])

        self.size_x = size_x
        self.size_y = size_y
        self.size_z = size_z

        self.tsdf = np.zeros((size_x,size_y,size_z))  # default value: -1
        self.tsdf_count = np.zeros((size_x,size_y,size_z)).astype(int)  # count how many times it's observed
        self.voxel_x, self.voxel_y, self.voxel_z = np.indices(self.tsdf.shape).astype(float)

        self.scale = scale
        self.max_v = scale#1.0
        self.min_v = -scale#-1.0 
        self.active = 0  
        # 0 means this object might be hidden by others. 1 means we can safely move the object now
        # we will only expand the object tsdf volume when it's hidden. Once it's active we will keep
        # the tsdf volume fixed

        self.transform = np.zeros((4,4))
        self.transform[:3,0] = self.axis_x
        self.transform[:3,1] = self.axis_y
        self.transform[:3,2] = self.axis_z
        self.transform[:3,3] = np.array([self.origin_x, self.origin_y, self.origin_z])
        self.transform[3,3] = 1.

        # self.transform = transform  # the transform of the voxel grid cooridnate system in the world as {world}T{voxel}
        self.world_in_voxel = np.linalg.inv(self.transform)
        self.world_in_voxel_rot = self.world_in_voxel[:3,:3]
        self.world_in_voxel_tran = self.world_in_voxel[:3,3]

    
    def get_optimistic_model(self):
        threshold = 1
        return (self.tsdf_count >= threshold) & (self.tsdf < self.max_v) & (self.tsdf > self.min_v)
    
    def get_conservative_model(self):
        # unseen parts below to the conservative model
        threshold = 1
        return (self.tsdf_count < threshold) | ((self.tsdf_count >= threshold) & (self.tsdf < self.max_v))

    def set_active(self):
        # when the object is no longer hidden by others, it can move
        self.active = 1
    
    def update_transform_from_relative(self, rel_transform):
        # when the object is moved, update the transform
        # previous: W T O1
        # relative transform: O1 T O2
        self.transform = rel_transform.dot(self.transform)
        # self.transform = self.transform.dot(rel_transform)

    def expand_model(self, new_xmin, new_ymin, new_zmin, new_xmax, new_ymax, new_zmax):
        """
        expand the model when new parts are seen
        """
        # compute the location of origin in the new voxel
        dx = self.xmin - new_xmin
        dy = self.ymin - new_ymin
        dz = self.zmin - new_zmin

        dx = np.round(dx, decimals=3)
        dy = np.round(dy, decimals=3)
        dz = np.round(dz, decimals=3)

        nx = int(np.ceil(dx / self.resol[0]))
        ny = int(np.ceil(dy / self.resol[1]))
        nz = int(np.ceil(dz / self.resol[2]))

        nx = max(0, nx)
        ny = max(0, ny)
        nz = max(0, nz)

        # update new lower-bound to be integer number of resolution
        new_xmin = self.xmin - nx * self.resol[0]
        new_ymin = self.ymin - ny * self.resol[1]
        new_zmin = self.zmin - nz * self.resol[2]


        new_xmax = max(new_xmax, self.xmax)
        new_ymax = max(new_ymax, self.ymax)
        new_zmax = max(new_zmax, self.zmax)

        new_size_x = (new_xmax - new_xmin) / self.resol[0]
        new_size_x = np.round(new_size_x, decimals=3)
        new_size_x = int(np.ceil(new_size_x))
        new_size_y = (new_ymax - new_ymin) / self.resol[1]
        new_size_y = np.round(new_size_y, decimals=3)
        new_size_y = int(np.ceil(new_size_y))
        new_size_z = (new_zmax - new_zmin) / self.resol[2]
        new_size_z = np.round(new_size_z, decimals=3)
        new_size_z = int(np.ceil(new_size_z))

        # update upper bound
        new_xmax = new_xmin + new_size_x * self.resol[0]
        new_ymax = new_ymin + new_size_y * self.resol[1]
        new_zmax = new_zmin + new_size_z * self.resol[2]


        # if values don't change, then no need to expand
        if (self.xmin == new_xmin) and (self.ymin == new_ymin) and (self.zmin == new_zmin) and \
            (self.xmax == new_xmax) and (self.ymax == new_ymax) and (self.zmax == new_zmax):
            return
        

        new_tsdf = np.zeros((new_size_x, new_size_y, new_size_z))
        new_tsdf_count = np.zeros((new_size_x, new_size_y, new_size_z)).astype(int)

        new_tsdf[nx:nx+self.size_x,ny:ny+self.size_y,nz:nz+self.size_z] = self.tsdf
        new_tsdf_count[nx:nx+self.size_x,ny:ny+self.size_y,nz:nz+self.size_z] = self.tsdf_count

        self.xmin = new_xmin
        self.ymin = new_ymin
        self.zmin = new_zmin
        self.xmax = new_xmax

        self.ymax = new_ymax
        self.zmax = new_zmax
        
        self.size_x = new_size_x
        self.size_y = new_size_y
        self.size_z = new_size_z

        self.origin_x = new_xmin
        self.origin_y = new_ymin
        self.origin_z = new_zmin
        self.transform[:3,3] = np.array([self.origin_x, self.origin_y, self.origin_z])
        
        self.tsdf = new_tsdf
        self.tsdf_count = new_tsdf_count
        self.voxel_x, self.voxel_y, self.voxel_z = np.indices(self.tsdf.shape).astype(float)

    
    def update_tsdf(self, depth_img, color_img, camera_extrinsics, camera_intrinsics):
        """
        given the *segmented* depth image belonging to the object, update tsdf
        if new parts are seen, expand the model (this only happens when this object is inactive, or hidden initially)
        """
        # obtain pixel locations for each of the voxels
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


        # handle valid space
        tsdf = np.zeros(self.tsdf.shape)
        tsdf = (voxel_depth - cam_to_voxel_depth)# * self.scale
        valid_space = (voxel_depth>0) & (tsdf > self.min_v) & valid_mask

        print('tsdf: ')
        print(((tsdf[valid_space]>self.min_v) & (tsdf[valid_space]<self.max_v)).astype(int).sum() / valid_space.astype(int).sum())

        self.tsdf[valid_space] = (self.tsdf[valid_space] * self.tsdf_count[valid_space] + tsdf[valid_space]) / (self.tsdf_count[valid_space] + 1)
        self.tsdf_count[valid_space] = self.tsdf_count[valid_space] + 1

        self.tsdf[self.tsdf>self.max_v] = self.max_v
        self.tsdf[self.tsdf<self.min_v] = self.min_v

        # handle invalid space: don't update
        invalid_space = ((voxel_depth <= 0) | (tsdf < self.min_v)) & valid_mask

        self.tsdf[self.tsdf_count==0] = 0.0
    

    def sample_pcd(self, mask, n_sample=10):
        # sample voxels in te mask
        # obtain sample in one voxel cell
        grid_sample = np.random.uniform(low=[0,0,0], high=[1,1,1], size=(n_sample, 3))
        voxel_x = self.voxel_x[mask]
        voxel_y = self.voxel_y[mask]
        voxel_z = self.voxel_z[mask]

        total_sample = np.zeros((len(voxel_x), n_sample, 3))
        print(total_sample.shape)
        total_sample = total_sample + grid_sample
        total_sample = total_sample + np.array([voxel_x, voxel_y, voxel_z]).T.reshape(len(voxel_x),1,3)

        total_sample = total_sample.reshape(-1, 3) * np.array(self.resol)

        return total_sample

    def sample_conservative_pcd(self, n_sample=10):
        # obtain the pcd of the conservative volume
        return self.sample_pcd(self.get_conservative_model(), n_sample)
    def sample_optimistic_pcd(self, n_sample=10):
        # obtain the pcd of the conservative volume
        return self.sample_pcd(self.get_optimistic_model(), n_sample)




def test():
    # test the module
    object = ObjectModel(0.0, 0.0, 0.0, 1.000, 1.000, 1.000, [0.01,0.01,0.01], 0.05)

    object.tsdf = object.tsdf + 1.0
    
    object.expand_model(-0.005, -0.005, -0.005, 1.01,1.01,1.01)

    print('new mins: ')
    print(object.xmin, object.ymin, object.zmin)
    print('new maxs: ')
    print(object.xmax, object.ymax, object.zmax)

    print(object.tsdf[1:,1:,1].sum())

if __name__ == "__main__":
    test()