"""
Naive implementation of the vision system.
- retrieve the maximum bounding box enclosing the object
- initialize the TSDF representation at beginning
- each time observe the image, and track the TSDF
- given the current TSDF representation, evaluate the uncertainty given a
- transformation of the TSDF (object)
"""

import numpy as np 
import skimage.measure as measure

from icp import icp
import transformations as tf

def grid_to_world(grid_i, grid_j, grid_k, grid_origin, grid_n, grid_dis):
    """
    grid_origin: frame of the grid (0,0,0). 4x4 homogenous transformation matrix
    """
    grid_x = grid_i * grid_dis[0]
    grid_y = grid_j * grid_dis[1]
    grid_z = grid_k * grid_dis[2]
    pts = np.transpose(np.array([grid_x,grid_y,grid_z,np.ones(grid_x.shape)]), axes=(1,2,3,0)).reshape(list(grid_x.shape)+[4,1])
    pts = np.dot(grid_origin, pts).reshape([4]+list(grid_x.shape)).transpose((1,2,3,0))
    grid_x = pts[...,0]
    grid_y = pts[...,1]
    grid_z = pts[...,2]
    return grid_x, grid_y, grid_z

def world_to_cam(xs, ys, zs, cam_intrinsics, cam_extrinsics):
    # project the world points to the pixel
    # input: ... * 1
    # cam_extrinsics: {world}T{cam}
    fx = cam_intrinsics[0][0]
    fy = cam_intrinsics[1][1]
    cx = cam_intrinsics[0][2]
    cy = cam_intrinsics[1][2]

    # transform the points to camera frame
    pts = np.array([xs, ys, zs, np.ones(xs.shape)]).transpose((1,2,3,0)).reshape(list(xs.shape)+[4,1])  # ... x 4 x 1
    print(np.dot(tf.inverse_matrix(cam_extrinsics), pts).reshape([4]+list(xs.shape)).shape)
    pts = np.dot(tf.inverse_matrix(cam_extrinsics), pts).reshape([4]+list(xs.shape)).transpose((1,2,3,0))  # ... x 4
    xs = pts[...,0]
    ys = pts[...,1]
    zs = pts[...,2]

    img_is = xs / zs * fx + cx
    img_js = ys / zs * fy + cy
    img_pts = np.transpose(np.array([img_js, img_is]), axes=(1,2,3,0))  # notice the order
    img_pts = np.round(img_pts).astype(int)
    return img_pts, xs, ys, zs # ... * 2

def cam_indices_val_mask(img_pts, img):
    # input: ... * 2
    # output: a vector of size N, that contains boolean values. True if inside boundary; False otherwise
    # print(img_pts)
    mask = (img_pts[...,0] >= 0) * (img_pts[...,0] < img.shape[0]) * (img_pts[...,1] >= 0) * (img_pts[...,1] < img.shape[1])
    return mask  # ...

def naive_tsdf_grid_diff(cam_intrinsics, cam_extrinsics, depth_img, color_img, grid, grid_origin, grid_n, grid_d, dist_lim):
    """
    compute the distance difference for each grid voxel that is valid
    output: grid, grid_mask
    remark: the input grid here is only used to extract the shape
    """
    # reconstruct using the provided information
    # cam_intrinsics: 3 x 4
    # map from world to image: p_i = K p
    fx = cam_intrinsics[0][0]
    fy = cam_intrinsics[1][1]
    cx = cam_intrinsics[0][2]
    cy = cam_intrinsics[1][2]

    # * parameters
    grid_dis = grid_d
    # grid_i -> value: (grid_i - (grid_n-1)/2) * grid_dis
    # factor = 5000.0
    depth_img = np.array(depth_img)

    # * compute the distance difference for update
    grid_indices_i, grid_indices_j, grid_indices_k = np.indices(grid.shape)

    grid_x, grid_y, grid_z = grid_to_world(grid_indices_i, grid_indices_j, grid_indices_k, grid_origin, grid_n, grid_dis)

    grid_img_pts, grid_cx, grid_cy, grid_cz = world_to_cam(grid_x, grid_y, grid_z, cam_intrinsics, cam_extrinsics)
    val_mask = cam_indices_val_mask(grid_img_pts, depth_img)
    grid_img_pts[~val_mask] = np.array([0,0])
    depth_grid = depth_img[grid_img_pts[...,0], grid_img_pts[...,1]]
    grid_dif = depth_grid - grid_z
    
    # truncate the distance: we cap the positive dist difference. For negative value, we let it grow and use mask to take care of
    grid_dif[grid_dif > dist_lim] = dist_lim
    #grid_dif[grid_dif < -dist_lim]
    # obtain valid TSDF grids
    # valid_grids_mask = val_mask * (depth_grid>0) * (grid_dif > -dist_lim) * (grid_dif < dist_lim)
    # depth_grid > 0 means too faraway, we assume they're empty
    # grid_dif[depth_grid==0] = 1.
    # valid_grids_mask = val_mask * (grid_dif >= -dist_lim) # allow only where information is in front of the surface (we can't see occlusion)


    color_dif = color_img[grid_img_pts[...,0], grid_img_pts[...,1]]
    return color_dif, grid_dif, val_mask, depth_grid>0, (grid_dif >= -dist_lim), (grid_dif <= dist_lim)

def pcd_from_depth(cam_intrinsics, cam_extrinsics, depth_img, color_img):
    # extract the point cloud (world position) from depth img
    factor = 1.0
    img_i, img_j = np.indices(depth_img.shape).astype(float)
    depth_img = np.array(depth_img) / factor
    fx = cam_intrinsics[0][0]
    fy = cam_intrinsics[1][1]
    cx = cam_intrinsics[0][2]
    cy = cam_intrinsics[1][2]
    img_x = (img_j - cx) / fx * depth_img
    img_y = (img_i - cy) / fy * depth_img
    # ** seems the x and y need to be reverse sign. Probably because the pinhole creates reverse image?
    # extrinsics: {world} T {cam}
    img_pts = np.transpose(np.array([img_x, img_y, depth_img, np.ones(depth_img.shape)]), axes=(1,2,0))
    # ** Notice here we should use the negative of the depth image value, since the "forward" vector points backward
    # https://stackoverflow.com/questions/59128880/getting-world-coordinates-from-opengl-depth-buffer
    img_pts = img_pts[..., np.newaxis]
    world_pts = cam_extrinsics.dot(img_pts)
    valid_mask = (depth_img > 0.001)
    # print('mask shape: ')
    # print(valid_mask.shape)
    print('before mask... world pts shape: ')
    print(world_pts.shape)
    world_pts = np.transpose(world_pts, axes=(1,2,0,3))
    world_pts = world_pts[valid_mask]

    # import cv2
    # import matplotlib.pyplot as plt
    # # Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
    # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # # Set flags (Just to avoid line break in the code)
    # flags = cv2.KMEANS_RANDOM_CENTERS
    # # Apply KMeans
    # compactness,labels,centers = cv2.kmeans(depth_img.flatten(),5,None,criteria,10,flags)
    # labels = labels.reshape(depth_img.shape)
    # # labels = np.transpose(np.array([labels, labels, labels]), axes=(1,2,0))
    # color_depth_img = np.zeros(list(labels.shape) + [3])
    # color_depth_img[labels==1] = [255,255,255]
    # color_depth_img[labels==2] = [255,0,0]
    # color_depth_img[labels==3] = [0,255,0]
    # color_depth_img[labels==4] = [0,0,255]
    # print('green points:')
    # print(depth_img[labels==3])
    # print('blue points:')

    # print(depth_img[labels==4])


    # color_depth_img[labels==5] = [255,0,0]



    # import matplotlib.pyplot as plt
    # plt.imshow(color_depth_img)
    # plt.show()

    # # world_pts = world_pts[(labels!=1) * (labels!=0)]
    world_pts = world_pts.reshape(-1,4)
    world_pts = world_pts[...,:3]
    # world_pts = world_pts[world_pts[...,2]>0.72]
    # print(world_pts[...,2].min())
    # print('world pts shape: ')
    # print(world_pts.shape)
    # print('pcd: ')
    # # print(world_pts[::100,2])
    # plt.boxplot(world_pts[:,2])
    # plt.show()
    # import open3d as o3d
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(world_pts)
    # o3d.visualization.draw_geometries([pcd])

    return world_pts

def pcd_from_occlusion(cam_intrinsics, cam_extrinsics, pcd, table_h):
    """
    extract the points at the intersection of camera ray and the table
    we assume that the camera is high enough so that the intersection always exists
    pcd frame: world
    """
    cam_pos = cam_extrinsics[:3,3]  # camera pos in the world frame
    # (cam_pos + (pcd-cam_pos) * lambda)_z = table_h
    vec = pcd - cam_pos
    print('height difference: ', table_h-cam_pos[2])
    print(np.abs(vec[:,2]).min())
    factor = (table_h - cam_pos[2]) / vec[:,2]
    occluded_points = cam_pos + vec * factor.reshape(-1,1)
    return occluded_points


def naive_uncertainty(cam_intrinsics, cam_extrinsics, tsdf, weight_ratio, grid_origin, grid_n, grid_d, window_shape=(400,600)):
    # * obtain occupancy grid: 1 when tsdf>0 and weight is large
    small_ratio = 0.1
    grid_n = tsdf.shape
    print(tsdf>=0)
    print(weight_ratio>small_ratio)
    occupancy_grid = ((tsdf >= 0) * (weight_ratio > small_ratio))
    # occupancy_grid[~occupancy_grid] = -1
    print(occupancy_grid)
    # * obtain indices matrix
    grid_i, grid_j, grid_k = np.indices(tsdf.shape)
    grid_x, grid_y, grid_z = grid_to_world(grid_i, grid_j, grid_k, grid_origin, grid_n, grid_d)
    img_pts, depths = world_to_cam(grid_x, grid_y, grid_z, cam_intrinsics, cam_extrinsics)

    cert_img = np.zeros(window_shape)
    img_val_mask = cam_indices_val_mask(img_pts, cert_img)

    # * zero-crossing on the occupancy_grid
    verts = measure.marching_cubes_lewiner(occupancy_grid.reshape(grid_n), level=0)[0]
    verts_ind = np.round(verts).astype(int)  # N x 3    

    # * obtain (img_i, img_j) -> min_d (grid_i, grid_j, grid_k, depth_to_cam, tsdf, weight_ratio)
    verts_img_pts = img_pts[verts_ind[:,0],verts_ind[:,1],verts_ind[:,2]]  # N * 2
    verts_depths = depths[verts_ind[:,0],verts_ind[:,1],verts_ind[:,2]]
    verts_tsdf = tsdf[verts_ind[:,0],verts_ind[:,1],verts_ind[:,2]]
    verts_weight_ratio = weight_ratio[verts_ind[:,0],verts_ind[:,1],verts_ind[:,2]]
    verts_img_val_mask = img_val_mask[verts_ind[:,0],verts_ind[:,1],verts_ind[:,2]] # N
    # extract the valid ones
    verts_ind = verts_ind[verts_img_val_mask]
    verts_img_pts = verts_img_pts[verts_img_val_mask] # k x 2
    verts_depths = verts_depths[verts_img_val_mask]
    verts_tsdf = verts_tsdf[verts_img_val_mask]
    verts_weight_ratio = verts_weight_ratio[verts_img_val_mask]

    window_min_depth = np.zeros(window_shape) + max(grid_dis) * max(grid_n)
    window_min_ind = -np.ones(window_shape).astype(int)
    window_min_grid_id = -np.zeros(list(window_shape)+[3])
    for i in range(len(verts_img_pts)):
        # find the min depth
        if verts_depths[i] < window_min_depth[verts_img_pts[i][0],verts_img_pts[i][1]]:
            window_min_depth[verts_img_pts[i][0],verts_img_pts[i][1]] = verts_depths[i]
            window_min_ind[verts_img_pts[i][0],verts_img_pts[i][1]] = i
            # uncertainty: weight_ratio small
            cert_img[verts_img_pts[i][0],verts_img_pts[i][1]] = float(verts_weight_ratio[i] < small_ratio)
            window_min_grid_id[verts_img_pts[i][0],verts_img_pts[i][1]] = verts_ind[i]

    return cert_img, window_min_grid_id  # visualization of the uncertainty img




class VisionSystem():
    def __init__(self, table_h, cam_intrinsics, cam_extrinsics, depth_img, color_img):
        """
        initialize the bounding box that includes the sensed part, and then occluded part
        initialize the TSDF based on the first observation
        """
        self.table_h = table_h
        self.cam_intrinsics = cam_intrinsics
        self.cam_extrinsics = cam_extrinsics
        # parameteres
        self.grid_d = [0.01, 0.01, 0.01]
        self.n_update = 0
        """
        TSDF is defined to be the transformation/frame of (0,0,0) relative to world, and the distance of each
        grid (dx,dy,dz), the number of grids (nx,ny,nz)
        # notice that we can then determine the depth of the grid by transformation to camera frame
        Bounding box is defined to be [[min_x,max_x],[min_y,max_y],[min_z,max_z]]
        """
        pcd = pcd_from_depth(cam_intrinsics, cam_extrinsics, depth_img, color_img)
        # mask out table
        pcd = pcd[pcd[...,2]>table_h]

        self.prev_pcd = pcd  # for tracking
        # obtain the occluded parts from the depth image by intersecting with the table (z=table_h)
        occ_pcd = pcd_from_occlusion(cam_extrinsics, cam_extrinsics, pcd, table_h)
        total_pcd = np.concatenate([pcd, occ_pcd], axis=0)
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.prev_pcd)
        o3d.visualization.draw_geometries([pcd])


        # extract bounding box from the point cloud
        xmin = total_pcd[:,0].min()
        xmax = total_pcd[:,0].max()
        ymin = total_pcd[:,1].min()
        ymax = total_pcd[:,1].max()
        zmin = total_pcd[:,2].min()
        zmax = total_pcd[:,2].max()
        print('xmax: ', xmax)
        print('xmin: ', xmin)
        self.grid_n = [int(np.ceil((xmax-xmin) / self.grid_d[0])),
                       int(np.ceil((ymax-ymin) / self.grid_d[1])),
                       int(np.ceil((zmax-zmin) / self.grid_d[2]))]
        self.tsdf_origin = np.eye(4)
        self.tsdf_origin[:3,3] = np.array([xmin,ymin,zmin])  # grid (0,0,0) in the world frame, {world} T {grid}
        self.dist_lim = 0.02
        # we add a virtual frame representing the origin of the tsdf, and estimate the transformation by tracking
        # (ICP or Particle Filter (Shiyang))
        self.tsdf = np.ones(self.grid_n)
        self.weight = np.zeros(self.grid_n)
        self.d_weight = np.ones(self.grid_n)
        self.color_grid = np.zeros(list(self.grid_n)+[3])
        color_dif, tsdf_dif, val_mask, depth_mask, n_dist_mask, dist_mask = \
            naive_tsdf_grid_diff(self.cam_intrinsics, self.cam_extrinsics, depth_img, color_img, \
                                self.tsdf, self.tsdf_origin, self.grid_n, self.grid_d, self.dist_lim)
        self.update_grid(color_dif, tsdf_dif, val_mask, depth_mask, n_dist_mask, dist_mask)
        
    def update_grid(self, color_dif, tsdf_dif, val_mask, depth_mask, n_dist_mask, dist_mask):
        tsdf_mask = val_mask * depth_mask * n_dist_mask * dist_mask
        # print('grid_weight shape')
        # print(grid_weight[grid_mask].shape)
        # print('weight shape')
        # print(weight[grid_mask].shape)
        # print('grid mask shape')
        # print(grid_mask.shape)
        # print('color grid shape')
        # print(color_grid[grid_mask].shape)
        # print('color_dif shape')
        # print(color_dif[grid_mask].shape)
        # print('grid shape')
        # print(grid[grid_mask].shape)
        # print((grid[grid_mask] * grid_weight[grid_mask]).shape)
        # * add new info to tsdf grid
        self.tsdf[tsdf_mask] = (self.tsdf[tsdf_mask] * self.weight[tsdf_mask] + tsdf_dif[tsdf_mask] * self.d_weight[tsdf_mask]) / (self.weight[tsdf_mask] + self.d_weight[tsdf_mask])
        new_grid_mask = (self.weight == 0) * (tsdf_mask)
        self.color_grid[new_grid_mask,0] = (self.color_grid[new_grid_mask,0] * self.weight[new_grid_mask] + color_dif[new_grid_mask,0] * self.d_weight[new_grid_mask]) / (self.weight[new_grid_mask] + self.d_weight[new_grid_mask])
        self.color_grid[new_grid_mask,1] = (self.color_grid[new_grid_mask,1] * self.weight[new_grid_mask] + color_dif[new_grid_mask,1] * self.d_weight[new_grid_mask]) / (self.weight[new_grid_mask] + self.d_weight[new_grid_mask])
        self.color_grid[new_grid_mask,2] = (self.color_grid[new_grid_mask,2] * self.weight[new_grid_mask] + color_dif[new_grid_mask,2] * self.d_weight[new_grid_mask]) / (self.weight[new_grid_mask] + self.d_weight[new_grid_mask])

        self.weight[tsdf_mask] = self.weight[tsdf_mask] + self.d_weight[tsdf_mask]


    def update(self, depth_img, color_img, transform):
        """
        given the transformation prior (which can come from robot movement), track and update the tsdf
        transform: {origin1} T {origin2}
        """
        pcd = pcd_from_depth(self.cam_intrinsics, self.cam_extrinsics, depth_img, color_img)
        pcd = pcd[pcd[...,2]>self.table_h]

        dT, distances, num_iter = icp(pcd[::10], self.prev_pcd[::10], init_pose=transform, max_iterations=20, tolerance=0.001)
        self.tsdf_origin = self.tsdf_origin.dot(dT)  # {world} T {origin1} . {origin1} T {origin2}
        color_dif, tsdf_dif, val_mask, depth_mask, n_dist_mask, dist_mask = \
            naive_tsdf_grid_diff(self.cam_intrinsics, self.cam_extrinsics, depth_img, color_img, \
                                self.tsdf, self.tsdf_origin, self.grid_n, self.grid_d, self.dist_lim)
        self.update_grid(color_dif, tsdf_dif, val_mask, depth_mask, n_dist_mask, dist_mask)
        self.n_update += 1

    def get_belief(self, depth_img, color_img, tsdf_origin):
        """
        get the belief value for a specific object pose
        """
        cert_img, window_min_grid_id = naive_uncertainty(self.cam_intrinsics, self.cam_extrinsics, self.tsdf, self.weight/self.n_update, \
                                                        self.tsdf_origin, self.grid_n, self.grid_d)
        return cert_img