"""
implementation of the uncertainty captured by a camera
- naive method:
    simply use boolean values (0/1) to represent the uncertainty
- advanced method (TODO):
    entropy, probability to better quantify
"""
import skimage.measure as measure
import numpy as np
import transformations as tf

def grid_to_world(grid_i, grid_j, grid_k, grid_n, grid_dis):
    grid_x = (grid_i - (grid_n[0]-1)/2) * grid_dis[0]
    grid_y = (grid_j - (grid_n[1]-1)/2) * grid_dis[1]
    grid_z = grid_k * grid_dis[2]
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
    return img_pts, zs # ... * 2
    # Notice we also output the depth-to-camera



def cam_indices_val_mask(img_pts, img):
    # input: ... * 2
    # output: a vector of size N, that contains boolean values. True if inside boundary; False otherwise
    # print(img_pts)
    mask = (img_pts[...,0] >= 0) * (img_pts[...,0] < img.shape[0]) * (img_pts[...,1] >= 0) * (img_pts[...,1] < img.shape[1])
    return mask  # ...


def naive_uncertainty(cam_intrinsics, cam_extrinsics, tsdf, weight_ratio, grid_dis, window_shape=(400,600)):
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
    grid_x, grid_y, grid_z = grid_to_world(grid_i, grid_j, grid_k, grid_n, grid_dis)
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


# test
# uncertainty_img = naive_uncertainty(cam_intrinsics, cam_extrinsics, tsdf, weight_ratio, grid_dis, window_shape=(400,600))
cam_intrinsics = [[1.0, 0, 2., 0.],
                  [0., 1.0, 2., 0.],
                  [0., 0., 1., 0.]]
cam_intrinsics = np.array(cam_intrinsics)
cam_extrinsics = np.eye(4)
window_shape = (5,5)
tsdf = np.ones((5,5,5))
tsdf[2,:,3] = 1.0
tsdf[0,:,3] = -1.0
tsdf[4,:,3] = -1.0
tsdf[0,:,4] = 1.0
tsdf[4,:,4] = 1.0
tsdf[2,:,4] = -1.0
weight_ratio = np.ones((5,5,5))
weight_ratio[2,:,3] = 0.01
weight_ratio[0,:,4] = 0.01
weight_ratio[4,:,4] = 0.01
weight_ratio[0,:,2] = 0.01
grid_dis = [1.0, 1.0, 1.0]
uncertainty_img, min_grid_id = naive_uncertainty(cam_intrinsics, cam_extrinsics, tsdf, weight_ratio, grid_dis, window_shape)
# visualize the nearest
print('min grid index: ')
s = ''
for i in range(len(uncertainty_img)):
    for j in range(len(uncertainty_img[i])):
        s += '(%f,%f,%f) ' % (min_grid_id[i,j,0],min_grid_id[i,j,1],min_grid_id[i,j,2])
    s += '\n'
print(s)
print('uncertainty image:')
print(uncertainty_img)