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

# from icp import icp
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
    # print(np.dot(tf.inverse_matrix(cam_extrinsics), pts).reshape([4]+list(xs.shape)).shape)
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
    img_i, img_j = np.indices(depth_img.shape).astype(float)
    depth_img = np.array(depth_img)
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
    # print('before mask... world pts shape: ')
    # print(world_pts.shape)
    world_pts = np.transpose(world_pts, axes=(1,2,0,3))
    world_pts = world_pts[valid_mask]
    depth_img = depth_img[valid_mask]
    # # world_pts = world_pts[(labels!=1) * (labels!=0)]
    world_pts = world_pts.reshape(-1,4)
    world_pts = world_pts[...,:3]
    depth_img = depth_img.reshape(-1)

    return world_pts, depth_img

def pcd_from_occlusion(cam_intrinsics, cam_extrinsics, pcd, table_h):
    """
    extract the points at the intersection of camera ray and the table
    we assume that the camera is high enough so that the intersection always exists
    pcd frame: world
    """
    cam_pos = cam_extrinsics[:3,3]  # camera pos in the world frame
    # (cam_pos + (pcd-cam_pos) * lambda)_z = table_h
    vec = pcd - cam_pos
    # print('height difference: ', table_h-cam_pos[2])
    # print(np.abs(vec[:,2]).min())
    factor = (table_h - cam_pos[2]) / vec[:,2]
    occluded_points = cam_pos + vec * factor.reshape(-1,1)
    return occluded_points

def ray_intersect(ray_orig, ray_vec, voxel_x, voxel_y, voxel_z):
    # assume we have a standarized voxel grid, compute the intersection mask
    sign_vec = ray_vec < 0
    tmin = (voxel_x + sign_vec[0] - ray_orig[0]) / ray_vec[0]

    tmax = (voxel_x + 1 - sign_vec[0] - ray_orig[0]) / ray_vec[0]

    tymin = (voxel_y + sign_vec[1] - ray_orig[1]) / ray_vec[1]
    tymax = (voxel_y + 1 - sign_vec[1] - ray_orig[1]) / ray_vec[1]
 
    mask = np.ones(voxel_x.shape).astype(bool)
    bool_vec = (tmin > tymax) | (tymin > tmax)
    mask[bool_vec] = False

    tmin[tymin > tmin] = tymin[tymin > tmin]
    tmax[tymax < tmax] = tymax[tymax < tmax]

    tzmin = (voxel_z + sign_vec[2] - ray_orig[2]) / ray_vec[2]
    tzmax = (voxel_z + 1 - sign_vec[2] - ray_orig[2]) / ray_vec[2]

    bool_vec = (tmin > tzmax) | (tzmin > tmax)
    mask[bool_vec] = False

    tmin[tzmin > tmin] = tzmin[tzmin > tmin]
    tmax[tzmax < tmax] = tzmax[tzmax < tmax]

    return mask, tmin, tmax


