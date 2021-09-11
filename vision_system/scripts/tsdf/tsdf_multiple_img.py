import numpy as np
import transformations as tf
from PIL import Image
import open3d as o3d
import skimage.measure as measure

from icp import icp

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
    test_i = 100
    test_j = 100
    test_k = 1
    print('for (%d, %d, %d): ' % (test_i, test_j, test_k))
    print('xs: ', xs[test_i,test_j,test_k])
    print('ys: ', ys[test_i,test_j,test_k])
    print('zs: ', zs[test_i,test_j,test_k])
    print('image index: ', img_pts[test_i,test_j,test_k])
    img_pts = np.round(img_pts).astype(int)
    return img_pts, xs, ys, zs # ... * 2

def cam_indices_val_mask(img_pts, img):
    # input: ... * 2
    # output: a vector of size N, that contains boolean values. True if inside boundary; False otherwise
    # print(img_pts)
    mask = (img_pts[...,0] >= 0) * (img_pts[...,0] < img.shape[0]) * (img_pts[...,1] >= 0) * (img_pts[...,1] < img.shape[1])
    return mask  # ...

def naive_tsdf_grid_diff(cam_intrinsics, cam_extrinsics, depth_img, color_img, grid):
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
    grid_dis = np.array([0.02,0.02,0.02])  # units: meter
    grid_n = [201,201,201]  # centered around (0,0,0)
    # grid_i -> value: (grid_i - (grid_n-1)/2) * grid_dis
    dist_lim = 0.08
    factor = 5000.0
    depth_img = np.array(depth_img)
    depth_img = depth_img / factor

    # * compute the distance difference for update
    grid_indices_i, grid_indices_j, grid_indices_k = np.indices(grid.shape)

    grid_x, grid_y, grid_z = grid_to_world(grid_indices_i, grid_indices_j, grid_indices_k, grid_n, grid_dis)

    grid_img_pts, grid_cx, grid_cy, grid_cz = world_to_cam(grid_x, grid_y, grid_z, cam_intrinsics, cam_extrinsics)
    val_mask = cam_indices_val_mask(grid_img_pts, depth_img)
    grid_img_pts[~val_mask] = np.array([0,0])
    depth_grid = depth_img[grid_img_pts[...,0], grid_img_pts[...,1]]

    # Notice that we are using the grid location in the *camera frame*, rather than the world frame.
    # This makes sense because the depth image is also relative to the camera
    grid_dif = depth_grid - grid_cz
    
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
    factor = 5000.0

    img_i, img_j = np.indices(depth_img.shape).astype(float)
    depth_img = np.array(depth_img) / factor
    fx = cam_intrinsics[0][0]
    fy = cam_intrinsics[1][1]
    cx = cam_intrinsics[0][2]
    cy = cam_intrinsics[1][2]
    img_x = (img_j - cx) / fx * depth_img
    img_y = (img_i - cy) / fy * depth_img
    # extrinsics: {world} T {cam}
    img_pts = np.transpose(np.array([img_x, img_y, depth_img, np.ones(depth_img.shape)]), axes=(1,2,0))
    img_pts = img_pts[..., np.newaxis]
    world_pts = cam_extrinsics.dot(img_pts)
    print('world_pts shape:')
    print(world_pts.shape)
    world_pts = np.transpose(world_pts, axes=(1,2,0,3))
    world_pts = world_pts.reshape(-1,4)
    world_pts = world_pts[...,:3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(world_pts)
    o3d.visualization.draw_geometries([pcd])

    return world_pts

def naive_tsdf_multiple(cam_intrinsics, depth_imgs, color_imgs, cam_trans):
    # reconstruct using the provided information
    # cam_intrinsics: 3 x 4
    # map from world to image: p_i = K p
    fx = cam_intrinsics[0][0]
    fy = cam_intrinsics[1][1]
    cx = cam_intrinsics[0][2]
    cy = cam_intrinsics[1][2]

    # * parameters
    grid_dis = np.array([0.02,0.02,0.02])  # units: meter
    grid_n = [201,201,201]  # centered around (0,0,0)
    # grid_i -> value: (grid_i - (grid_n-1)/2) * grid_dis
    dist_lim = 0.05
    factor = 5000.0

    # * initialization
    grid = np.ones(grid_n)
    color_grid = np.zeros(list(grid_n)+[3])
    grid_weight = np.zeros(list(grid_n))#+[1])
    weight = np.ones(list(grid_n))#+[1])
    
    # * use the first cam transformation as world frame
    # * future relative transformation will denote {world}T{cam}
    home_tran = cam_trans[0]
    current_tran = np.eye(4)  # {world}T{cam0}
    pcds = []
    for i in range(len(depth_imgs)):
        # * compute relative transformation
        # given: {measure} T {cam0}, {measure} T {cam1}
        # relative: {cam0} T {cam1} = left inverse, dot right
        if i == 0:
            cam_extrinsics = np.eye(4)
            pcd = pcd_from_depth(cam_intrinsics, cam_extrinsics, depth_imgs[i], color_imgs[i])

        else:
            rel_tran = tf.inverse_matrix(cam_trans[i-1]).dot(cam_trans[i])
            # {world} T {cam1} = {world} T {cam0} {cam0} T {cam1}
            cam_extrinsics = current_tran.dot(rel_tran)
            # compute the delta transformation to handle noise
            # Warn: currently doesn't seem to work well
            prev_pcd = pcd_from_depth(cam_intrinsics, current_tran, depth_imgs[i-1], color_imgs[i-1])
            pcd = pcd_from_depth(cam_intrinsics, cam_extrinsics, depth_imgs[i], color_imgs[i])
            dT, distances, num_iter = icp(pcd[::10], prev_pcd[::10], max_iterations=20, tolerance=0.001)
            print('distance: ', distances.max())
            cam_extrinsics = dT.dot(cam_extrinsics)
            print('transformation: ')
            print(dT)

        current_tran = cam_extrinsics
        color_dif, grid_dif, val_mask, depth_mask, n_dist_mask, dist_mask = \
            naive_tsdf_grid_diff(cam_intrinsics, cam_extrinsics, depth_imgs[i], color_imgs[i], grid)

        grid_mask = val_mask * depth_mask * n_dist_mask * dist_mask
        print('grid_weight shape')
        print(grid_weight[grid_mask].shape)
        print('weight shape')
        print(weight[grid_mask].shape)
        print('grid mask shape')
        print(grid_mask.shape)
        print('color grid shape')
        print(color_grid[grid_mask].shape)
        print('color_dif shape')
        print(color_dif[grid_mask].shape)
        print('grid shape')
        print(grid[grid_mask].shape)
        print((grid[grid_mask] * grid_weight[grid_mask]).shape)
        # * add new info to tsdf grid
        grid[grid_mask] = (grid[grid_mask] * grid_weight[grid_mask] + grid_dif[grid_mask] * weight[grid_mask]) / (grid_weight[grid_mask] + weight[grid_mask])
        new_grid_mask = (grid_weight == 0) * (grid_mask)
        color_grid[new_grid_mask,0] = (color_grid[new_grid_mask,0] * grid_weight[new_grid_mask] + color_dif[new_grid_mask,0] * weight[new_grid_mask]) / (grid_weight[new_grid_mask] + weight[new_grid_mask])
        color_grid[new_grid_mask,1] = (color_grid[new_grid_mask,1] * grid_weight[new_grid_mask] + color_dif[new_grid_mask,1] * weight[new_grid_mask]) / (grid_weight[new_grid_mask] + weight[new_grid_mask])
        color_grid[new_grid_mask,2] = (color_grid[new_grid_mask,2] * grid_weight[new_grid_mask] + color_dif[new_grid_mask,2] * weight[new_grid_mask]) / (grid_weight[new_grid_mask] + weight[new_grid_mask])

        grid_weight[grid_mask] = grid_weight[grid_mask] + weight[grid_mask]

        # visualize the surface
        verts = measure.marching_cubes_lewiner(grid.reshape(grid_n), level=0, mask=(weight>0))[0]
        verts_ind = np.round(verts).astype(int)
        # convert from grid index to xyz
        verts[:,0] = (verts[:,0] - (grid_n[0]-1)/2) * grid_dis[0]
        verts[:,1] = (verts[:,1] - (grid_n[1]-1)/2) * grid_dis[1]
        verts[:,2] = verts[:,2] * grid_dis[2]
        colors = []
        for k in range(len(verts)):
            colors.append(color_grid[verts_ind[k,0],verts_ind[k,1],verts_ind[k,2]]/255.)

        if i % 100 == 0:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(verts)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            pcds.append(pcd)
            o3d.visualization.draw_geometries([pcd])
        
            input('Enter for next...')


def parse_files(num=400):
    # parse the file to get the depth images, color images, transformations
    import os
    folder = 'rgbd_dataset_freiburg1_xyz/'
    depth_fname = 'combined.txt'
    depth_file = open(folder+depth_fname, 'r')
    content = depth_file.readlines()[3:][:num]
    depth_content = content
    depth_fnames = [di.strip().split()[1] for di in depth_content]

    depth_imgs = []
    for i in range(len(depth_fnames)):
        depth_img = np.array(Image.open(os.path.join(folder, depth_fnames[i])))
        depth_imgs.append(depth_img)
    print(len(depth_imgs))

    # color_fname = 'rgb.txt'
    # color_file = open(folder+color_fname, 'r')
    # color_content = color_file.readlines()[3:][:num]
    color_content = content
    color_fnames = [ci.strip().split()[3] for ci in color_content]
    
    color_imgs = []
    for i in range(len(color_fnames)):
        color_img = np.array(Image.open(os.path.join(folder, color_fnames[i])))
        color_imgs.append(color_img)

    print(len(color_imgs))

    # gt_fname = 'groundtruth.txt'
    # gt_file = open(folder+gt_fname, 'r')
    # gt_content = gt_file.readlines()[3:][:num]
    gt_content = content
    gt_raw_trans = [gi.strip().split()[5:] for gi in gt_content]

    cam_trans = []
    # tx ty tz qx qy qz qw
    for i in range(len(gt_raw_trans)):
        T = tf.translation_matrix(gt_raw_trans[i][:3])
        q = gt_raw_trans[i][3:]  # x y z w
        R = tf.quaternion_matrix([q[3],q[0],q[1],q[2]])
        M = tf.concatenate_matrices(T, R)  # not sure about the order
        cam_trans.append(M)

    return depth_imgs, color_imgs, cam_trans

depth_imgs, color_imgs, cam_trans = parse_files()
cam_intrinsics = [[525.0, 0, 319.5, 0.],
                  [0., 525.0, 239.5, 0.],
                  [0., 0., 1., 0.]]
cam_intrinsics = np.array(cam_intrinsics)
naive_tsdf_multiple(cam_intrinsics, depth_imgs, color_imgs, cam_trans)
