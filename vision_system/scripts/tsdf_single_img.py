"""
object reconstruction using TSDF for single image
- Naive Method: fixed # of grids, fixed grid size
- TSDF 2.0: dynamic TSDF, with fixed grid size (need better DS)
- TSDF 3.0: in GPU, C++ (probably)
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# reference: https://github.com/andyzeng/tsdf-fusion-python/blob/master/fusion.py

def grid_to_world(grid_i, grid_j, grid_k, grid_n, grid_dis):
    grid_x = (grid_i - (grid_n[0]-1)/2) * grid_dis[0]
    grid_y = (grid_j - (grid_n[1]-1)/2) * grid_dis[1]
    grid_z = grid_k * grid_dis[2]
    return grid_x, grid_y, grid_z

def world_to_cam(xs, ys, zs, cam_intrinsics):
    # project the world points to the pixel
    # input: ... * 1
    #pts = np.array([xs, ys, zs, np.ones(xs.shape)])
    # pts = np.dstack([xs, ys, zs, np.ones(xs.shape)]).reshape(list(xs.shape)+[4])
    # img_pts = cam_intrinsics.dot(pts.reshape(-1,4).T).T
    # img_pts = img_pts.reshape(list(xs.shape)+[3])
    # print(img_pts[100,100,2,:3])
    # print(img_pts[50,100,2,:3])
    fx = cam_intrinsics[0][0]
    fy = cam_intrinsics[1][1]
    cx = cam_intrinsics[0][2]
    cy = cam_intrinsics[1][2]

    img_is = xs / zs * fx + cx
    img_js = ys / zs * fy + cy
    img_pts = np.transpose(np.array([img_is, img_js]), axes=(1,2,3,0))
    test_i = 100
    test_j = 100
    test_k = 1
    print('for (%d, %d, %d): ' % (test_i, test_j, test_k))
    print('xs: ', xs[test_i,test_j,test_k])
    print('ys: ', ys[test_i,test_j,test_k])
    print('zs: ', zs[test_i,test_j,test_k])
    print('image index: ', img_pts[test_i,test_j,test_k])
    img_pts = np.round(img_pts).astype(int)
    return img_pts # ... * 2

def cam_indices_val_mask(img_pts, img):
    # input: ... * 2
    # output: a vector of size N, that contains boolean values. True if inside boundary; False otherwise
    # print(img_pts)
    mask = (img_pts[...,0] >= 0) * (img_pts[...,0] < img.shape[1]) * (img_pts[...,1] >= 0) * (img_pts[...,1] < img.shape[0])
    return mask  # ...

def naive_tsdf(cam_intrinsics, depth_img, color_img):
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
    dist_lim = 0.5
    factor = 5000.0
    depth_img = np.array(depth_img)
    depth_img = depth_img / factor
    print(depth_img)
    print(depth_img[depth_img!=0].max())

    # * update grids based on image
    # initialize the grids
    color_grids = grid_n+ [3]
    grids = np.ones(grid_n)
    grid_indices_i, grid_indices_j, grid_indices_k = np.indices(grids.shape)

    grid_x, grid_y, grid_z = grid_to_world(grid_indices_i, grid_indices_j, grid_indices_k, grid_n, grid_dis)

    grid_img_pts = world_to_cam(grid_x, grid_y, grid_z, cam_intrinsics)
    val_mask = cam_indices_val_mask(grid_img_pts, depth_img)
    grid_img_pts[~val_mask] = np.array([0,0])
    grid_dif = depth_img[grid_img_pts[...,1], grid_img_pts[...,0]] - grid_z
    depth_grid = depth_img[grid_img_pts[...,1], grid_img_pts[...,0]]

    # obtain valid TSDF grids
    valid_grids_mask = val_mask * (depth_grid>0) * (grid_dif > -dist_lim) * (grid_dif < dist_lim)

    grids[valid_grids_mask] = grid_dif[valid_grids_mask]


    color_grids = color_img[grid_img_pts[...,1], grid_img_pts[...,0]]
    color_grids[~val_mask] = np.array([0.,0.,0.])


    # # extract surface: gradient computation
    # WARNING: this version is not working!
    # grids_signed = (grids >= 0).astype(int)
    # grid_dx = grids_signed[:-1,:-1,:-1] - grids_signed[1:,:-1,:-1]
    # dx_mask = (grid_dx != 0)
    # grid_dy = grids_signed[:-1,:-1,:-1] - grids_signed[:-1,1:,:-1]
    # dy_mask = (grid_dy != 0)
    # grid_dz = grids_signed[:-1,:-1,:-1] - grids_signed[:-1,:-1,1:]
    # dz_mask = (grid_dz != 0)
    # surface_mask = dx_mask + dy_mask + dz_mask
    # surface = [grid_indices_i[:-1,:-1,:-1][surface_mask], grid_indices_j[:-1,:-1,:-1][surface_mask], grid_indices_k[:-1,:-1,:-1][surface_mask]]
    # surface = np.array(surface).T
    # print(surface.shape)
    # verts = surface

    import skimage.measure as measure
    verts = measure.marching_cubes_lewiner(grids, level=0)[0]
    verts_ind = np.round(verts).astype(int)
    # convert from grid index to xyz
    verts[:,0] = (verts[:,0] - (grid_n[0]-1)/2) * grid_dis[0]
    verts[:,1] = (verts[:,1] - (grid_n[1]-1)/2) * grid_dis[1]
    verts[:,2] = verts[:,2] * grid_dis[2]
    colors = []
    for i in range(len(verts)):
        colors.append(color_grids[verts_ind[i,0],verts_ind[i,1],verts_ind[i,2]]/255.)
    # print(colors)
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(verts)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])

def naive_tsdf_bk(cam_intrinsics, depth_img, color_img):
    # reconstruct using the provided information
    # cam_intrinsics: 3 x 4
    # map from world to image: p_i = K p
    fx = cam_intrinsics[0][0]
    fy = cam_intrinsics[1][1]
    cx = cam_intrinsics[0][2]
    cy = cam_intrinsics[1][2]

    # initialize the grids
    grid_dis = 0.02  # units: meter
    grid_n = 101  # centered around (0,0,0)
    # grid_i -> value: (grid_i - (grid_n-1)/2) * grid_dis
    dist_lim = 0.1
    factor = 5000
    depth_img = np.array(depth_img)
    depth_img = depth_img / factor
    # update the grid based on image
    ind_is, ind_js = np.indices(depth_img.shape)
    xs = (ind_is-cx) / fx * depth_img
    ys = (ind_js-cy) / fy * depth_img
    front_xs = xs - xs / depth_img * dist_lim
    back_xs = xs + xs / depth_img * dist_lim
    front_ys = ys - ys / depth_img * dist_lim
    back_ys = ys + ys / depth_img * dist_lim
    front_zs = depth_img - dist_lim
    back_zs = depth_img + dist_lim

    front_grid_is = front_xs / grid_dis + (grid_n-1)/2
    front_grid_js = front_ys / grid_dis + (grid_n-1)/2
    front_grid_ks = front_zs / grid_dis

    back_grid_is = back_xs / grid_dis + (grid_n-1)/2
    back_grid_js = back_ys / grid_dis + (grid_n-1)/2
    back_grid_ks = back_zs / grid_dis


    imin = np.floor(min(front_grid_is[depth_img!=0].min(),back_grid_is[depth_img!=0].min()))
    imax = np.ceil(max(front_grid_is[depth_img!=0].max(),back_grid_is[depth_img!=0].max()))
    jmin = np.floor(min(front_grid_js[depth_img!=0].min(),back_grid_js[depth_img!=0].min()))
    jmax = np.ceil(max(front_grid_js[depth_img!=0].max(),back_grid_js[depth_img!=0].max()))
    kmin = np.floor(min(front_grid_ks[depth_img!=0].min(),back_grid_ks[depth_img!=0].min()))
    kmax = np.ceil(max(front_grid_ks[depth_img!=0].max(),back_grid_ks[depth_img!=0].max()))

    imin = int(imin)
    imax = int(imax)
    jmin = int(jmin)
    jmax = int(jmax)
    kmin = int(kmin)
    kmax = int(kmax)

    imin = max(0, imin)
    imax = min(imax, grid_n-1)
    jmin = max(0, jmin)
    jmax = min(jmax, grid_n-1)
    kmin = max(0, kmin)
    kmax = min(kmax, grid_n-1)

    print('imin, imax, jmin, jmax, kmin, kmax: ', imin, imax, jmin, jmax, kmin, kmax)
    # algorithm choice:
    # 1. loop over each grid, and find its projection
    #    O(G*G*G)
    # 2. loop over each ray, use the grid to find all grids that it passes through, and computes the min absolute distance for each grid
    #    O(N*M*G)

    grids = np.zeros((grid_n, grid_n, grid_n)) + 1.0
    grid_indices_i, grid_indices_j, grid_indices_k = np.indices(grids.shape)
    grid_xs = (grid_indices_i - (grid_n-1)/2) * grid_dis
    grid_ys = (grid_indices_j - (grid_n-1)/2) * grid_dis
    grid_zs = grid_indices_k * grid_dis

    print('grid_xs: ')
    print(grid_xs)
    print('grid_ys: ')
    print(grid_ys)
    color_grids = np.zeros((grid_n, grid_n, grid_n, 3))
    for i in range(imin, imax+1):
        for j in range(jmin, jmax+1):
            for k in range(kmin, kmax):
                # project the grid center point to the depth image
                # use the depth value to calculate the the SDF (distance between the grid and the object)
                # projection: find the depth of the pixel, and then the distance is just (depth_of_pixel - k * grid_dis)
                x = (i-(grid_n-1)/2) * grid_dis
                y = (j-(grid_n-1)/2) * grid_dis
                z = k*grid_dis
                img_p = cam_intrinsics.dot(np.array([x,y,z,1]))
                img_i = int(np.round(img_p[0]))
                img_j = int(np.round(img_p[1]))
                if img_i < 0 or img_i >= depth_img.shape[0]:
                    continue
                if img_j < 0 or img_j >= depth_img.shape[1]:
                    continue
                grids[i,j,k] = depth_img[img_i,img_j] - z
                color_grids[i,j,k] = color_img[img_i,img_j]
    # extract surface: gradient computation
    # grids_signed = (grids >= 0).astype(int)
    # grid_dx = grids_signed[:-1,:-1,:-1] - grids_signed[1:,:-1,:-1]
    # dx_mask = (grid_dx != 0)
    # grid_dy = grids_signed[:-1,:-1,:-1] - grids_signed[:-1,1:,:-1]
    # dy_mask = (grid_dy != 0)
    # grid_dz = grids_signed[:-1,:-1,:-1] - grids_signed[:-1,:-1,1:]
    # dz_mask = (grid_dz != 0)
    # surface_mask = dx_mask + dy_mask + dz_mask
    # surface = [grid_xs[:-1,:-1,:-1][surface_mask], grid_ys[:-1,:-1,:-1][surface_mask], grid_zs[:-1,:-1,:-1][surface_mask]]
    # surface = np.array(surface).T
    # print(surface.shape)

    import skimage.measure as measure
    verts = measure.marching_cubes_lewiner(grids, level=0)[0]
    verts_ind = np.round(verts).astype(int)
    # convert from grid index to xyz
    verts[:,0] = (verts[:,0] - (grid_n-1)/2) * grid_dis
    verts[:,1] = (verts[:,1] - (grid_n-1)/2) * grid_dis
    verts[:,2] = verts[:,2] * grid_dis
    colors = []
    for i in range(len(verts)):
        colors.append(color_grids[verts_ind[i,0],verts_ind[i,1],verts_ind[i,2]]/255.)
    # print(colors)
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(verts)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])

    # ax = plt.figure().add_subplot(projection='3d')
    # grids = (grids >= -dist_lim) * (grids <= dist_lim)
    # ax.voxels(grids)
    # plt.show()


def generate_test_img():
    color_img = np.zeros((10,10,3))
    depth_img = np.zeros((10,10))
    color_img[3:8,3:8] = np.array([1.,0,0])
    depth_img[3:8,3:8] = .02 * 5000
    np.save('color_test.npy', color_img)
    np.save('depth_test.npy', depth_img)

from PIL import Image
color = np.array(Image.open('color.png'))
depth = np.array(Image.open('depth.png'))
generate_test_img()
# color = np.load('color_test.npy')
# depth = np.load('depth_test.npy')
print(color.shape)
print(depth.shape)
cam_intrinsics = [[525.0, 0, 319.5, 0.],
                  [0., 525.0, 239.5, 0.],
                  [0., 0., 1., 0.]]
cam_intrinsics = np.array(cam_intrinsics)
naive_tsdf(cam_intrinsics, depth, color)