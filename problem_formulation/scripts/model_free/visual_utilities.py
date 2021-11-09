import open3d as o3d
import numpy as np
def visualize_pcd(pcd, color):
    pcd_pcd = o3d.geometry.PointCloud()
    pcd_pcd.points = o3d.utility.Vector3dVector(pcd)
    colors = np.zeros(pcd.shape)
    colors[:,0] = color[0]
    colors[:,1] = color[1]
    colors[:,2] = color[2]
    pcd_pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd_pcd

def visualize_voxel(voxel_x, voxel_y, voxel_z, filter, color):
    pcd = o3d.geometry.PointCloud()
    voxel_x = voxel_x[filter].reshape(-1,1)
    voxel_y = voxel_y[filter].reshape(-1,1)
    voxel_z = voxel_z[filter].reshape(-1,1)
    pcd_points = np.concatenate([voxel_x+0.5, voxel_y+0.5, voxel_z+0.5], axis=1)
    pcd.points = o3d.utility.Vector3dVector(pcd_points)
    colors = np.zeros(pcd_points.shape)
    colors[:,0] = color[0]
    colors[:,1] = color[1]
    colors[:,2] = color[2]
    pcd.colors = o3d.utility.Vector3dVector(colors)

    min_bound = [voxel_x.min(),voxel_y.min(),voxel_z.min()]
    min_bound = np.array(min_bound)
    max_bound = [voxel_x.max(), voxel_y.max(), voxel_z.max()]
    max_bound = np.array(max_bound)+1.0
    voxel = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(pcd, 1., min_bound, max_bound)
    print('min_bound: ', voxel.get_min_bound())
    print('max_bound: ', voxel.get_max_bound())


    # bbox = voxel.get_axis_aligned
    return voxel

def visualize_bbox(voxel_x, voxel_y, voxel_z):
    min_bound = [voxel_x.min(),voxel_y.min(),voxel_z.min()]
    min_bound = np.array(min_bound)
    max_bound = [voxel_x.max(), voxel_y.max(), voxel_z.max()]
    max_bound = np.array(max_bound)+1.0

    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    return bbox