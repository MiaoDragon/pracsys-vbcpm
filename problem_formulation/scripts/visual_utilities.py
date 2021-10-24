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
    pcd_points = np.concatenate([voxel_x, voxel_y, voxel_z], axis=1)
    pcd.points = o3d.utility.Vector3dVector(pcd_points)
    colors = np.zeros(pcd_points.shape)
    colors[:,0] = color[0]
    colors[:,1] = color[1]
    colors[:,2] = color[2]
    pcd.colors = o3d.utility.Vector3dVector(colors)

    voxel = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, 1.)
    return voxel