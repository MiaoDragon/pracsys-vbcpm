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


    # bbox = voxel.get_axis_aligned
    return voxel

def visualize_bbox(voxel_x, voxel_y, voxel_z):
    min_bound = [voxel_x.min(),voxel_y.min(),voxel_z.min()]
    min_bound = np.array(min_bound)
    max_bound = [voxel_x.max(), voxel_y.max(), voxel_z.max()]
    max_bound = np.array(max_bound)+1.0

    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    return bbox

def visualize_coordinate_frame_centered(size=1.0, transform=np.eye(4)):
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size).transform(transform)
    return frame

def get_color_picks():
    color_pick = np.zeros((8,3))
    color_pick[0] = np.array([1., 0., 0.])
    color_pick[1] = np.array([0., 1.0, 0.])
    color_pick[2] = np.array([0., 0., 1.])
    color_pick[3] = np.array([252/255, 169/255, 3/255])
    color_pick[4] = np.array([252/255, 3/255, 252/255])
    color_pick[5] = np.array([20/255, 73/255, 82/255])
    color_pick[6] = np.array([22/255, 20/255, 82/255])
    color_pick[7] = np.array([60/255, 73/255, 10/255])
    return color_pick



