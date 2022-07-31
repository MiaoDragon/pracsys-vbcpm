import pybullet as p
import rospkg

# bullet_id = p.connect(p.GUI)

import json
import os
import cv2
import time
import numpy as np
import transformations as tf
from retrieval_env_simple_3d import Environment
import open3d as o3d
import skimage.measure as measure

from occlusion_3d import Occlusion
from pose_distribution_3d import DiscreteDistribution

import time
import utility
import cam_utilities

"""
Construct environment
"""
rp = rospkg.RosPack()
package_path = rp.get_path('vbcpm_execution_system')
# f = os.path.join(package_path, 'prob1.json')  # problem configuration file (JSON)
f = 'prob1.json'
prob_config_dict = utility.prob_config_parser(f)



env = Environment(prob_config_dict)
depth_img, rgb_img, obj_poses = env.sense()

print('obj_poses: ')
print(obj_poses)

resol = np.array([0.01, 0.01, 0.01])

occlusion = Occlusion(env.workspace_high[0]-env.workspace_low[0], \
                        env.workspace_high[1]-env.workspace_low[1], \
                        env.workspace_high[2]-env.workspace_low[2], \
                        resol, \
                        env.workspace_low[0],
                        env.workspace_low[1], 
                        env.workspace_low[2], \
                        np.array([1.,0,0]), np.array([0,1.,0]), np.array([0,0,1.]))
occluded, obj_pcd_indices = occlusion.get_occlusions(depth_img, rgb_img, env.camera, obj_poses, env.obj_pcds)

# occluded = occlusion.get_total_occlusions_v2(depth_img, rgb_img, env.camera)


# obtain object height
dist = DiscreteDistribution(np.array([env.workspace_high[0]-env.workspace_low[0],
                                      env.workspace_high[1]-env.workspace_low[1],
                                      env.workspace_high[2]-env.workspace_low[2]]),
                            [env.workspace_low[0], env.workspace_low[1], env.workspace_low[2]],
                            [np.array([1.,0,0]), np.array([0,1.,0]), np.array([0,0,1.])],
                            np.array([resol[0],resol[1],5.0*np.pi/180]), resol,
                            env.objs[1], env.obj_heights[1], env.obj_pcds[1])
occupancy, return_pcds = dist.compute_distribution(occluded, 0.9)
# check where the object might locate
occupancy_i, occupancy_j, occupancy_k = np.indices(occupancy.shape).astype(float)
occupancy_i = occupancy_i[occupancy]
occupancy_j = occupancy_j[occupancy]
occupancy_k = occupancy_k[occupancy]
xs = occupancy_i * resol[0] + env.workspace_low[0]
ys = occupancy_j * resol[1] + env.workspace_low[1]
ks = occupancy_k * 5.0 - 180
poses = np.array([xs,ys,ks]).T
print('possible poses: ')
print(poses)

"""
Create visualization
"""

pcd, _ = cam_utilities.pcd_from_depth(env.camera['intrinsics'], env.camera['extrinsics'], depth_img, rgb_img)

pcd = np.concatenate([pcd, np.ones(pcd.shape[0]).reshape((-1,1))],axis=1)
# transform the ray vector into the voxel_grid space
# notice that we have to divide by the resolution vector
pcd = np.linalg.inv(occlusion.transform).dot(pcd.T).T
pcd = pcd[:,:3] / occlusion.resol


# vol_box = o3d.geometry.OrientedBoundingBox()
# vol_box.center = vol_bnds.mean(1)
# vol_box.extent = vol_bnds[:, 1] - vol_bnds[:, 0]
voxel_pcd = o3d.geometry.PointCloud()
print((occluded==1).sum())
voxel_x = occlusion.voxel_x[occluded==1].reshape(-1,1)
voxel_y = occlusion.voxel_y[occluded==1].reshape(-1,1)
voxel_z = occlusion.voxel_z[occluded==1].reshape(-1,1)

voxel_pcd.points = o3d.utility.Vector3dVector(np.concatenate([voxel_x, voxel_y, voxel_z], axis=1))

voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(voxel_pcd, 1.)

pcd_pcd = o3d.geometry.PointCloud()
pcd_pcd.points = o3d.utility.Vector3dVector(pcd)
colors = np.zeros(pcd.shape)
colors[:,0] = 1.
pcd_pcd.colors = o3d.utility.Vector3dVector(colors)



obj_pcd = o3d.geometry.PointCloud()
obj_pcd.points = o3d.utility.Vector3dVector(obj_pcd_indices)
colors = np.zeros(obj_pcd_indices.shape)
# colors[:,0] =
obj_pcd.colors = o3d.utility.Vector3dVector(colors)

obj_voxel = o3d.geometry.VoxelGrid.create_from_point_cloud(obj_pcd, 1)



total_voxel_pcd = o3d.geometry.PointCloud()
total_voxel_x = occlusion.voxel_x.reshape(-1,1)
total_voxel_y = occlusion.voxel_y.reshape(-1,1)
total_voxel_z = occlusion.voxel_z.reshape(-1,1)
total_voxel_pcd.points = o3d.utility.Vector3dVector(np.concatenate([total_voxel_x, total_voxel_y, total_voxel_z], axis=1))
total_voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(total_voxel_pcd, 1.)



# return_pcds


occupancy_pcd = o3d.geometry.PointCloud()
occupancy_pcd.points = o3d.utility.Vector3dVector(return_pcds)
colors = np.zeros(return_pcds.shape)
# colors[:,0] =
occupancy_pcd.colors = o3d.utility.Vector3dVector(colors)

occupancy_voxel = o3d.geometry.VoxelGrid.create_from_point_cloud(occupancy_pcd, 1)



# o3d.visualization.draw_geometries([pcd_pcd, obj_pcd])
# o3d.visualization.draw_geometries([voxel_grid, pcd_pcd])
o3d.visualization.draw_geometries([pcd_pcd, occupancy_voxel])
