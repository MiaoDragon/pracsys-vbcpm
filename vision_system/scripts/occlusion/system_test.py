"""
construct a PyBullet environment where we
1. initialize the camera and the object
2. update while rotating or translating the object with a rigid transformation
for several times
3. visualize the TSDF
4. obtain the information gain at certain angles
"""
# from vision_system.scripts.tsdf.naive_vision_system import pcd_from_depth
import pybullet as p
import rospkg

bullet_id = p.connect(p.GUI)

import json
import os
import cv2
import time
import numpy as np
import transformations as tf

def prob_config_parser(fname):
    # * parse the problem configuration file
    """
    format: 
        {'robot': {'pose': pose, 'urdf': urdf},
         'table': {'pose': pose, 'urdf': urdf},
         'objects': [{'pose': pose, 'urdf': urdf}],
         'camera': {'pose': pose, 'urdf': urdf},
         'placement': [(x1,y1),(x2,y2),(x3,y3),(x4,y4)]}
    """
    f = open(fname, 'r')
    data = json.load(f)
    return data


rp = rospkg.RosPack()
package_path = rp.get_path('vbcpm_execution_system')

prob_config_dict = prob_config_parser(os.path.join(package_path, 'data/configs/prob1.json'))


# table
table_pos = prob_config_dict['table']['pose']['pos']
table_ori = prob_config_dict['table']['pose']['ori']
table_id = p.loadURDF(os.path.join(package_path, prob_config_dict['table']['urdf']),table_pos, table_ori, useFixedBase=True)

cam_pos = np.array([0.35, 0., 1.00])
look_at = np.array([1.35, 0., 0.58])
up_vec = np.array([1.00-0.58, 0., 1.35-0.35])

view_mat = p.computeViewMatrix(
    cameraEyePosition=cam_pos,
    cameraTargetPosition=look_at,
    cameraUpVector=up_vec
)


L = look_at - cam_pos
L = L / np.linalg.norm(L)
s = np.cross(L, up_vec)

s = s / np.linalg.norm(s)
u_prime = np.cross(s, L)
print('u_prime: ', u_prime)
print('up_vector: ', up_vec/np.linalg.norm(up_vec))

# p.addUserDebugLine(lineFromXYZ=cam_pos, lineToXYZ=cam_pos+s, lineColorRGB=[255,0,0])
# p.addUserDebugLine(lineFromXYZ=cam_pos, lineToXYZ=cam_pos+u_prime, lineColorRGB=[0,255,0])
# p.addUserDebugLine(lineFromXYZ=cam_pos, lineToXYZ=cam_pos-L, lineColorRGB=[0,0,255])

# transformation matrix: rotation
rot_mat = np.array([s, -u_prime, L])#.T  # three rows: Right, Up, Forward (column-major form). After transpose, row-major form
# https://www.scratchapixel.com/lessons/mathematics-physics-for-computer-graphics/lookat-function
tran_mat = -rot_mat.dot(cam_pos)

T_mat = np.eye(4)
T_mat[:3,:3] = rot_mat
T_mat[:3,3] = tran_mat

T_mat = tf.inverse_matrix(T_mat)
# assert False

# https://stackoverflow.com/questions/60430958/understanding-the-view-and-projection-matrix-from-pybullet
# https://stackoverflow.com/questions/39992968/how-to-calculate-field-of-view-of-the-camera-from-camera-intrinsic-matrix
# https://github.com/bulletphysics/bullet3/blob/master/examples/SharedMemory/PhysicsClientC_API.cpp#L4372
fov = 90
near = 0.01
far = 1.2
proj_mat = p.computeProjectionMatrixFOV(
    fov=fov,
    aspect=1,
    nearVal=near,
    farVal=far
)

# objects
objs = []
for obj_dict in prob_config_dict['objects']:
    obj_i_c = p.createCollisionShape(shapeType=p.GEOM_MESH, meshScale=obj_dict['scale'], \
                                    fileName=os.path.join(package_path, obj_dict['collision_mesh']))
    obj_i_v = p.createVisualShape(shapeType=p.GEOM_MESH, meshScale=obj_dict['scale'], \
                                    fileName=os.path.join(package_path, obj_dict['visual_mesh']))
    obj_i = p.createMultiBody(baseCollisionShapeIndex=obj_i_c, baseVisualShapeIndex=obj_i_v, \
                    basePosition=obj_dict['pose']['pos'], baseOrientation=obj_dict['pose']['ori'],
                    baseMass=obj_dict['mass'])
    abmin, abmax = p.getAABB(obj_i)
    print('aabbmin: ', abmin)
    print('aabbmax: ', abmax)
    # assert False
    objs.append(obj_i)
    break
abmin, abmax = p.getAABB(table_id)
print('aabbmin: ', abmin)
print('aabbmax: ', abmax)

sim_time = 1/240.

# * initialize the scene
# * publish camera info

width, height, rgb_img, depth_img, seg_img = p.getCameraImage(
    width=128,
    height=128,
    viewMatrix=view_mat,
    projectionMatrix=proj_mat)
# cv2.imshow('camera_rgb', rgb_img)
depth_img = far * near / (far-(far-near)*depth_img)
print('depth image: ')
print(depth_img[::40,::40])
print(depth_img.max())
depth_img[depth_img>=far] = 0.
depth_img[depth_img<=near]=0.


# from naive_vision_system import *

table_h = 0.62  # Found out that the point cloud is tilted, and need more height than usual to filter out the table
focal = 128 / np.tan(fov * np.pi/180 / 2)/2
# focal = 320
print('focal:')
print(focal)
cam_intrinsics = [[focal, 0, 64],
                  [0, focal, 64],
                  [0, 0, 1.]]
# cam_extrinsics = [[-0.0182505, -0.724286, 0.689259, 0.329174],[-0.999453, 0.0322427, 0.00741728, -0.036492],[-0.0275958, -0.688746, -0.724478, 1.24839],[0.0, 0.0, 0.0, 1.0]]
cam_intrinsics = np.array(cam_intrinsics)
cam_extrinsics = np.array(T_mat)

# p.addUserDebugLine(lineFromXYZ=cam_pos, lineToXYZ=cam_pos+T_mat[:3,0], lineColorRGB=[255,0,0])
# p.addUserDebugLine(lineFromXYZ=cam_pos, lineToXYZ=cam_pos+T_mat[:3,1], lineColorRGB=[0,255,0])
# p.addUserDebugLine(lineFromXYZ=cam_pos, lineToXYZ=cam_pos+T_mat[:3,2], lineColorRGB=[0,0,255])


p.stepSimulation()
time.sleep(1/240)


width, height, rgb_img, depth_img, seg_img = p.getCameraImage(
    width=128,
    height=128,
    viewMatrix=view_mat,
    projectionMatrix=proj_mat)

depth_img = far * near / (far-(far-near)*depth_img)
depth_img[depth_img>=far] = 0.
depth_img[depth_img<=near]=0.

import cam_utilities

# occ_pcd = pcd_from_occlusion(cam_extrinsics, cam_extrinsics, pcd, table_h)
# total_pcd = np.concatenate([pcd, occ_pcd], axis=0)
# pcd = total_pcd

# vis_pcd, _ = cam_utilities.pcd_from_depth(cam_intrinsics, cam_extrinsics, depth_img, rgb_img)
# for i in range(0,len(vis_pcd),10):
#     p.addUserDebugLine(lineFromXYZ=T_mat[:3,3], lineToXYZ=vis_pcd[i], lineColorRGB=[255,255,255])

# visualize the initial TSDF
from voxel_grid_representation import Occlusion
import open3d as o3d
import skimage.measure as measure

occlusion = Occlusion(abmin[0], abmin[1], abmax[2]-0.05, np.array([1.,0,0]), np.array([0.,1,0]), np.array([0.,0,1]), np.array([0.01,0.01,0.01]), 100, 100, 20)
cam_center = T_mat[:3,3]
occupied_voxel, occluded_voxel = occlusion.observe_model_based(cam_center, cam_intrinsics, cam_extrinsics, depth_img, rgb_img, None)

pcd, _ = cam_utilities.pcd_from_depth(cam_intrinsics, cam_extrinsics, depth_img, rgb_img)
print('pcd shape: ', pcd.shape)
pcd = pcd[pcd[...,2]>table_h-0.1]
print('pcd shape: ', pcd.shape)
pcd = np.concatenate([pcd, np.ones(pcd.shape[0]).reshape((-1,1))],axis=1)
# transform the ray vector into the voxel_grid space
# notice that we have to divide by the resolution vector
pcd = np.linalg.inv(occlusion.T).dot(pcd.T).T
pcd = pcd[:,:3] / occlusion.resol


# vol_box = o3d.geometry.OrientedBoundingBox()
# vol_box.center = vol_bnds.mean(1)
# vol_box.extent = vol_bnds[:, 1] - vol_bnds[:, 0]
voxel_pcd = o3d.geometry.PointCloud()
print((occluded_voxel==1).sum())
voxel_x = occlusion.voxel_x[occluded_voxel==1].reshape(-1,1)
voxel_y = occlusion.voxel_y[occluded_voxel==1].reshape(-1,1)
voxel_z = occlusion.voxel_z[occluded_voxel==1].reshape(-1,1)

voxel_pcd.points = o3d.utility.Vector3dVector(np.concatenate([voxel_x, voxel_y, voxel_z], axis=1))

voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(voxel_pcd, 1.)

pcd_pcd = o3d.geometry.PointCloud()
pcd_pcd.points = o3d.utility.Vector3dVector(pcd)
colors = np.zeros(pcd.shape)
colors[:,0] = 1.
pcd_pcd.colors = o3d.utility.Vector3dVector(colors)


total_voxel_pcd = o3d.geometry.PointCloud()
total_voxel_x = occlusion.voxel_x.reshape(-1,1)
total_voxel_y = occlusion.voxel_y.reshape(-1,1)
total_voxel_z = occlusion.voxel_z.reshape(-1,1)
total_voxel_pcd.points = o3d.utility.Vector3dVector(np.concatenate([total_voxel_x, total_voxel_y, total_voxel_z], axis=1))
total_voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(total_voxel_pcd, 1.)

o3d.visualization.draw_geometries([voxel_grid, pcd_pcd])

input('Enter for next...')
# abmin, abmax


time.sleep(100000.)
# * move the object by a rigid transformation, and update the TSDF

# * obtain uncertainty image of the object when it's at certain transformation3
