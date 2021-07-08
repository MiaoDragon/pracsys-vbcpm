"""
construct a PyBullet environment where we
1. initialize the camera and the object
2. update while rotating or translating the object with a rigid transformation
for several times
3. visualize the TSDF
4. obtain the information gain at certain angles
"""
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
# T_mat = np.block([[rot_mat, np.zeros((3,1))], [tran_mat.reshape(1,3), 1]]).T
# tran_mat = cam_pos
# T_mat = np.block([[rot_mat, np.zeros((3,1))], [np.zeros((1,3)), 1]])
# trans_mat = np.eye(4)
# trans_mat[0,3] = -cam_pos[0]
# trans_mat[1,3] = -cam_pos[1]
# trans_mat[2,3] = -cam_pos[2]

# T_mat = T_mat.dot(trans_mat)

# T_mat = np.block([[rot_mat, np.zeros((3,1))], [tran_mat.reshape(1,3), 1]]).T
# T_mat = np.block([[rot_mat, tran_mat.reshape(3,1)],[np.zeros(3),1]])
T_mat = np.eye(4)
T_mat[:3,:3] = rot_mat
T_mat[:3,3] = tran_mat

print("T_mat shape,:")
print(T_mat.shape)
print('before inverse: ')
print(T_mat)
T_mat = tf.inverse_matrix(T_mat)
print(T_mat)
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
    width=640,
    height=640,
    viewMatrix=view_mat,
    projectionMatrix=proj_mat)
# cv2.imshow('camera_rgb', rgb_img)
depth_img = far * near / (far-(far-near)*depth_img)
print('depth image: ')
print(depth_img[::40,::40])
print(depth_img.max())
depth_img[depth_img>=far] = 0.
depth_img[depth_img<=near]=0.


from naive_vision_system import *

table_h = 0.62  # Found out that the point cloud is tilted, and need more height than usual to filter out the table
focal = 640 / np.tan(fov * np.pi/180 / 2)/2
# focal = 320
print('focal:')
print(focal)
cam_intrinsics = [[focal, 0, 320],
                  [0, focal, 320],
                  [0, 0, 1.]]
# cam_extrinsics = [[-0.0182505, -0.724286, 0.689259, 0.329174],[-0.999453, 0.0322427, 0.00741728, -0.036492],[-0.0275958, -0.688746, -0.724478, 1.24839],[0.0, 0.0, 0.0, 1.0]]
cam_intrinsics = np.array(cam_intrinsics)
cam_extrinsics = np.array(T_mat)

p.addUserDebugLine(lineFromXYZ=cam_pos, lineToXYZ=cam_pos+T_mat[:3,0], lineColorRGB=[255,0,0])
p.addUserDebugLine(lineFromXYZ=cam_pos, lineToXYZ=cam_pos+T_mat[:3,1], lineColorRGB=[0,255,0])
p.addUserDebugLine(lineFromXYZ=cam_pos, lineToXYZ=cam_pos+T_mat[:3,2], lineColorRGB=[0,0,255])


p.stepSimulation()
time.sleep(1/240)


width, height, rgb_img, depth_img, seg_img = p.getCameraImage(
    width=640,
    height=640,
    viewMatrix=view_mat,
    projectionMatrix=proj_mat)

depth_img = far * near / (far-(far-near)*depth_img)
depth_img[depth_img>=far] = 0.
depth_img[depth_img<=near]=0.
pcd = pcd_from_depth(cam_intrinsics, cam_extrinsics, depth_img, rgb_img)

pcd = pcd[pcd[...,2]>table_h]
# occ_pcd = pcd_from_occlusion(cam_extrinsics, cam_extrinsics, pcd, table_h)
# total_pcd = np.concatenate([pcd, occ_pcd], axis=0)
# pcd = total_pcd
for i in range(0,len(pcd),10):
    p.addUserDebugLine(lineFromXYZ=cam_pos, lineToXYZ=pcd[i], lineColorRGB=[255,255,255])



vision_system = VisionSystem(table_h, cam_intrinsics, cam_extrinsics, depth_img, rgb_img)


# visualize the initial TSDF
import open3d as o3d
import skimage.measure as measure



# visualize the surface
verts = measure.marching_cubes_lewiner(vision_system.tsdf.reshape(vision_system.grid_n), level=0, mask=(vision_system.weight>0))[0]
verts_ind = np.round(verts).astype(int)
# convert from grid index to xyz
verts[:,0] = (verts[:,0] - (vision_system.grid_n[0]-1)/2) * vision_system.grid_d[0]
verts[:,1] = (verts[:,1] - (vision_system.grid_n[1]-1)/2) * vision_system.grid_d[1]
verts[:,2] = verts[:,2] * vision_system.grid_d[2]

grid_x = verts[:,0] * vision_system.grid_d[0]
grid_y = verts[:,1] * vision_system.grid_d[1]
grid_z = verts[:,2] * vision_system.grid_d[2]
pts = np.array([grid_x,grid_y,grid_z,np.ones(grid_x.shape)])
pts = np.dot(vision_system.tsdf_origin, pts).T
verts = pts[:,:3]


colors = []
for k in range(len(verts)):
    colors.append(vision_system.color_grid[verts_ind[k,0],verts_ind[k,1],verts_ind[k,2]]/255.)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(verts)
# pcd.colors = o3d.utility.Vector3dVector(colors)
o3d.visualization.draw_geometries([pcd])

input('Enter for next...')



time.sleep(100000.)
# * move the object by a rigid transformation, and update the TSDF

# * obtain uncertainty image of the object when it's at certain transformation3
