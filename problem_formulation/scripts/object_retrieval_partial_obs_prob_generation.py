"""
generate a problem for the object retrieval problem under partial observation

objects are randomly placed on the shelf, and ensure stable placing and collision-free
the target object is hidden by other objects

level: define the difficulty level of the scene. Can be the number of objects in the scene,
       the clutterness of the scene.
num: number of objects in the scene.


level-1: simple geometry objects. not cluttered. number of objects is less.

"""

import json
import random
import numpy as np
from numpy.core.fromnumeric import size
import rospkg
import pybullet as p
from workspace import Workspace
from robot import Robot
import os
import time
from camera import Camera
from occlusion_3d import Occlusion
from occlusion_scene import OcclusionScene

import open3d as o3d
import cam_utilities

from occlusion_share_graph import OcclusionShareGraph
from greedy_baseline import greedy_baseline_snapshot
from pipeline_baseline import PipelineBaseline
from visual_utilities import *
def random_one_problem(scene, level, num_objs, num_hiding_objs):
    """
    generate one random instance of the problem
    last one object is the target object
    """
    # load scene definition file
    pid = p.connect(p.GUI)
    f = open(scene, 'r')
    scene_dict = json.load(f)

    rp = rospkg.RosPack()
    package_path = rp.get_path('vbcpm_execution_system')
    urdf_path = os.path.join(package_path,scene_dict['robot']['urdf'])
    joints = [0.] * 16
    robot = Robot(urdf_path, scene_dict['robot']['pose']['pos'], scene_dict['robot']['pose']['ori'], pid)



    workspace_low = scene_dict['workspace']['region_low']
    workspace_high = scene_dict['workspace']['region_high']
    
    workspace = Workspace(scene_dict['workspace']['pos'], scene_dict['workspace']['ori'], \
                            scene_dict['workspace']['components'], workspace_low, workspace_high, \
                            pid)

    # camera
    camera = Camera()
    resol = np.array([0.01,0.01,0.01])
    occlusion = OcclusionScene(workspace_high[0]-workspace_low[0],workspace_high[1]-workspace_low[1], 
                          workspace_high[2]-workspace_low[2],resol,workspace_low[0],workspace_low[1],workspace_low[2],
                          np.array([1.0,0.,0.]),np.array([0.0,1.,0.]),np.array([0.0,0.,1.]))

    n_samples = 12000
    if level == 1:
        obj_list = ['cube', 'cylinder']
        # pcd_cube_face1 = np.zeros((n_samples//6,3))
        # pcd_cube_face1[:,0] = -0.5
        # pcd_cube_face1[:,1:3] = np.random.uniform(low=[-0.5,-0.5],high=[0.5,0.5], size=(n_samples//6,2))
        # pcd_cube_face2 = np.zeros((n_samples//6,3))
        # pcd_cube_face2[:,0] = 0.5
        # pcd_cube_face2[:,1:3] = np.random.uniform(low=[-0.5,-0.5],high=[0.5,0.5], size=(n_samples//6,2))
        # pcd_cube_face3 = np.zeros((n_samples//6,3))
        # pcd_cube_face3[:,1] = -0.5
        # pcd_cube_face = np.random.uniform(low=[-0.5,-0.5],high=[0.5,0.5], size=(n_samples//6,2))
        # pcd_cube_face3[:,0] = pcd_cube_face[:,0]
        # pcd_cube_face3[:,2] = pcd_cube_face[:,1]
        # pcd_cube_face4 = np.zeros((n_samples//6,3))
        # pcd_cube_face4[:,1] = 0.5
        # pcd_cube_face = np.random.uniform(low=[-0.5,-0.5],high=[0.5,0.5], size=(n_samples//6,2))
        # pcd_cube_face4[:,0] = pcd_cube_face[:,0]
        # pcd_cube_face4[:,2] = pcd_cube_face[:,1]
        # pcd_cube_face5 = np.zeros((n_samples//6,3))
        # pcd_cube_face5[:,2] = -0.5
        # pcd_cube_face = np.random.uniform(low=[-0.5,-0.5],high=[0.5,0.5], size=(n_samples//6,2))
        # pcd_cube_face5[:,0] = pcd_cube_face[:,0]
        # pcd_cube_face5[:,1] = pcd_cube_face[:,1]
        # pcd_cube_face6 = np.zeros((n_samples//6,3))
        # pcd_cube_face6[:,2] = 0.5
        # pcd_cube_face = np.random.uniform(low=[-0.5,-0.5],high=[0.5,0.5], size=(n_samples//6,2))
        # pcd_cube_face6[:,0] = pcd_cube_face[:,0]
        # pcd_cube_face6[:,1] = pcd_cube_face[:,1]
        # pcd_cube = np.concatenate([pcd_cube_face1,pcd_cube_face2,pcd_cube_face3,pcd_cube_face4,pcd_cube_face5,pcd_cube_face6], axis=0)
        pcd_cube = np.random.uniform(low=[-0.5,-0.5,-0.5],high=[0.5,0.5,0.5], size=(n_samples,3))
        # cylinder we generate by first sampling r theta representing the circle, and then the height
        # # pcd_cylinder_r = np.random.uniform(low=0, high=0.5, size=n_samples)
        # pcd_cylinder_r = np.random.triangular(left=0., mode=0.5, right=0.5, size=n_samples//4)
        # pcd_cylinder_xy = np.random.normal(loc=[0.,0.], scale=[1.,1.], size=(n_samples//4,2))
        # pcd_cylinder_xy = pcd_cylinder_xy / np.linalg.norm(pcd_cylinder_xy, axis=1).reshape(-1,1)
        # # pcd_cylinder_xy = pcd_cylinder_xy * pcd_cylinder_r.reshape(-1,1)

        # pcd_cylinder_up = np.zeros((n_samples//4, 3))
        # pcd_cylinder_up[:,2] = 0.5
        # pcd_cylinder_up[:,0] = pcd_cylinder_xy[:,0]
        # pcd_cylinder_up[:,1] = pcd_cylinder_xy[:,1]
        # pcd_cylinder_down = np.zeros((n_samples//4, 3))
        # pcd_cylinder_down[:,2] = -0.5
        # pcd_cylinder_down[:,0] = pcd_cylinder_xy[:,0]
        # pcd_cylinder_down[:,1] = pcd_cylinder_xy[:,1]


        # pcd_cylinder_h = np.random.uniform(low=-0.5, high=0.5, size=n_samples//2)
        # # pcd_cylinder_h = pcd_cylinder_h.reshape(-1,1)


        # pcd_cylinder_surface_xy = np.random.normal(loc=[0.,0.], scale=[1.,1.], size=(n_samples//2,2))
        # pcd_cylinder_surface_xy = pcd_cylinder_surface_xy / np.linalg.norm(pcd_cylinder_surface_xy, axis=1).reshape(-1,1)

        # pcd_cylinder_surface = np.zeros((n_samples//2, 3))
        # pcd_cylinder_surface[:,2] = pcd_cylinder_h
        # pcd_cylinder_surface[:,0] = pcd_cylinder_surface_xy[:,0]
        # pcd_cylinder_surface[:,1] = pcd_cylinder_surface_xy[:,1]

        # # pcd_cylinder_down[:,1] = pcd_cylinder_xy[:,1]
        # pcd_cylinder = np.concatenate([pcd_cylinder_up, pcd_cylinder_down, pcd_cylinder_surface], axis=0)

        pcd_cylinder_r = np.random.uniform(low=0, high=0.5, size=n_samples)
        pcd_cylinder_r = np.random.triangular(left=0., mode=0.5, right=0.5, size=n_samples)
        pcd_cylinder_xy = np.random.normal(loc=[0.,0.], scale=[1.,1.], size=(n_samples,2))
        pcd_cylinder_xy = pcd_cylinder_xy / np.linalg.norm(pcd_cylinder_xy, axis=1).reshape(-1,1)
        pcd_cylinder_xy = pcd_cylinder_xy * pcd_cylinder_r.reshape(-1,1)

        pcd_cylinder_h = np.random.uniform(low=-0.5, high=0.5, size=n_samples)
        pcd_cylinder_h = pcd_cylinder_h.reshape(-1,1)
        pcd_cylinder = np.concatenate([pcd_cylinder_xy, pcd_cylinder_h], axis=1)
        print('pcd cube:')
        print(pcd_cube)
        print('pcd cylinder: ')
        print(pcd_cylinder)
        # basic shape: cube of size 1, shpere of size 1

        # assuming the workspace coordinate system is at the center of the world
        # * sample random objects on the workspace
        obj_ids = []
        obj_poses = []
        obj_pcds = []
        for i in range(num_objs):
            # randomly pick one object shape
            obj_shape = random.choice(obj_list)
            # randomly scale the object
            if i == 0:
                x_scales = np.arange(0.4, 0.7, 0.1)/10
                y_scales = np.arange(0.4, 0.7, 0.1)/10
                z_scales = np.arange(0.5, 0.9, 0.1)/10
                # put it slightly inside
            else:
                x_scales = np.arange(0.5, 1.2, 0.1)/10
                y_scales = np.arange(0.5, 1.2, 0.1)/10
                z_scales = np.arange(0.5, 1.5, 0.1)/10
                x_low_offset = 0
            if i == 0:
                color = [1.0,0.,0.,1]
            else:
                color = [1,1,1,1]
            if obj_shape == 'cube':
                while True:
                    x_size = x_scales[np.random.choice(len(x_scales))]
                    y_size = y_scales[np.random.choice(len(y_scales))]
                    z_size = z_scales[np.random.choice(len(z_scales))]

                    if obj_shape == 'cylinder':
                        y_size = x_size
                    # sample a pose in the workspace
                    if i == 0:
                        x_low_offset = (workspace_high[0]-workspace_low[0]-x_size)/2
                    else:
                        x_low_offset = 0
                    pcd = pcd_cube * np.array([x_size, y_size, z_size])

                    x = np.random.uniform(low=workspace_low[0]+x_size/2+x_low_offset, high=workspace_high[0]-x_size/2)
                    y = np.random.uniform(low=workspace_low[1]+y_size/2, high=workspace_high[1]-y_size/2)
                    z_offset = 0.#0.01
                    z = workspace_low[2] + z_size/2 + z_offset
                    cid = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=[x_size/2,y_size/2,z_size/2])
                    vid = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=[x_size/2,y_size/2,z_size/2], rgbaColor=color)
                    bid = p.createMultiBody(baseCollisionShapeIndex=cid, baseVisualShapeIndex=vid, basePosition=[x,y,z], baseOrientation=[0,0,0,1])
                    # check collision with scene
                    collision = False
                    for comp_name, comp_id in workspace.components.items():
                        contacts = p.getClosestPoints(bid, comp_id, distance=0.,physicsClientId=pid)
                        if len(contacts):
                            collision = True
                            break
                    for obj_id in obj_ids:
                        contacts = p.getClosestPoints(bid, obj_id, distance=0.,physicsClientId=pid)
                        if len(contacts):
                            collision = True
                            break                    
                    if collision:
                        p.removeBody(bid)
                        continue
                    if i == num_hiding_objs:
                        # for the target, need to be hide by other objects
                        # Method 1: use camera segmentation to see if the target is unseen
                        width, height, rgb_img, depth_img, seg_img = p.getCameraImage(
                            width=camera.info['img_size'],
                            height=camera.info['img_size'],
                            viewMatrix=camera.info['view_mat'],
                            projectionMatrix=camera.info['proj_mat'])
                        # cv2.imshow('camera_rgb', rgb_img)
                        depth_img = depth_img / camera.info['factor']
                        far = camera.info['far']
                        near = camera.info['near']
                        depth_img = far * near / (far-(far-near)*depth_img)
                        depth_img[depth_img>=far] = 0.
                        depth_img[depth_img<=near]=0.
                        seen_obj_ids = set(np.array(seg_img).astype(int).reshape(-1).tolist())
                        if obj_ids[0] in seen_obj_ids:
                            p.removeBody(bid)
                            continue
                        # Method 2: use occlusion
                    obj_ids.append(bid)
                    pose = np.zeros((4,4))
                    pose[:3,:3] = np.eye(3)
                    pose[:3,3] = np.array([x, y, z])
                    obj_poses.append(pose)
                    obj_pcds.append(pcd)
                    break
            else:
                while True:
                    x_size = x_scales[np.random.choice(len(x_scales))]
                    y_size = y_scales[np.random.choice(len(y_scales))]
                    z_size = z_scales[np.random.choice(len(z_scales))]
                    if obj_shape == 'cylinder':
                        y_size = x_size
                    if i == 0:
                        x_low_offset = (workspace_high[0]-workspace_low[0]-x_size)/2
                    else:
                        x_low_offset = 0

                    pcd = pcd_cylinder * np.array([x_size, y_size, z_size])
                    # sample a pose in the workspace
                    x = np.random.uniform(low=workspace_low[0]+x_size/2+x_low_offset, high=workspace_high[0]-x_size/2)
                    y = np.random.uniform(low=workspace_low[1]+y_size/2, high=workspace_high[1]-y_size/2)
                    z_offset = 0.
                    z = workspace_low[2] + z_size/2 + z_offset
                    cid = p.createCollisionShape(shapeType=p.GEOM_CYLINDER, height=z_size, radius=x_size/2)
                    vid = p.createVisualShape(shapeType=p.GEOM_CYLINDER,  length=z_size, radius=x_size/2, rgbaColor=color)
                    bid = p.createMultiBody(baseCollisionShapeIndex=cid, baseVisualShapeIndex=vid, basePosition=[x,y,z], baseOrientation=[0,0,0,1])
                    # check collision with scene
                    collision = False
                    for comp_name, comp_id in workspace.components.items():
                        contacts = p.getClosestPoints(bid, comp_id, distance=0.,physicsClientId=pid)
                        if len(contacts):
                            collision = True
                            break
                    for obj_id in obj_ids:
                        contacts = p.getClosestPoints(bid, obj_id, distance=0.,physicsClientId=pid)
                        if len(contacts):
                            collision = True
                            break                    
                    if collision:
                        p.removeBody(bid)
                        continue
                    if i == num_hiding_objs:
                        # for the target, need to be hide by other objects
                        # Method 1: use camera segmentation to see if the target is unseen
                        width, height, rgb_img, depth_img, seg_img = p.getCameraImage(
                            width=camera.info['img_size'],
                            height=camera.info['img_size'],
                            viewMatrix=camera.info['view_mat'],
                            projectionMatrix=camera.info['proj_mat'])
                        # cv2.imshow('camera_rgb', rgb_img)
                        depth_img = depth_img / camera.info['factor']
                        far = camera.info['far']
                        near = camera.info['near']
                        depth_img = far * near / (far-(far-near)*depth_img)
                        depth_img[depth_img>=far] = 0.
                        depth_img[depth_img<=near]=0.
                        seen_obj_ids = set(np.array(seg_img).astype(int).reshape(-1).tolist())
                        if obj_ids[0] in seen_obj_ids:
                            p.removeBody(bid)
                            continue
                        # Method 2: use occlusion
                    obj_ids.append(bid)
                    pose = np.zeros((4,4))
                    pose[:3,:3] = np.eye(3)
                    pose[:3,3] = np.array([x, y, z])
                    obj_poses.append(pose)
                    obj_pcds.append(pcd)

                    break

    obj_pcd_indices_list = []
    for i in range(len(obj_poses)):
        if i != 0:
            obj_pcd = obj_poses[i][:3,:3].dot(obj_pcds[i].T).T + obj_poses[i][:3,3]
            obj_pcd_indices = obj_pcd
            obj_pcd_indices_list.append(obj_pcd_indices)
    # TODO: testing


    width, height, rgb_img, depth_img, seg_img = p.getCameraImage(
        width=camera.info['img_size'],
        height=camera.info['img_size'],
        viewMatrix=camera.info['view_mat'],
        projectionMatrix=camera.info['proj_mat'])
    # cv2.imshow('camera_rgb', rgb_img)
    depth_img = depth_img / camera.info['factor']
    far = camera.info['far']
    near = camera.info['near']
    depth_img = far * near / (far-(far-near)*depth_img)
    depth_img[depth_img>=far] = 0.
    depth_img[depth_img<=near]=0.


    occluded = occlusion.scene_occlusion(depth_img, rgb_img, camera.info['extrinsics'], camera.info['intrinsics'])
    # occluded = occlusion.get_occlusion_single_obj(camera.info['extrinsics'], obj_poses[-1], obj_pcds[-1])
    occlusion_label, occupied_label, occluded_list = occlusion.label_scene_occlusion(occluded, camera.info['extrinsics'], camera.info['intrinsics'],
                                                                        obj_poses[1:], obj_pcds[1:])


    pcd, _ = cam_utilities.pcd_from_depth(camera.info['intrinsics'], camera.info['extrinsics'], depth_img, rgb_img)

    pcd = np.concatenate([pcd, np.ones(pcd.shape[0]).reshape((-1,1))],axis=1)
    # transform the ray vector into the voxel_grid space
    # notice that we have to divide by the resolution vector
    pcd = np.linalg.inv(occlusion.transform).dot(pcd.T).T
    pcd = pcd[:,:3] / occlusion.resol


    # vol_box = o3d.geometry.OrientedBoundingBox()
    # vol_box.center = vol_bnds.mean(1)
    # vol_box.extent = vol_bnds[:, 1] - vol_bnds[:, 0]
    voxel_grids = []
    color_pick = np.zeros((6,3))
    color_pick[0] = np.array([1., 0., 0.])
    color_pick[1] = np.array([0., 1.0, 0.])
    color_pick[2] = np.array([0., 0., 1.])
    color_pick[3] = np.array([252/255, 169/255, 3/255])
    color_pick[4] = np.array([252/255, 3/255, 252/255])
    color_pick[5] = np.array([20/255, 73/255, 82/255])


    occupied_grids = []
    voxel_pcds_indices = list(range(1,len(obj_poses))) + [-1]
    for i in voxel_pcds_indices:
        voxel_pcd = o3d.geometry.PointCloud()
        voxel_x = occlusion.voxel_x[occlusion_label==i].reshape(-1,1)
        voxel_y = occlusion.voxel_y[occlusion_label==i].reshape(-1,1)
        voxel_z = occlusion.voxel_z[occlusion_label==i].reshape(-1,1)
        
        voxel_pcd_points = np.concatenate([voxel_x, voxel_y, voxel_z], axis=1)
        voxel_pcd.points = o3d.utility.Vector3dVector(voxel_pcd_points)
        colors = np.zeros(voxel_pcd_points.shape)
        colors = colors + color_pick[(i-1)%len(color_pick)]
        voxel_pcd.colors = o3d.utility.Vector3dVector(colors)

        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(voxel_pcd, 1.)
        voxel_grids.append(voxel_grid)
    

        occupied_pcd = o3d.geometry.PointCloud()
        voxel_x = occlusion.voxel_x[occupied_label==i].reshape(-1,1)
        voxel_y = occlusion.voxel_y[occupied_label==i].reshape(-1,1)
        voxel_z = occlusion.voxel_z[occupied_label==i].reshape(-1,1)
        occupied_pcd_points = np.concatenate([voxel_x, voxel_y, voxel_z], axis=1)
        occupied_pcd.points = o3d.utility.Vector3dVector(occupied_pcd_points)
        colors = np.zeros(voxel_pcd_points.shape)
        colors = colors + color_pick[(i-1)%len(color_pick)]
        occupied_pcd.colors = o3d.utility.Vector3dVector(colors)

        occupied_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(occupied_pcd, 1.)
        occupied_grids.append(occupied_grid)

    pcd_pcd = o3d.geometry.PointCloud()
    pcd_pcd.points = o3d.utility.Vector3dVector(pcd)
    colors = np.zeros(pcd.shape)
    colors[:,0] = .1
    colors[:,1] = .1
    colors[:,2] = .1
    pcd_pcd.colors = o3d.utility.Vector3dVector(colors)


    obj_voxel_list = []
    for i in range(len(obj_pcd_indices_list)):


        obj_pcd_indices = np.concatenate([obj_pcd_indices_list[i], np.ones(obj_pcd_indices_list[i].shape[0]).reshape((-1,1))],axis=1)
        # transform the ray vector into the voxel_grid space
        # notice that we have to divide by the resolution vector
        obj_pcd_indices = np.linalg.inv(occlusion.transform).dot(obj_pcd_indices.T).T
        obj_pcd_indices = obj_pcd_indices[:,:3] / occlusion.resol


        obj_pcd = o3d.geometry.PointCloud()
        obj_pcd.points = o3d.utility.Vector3dVector(obj_pcd_indices)
        colors = np.zeros(obj_pcd_indices.shape)
        colors[:,0] = 1.
        colors[:,1] = .0
        colors[:,2] = .0

        obj_pcd.colors = o3d.utility.Vector3dVector(colors)

        obj_voxel = o3d.geometry.VoxelGrid.create_from_point_cloud(obj_pcd, 1)
        # print('object pcd: ')
        # print(obj_pcd_indices)
        obj_voxel_list.append(obj_voxel)
    # return_pcds



    # o3d.visualization.draw_geometries([pcd_pcd]+obj_voxel_list)
    o3d.visualization.draw_geometries([pcd_pcd]+voxel_grids)
    # o3d.visualization.draw_geometries([pcd_pcd, occupancy_voxel])


    return pid, scene_dict, robot, workspace, camera, occlusion, obj_poses[1:], obj_pcds[1:], obj_ids[1:], obj_poses[0], obj_pcds[0], obj_ids[0]

pid, scene_dict, robot, workspace, camera, occlusion, obj_poses, obj_pcds, obj_ids, target_pose, target_pcd, target_obj_id = \
        random_one_problem(scene='scene1.json', level=1, num_objs=7, num_hiding_objs=1)
# construct the occlusion constraint graph
# occluded_list = []
# for i in range(len(obj_poses)):
#     occluded = occlusion.get_occlusion_single_obj(camera.info['extrinsics'], obj_poses[i], obj_pcds[i])
#     occluded_list.append(occluded)
# occlusion_graph = OcclusionShareGraph(obj_poses, occluded_list)
# print('occlusion graph: ')
# print(occlusion_graph.connected)


width, height, rgb_img, depth_img, seg_img = p.getCameraImage(
    width=camera.info['img_size'],
    height=camera.info['img_size'],
    viewMatrix=camera.info['view_mat'],
    projectionMatrix=camera.info['proj_mat'])
# cv2.imshow('camera_rgb', rgb_img)
depth_img = depth_img / camera.info['factor']
far = camera.info['far']
near = camera.info['near']
depth_img = far * near / (far-(far-near)*depth_img)
depth_img[depth_img>=far] = 0.
depth_img[depth_img<=near]=0.

occluded = occlusion.scene_occlusion(depth_img, rgb_img, camera.info['extrinsics'], camera.info['intrinsics'])
# occluded = occlusion.get_occlusion_single_obj(camera.info['extrinsics'], obj_poses[-1], obj_pcds[-1])
occlusion_label, occupied_label, occluded_list = occlusion.label_scene_occlusion(occluded, camera.info['extrinsics'], camera.info['intrinsics'],
                                                                    obj_poses, obj_pcds)

intersected, shadow_occupancy = occlusion.shadow_occupancy_single_obj(occlusion_label>0, None, None, target_pcd)
voxel1 = visualize_voxel(occlusion.voxel_x, occlusion.voxel_y, occlusion.voxel_z, occlusion_label>0, [0,0,0])

voxel_x, voxel_y, voxel_z = np.indices(intersected.shape).astype(float)

voxel2 = visualize_voxel(voxel_x, voxel_y, voxel_z, intersected, [1,0,0])

voxel3 = visualize_voxel(occlusion.voxel_x, occlusion.voxel_y, occlusion.voxel_z, shadow_occupancy, [0,1,0])
o3d.visualization.draw_geometries([voxel1, voxel3])
o3d.visualization.draw_geometries([voxel1, voxel2])

print('obj_poses length: ', len(obj_poses))
print('occluded_list length: ', len(occluded_list))

input('input...')


problem_def = {}
problem_def['pid'] = pid
problem_def['scene_dict'] = scene_dict
problem_def['robot'] = robot
problem_def['workspace'] = workspace
problem_def['camera'] = camera
problem_def['occlusion'] = occlusion
problem_def['obj_pcds'] = obj_pcds
problem_def['obj_ids'] = obj_ids
problem_def['target_obj_pcd'] = target_pcd
problem_def['target_obj_id'] = target_obj_id

pipeline_baseline = PipelineBaseline(problem_def)
pipeline_baseline.solve(greedy_baseline_snapshot)