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
import rospkg
import pybullet as p
from workspace import Workspace
from robot import Robot
import os
import time
from camera import Camera

import open3d as o3d

import transformations as tf

from visual_utilities import *
def random_one_problem(scene, level, num_objs, num_hiding_objs, safety_padding=0.01):
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

    ll = [-1.58, \
        -3.13, -1.90, -2.95, -2.36, -3.13, -1.90, -3.13, \
        -3.13, -1.90, -2.95, -2.36, -3.13, -1.90, -3.13] +  \
        [0.0, -0.8757, 0.0, 0.0, -0.8757, 0.0]
    ### upper limits for null space
    ul = [1.58, \
        3.13, 1.90, 2.95, 2.36, 3.13, 1.90, 3.13, \
        3.13, 1.90, 2.95, 2.36, 3.13, 1.90, 3.13] + \
        [0.8, 0.0, 0.8757, 0.81, 0.0, 0.8757]
    ### joint ranges for null space
    jr = [1.58*2, \
        6.26, 3.80, 5.90, 4.72, 6.26, 3.80, 6.26, \
        6.26, 3.80, 5.90, 4.72, 6.26, 3.80, 6.26] + \
        [0.8, 0.8757, 0.8757, 0.81, 0.8757, 0.8757]

    robot = Robot(urdf_path, scene_dict['robot']['pose']['pos'], scene_dict['robot']['pose']['ori'], 
                    ll, ul, jr, 'motoman_left_ee', 0.3015, pid, [1,1,1,1,1,1,1,1,0,0,0,0,0,0,0])


    joints = [0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ,0.0,  # left (suction)
            1.75, 0.8, 0.0, -0.66, 0.0, 0.0 ,0.0,  # right
            ]
    robot.set_joints(joints)  # 

    workspace_low = scene_dict['workspace']['region_low']
    workspace_high = scene_dict['workspace']['region_high']
    padding = scene_dict['workspace']['padding']
    workspace = Workspace(scene_dict['workspace']['pos'], scene_dict['workspace']['ori'], \
                            scene_dict['workspace']['components'], workspace_low, workspace_high, padding, \
                            pid)
    workspace_low = workspace.region_low
    workspace_high = workspace.region_high
    # camera
    camera = Camera()

    n_samples = 12000
    if level == 1:
        obj_list = ['cube', 'cylinder']

        pcd_cube = np.random.uniform(low=[-0.5,-0.5,-0.5],high=[0.5,0.5,0.5], size=(n_samples,3))


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
        obj_shapes = []
        obj_sizes = []
        for i in range(num_objs):
            # randomly pick one object shape
            obj_shape = random.choice(obj_list)
            obj_shapes.append(obj_shape)
            # randomly scale the object
            if i == 0:
                x_scales = np.arange(0.4, 0.9, 0.1)/10
                y_scales = np.arange(0.4, 0.9, 0.1)/10
                z_scales = np.arange(0.8, 1.2, 0.1)/10
                # put it slightly inside
            else:
                x_scales = np.arange(0.5, 1.1, 0.1)/10
                y_scales = np.arange(0.5, 1.1, 0.1)/10
                z_scales = np.arange(1.2, 1.5, 0.1)/10
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
                    for comp_name, comp_id in workspace.component_id_dict.items():
                        contacts = p.getClosestPoints(bid, comp_id, distance=0.,physicsClientId=pid)
                        if len(contacts):
                            collision = True
                            break
                    for obj_id in obj_ids:
                        # add some distance between objects
                        contacts = p.getClosestPoints(bid, obj_id, distance=safety_padding,physicsClientId=pid)
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
                    pose[3,3] = 1.0
                    pose[:3,:3] = np.eye(3)
                    pose[:3,3] = np.array([x, y, z])
                    obj_poses.append(pose)
                    obj_pcds.append(pcd)
                    obj_sizes.append([x_size, y_size, z_size])

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
                    for comp_name, comp_id in workspace.component_id_dict.items():
                        contacts = p.getClosestPoints(bid, comp_id, distance=0.,physicsClientId=pid)
                        if len(contacts):
                            collision = True
                            break
                    for obj_id in obj_ids:
                        contacts = p.getClosestPoints(bid, obj_id, distance=safety_padding,physicsClientId=pid)
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
                    pose[3,3] = 1.0
                    pose[:3,:3] = np.eye(3)
                    pose[:3,3] = np.array([x, y, z])
                    obj_poses.append(pose)
                    obj_pcds.append(pcd)
                    obj_sizes.append([x_size, y_size, z_size])

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

    return pid, scene, robot, workspace, camera, obj_poses, obj_pcds, obj_ids, obj_shapes, obj_sizes, \
        obj_poses[0], obj_pcds[0], obj_ids[0], obj_shapes[0], obj_sizes[0]


def load_problem(scene, obj_poses, obj_pcds, obj_shapes, obj_sizes,
                target_pose, target_pcd, target_obj_shape, target_obj_size):

    # load scene definition file
    pid = p.connect(p.GUI)
    f = open(scene, 'r')
    scene_dict = json.load(f)

    rp = rospkg.RosPack()
    package_path = rp.get_path('vbcpm_execution_system')
    urdf_path = os.path.join(package_path,scene_dict['robot']['urdf'])
    joints = [0.] * 16

    ll = [-1.58, \
        -3.13, -1.90, -2.95, -2.36, -3.13, -1.90, -3.13, \
        -3.13, -1.90, -2.95, -2.36, -3.13, -1.90, -3.13] +  \
        [0.0, -0.8757, 0.0, 0.0, -0.8757, 0.0]
    ### upper limits for null space
    ul = [1.58, \
        3.13, 1.90, 2.95, 2.36, 3.13, 1.90, 3.13, \
        3.13, 1.90, 2.95, 2.36, 3.13, 1.90, 3.13] + \
        [0.8, 0.0, 0.8757, 0.81, 0.0, 0.8757]
    ### joint ranges for null space
    jr = [1.58*2, \
        6.26, 3.80, 5.90, 4.72, 6.26, 3.80, 6.26, \
        6.26, 3.80, 5.90, 4.72, 6.26, 3.80, 6.26] + \
        [0.8, 0.8757, 0.8757, 0.81, 0.8757, 0.8757]

    robot = Robot(urdf_path, scene_dict['robot']['pose']['pos'], scene_dict['robot']['pose']['ori'], 
                    ll, ul, jr, 'motoman_left_ee', 0.3015, pid, [1,1,1,1,1,1,1,1,0,0,0,0,0,0,0])


    joints = [0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ,0.0,  # left (suction)
            1.75, 0.8, 0.0, -0.66, 0.0, 0.0 ,0.0,  # right
            ]
    robot.set_joints(joints)  # 

    workspace_low = scene_dict['workspace']['region_low']
    workspace_high = scene_dict['workspace']['region_high']
    padding = scene_dict['workspace']['padding']
    workspace = Workspace(scene_dict['workspace']['pos'], scene_dict['workspace']['ori'], \
                            scene_dict['workspace']['components'], workspace_low, workspace_high, padding, \
                            pid)
    workspace_low = workspace.region_low
    workspace_high = workspace.region_high
    # camera
    camera = Camera()
    obj_ids = []

    for i in range(len(obj_shapes)):
        # randomly pick one object shape
        obj_shape = obj_shapes[i]
        # randomly scale the object
        if i == 0:
            color = [1.0,0.,0.,1]
        else:
            color = [1,1,1,1]
        x_size, y_size, z_size = obj_sizes[i]
        x, y, z = obj_poses[i][:3,3]
        if obj_shape == 'cube':
            # sample a pose in the workspace
            cid = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=[x_size/2,y_size/2,z_size/2])
            vid = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=[x_size/2,y_size/2,z_size/2], rgbaColor=color)
            bid = p.createMultiBody(baseCollisionShapeIndex=cid, baseVisualShapeIndex=vid, basePosition=[x,y,z], baseOrientation=[0,0,0,1])
        else:
            cid = p.createCollisionShape(shapeType=p.GEOM_CYLINDER, height=z_size, radius=x_size/2)
            vid = p.createVisualShape(shapeType=p.GEOM_CYLINDER,  length=z_size, radius=x_size/2, rgbaColor=color)
            bid = p.createMultiBody(baseCollisionShapeIndex=cid, baseVisualShapeIndex=vid, basePosition=[x,y,z], baseOrientation=[0,0,0,1])
        obj_ids.append(bid)

    return pid, scene, robot, workspace, camera, obj_poses, obj_pcds, obj_ids, obj_poses[0], obj_pcds[0], obj_ids[0]