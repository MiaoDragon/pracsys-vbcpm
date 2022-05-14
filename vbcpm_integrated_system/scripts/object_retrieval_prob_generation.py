"""
generate a problem for the object retrieval problem under partial observation

objects are randomly placed on the shelf, and ensure stable placing and collision-free
the target object is hidden by other objects

level: define the difficulty level of the scene. Can be the number of objects in the scene,
       the clutterness of the scene.
num: number of objects in the scene.


level-1: simple geometry objects. not cluttered. number of objects is less.

"""

import pickle
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
def random_one_problem(scene, level, num_objs, num_hiding_objs, safety_padding=0.015):
    """
    generate one random instance of the problem
    last one object is the target object

    level = 1: easy
    level = 2: medium
    level = 3: hard
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
                            print('collision with ', comp_name)
                            break
                    for obj_id in obj_ids:
                        # add some distance between objects
                        contacts = p.getClosestPoints(bid, obj_id, distance=safety_padding,physicsClientId=pid)
                        if len(contacts):
                            collision = True
                            break                    
                    if collision:
                        print('collision...')
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
                            print('collision with ', comp_name)
                            break
                    for obj_id in obj_ids:
                        contacts = p.getClosestPoints(bid, obj_id, distance=safety_padding,physicsClientId=pid)
                        if len(contacts):
                            collision = True
                            break                    
                    if collision:
                        p.removeBody(bid)
                        print('collision')
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






def random_one_problem_level(scene, level, num_objs, num_hiding_objs=1, safety_padding=0.015):
    """
    generate one random instance of the problem
    last one object is the target object

    small scale: 0.05-0.07
    medium scale: 0.08-0.10
    large scale: 0.11-0.13

    level = 1: easy
    100% small

    level = 2: medium
    60% small, 30% medium, 10% large
    0-10% (round): large
    10%-30%: medium

    level = 3: hard
    25% small, 50% medium, 25% large


    TODO: remove the first object since we are focusing on environment exploration
    TODO: construct new object sets for each level of difficulty
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

    small_scale = np.array([0.04, 0.05])
    medium_scale = np.array([0.07, 0.08])
    large_scale = np.array([0.10, 0.11])
    # height_scale = [0.06, 0.10, 0.14]
    height_scale = [0.09, 0.12, 0.15]
    small_obj = [[1,1]]
    medium_obj = [[1,2], [2,1], [2,2]]
    large_obj = [[1,3], [3,1], [2,3], [3,2], [3,3]]

    if num_objs >= 10:
        # scale down the size of the objects for cluttered case
        small_scale = small_scale * 0.9
        medium_scale = medium_scale * 0.9
        large_scale = large_scale * 0.9


    def generate_obj_scale(scale_idx):
        if scale_idx == 1:
            return np.random.choice(small_scale)
        elif scale_idx == 2:
            return np.random.choice(medium_scale)
        elif scale_idx == 3:
            return np.random.choice(large_scale)

    def generate_obj_size(obj_type_idx):
        # generate an object size with the given object size index
        if obj_type_idx == 1:
            obj_type = small_obj
        elif obj_type_idx == 2:
            obj_type = medium_obj
        elif obj_type_idx == 3:
            obj_type = large_obj
        idx = np.random.choice(len(obj_type))
        x = generate_obj_scale(obj_type[idx][0])
        y = generate_obj_scale(obj_type[idx][1])
        z = np.random.choice(height_scale)
        return [x, y, z]

    n_samples = 12000
    obj_sizes = []
    if level == 1:
        # all objects are small objects
        # construct object scales
        for i in range(num_objs):
            obj_sizes.append(generate_obj_size(1))
    elif level == 2:
        for i in range(num_objs):
            if i < int(np.round(0.1 * num_objs)):
                obj_sizes.append(generate_obj_size(3))
            elif i < int(np.round(0.4 * num_objs)):
                obj_sizes.append(generate_obj_size(2))
            else:
                obj_sizes.append(generate_obj_size(1))
    elif level == 3:
        for i in range(num_objs):
            if i < int(np.round(0.25 * num_objs)):
                obj_sizes.append(generate_obj_size(3))
            elif i < int(np.round(0.75 * num_objs)):
                obj_sizes.append(generate_obj_size(2))
            else:
                obj_sizes.append(generate_obj_size(1))
    obj_sizes = obj_sizes[::-1]
    obj_sizes[0][1] = small_scale[0]
    obj_sizes[0][2] = height_scale[0]

    obj_sizes[1][1] = max(0.05, obj_sizes[1][1])
    obj_sizes[1][2] = np.random.choice([height_scale[1], height_scale[2]])
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

    # basic shape: cube of size 1, shpere of size 1

    # assuming the workspace coordinate system is at the center of the world
    # * sample random objects on the workspace
    obj_ids = []
    obj_poses = []
    obj_pcds = []
    obj_shapes = []
    for i in range(num_objs):
        # randomly pick one object shape
        while True:
            obj_shape = random.choice(obj_list)

            color = [1,1,1,1]

            if obj_shape == 'cube':
                x_size = obj_sizes[i][0]
                y_size = obj_sizes[i][1]
                z_size = obj_sizes[i][2]
                # sample a pose in the workspace
                if i == 0:
                    x_low_offset = (workspace_high[0]-workspace_low[0]-x_size)/2
                else:
                    x_low_offset = 0

                pcd = pcd_cube * np.array([x_size, y_size, z_size])

                x = np.random.uniform(low=workspace_low[0]+x_size/2+x_low_offset, high=workspace_high[0]-x_size/2)
                y = np.random.uniform(low=workspace_low[1]+y_size/2, high=workspace_high[1]-y_size/2)
                z_offset = 0.001#0.01
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
                        print('collision with ', comp_name)
                        break
                for obj_id in obj_ids:
                    # add some distance between objects
                    contacts = p.getClosestPoints(bid, obj_id, distance=safety_padding,physicsClientId=pid)
                    if len(contacts):
                        collision = True
                        break                    
                if collision:
                    print('collision...')
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
                obj_shapes.append(obj_shape)
                break
            else:
                x_size = obj_sizes[i][0]
                y_size = obj_sizes[i][1]
                z_size = obj_sizes[i][2]
                if i == 0:
                    x_size = y_size
                else:
                    x_size = max(x_size, y_size)
                    y_size = x_size
                if i == 0:
                    x_low_offset = (workspace_high[0]-workspace_low[0]-x_size)/2
                else:
                    x_low_offset = 0
                    
                pcd = pcd_cylinder * np.array([x_size, y_size, z_size])
                # sample a pose in the workspace
                x = np.random.uniform(low=workspace_low[0]+x_size/2+x_low_offset, high=workspace_high[0]-x_size/2)
                y = np.random.uniform(low=workspace_low[1]+y_size/2, high=workspace_high[1]-y_size/2)
                z_offset = 0.001
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
                        print('collision with ', comp_name)
                        break
                for obj_id in obj_ids:
                    contacts = p.getClosestPoints(bid, obj_id, distance=safety_padding,physicsClientId=pid)
                    if len(contacts):
                        collision = True
                        break                    
                if collision:
                    p.removeBody(bid)
                    print('collision')
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
                obj_shapes.append(obj_shape)
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


def load_problem_level(scene, obj_poses, obj_pcds, obj_shapes, obj_sizes,
                target_pose, target_pcd, target_obj_shape, target_obj_size):

    # load scene definition file
    pid = p.connect(p.GUI)
    # pid = p.connect(p.DIRECT)

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

    old_height_scale = [0.06, 0.10, 0.14]
    height_scale = [0.09, 0.12, 0.15]

    for i in range(len(obj_shapes)):
        # randomly pick one object shape
        obj_shape = obj_shapes[i]
        # randomly scale the object
        if i == 0:
            # color = [1.0,0.,0.,1]
            color = [1,1,1,1]
        else:
            color = [1,1,1,1]
        x_size, y_size, z_size = obj_sizes[i]

        if np.round(z_size, 2) == 0.06:
            z_size = height_scale[0]
        elif np.round(z_size, 2) == 0.10:
            z_size = height_scale[1]
        elif np.round(z_size, 2) == 0.14:
            z_size = height_scale[2]

        x, y, z = obj_poses[i][:3,3]
        z_offset = 0.001
        z = workspace_low[2] + z_size/2 + z_offset
                
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






def random_one_problem_ycb(scene, level, num_objs, num_hiding_objs, safety_padding=0.01):
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
    model_folder = os.path.join(package_path, "data/models/objects/ocrtoc/")
    # camera
    camera = Camera()
    

    if level == 1:

        if os.path.exists('ycb_database.pkl'):
            f = open('ycb_database.pkl', 'rb')
            obj_name_list, obj_mins, obj_maxs = pickle.load(f)
            f.close()

        else:
            # import pywavefront
            # import trimesh
            # obj_id_list = [1,2,3,4,5,6,9,10,19,21,23,25,65,70,77]
            obj_name_list = os.listdir(model_folder)
            obj_name_list = [
                "a_cups",
                "b_cups",
                "bleach_cleanser",
                "c_cups",
                "conditioner",
                "cracker_box",
                "cup_small",
                "d_cups",
                "doraemon_cup",
                "e_cups",
                "f_cups",
                "foam_brick",
                "g_cups",
                "gelatin_box",
                "h_cups",
                "hello_kitty_cup",
                "i_cups",
                "j_cups",
                "jenga",
                "master_chef_can",
                "mug",
                "mustard_bottle",
                "pepsi",
                "pigeon",
                "pitcher_base",
                "potted_meat_can",
                "pure_zhen",
                "realsense_box",
                "redbull",
                # "rubiks_cube",
                "sugar_box",
                "tea_can1",
                "tomato_soup_can",
                "tuna_fish_can",
                "wood_block"#,
                # "wooden_puzzle1",
                # "wooden_puzzle2",
                # "wooden_puzzle3"
            ]

            new_obj_name_list = []
            obj_mins = []
            obj_maxs = []
            for i in range(len(obj_name_list)):
                if not os.path.isdir(os.path.join(model_folder, obj_name_list[i])):
                    continue
                if not os.path.exists(os.path.join(model_folder, obj_name_list[i], 'collision_meshes')):
                    continue
                # obj_i = pywavefront.Wavefront(os.path.join(model_folder, obj_name_list[i], 'google_16k/textured.obj'), parse=True, cache=False)
                # total_vertices = np.array(obj_i.vertices)
                # mesh = trimesh.load_mesh(os.path.join(model_folder, obj_name_list[i], 'google_16k/nontextured.stl'))
                # total_vertices = np.array(mesh.vertices)
                if obj_name_list[i] == 'cracker_box':
                    ori = [ 0, 0.7071068, 0, 0.7071068 ]
                elif obj_name_list[i] in ['mug', 'pitcher_base', 'sugar_box', 'bleach_cleanser']:
                    ori = [ 0, 0, 1, 0 ]  # make sure the handle is behind the object
                else:
                    ori = [0, 0, 0, 1]
                cid = p.createCollisionShape(shapeType=p.GEOM_MESH, fileName=os.path.join(model_folder, obj_name_list[i], 'collision_meshes/collision.obj'),
                                                meshScale=[1,1,1],collisionFrameOrientation=ori)#, flags=p.GEOM_FORCE_CONCAVE_TRIMESH)
                vid = p.createVisualShape(shapeType=p.GEOM_MESH, fileName=os.path.join(model_folder, obj_name_list[i], 'meshes/visual.obj'),
                                                meshScale=[1,1,1],visualFrameOrientation=ori)
                bid = p.createMultiBody(baseCollisionShapeIndex=cid, baseVisualShapeIndex=vid, basePosition=[0,0,0], baseOrientation=[0,0,0,1])
                aabbmin, aabbmax = p.getAABB(bodyUniqueId=bid)
                p.removeBody(bid)
                print('aabbmin: ', aabbmin)
                print('aabbmax: ', aabbmax)
                if aabbmax[2] - aabbmin[2] < 0.07:
                    print('height smaller than threshold')
                    continue
                new_obj_name_list.append(obj_name_list[i])                
                obj_mins.append([aabbmin[0],aabbmin[1],aabbmin[2]])
                obj_maxs.append([aabbmax[0],aabbmax[1],aabbmax[2]])
            obj_name_list = new_obj_name_list

            f = open('ycb_database.pkl', 'wb')
            pickle.dump((obj_name_list, obj_mins, obj_maxs), f)
            f.close()
        # check the size of each object


        # TODO: load the mesh of objects
        # assuming the workspace coordinate system is at the center of the world
        # * sample random objects on the workspace
        obj_name_to_id = {}
        for i in range(len(obj_name_list)):
            obj_name_to_id[obj_name_list[i]] = i
        obj_ids = []
        obj_poses = []
        obj_shapes = []
        print('names: ')
        print(obj_name_list)
        for i in range(num_objs):
            # randomly pick one object shape
            if i == 0:
                color = [1.0,0.,0.,1]
            else:
                color = [1,1,1,1]
            while True:
                obj_shape_i = np.random.choice(len(obj_name_list))

                # randomly scale the object
                print('sampling object %d' % (i))
                print('obj shape: ', obj_name_list[obj_shape_i])
                if obj_name_list[obj_shape_i] in obj_shapes:
                    # don't allow repeat shape
                    continue
                # sample a pose in the workspace
                if i == 0:
                    x_low_offset = (workspace_high[0]-workspace_low[0])/2 + (obj_maxs[obj_shape_i][0]-obj_mins[obj_shape_i][0])/2
                    if obj_maxs[obj_shape_i][2] - obj_mins[obj_shape_i][2] > 0.14:
                        continue

                    # x_low_offset = 0
                else:
                    x_low_offset = 0
                x = np.random.uniform(low=workspace_low[0]-obj_mins[obj_shape_i][0]+x_low_offset, high=workspace_high[0]-obj_maxs[obj_shape_i][0])
                y = np.random.uniform(low=workspace_low[1]-obj_mins[obj_shape_i][1], high=workspace_high[1]-obj_maxs[obj_shape_i][1])
                z_offset = 0.001#0.01
                z = workspace_low[2] - np.floor(1000*obj_mins[obj_shape_i][2])/1000 + z_offset

                ###########################################
                # * specific setting for paper illustration
                # num_hiding_objs = -1
                # if i == 0:
                #     obj_shape_i = obj_name_to_id['mug']
                #     x = min(workspace_low[0] - obj_mins[obj_shape_i][0] + 0.15, workspace_high[0]-obj_maxs[obj_shape_i][0])
                #     y = 0.15#workspace_low[1] - obj_mins[obj_shape_i][1]  
                #     z = workspace_low[2] - np.floor(1000*obj_mins[obj_shape_i][2])/1000 + z_offset
                    
                # if i == 1:
                #     obj_shape_i = obj_name_to_id['sugar_box']
                #     x = workspace_low[0] - obj_mins[obj_shape_i][0] + 0.03
                #     y = 0.205
                #     z = workspace_low[2] - np.floor(1000*obj_mins[obj_shape_i][2])/1000 + z_offset
                # if i == 2:
                #     obj_shape_i = obj_name_to_id['redbull']
                #     x = workspace_low[0] - obj_mins[obj_shape_i][0] + 0.07
                #     y = 0.09
                #     z = workspace_low[2] - np.floor(1000*obj_mins[obj_shape_i][2])/1000 + z_offset                    
                #     pass
                # if i == 3:
                #     obj_shape_i = obj_name_to_id['jenga']
                #     x = min(workspace_low[0] - obj_mins[obj_shape_i][0] + 0.15, workspace_high[0]-obj_maxs[obj_shape_i][0])
                #     y = -0.175
                #     z = workspace_low[2] - np.floor(1000*obj_mins[obj_shape_i][2])/1000 + z_offset                    
                #     pass
                # if i == 4:
                #     obj_shape_i = obj_name_to_id['bleach_cleanser']
                #     x = min(workspace_low[0] - obj_mins[obj_shape_i][0] + 0.03, workspace_high[0]-obj_maxs[obj_shape_i][0])
                #     y = -0.16
                #     z = workspace_low[2] - np.floor(1000*obj_mins[obj_shape_i][2])/1000 + z_offset                    
                #     pass

                ###########################################
                if obj_name_list[obj_shape_i] == 'cracker_box':
                    ori = [ 0, 0.7071068, 0, 0.7071068 ]
                elif obj_name_list[obj_shape_i] in ['mug', 'pitcher_base', 'sugar_box', 'bleach_cleanser']:
                    ori = [ 0, 0, 1, 0 ]  # make sure the handle is behind the object
                else:
                    ori = [0, 0, 0, 1]
                
                cid = p.createCollisionShape(shapeType=p.GEOM_MESH, fileName=os.path.join(model_folder, obj_name_list[obj_shape_i], 'collision_meshes/collision.obj'),
                                                meshScale=[1,1,1],collisionFrameOrientation=ori)#, flags=p.GEOM_FORCE_CONCAVE_TRIMESH)
                vid = p.createVisualShape(shapeType=p.GEOM_MESH, fileName=os.path.join(model_folder, obj_name_list[obj_shape_i], 'meshes/visual.obj'),
                                                meshScale=[1,1,1],visualFrameOrientation=ori)
                bid = p.createMultiBody(baseCollisionShapeIndex=cid, baseVisualShapeIndex=vid, basePosition=[x,y,z], baseOrientation=[0,0,0,1])
                # check collision with scene
                collision = False
                for comp_name, comp_id in workspace.component_id_dict.items():
                    contacts = p.getClosestPoints(bid, comp_id, distance=0.,physicsClientId=pid)
                    if len(contacts):
                        collision = True
                        print('colliding with workspace ', comp_name)
                        break
                for obj_id in obj_ids:
                    # add some distance between objects
                    contacts = p.getClosestPoints(bid, obj_id, distance=safety_padding,physicsClientId=pid)
                    if len(contacts):
                        collision = True
                        print('colliding with other objects')

                        break                    
                if collision:
                    p.removeBody(bid)
                    continue
                if i <= num_hiding_objs:
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
                if i == 0 and num_hiding_objs > 0:
                    ori_depth_img = np.array(depth_img)
                    ori_seg_img = np.array(seg_img)
                if i > 0 and i < num_hiding_objs:                    
                    if ((ori_seg_img == obj_ids[0]) & (np.array(seg_img) == bid)).sum() == 0:
                        # no intersection in the segmentation image
                        p.removeBody(bid)
                        continue
                if i == num_hiding_objs:
                    # for the target, need to be hide by other objects
                    # Method 1: use camera segmentation to see if the target is unseen
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
                # obj_sizes.append([x_size, y_size, z_size])
                obj_shapes.append(obj_name_list[obj_shape_i])
                break

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
    print('visible objects: ')
    for i in range(len(obj_ids)):
        if obj_ids[i] in seen_obj_ids:
            print(obj_shapes[i])

    return pid, scene, robot, workspace, camera, obj_poses, obj_ids, obj_shapes, \
        obj_poses[0], obj_ids[0], obj_shapes[0]


def load_problem_ycb(scene, obj_poses, obj_shapes,
                target_pose, target_obj_shape):

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
    camera = Camera(visualize=False)
    obj_ids = []
    model_folder = os.path.join(package_path, "data/models/objects/ocrtoc/")

    for i in range(len(obj_shapes)):
        # randomly pick one object shape
        obj_shape = obj_shapes[i]
        print('obj shape: ', obj_shape)
        # randomly scale the object
        if i == 0:
            color = [1.0,0.,0.,1]
        else:
            color = [1,1,1,1]
        x, y, z = obj_poses[i][:3,3]
        if obj_shape == 'cracker_box':
            ori = [ 0, 0.7071068, 0, 0.7071068 ]
        elif obj_shape in ['mug', 'pitcher_base', 'sugar_box', 'bleach_cleanser']:
            ori = [ 0, 0, 1, 0 ]  # make sure the handle is behind the object
        else:
            ori = [0, 0, 0, 1]            

        cid = p.createCollisionShape(shapeType=p.GEOM_MESH, fileName=os.path.join(model_folder, obj_shape, 'collision_meshes/collision.obj'),
                                        meshScale=[1,1,1],collisionFrameOrientation=ori)#, flags=p.GEOM_FORCE_CONCAVE_TRIMESH)
        vid = p.createVisualShape(shapeType=p.GEOM_MESH, fileName=os.path.join(model_folder, obj_shape, 'meshes/visual.obj'),
                                        meshScale=[1,1,1],visualFrameOrientation=ori)        
        bid = p.createMultiBody(baseCollisionShapeIndex=cid, baseVisualShapeIndex=vid, basePosition=[x,y,z], baseOrientation=[0,0,0,1])

        obj_ids.append(bid)

    return pid, scene, robot, workspace, camera, obj_poses, obj_ids, obj_poses[0], obj_ids[0]