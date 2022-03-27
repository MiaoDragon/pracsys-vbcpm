"""
objects are randomly placed on the table while ensuring that placements are
stable, collision-free, and that the target object is hidden by other objects

num: number of objects in the scene.
"""

import os
import sys
import time
import json
import random

import cv2
import numpy as np
import pybullet as p
import open3d as o3d

import rospy
import rospkg
from std_msgs.msg import Header
from moveit_msgs.msg import RobotState
from sensor_msgs.msg import JointState
from pybullet_scene_publisher import PybulletScenePublisher

import cam_utilities
from robot import Robot
from camera import Camera
from pipeline import Pipeline
from visual_utilities import *
from dep_graph import DepGraph
from workspace import Workspace
from occlusion_scene import OcclusionScene
from baxter_planner import BaxterPlanner as Planner


def random_one_problem(scene, num_objs, num_hiding_objs):
    """
    generate one random instance of the problem
    last one object is the target object
    """
    # load scene definition file
    pid = p.connect(p.GUI_SERVER)
    f = open(scene, 'r')
    scene_dict = json.load(f)

    rp = rospkg.RosPack()
    package_path = rp.get_path('vbcpm_execution_system')
    urdf_path = os.path.join(package_path, scene_dict['robot']['urdf'])
    joints = [0.] * 16
    robot = Robot(
        urdf_path,
        scene_dict['robot']['pose']['pos'],
        scene_dict['robot']['pose']['ori'],
        pid,
    )

    workspace_low = scene_dict['workspace']['region_low']
    workspace_high = scene_dict['workspace']['region_high']

    workspace = Workspace(
        scene_dict['workspace']['pos'], scene_dict['workspace']['ori'],
        scene_dict['workspace']['components'], workspace_low, workspace_high, pid
    )

    # camera
    camera = Camera()
    resol = np.array([0.01, 0.01, 0.01])
    occlusion = OcclusionScene(
        workspace_high[0] - workspace_low[0], workspace_high[1] - workspace_low[1],
        workspace_high[2] - workspace_low[2], resol, workspace_low[0], workspace_low[1],
        workspace_low[2], np.array([1.0, 0., 0.]), np.array([0.0, 1., 0.]),
        np.array([0.0, 0., 1.])
    )

    n_samples = 12000
    obj_list = ['cube', 'wall', 'cylinder', 'cylinder', 'ontop', 'ontop']
    obj_list = ['cube', 'wall', 'ontop', 'ontop', 'ontop', 'cylinder']

    pcd_cube = np.random.uniform(
        low=[-0.5, -0.5, -0.5], high=[0.5, 0.5, 0.5], size=(n_samples, 3)
    )

    pcd_cylinder_r = np.random.uniform(low=0, high=0.5, size=n_samples)
    pcd_cylinder_r = np.random.triangular(left=0., mode=0.5, right=0.5, size=n_samples)
    pcd_cylinder_xy = np.random.normal(loc=[0., 0.], scale=[1., 1.], size=(n_samples, 2))
    pcd_cylinder_xy = pcd_cylinder_xy / np.linalg.norm(
        pcd_cylinder_xy, axis=1
    ).reshape(-1, 1)
    pcd_cylinder_xy = pcd_cylinder_xy * pcd_cylinder_r.reshape(-1, 1)

    pcd_cylinder_h = np.random.uniform(low=-0.5, high=0.5, size=n_samples)
    pcd_cylinder_h = pcd_cylinder_h.reshape(-1, 1)
    pcd_cylinder = np.concatenate([pcd_cylinder_xy, pcd_cylinder_h], axis=1)
    # print('pcd cube:')
    # print(pcd_cube)
    # print('pcd cylinder: ')
    # print(pcd_cylinder)
    # basic shape: cube of size 1, cylinder of size 1

    # assuming the workspace coordinate system is at the center of the world
    # * sample random objects on the workspace
    obj_ids = []
    obj_poses = []
    obj_pcds = []
    obj_tops = []
    obj_colors = []
    for i in range(num_objs):
        # randomly pick one object shape
        obj_shape = random.choice(obj_list)
        if i == num_hiding_objs:
            obj_shape = 'wall'
        if i == 0:
            obj_shape = 'cube'
        # obj_shape = obj_list[i%len(obj_list)]
        # randomly scale the object
        if obj_shape == 'cube':
            x_scales = np.arange(0.30, 0.40, 0.05) / 10
            y_scales = np.arange(0.30, 0.40, 0.05) / 10
            z_scales = np.arange(0.65, 1.2, 0.05) / 10
        elif obj_shape == 'ontop':
            x_scales = np.arange(0.30, 0.40, 0.05) / 10
            y_scales = np.arange(0.30, 0.40, 0.05) / 10
            z_scales = np.arange(0.65, 1.2, 0.05) / 10
        elif obj_shape == 'cylinder':
            x_scales = np.arange(0.25, 0.40, 0.05) / 10
            y_scales = np.arange(0.25, 0.40, 0.05) / 10
            z_scales = np.arange(1.0, 1.5, 0.05) / 10
        elif obj_shape == 'wall':
            x_scales = np.arange(0.30, 0.40, 0.05) / 10
            y_scales = np.arange(2.0, 2.5, 0.05) / 10
            z_scales = np.arange(1.5, 2.0, 0.05) / 10

        # if i == 0:
        #     color = [1.0, 0., 0., 1]
        # else:
        #     color = [*select_color(i), 1]
        color = list(np.round([*from_color_map(i, num_objs), 1], 5))

        # scale base object and transform until it satisfies constraints
        while True:
            x_size = x_scales[np.random.choice(len(x_scales))]
            y_size = y_scales[np.random.choice(len(y_scales))]
            z_size = z_scales[np.random.choice(len(z_scales))]
            if obj_shape == 'cylinder':
                y_size = x_size

            # sample a pose in the workspace
            if i < num_hiding_objs:
                x_low_offset = (workspace_high[0] - workspace_low[0] - x_size) / 2
            else:
                x_low_offset = 0

            if obj_shape == 'cube' or obj_shape == 'wall' or obj_shape == 'ontop':
                pcd = pcd_cube * np.array([x_size, y_size, z_size])
            elif obj_shape == 'cylinder':
                pcd = pcd_cylinder * np.array([x_size, y_size, z_size])

            if obj_shape == 'ontop':
                prev_ind = random.randint(0, i - 1)
                x, y = obj_poses[prev_ind][:2, 3]
                z = 0
                z += obj_tops[prev_ind] + z_size
            else:
                x = np.random.uniform(
                    low=workspace_low[0] + x_size / 2 + x_low_offset,
                    high=workspace_high[0] - x_size / 2
                )
                y = np.random.uniform(
                    low=workspace_low[1] + y_size / 2,
                    high=workspace_high[1] - y_size / 2
                )
                z = 0.001
                z += workspace_low[2] + z_size

            # save top coord for later and adjust current z
            ztop = z
            z -= z_size / 2

            if obj_shape == 'cube' or obj_shape == 'wall' or obj_shape == 'ontop':
                cid = p.createCollisionShape(
                    shapeType=p.GEOM_BOX,
                    halfExtents=[x_size / 2, y_size / 2, z_size / 2]
                )
                vid = p.createVisualShape(
                    shapeType=p.GEOM_BOX,
                    halfExtents=[x_size / 2, y_size / 2, z_size / 2],
                    rgbaColor=color
                )
            elif obj_shape == 'cylinder':
                cid = p.createCollisionShape(
                    shapeType=p.GEOM_CYLINDER, height=z_size, radius=x_size / 2
                )
                vid = p.createVisualShape(
                    shapeType=p.GEOM_CYLINDER,
                    length=z_size,
                    radius=x_size / 2,
                    rgbaColor=color
                )
            bid = p.createMultiBody(
                # baseMass=0.01,
                baseMass=0.0001,
                baseCollisionShapeIndex=cid,
                baseVisualShapeIndex=vid,
                basePosition=[x, y, z],
                baseOrientation=[0, 0, 0, 1]
                # baseOrientation=[0, 0, 0.5, 0.5]
            )
            # check collision with scene
            collision = False
            for comp_name, comp_id in workspace.components.items():
                contacts = p.getClosestPoints(
                    bid, comp_id, distance=0., physicsClientId=pid
                )
                if len(contacts):
                    collision = True
                    break
            for obj_id in obj_ids:
                contacts = p.getClosestPoints(
                    bid, obj_id, distance=0., physicsClientId=pid
                )
                if len(contacts):
                    collision = True
                    break
            if collision:
                p.removeBody(bid)
                continue
            if i == num_hiding_objs and num_hiding_objs > 0:
                # for the target, need to be hide by other objects
                # Method 1: use camera segmentation to see if the target is unseen
                width, height, rgb_img, depth_img, seg_img = p.getCameraImage(
                    width=camera.info['img_size'],
                    height=camera.info['img_size'],
                    viewMatrix=camera.info['view_mat'],
                    projectionMatrix=camera.info['proj_mat']
                )
                # cv2.imshow('camera_rgb', rgb_img)
                depth_img = depth_img / camera.info['factor']
                far = camera.info['far']
                near = camera.info['near']
                depth_img = far * near / (far - (far - near) * depth_img)
                depth_img[depth_img >= far] = 0.
                depth_img[depth_img <= near] = 0.
                seen_obj_ids = set(np.array(seg_img).astype(int).reshape(-1).tolist())
                if obj_ids[0] in seen_obj_ids:
                    p.removeBody(bid)
                    continue
                # Method 2: use occlusion

            obj_ids.append(bid)
            pose = np.zeros((4, 4))
            pose[:3, :3] = np.eye(3)
            pose[:3, 3] = np.array([x, y, z])
            obj_poses.append(pose)
            obj_pcds.append(pcd)
            obj_tops.append(ztop)
            obj_colors.append(color)
            break

    return (
        pid,
        scene_dict,
        robot,
        workspace,
        camera,
        occlusion,
        obj_poses,
        obj_pcds,
        obj_ids,
        obj_colors,
    )


pid, scene_dict, robot, workspace, camera, occlusion, obj_poses, obj_pcds, obj_ids, obj_colors = random_one_problem(
    scene='scene_table.json',
    num_objs=7,
    num_hiding_objs=1,
)

obj_names = ['red', 'purple', 'dark_green', 'cyan', 'light_green', 'pink', 'maroon']

pipeline = Pipeline(
    robot,
    workspace,
    camera,
    occlusion,
    obj_poses,
    obj_pcds,
    obj_ids,
    obj_colors,
    obj_names,
    pid,
)

input('Run pybullet_scene_publisher.py then hit Enter...\nReady?\n')

### Set "Stable" Physics Parameters ###
lfl = 100000000000000000.0
sfl = 100000000000000000.0
rfl = 100000000000000000.0
lfr = 100000000000000000.0
sfr = 100000000000000000.0
rfr = 100000000000000000.0
# lfl = 0.001
# sfl = 0.1
# rfl = 0.001
# lfr = 0.001
# sfr = 0.1
# rfr = 0.001
lfg = 0.0
sfg = 0.0000
rfg = 0.0000
lf = 15.0
sf = 0.0001
rf = 0.0001
eemass = 0.0

p.changeDynamics(
    0,
    robot.left_gripper_id,
    mass=eemass,
    lateralFriction=lfg,
    spinningFriction=sfg,
    rollingFriction=rfg,
    physicsClientId=pid,
)
p.changeDynamics(
    0,
    robot.right_gripper_id,
    mass=eemass,
    lateralFriction=lfg,
    spinningFriction=sfg,
    rollingFriction=rfg,
    physicsClientId=pid,
)
p.changeDynamics(
    0,
    robot.left_fingers[0],
    mass=eemass,
    lateralFriction=lfl,
    spinningFriction=sfl,
    rollingFriction=rfl,
    physicsClientId=pid,
)
p.changeDynamics(
    0,
    robot.left_fingers[1],
    mass=eemass,
    lateralFriction=lfl,
    spinningFriction=sfl,
    rollingFriction=rfl,
    physicsClientId=pid,
)
p.changeDynamics(
    0,
    robot.right_fingers[0],
    mass=eemass,
    lateralFriction=lfr,
    spinningFriction=sfr,
    rollingFriction=rfr,
    physicsClientId=pid,
)
p.changeDynamics(
    0,
    robot.right_fingers[1],
    mass=eemass,
    lateralFriction=lfr,
    spinningFriction=sfr,
    rollingFriction=rfr,
    physicsClientId=pid,
)
# print("Dynamics:", p.getDynamicsInfo(0, robot.left_fingers[0], physicsClientId=pid))
# print("Dynamics:", p.getDynamicsInfo(0, robot.left_fingers[1], physicsClientId=pid))
# print("Dynamics:", p.getDynamicsInfo(0, robot.right_fingers[0], physicsClientId=pid))
# print("Dynamics:", p.getDynamicsInfo(0, robot.right_fingers[1], physicsClientId=pid))
# print("Dynamics:", p.getDynamicsInfo(0, robot.left_gripper_id, physicsClientId=pid))
# print("Dynamics:", p.getDynamicsInfo(0, robot.right_gripper_id, physicsClientId=pid))
# print("Dynamics:", p.getDynamicsInfo(1, -1))
for obj in obj_ids:
    p.changeDynamics(
        obj,
        -1,
        lateralFriction=lf,
        spinningFriction=sf,
        rollingFriction=rf,
        physicsClientId=pid,
    )
    # print("Dynamics:", p.getDynamicsInfo(obj, -1))

p.setGravity(0, 0, -9.81)
# p.setRealTimeSimulation(1)
### End Physics Parameters ###

try:
    pipeline.pick_and_place()
    pipeline.retrieve()
except KeyboardInterrupt:
    sys.exit()
