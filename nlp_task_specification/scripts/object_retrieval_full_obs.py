"""
generate a problem for the object retrieval problem under partial observation

objects are randomly placed on the shelf, and ensure stable placing and collision-free
the target object is hidden by other objects

level: define the difficulty level of the scene. Can be the number of objects in the scene,
       the clutterness of the scene.
num: number of objects in the scene.


level-1: simple geometry objects. not cluttered. number of objects is less.

"""

import os
import time
import json
import random
import numpy as np
from numpy.core.fromnumeric import size

import rospy
import rospkg
import pybullet as p
import open3d as o3d

import cam_utilities
from robot import Robot
from camera import Camera
from visual_utilities import *
from dep_graph import DepGraph
from workspace import Workspace
from occlusion_scene import OcclusionScene

from baxter_planner import BaxterPlanner as Planner
from planit.msg import PercievedObject
from pybullet_scene_publisher import PybulletScenePublisher


def random_one_problem(scene, level, num_objs, num_hiding_objs):
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
    if level == 1:
        obj_list = ['cube', 'wall', 'cylinder', 'cylinder', 'ontop', 'ontop']

        pcd_cube = np.random.uniform(
            low=[-0.5, -0.5, -0.5], high=[0.5, 0.5, 0.5], size=(n_samples, 3)
        )

        pcd_cylinder_r = np.random.uniform(low=0, high=0.5, size=n_samples)
        pcd_cylinder_r = np.random.triangular(
            left=0., mode=0.5, right=0.5, size=n_samples
        )
        pcd_cylinder_xy = np.random.normal(
            loc=[0., 0.], scale=[1., 1.], size=(n_samples, 2)
        )
        pcd_cylinder_xy = pcd_cylinder_xy / np.linalg.norm(
            pcd_cylinder_xy, axis=1
        ).reshape(-1, 1)
        pcd_cylinder_xy = pcd_cylinder_xy * pcd_cylinder_r.reshape(-1, 1)

        pcd_cylinder_h = np.random.uniform(low=-0.5, high=0.5, size=n_samples)
        pcd_cylinder_h = pcd_cylinder_h.reshape(-1, 1)
        pcd_cylinder = np.concatenate([pcd_cylinder_xy, pcd_cylinder_h], axis=1)
        print('pcd cube:')
        print(pcd_cube)
        print('pcd cylinder: ')
        print(pcd_cylinder)
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
                x_scales = np.arange(0.25, 0.40, 0.05) / 10
                y_scales = np.arange(0.25, 0.40, 0.05) / 10
                z_scales = np.arange(0.5, 1.0, 0.05) / 10
            elif obj_shape == 'ontop':
                x_scales = np.arange(0.25, 0.40, 0.05) / 10
                y_scales = np.arange(0.25, 0.40, 0.05) / 10
                z_scales = np.arange(0.5, 1.0, 0.05) / 10
            elif obj_shape == 'cylinder':
                x_scales = np.arange(0.25, 0.40, 0.05) / 10
                y_scales = np.arange(0.25, 0.40, 0.05) / 10
                z_scales = np.arange(1.0, 1.5, 0.05) / 10
            elif obj_shape == 'wall':
                x_scales = np.arange(0.25, 0.40, 0.05) / 10
                y_scales = np.arange(2.0, 2.5, 0.05) / 10
                z_scales = np.arange(1.5, 2.0, 0.05) / 10

            # if i == 0:
            #     color = [1.0, 0., 0., 1]
            # else:
            #     color = [*select_color(i), 1]
            color = [*from_color_map(i, num_objs), 1]

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
                    baseMass=1.0,
                    baseCollisionShapeIndex=cid,
                    baseVisualShapeIndex=vid,
                    basePosition=[x, y, z],
                    baseOrientation=[0, 0, 0, 1]
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
    level=1,
    num_objs=7,
    num_hiding_objs=1,
)

true_obj_poses = obj_poses.copy()

width, height, rgb_img, depth_img, seg_img = p.getCameraImage(
    width=camera.info['img_size'],
    height=camera.info['img_size'],
    viewMatrix=camera.info['view_mat'],
    projectionMatrix=camera.info['proj_mat']
)

depth_img = depth_img / camera.info['factor']
far = camera.info['far']
near = camera.info['near']
depth_img = far * near / (far - (far - near) * depth_img)
depth_img[depth_img >= far] = 0.
depth_img[depth_img <= near] = 0.

# simulated sensing
real = False
if real:
    obj_ind = list(range(1, len(obj_poses) + 1))
    rgb_img, depth_img, _tmp, obj_poses, target_obj_pose = camera.sense(
        obj_pcds[1:],
        obj_pcds[0],
        obj_ind[1:],
        obj_ind[0],
    )

occluded = occlusion.scene_occlusion(
    depth_img, rgb_img, camera.info['extrinsics'], camera.info['intrinsics']
)
occlusion_label, occupied_label, occluded_list = occlusion.label_scene_occlusion(
    occluded,
    camera.info['extrinsics'],
    camera.info['intrinsics'],
    obj_poses,
    obj_pcds,
    depth_nn=1
)
# intersected, shadow_occupancy = occlusion.shadow_occupancy_single_obj(occlusion_label > 0, None, None, target_pcd)

# fake perception
if not real:
    hidden_objs = set()
    for i in range(len(obj_poses)):
        obj_i = i + 1
        obj_i_vox = occupied_label == obj_i
        obj_i_vol = obj_i_vox.sum()
        obj_i_occ_vol = 0
        for j in range(len(obj_poses)):
            # if i == j:
            #     continue
            obj_j = j + 1
            occ_j_vox = (
                occlusion_label == obj_j
            )  #| (occupied_label == -1)  #| (occupied_label == obj_j)

            obj_i_occ_vol += (obj_i_vox & occ_j_vox).sum()
        print(obj_i, obj_i_occ_vol, obj_i_occ_vol / obj_i_vol)
        if obj_i_occ_vol / obj_i_vol > 0.9:
            hidden_objs.add(obj_i)
            obj_poses[obj_i - 1] = None
    print(hidden_objs)

for obj in obj_ids:
    p.changeDynamics(
        obj,
        -1,
        lateralFriction=10.0,
        spinningFriction=10.0,
        rollingFriction=10.0,
    )
    print("Dynamics:", p.getDynamicsInfo(obj, -1))

robot.set_gripper('left', 'open', reset=True)
robot.set_gripper('right', 'open', reset=True)
# p.setGravity(0, 0, 0)
# p.setRealTimeSimulation(1)
# pybullet_scene_pub = PybulletScenePublisher(pid)
# pybullet_scene_pub.publish()

### Grasp Sampling Test ###
rest_joints = robot.get_joints()
pose_ind = input("Please Enter Pose Index: ")
# for i, pose in enumerate(true_obj_poses):
#     obj_i = i + 1
while pose_ind != 'q':
    try:
        obj_i = int(pose_ind[0])
    except IndexError:
        pose_ind = input("Please Enter Pose Index: ")
        continue
    i = obj_i - 1
    t0 = time.time()
    poses = robot.getGrasps(obj_ids[i])
    filteredPoses = robot.filterGrasps(robot.left_gripper_id, poses)
    filteredPoses += robot.filterGrasps(robot.right_gripper_id, poses)
    t1 = time.time()

    print("Time: ", t1 - t0)
    for pose, cols, sparse_pose in filteredPoses:
        input("Next?")
        # for iters in range(1000):
        #     robot.setMotors(sparse_pose)
        #     p.stepSimulation()
        robot.set_joints(pose)
        print(i + 1, [obj_ids.index(x) + 1 for x in cols])
    robot.set_joints(rest_joints)
    pose_ind = input("Please Enter Pose Index: ")
### Grasp Sampling Test End ###

### Pick Test ###
rospy.init_node("planit", anonymous=False)
planner = Planner(robot, is_sim=True)
perception_sub = rospy.Subscriber(
    '/perception', PercievedObject, planner.scene.updatePerception
)
time.sleep(2)

pose_ind = input("Please Enter Pose Index: ")
while pose_ind != 'q':
    try:
        obj_i = int(pose_ind[0]) + 1
    except IndexError:
        pose_ind = input("Please Enter Pose Index: ")
        continue

    object_name = f'Obj_{obj_i}'
    for chirality in ('left', 'right'):
        ### pick up red cylinder ###
        res = planner.pick(
            object_name,
            v_scale=0.50,
            a_scale=1.0,
            grasping_group=chirality + "_hand",
            group_name=chirality + "_arm",
        )
        print(res, type(res))
        if res is True:
            break
        #TODO back to rest pose

    pose_ind = input("Please Enter Pose Index: ")
### Pick End ###

dg = DepGraph(obj_poses, obj_colors, occlusion, occupied_label, occlusion_label)

# visualize occupied voxel grid
vox_occupied = []
vox_occluded = []
vox_revealed = []
vox_ups = []
for i in range(len(obj_poses)):
    obj_i = i + 1
    voxel1 = visualize_voxel(
        occlusion.voxel_x,
        occlusion.voxel_y,
        occlusion.voxel_z,
        occupied_label == obj_i,
        obj_colors[i],
    )
    vox_occupied.append(voxel1)
    voxel2 = visualize_voxel(
        occlusion.voxel_x,
        occlusion.voxel_y,
        occlusion.voxel_z,
        occlusion_label == obj_i,
        obj_colors[i],
    )
    vox_occluded.append(voxel2)
    voxel3 = visualize_voxel(
        occlusion.voxel_x,
        occlusion.voxel_y,
        occlusion.voxel_z,
        occupied_label == obj_i,
        [0, 0, 0],
    )
    if not real:
        vox_revealed.append(voxel3 if obj_i in hidden_objs else voxel1)
    voxel4 = visualize_voxel(
        occlusion.voxel_x,
        occlusion.voxel_y,
        occlusion.voxel_z,
        dg.upmasks[i],
        obj_colors[i],
    )
    vox_ups.append(voxel4)
    # o3d.visualization.draw_geometries([voxel1, voxel2])

o3d.visualization.draw_geometries(vox_occupied)
# o3d.visualization.draw_geometries(
#     [
#         visualize_voxel(
#             occlusion.voxel_x, occlusion.voxel_y, occlusion.voxel_z,
#             occlusion_label == -1, [0, 0, 0]
#         )
#     ]
# )
# o3d.visualization.draw_geometries(vox_occluded)
if not real:
    o3d.visualization.draw_geometries(vox_revealed)
# o3d.visualization.draw_geometries(vox_ups)

dg.draw_graph()
suggestion = int(input("Suggest region"))
result = dg.update_target_confidence(1, suggestion, 1000)
while not result:
    dg.draw_graph()
    suggestion = int(input("Suggest region"))
    result = dg.update_target_confidence(1, suggestion, 1000)
print("Success!", dg.pick_order(result))
dg.draw_graph()

print('obj_poses length: ', len(obj_poses))
print('occluded_list length: ', len(occluded_list))
