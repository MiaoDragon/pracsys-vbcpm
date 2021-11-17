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
from perception_pipeline import PerceptionPipeline

import transformations as tf

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
                    pose[3,3] = 1.0
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
                    pose[3,3] = 1.0
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

    return pid, scene_dict, robot, workspace, camera, obj_poses, obj_pcds, obj_ids, obj_poses[0], obj_pcds[0], obj_ids[0]


def test():
    pid, scene_dict, robot, workspace, camera, obj_poses, obj_pcds, obj_ids, target_pose, target_pcd, target_obj_id = \
            random_one_problem(scene='scene1.json', level=1, num_objs=7, num_hiding_objs=1)


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


    workspace_low = scene_dict['workspace']['region_low']
    workspace_high = scene_dict['workspace']['region_high']

    resol = np.array([0.01,0.01,0.01])

    world_x = workspace_high[0]-workspace_low[0]
    world_y = workspace_high[1]-workspace_low[1]
    world_z = workspace_high[2]-workspace_low[2]
    x_base = workspace_low[0]
    y_base = workspace_low[1]
    z_base = workspace_low[2]
    x_vec = np.array([1.0,0.,0.])
    y_vec = np.array([0.,1,0.])
    z_vec = np.array([0,0,1.])

    occlusion_params = {'world_x': world_x, 'world_y': world_y, 'world_z': world_z, 'x_base': x_base, 'y_base': y_base,
                        'z_base': z_base, 'resol': resol, 'x_vec': x_vec, 'y_vec': y_vec, 'z_vec': z_vec}
    object_params = {'resol': resol, 'scale': 0.01}  # scale is used to scale the depth, not the voxel

    perception_pipeline = PerceptionPipeline(occlusion_params, object_params)

    for i in range(10):
        if i >0:
            voxel0 = visualize_voxel(perception_pipeline.slam_system.occlusion.voxel_x, 
                                    perception_pipeline.slam_system.occlusion.voxel_y,
                                    perception_pipeline.slam_system.occlusion.voxel_z,
                                    perception_pipeline.slam_system.filtered_occluded, [0,0,0])

        perception_pipeline.pipeline_sim(camera, [robot.robot_id], workspace.component_ids)

        # visualize the updated occlusion
        occluded = perception_pipeline.slam_system.occluded_t
        occlusion_label = perception_pipeline.slam_system.occlusion_label_t
        occupied_label = perception_pipeline.slam_system.occupied_label_t
        occluded_dict = perception_pipeline.slam_system.occluded_dict_t

        # voxel0 = visualize_voxel(perception_pipeline.slam_system.occlusion.voxel_x, 
        #                         perception_pipeline.slam_system.occlusion.voxel_y,
        #                         perception_pipeline.slam_system.occlusion.voxel_z,
        #                         occluded, [0,0,0])
        

        voxel1 = visualize_voxel(perception_pipeline.slam_system.occlusion.voxel_x, 
                                perception_pipeline.slam_system.occlusion.voxel_y,
                                perception_pipeline.slam_system.occlusion.voxel_z,
                                perception_pipeline.slam_system.filtered_occluded, [1,0,0])


        # print('tsdf > 0: ', (perception_pipeline.slam_system.objects[0].tsdf>0).astype(int).sum())
        # print('tsdf < 0: ', (perception_pipeline.slam_system.objects[0].tsdf<0).astype(int).sum())
        # print('tsdf =min: ', (perception_pipeline.slam_system.objects[0].tsdf_count==0).astype(int).sum())

        voxel2 = visualize_voxel(perception_pipeline.slam_system.objects[0].voxel_x, 
                                perception_pipeline.slam_system.objects[0].voxel_y,
                                perception_pipeline.slam_system.objects[0].voxel_z,
                                perception_pipeline.slam_system.objects[0].get_conservative_model(), [0,0,0])

        voxel3 = visualize_voxel(perception_pipeline.slam_system.objects[0].voxel_x, 
                                perception_pipeline.slam_system.objects[0].voxel_y,
                                perception_pipeline.slam_system.objects[0].voxel_z,
                                perception_pipeline.slam_system.objects[0].get_optimistic_model(), [1,0,0])
        
        opt_pcd = perception_pipeline.slam_system.objects[0].sample_optimistic_pcd()
        opt_pcd = perception_pipeline.slam_system.objects[0].transform[:3,:3].dot(opt_pcd.T).T + perception_pipeline.slam_system.objects[0].transform[:3,3]
        opt_pcd = perception_pipeline.slam_system.occlusion.world_in_voxel_rot.dot(opt_pcd.T).T + perception_pipeline.slam_system.occlusion.world_in_voxel_tran
        opt_pcd = opt_pcd / perception_pipeline.slam_system.occlusion.resol

        pcd1 = visualize_pcd(opt_pcd, [0,1,0])

        bbox = visualize_bbox(perception_pipeline.slam_system.objects[0].voxel_x, 
                            perception_pipeline.slam_system.objects[0].voxel_y,
                            perception_pipeline.slam_system.objects[0].voxel_z)
        # voxel_x, voxel_y, voxel_z = np.indices(intersected.shape).astype(float)

        # voxel2 = visualize_voxel(voxel_x, voxel_y, voxel_z, intersected, [1,0,0])

        # voxel3 = visualize_voxel(occlusion.voxel_x, occlusion.voxel_y, occlusion.voxel_z, shadow_occupancy, [0,1,0])
        if i==0:
            o3d.visualization.draw_geometries([voxel1,pcd1])
        else:
            o3d.visualization.draw_geometries([voxel0, voxel1,pcd1])

        o3d.visualization.draw_geometries([voxel2, voxel3, bbox])
        o3d.visualization.draw_geometries([voxel3, bbox])



        input('input...')
        # randomly change one object's location
        while True:
            # i = np.random.choice(len(obj_ids))
            i = 0
            obj_id = obj_ids[i]
            past_x = obj_poses[i][0,3]
            past_y = obj_poses[i][1,3]

            past_obj_pose = np.array(obj_poses[i])

            x = np.random.uniform(low=workspace_low[0], high=workspace_high[0])
            y = np.random.uniform(low=workspace_low[1], high=workspace_high[1])
            angle = 2*np.pi / 10 * i -np.pi
            # angle = np.random.uniform(low=-np.pi,high=np.pi)
            rot = tf.rotation_matrix(angle, (0,0,1))
            quat = tf.quaternion_from_matrix(rot) # w,x,y,z
            p.resetBasePositionAndOrientation(obj_id, [x,y,obj_poses[i][2,3]], [quat[1],quat[2],quat[3],quat[0]], physicsClientId=pid)

            collision = False
            for comp_name, comp_id in workspace.components.items():
                contacts = p.getClosestPoints(obj_id, comp_id, distance=0.,physicsClientId=pid)
                if len(contacts):
                    collision = True
                    print('colliding with workspace')
                    break
            for obj_id_ in obj_ids:
                if obj_id_ == obj_id:
                    continue
                contacts = p.getClosestPoints(obj_id, obj_id_, distance=0.,physicsClientId=pid)
                if len(contacts):
                    print('colliding with others')
                    collision = True
                    break                    
            contacts = p.getClosestPoints(obj_id, target_obj_id, distance=0.,physicsClientId=pid)
            if len(contacts):
                collision = True
                print('colliding with others')
            
            if not collision:
                break
            
            rot_mat = np.zeros((4,4))
            rot_mat[3,3] = 1
            rot_mat[:3,:3] = past_obj_pose[:3,:3]
            quat = tf.quaternion_from_matrix(rot_mat) # w,x,y,z
            p.resetBasePositionAndOrientation(obj_id, past_obj_pose[:3,3], [quat[1],quat[2],quat[3],quat[0]], physicsClientId=pid)


        prev_obj_pose = np.array(obj_poses[i])

        obj_poses[i][:3,:3] = rot[:3,:3]
        obj_poses[i][0,3] = x
        obj_poses[i][1,3] = y

        # prev_obj_pose: W T O1
        # obj_pose: W T O2
        # relative transform: O1 T O2
        relative_transform = obj_poses[i].dot(np.linalg.inv(prev_obj_pose))
        # relative_transform = np.linalg.inv(prev_obj_pose).dot(obj_poses[i])

        obj_id = perception_pipeline.last_assoc[obj_id]
        if obj_id in perception_pipeline.slam_system.objects.keys():
            perception_pipeline.slam_system.objects[obj_id].set_active()
            perception_pipeline.slam_system.objects[obj_id].update_transform_from_relative(relative_transform)
        # problem_def = {}
        # problem_def['pid'] = pid
        # problem_def['scene_dict'] = scene_dict
        # problem_def['robot'] = robot
        # problem_def['workspace'] = workspace
        # problem_def['camera'] = camera
        # problem_def['occlusion'] = occlusion
        # problem_def['obj_pcds'] = obj_pcds
        # problem_def['obj_ids'] = obj_ids
        # problem_def['target_obj_pcd'] = target_pcd
        # problem_def['target_obj_id'] = target_obj_id

        # pipeline_baseline = PipelineBaseline(problem_def)
        # pipeline_baseline.solve(greedy_baseline_snapshot)