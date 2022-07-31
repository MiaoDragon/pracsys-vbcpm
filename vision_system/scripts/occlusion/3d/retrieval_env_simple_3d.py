"""
construct the planning problem in a simulated environment in 3D

Problem: retrieval of target object from the shelf where there are multiple objects
Simulator: PyBullet
Camera: RGBD
Object Modeling: have access to the object 3D model
    (future: extend to model-free problems, where we don't have the object model,
        and only have some clues of the target object.)
"""
import rospkg
import rospy
import os
import pybullet as p
from robot import Robot
import numpy as np
import transformations as tf
import time
import open3d as o3d

class Environment():
    def __init__(self, prob_config_dict):
        # construct the environment from the problem definition file, or through ROS
        rp = rospkg.RosPack()
        package_path = rp.get_path('vbcpm_execution_system')
        self.package_path = package_path

        # * initialize the PyBullet scene
        cid = p.connect(p.GUI)
        # * load objects into the scene

        # workspace
        self.workspace = {}
        for component_name, component in prob_config_dict['workspace']['components'].items():
            pos = component['pose']['pos']
            ori = component['pose']['ori']
            shape = component['shape']
            shape = np.array(shape)
            col_id = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=shape/2)
            vis_id = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=shape/2)
            comp_id = p.createMultiBody(baseCollisionShapeIndex=col_id, baseVisualShapeIndex=vis_id,
                                        basePosition=pos, baseOrientation=ori)
            self.workspace[component_name] = comp_id
        self.workspace_pos = prob_config_dict['workspace']['pos']
        self.workspace_ori = prob_config_dict['workspace']['ori']
        self.workspace_low = prob_config_dict['workspace']['region_low']
        self.workspace_high = prob_config_dict['workspace']['region_high']

        # camera
        # TODO: parameterize the camera position
        cam_pos = np.array([0.35, 0., 1.25])
        look_at = np.array([1.35, 0., 0.58])
        up_vec = np.array([cam_pos[2]-look_at[2], 0., look_at[0]-cam_pos[0]])
        # up_vec = np.array([1.25-0.58, 0., 1.35-0.35])

        view_mat = p.computeViewMatrix(
            cameraEyePosition=cam_pos,
            cameraTargetPosition=look_at,
            cameraUpVector=up_vec
        )

        # https://stackoverflow.com/questions/60430958/understanding-the-view-and-projection-matrix-from-pybullet
        # https://stackoverflow.com/questions/39992968/how-to-calculate-field-of-view-of-the-camera-from-camera-intrinsic-matrix
        # https://github.com/bulletphysics/bullet3/blob/master/examples/SharedMemory/PhysicsClientC_API.cpp#L4372
        fov = 90
        img_size = 320
        near = 0.01
        far = 1.2
        proj_mat = p.computeProjectionMatrixFOV(
            fov=fov,
            aspect=1,
            nearVal=near,
            farVal=far
        )

        L = look_at - cam_pos
        L = L / np.linalg.norm(L)
        s = np.cross(L, up_vec)

        s = s / np.linalg.norm(s)
        u_prime = np.cross(s, L)
        print('u_prime: ', u_prime)
        print('up_vector: ', up_vec/np.linalg.norm(up_vec))

        # transformation matrix: rotation
        rot_mat = np.array([s, -u_prime, L])#.T  # three rows: Right, Up, Forward (column-major form). After transpose, row-major form
        # https://www.scratchapixel.com/lessons/mathematics-physics-for-computer-graphics/lookat-function
        tran_mat = -rot_mat.dot(cam_pos)
        T_mat = np.eye(4)
        T_mat[:3,:3] = rot_mat
        T_mat[:3,3] = tran_mat

        T_mat = tf.inverse_matrix(T_mat)
        print(T_mat)

        focal = img_size / np.tan(fov * np.pi/180 / 2)/2
        cam_intrinsics = [[focal, 0, img_size/2],
                        [0, focal, img_size/2],
                        [0, 0, 1.]]
        cam_extrinsics = T_mat  # cam_extrinsics: {world}T{cam}
        self.camera = {}
        self.camera['view_mat'] = view_mat
        self.camera['proj_mat'] = proj_mat
        self.camera['intrinsics'] = cam_intrinsics
        self.camera['extrinsics'] = cam_extrinsics
        self.camera['factor'] = 1.0  # To be parameterized
        self.camera['img_size'] = img_size
        self.camera['far'] = far
        self.camera['near'] = near
        # objects
        objs = []
        obj_pcds = []
        obj_heights = []
        for obj_dict in prob_config_dict['objects']:
            obj_i_c = p.createCollisionShape(shapeType=p.GEOM_MESH, meshScale=obj_dict['scale'], \
                                            fileName=os.path.join(package_path, obj_dict['collision_mesh']), \
                                            physicsClientId=cid)
            obj_i_v = p.createVisualShape(shapeType=p.GEOM_MESH, meshScale=obj_dict['scale'], \
                                            fileName=os.path.join(package_path, obj_dict['visual_mesh']), \
                                            physicsClientId=cid)
            obj_i = p.createMultiBody(baseCollisionShapeIndex=obj_i_c, baseVisualShapeIndex=obj_i_v, \
                            basePosition=obj_dict['pose']['pos'], baseOrientation=obj_dict['pose']['ori'],
                            baseMass=obj_dict['mass'], physicsClientId=cid)
            objs.append(obj_i)
            pcd = o3d.io.read_point_cloud(os.path.join(package_path, obj_dict['pcd']))
            obj_pcds.append(np.asarray(pcd.points)*np.array(obj_dict['scale']))
            pcd = np.asarray(pcd.points)*np.array(obj_dict['scale'])
            obj_heights.append(np.max(pcd[:,2]) - np.min(pcd[:,2]))
        self.objs = objs
        self.obj_pcds = obj_pcds
        self.obj_heights = obj_heights

        self.target_idx = prob_config_dict['target_idx']


        # robot
        self.robot = Robot(prob_config_dict['robot']['pose']['pos'], prob_config_dict['robot']['pose']['ori'], \
                            os.path.join(package_path,prob_config_dict['robot']['urdf']), cid)
        p.stepSimulation()
        time.sleep(1/240)

    def sense(self):
        # provide a raw sensed information from the environment
        # * publish camera info
        width, height, rgb_img, depth_img, seg_img = p.getCameraImage(
            width=self.camera['img_size'],
            height=self.camera['img_size'],
            viewMatrix=self.camera['view_mat'],
            projectionMatrix=self.camera['proj_mat'])
        # cv2.imshow('camera_rgb', rgb_img)
        depth_img = depth_img / self.camera['factor']
        far = self.camera['far']
        near = self.camera['near']
        depth_img = far * near / (far-(far-near)*depth_img)
        depth_img[depth_img>=far] = 0.
        depth_img[depth_img<=near]=0.

        seen_obj_ids = set(np.array(seg_img).astype(int).reshape(-1).tolist())

        # * Fake Perception *
        # for the seen objects, return the exact pose
        obj_poses = []
        for i in range(len(self.objs)):
            # if the object can be seen
            if self.objs[i] not in seen_obj_ids:
                obj_poses.append(None)
                continue
            pos, ori = p.getBasePositionAndOrientation(self.objs[i])
            pose = np.zeros((4,4))
            pose[3,3] = 1
            pose[:3,:3] = tf.quaternion_matrix([ori[3],ori[0],ori[1],ori[2]])[:3,:3]  # order: w, x, y, z    
            pose[:3,3] = pos
            obj_poses.append(pose)
        return depth_img, rgb_img, obj_poses
# import time
# import utility

# rp = rospkg.RosPack()
# package_path = rp.get_path('vbcpm_execution_system')
# # f = os.path.join(package_path, 'prob1.json')  # problem configuration file (JSON)
# f = 'prob1.json'
# prob_config_dict = utility.prob_config_parser(f)



# env = Environment(prob_config_dict)
# env.sense()
# time.sleep(10000)