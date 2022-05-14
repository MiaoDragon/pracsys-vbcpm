from re import S
import numpy as np
import pybullet as p
import transformations as tf
import time

class Camera():
    def __init__(self, cam_pos=[0.39,0.0,1.3], look_at=[1.44,0,0.7], fov=90, far=1.2, img_size=320, visualize=False):
        # TODO: parameterize the camera position
        # cam_pos = np.array([0.39, 0., 1.3])
        # look_at = np.array([1.44, 0., 0.7])
        cam_pos = np.array(cam_pos)
        look_at = np.array(look_at)
        
        if cam_pos[1] == 0.0 and look_at[1] == 0.0:
            up_vec = np.array([cam_pos[2]-look_at[2], 0., look_at[0]-cam_pos[0]])
        else:
            s = np.array([cam_pos[1]-look_at[1], look_at[0]-cam_pos[0], 0])
            up_vec = -np.cross(cam_pos, s)

        # up_vec = np.array([1.25-0.58, 0., 1.35-0.35])

        view_mat = p.computeViewMatrix(
            cameraEyePosition=cam_pos,
            cameraTargetPosition=look_at,
            cameraUpVector=up_vec
        )

        # https://stackoverflow.com/questions/60430958/understanding-the-view-and-projection-matrix-from-pybullet
        # https://stackoverflow.com/questions/39992968/how-to-calculate-field-of-view-of-the-camera-from-camera-intrinsic-matrix
        # https://github.com/bulletphysics/bullet3/blob/master/examples/SharedMemory/PhysicsClientC_API.cpp#L4372
        # fov = 90
        # img_size = 320
        near = 0.01
        # far = 1.2
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
        # print('u_prime: ', u_prime)
        # print('up_vector: ', up_vec/np.linalg.norm(up_vec))

        # transformation matrix: rotation
        rot_mat = np.array([s, -u_prime, L])#.T  # three rows: Right, Up, Forward (column-major form). After transpose, row-major form
        # https://www.scratchapixel.com/lessons/mathematics-physics-for-computer-graphics/lookat-function
        tran_mat = -rot_mat.dot(cam_pos)
        T_mat = np.eye(4)
        T_mat[:3,:3] = rot_mat
        T_mat[:3,3] = tran_mat

        T_mat = tf.inverse_matrix(T_mat)
        # print(T_mat)

        focal = img_size / np.tan(fov * np.pi/180 / 2)/2
        cam_intrinsics = [[focal, 0, img_size/2],
                        [0, focal, img_size/2],
                        [0, 0, 1.]]

        cam_extrinsics = T_mat  # cam_extrinsics: {world}T{cam}
        self.info = {}
        self.info['view_mat'] = view_mat
        self.info['proj_mat'] = proj_mat
        self.info['intrinsics'] = cam_intrinsics
        self.info['extrinsics'] = cam_extrinsics
        self.info['factor'] = 1.0  # To be parameterized
        self.info['img_size'] = img_size
        self.info['img_shape'] = [img_size, img_size]
        self.info['far'] = far
        self.info['near'] = near
        self.info['pos'] = cam_pos
        self.info['look_at'] = look_at

        if visualize:
            # visualize the camera in the scene: shoot rays to represent the visible range
            vid = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.01,
                                      rgbaColor=[0,0,0,1])
            bid = p.createMultiBody(baseVisualShapeIndex=vid, basePosition=cam_pos)
            self.bid = bid
            # vid = p.addUserDebugPoints(pointPositions=cam_pos, pointColorsRGB=[1,0,0,1], pointSize=0.01)
            # self.vid = vid
            # draw lines to the far point to represent the visible regions of the camera
            # compute the four corners using image size
            # the pixel location is defined to be [0, 0], [0, img_size-1], [img_size-1, 0], [img_size-1, img_size-1]
            # depth is far value


            # corners = np.array([[0,0],[0,img_size-1],[img_size-1,img_size-1] ,[img_size-1,0]])  # 4x2
            # depth = 1.2
            # fx = cam_intrinsics[0][0]
            # fy = cam_intrinsics[1][1]
            # cx = cam_intrinsics[0][2]
            # cy = cam_intrinsics[1][2]
            # transformed_corners = np.zeros((len(corners),3))
            # transformed_corners[:,0] = (corners[:,1] - cx) / fx * depth
            # transformed_corners[:,1] = (corners[:,0] - cy) / fy * depth
            # transformed_corners[:,2] = depth
            # # transform into world frame
            # transformed_corners = cam_extrinsics[:3,:3].dot(transformed_corners.T).T + cam_extrinsics[:3,3]
            # for i in range(len(transformed_corners)):
            #     p.addUserDebugLine(lineFromXYZ=cam_pos, lineToXYZ=transformed_corners[i], lineColorRGB=[0,0,0],lineWidth=0.01)
            #     p.addUserDebugLine(lineFromXYZ=transformed_corners[i], lineToXYZ=transformed_corners[(i+1)%len(transformed_corners)],
            #                         lineColorRGB=[0,0,0],lineWidth=0.002)

        else:
            self.vid = None
    def sense_with_perceive(self, obj_pcds, target_obj_pcd, obj_ids, target_obj_id):
        """
        sense the environment. If an object is seen, give the pose.
        return the pose
        """
        width, height, rgb_img, depth_img, seg_img = p.getCameraImage(
            width=self.info['img_size'],
            height=self.info['img_size'],
            viewMatrix=self.info['view_mat'],
            projectionMatrix=self.info['proj_mat'])
        # cv2.imshow('camera_rgb', rgb_img)
        depth_img = depth_img / self.info['factor']
        far = self.info['far']
        near = self.info['near']
        depth_img = far * near / (far-(far-near)*depth_img)
        depth_img[depth_img>=far] = 0.
        depth_img[depth_img<=near]=0.

        seen_obj_ids = set(np.array(seg_img).astype(int).reshape(-1).tolist())

        obj_poses = []
        for i in range(len(obj_pcds)):
            if obj_ids[i] in seen_obj_ids:
                # obtain the pose of the object from pybullet
                pos, ori = p.getBasePositionAndOrientation(obj_ids[i])
                ori_mat = tf.quaternion_matrix([ori[3],ori[0],ori[1],ori[2]])[:3,:3]  # w x y z
                transform = np.zeros((4,4))
                transform[:3,:3] = ori_mat
                transform[:3,3] = pos
                obj_poses.append(transform)
            else:
                obj_poses.append(None)
        if target_obj_id in seen_obj_ids:
            pos, ori = p.getBasePositionAndOrientation(target_obj_id)
            ori_mat = tf.quaternion_matrix([ori[3],ori[0],ori[1],ori[2]])[:3,:3]  # w x y z
            transform = np.zeros((4,4))
            transform[:3,:3] = ori_mat
            transform[:3,3] = pos
            target_obj_pose = transform
        else:
            target_obj_pose = None
        return rgb_img, depth_img, seg_img, obj_poses, target_obj_pose


    def sense(self):
        """
        sense the environment. If an object is seen, give the pose.
        return the pose
        """
        width, height, rgb_img, depth_img, seg_img = p.getCameraImage(
            width=self.info['img_size'],
            height=self.info['img_size'],
            viewMatrix=self.info['view_mat'],
            projectionMatrix=self.info['proj_mat'])
        # cv2.imshow('camera_rgb', rgb_img)
        depth_img = depth_img / self.info['factor']
        far = self.info['far']
        near = self.info['near']
        depth_img = far * near / (far-(far-near)*depth_img)
        depth_img[depth_img>=far] = 0.
        depth_img[depth_img<=near]=0.



        return rgb_img, depth_img, seg_img