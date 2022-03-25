import cv2
import open3d as o3d
import pybullet as p

import cam_utilities
from robot import Robot
from camera import Camera
from visual_utilities import *
from dep_graph import DepGraph
from workspace import Workspace
from occlusion_scene import OcclusionScene
from baxter_planner import BaxterPlanner as Planner


class Pipeline():

    def __init__(
        self,
        robot,
        workspace,
        camera,
        occlusion,
        obj_poses,
        obj_pcds,
        obj_ids,
        obj_colors,
        pid=0
    ):
        self.robot = robot
        self.workspace = workspace
        self.camera = camera
        self.occlusion = occlusion
        self.obj_poses = obj_poses
        self.obj_pcds = obj_pcds
        self.obj_ids = obj_ids
        self.obj_colors = obj_colors
        self.pid = pid

        rospy.init_node("planit", anonymous=False)
        self.planner = Planner(self.robot, is_sim=True)
        perception_sub = rospy.Subscriber(
            '/perception', PercievedObject, planner.scene.updatePerception
        )

    def sense(self, fake=True):
        width, height, rgb_img, depth_img, seg_img = p.getCameraImage(
            width=self.camera.info['img_size'],
            height=self.camera.info['img_size'],
            viewMatrix=self.camera.info['view_mat'],
            projectionMatrix=self.camera.info['proj_mat']
        )

        depth_img = depth_img / self.camera.info['factor']
        far = self.camera.info['far']
        near = self.camera.info['near']
        depth_img = far * near / (far - (far - near) * depth_img)
        depth_img[depth_img >= far] = 0.
        depth_img[depth_img <= near] = 0.

        if fake:
            for i, obj in enumerate(self.obj_ids):
                pos, rot = p.getBasePositionAndOrientation(obj)
                pose = np.zeros((4, 4))
                pose[:3, :3] = np.reshape(p.getMatrixFromQuaternion(rot), (3, 3))
                pose[:3, 3] = pos
                self.obj_poses[i] = pose
        else:
            rgb_img, depth_img, _tmp, self.obj_poses, target_obj_pose = camera.sense(
                self.obj_pcds[1:],
                self.obj_pcds[0],
                self.obj_ids[1:],
                self.obj_ids[0],
            )

        occluded = occlusion.scene_occlusion(
            depth_img, rgb_img, self.camera.info['extrinsics'],
            self.camera.info['intrinsics']
        )
        occlusion_label, occupied_label, occluded_list = occlusion.label_scene_occlusion(
            occluded,
            self.camera.info['extrinsics'],
            self.camera.info['intrinsics'],
            self.obj_poses,
            self.obj_pcds,
            depth_nn=1
        )

        return occlusion_label, occupied_label, occluded_list

    def free_space_grid(self, obj_i):
        ws_low = self.workspace.region_low
        ws_high = self.workspace.region_high

        # get z coord for object placement
        obj_id = self.obj_ids[obj_i - 1]
        mins, maxs = p.getAABB(obj_id)
        z = mins[2] - ws_low[2] - 0.005

        # sense the scene
        occlusion_label, occupied_label, occluded_list = self.sense()

        # TODO find better kernel generation
        obj_x, obj_y = np.where((occupied_label == obj_i).any(2))
        obj_x -= min(obj_x)
        obj_y -= min(obj_y)
        kernel = np.zeros((max(obj_x) + 1, max(obj_y) + 1)).astype('uint8')
        kernel[obj_x, obj_y] = 1
        print(f"Obj{obj_i}:")
        print(kernel[:, :])

        free_x, free_y = np.where(((occlusion_label <= 0) & (occupied_label == 0)).all(2))
        shape = self.occlusion.occlusion.shape
        img = 255 * np.ones(shape[0:2]).astype('uint8')
        img[free_x, free_y] = 0
        # cv2.imshow("Test0", img)
        # cv2.waitKey(0)
        fimg = cv2.filter2D(img, -1, kernel)
        # cv2.imshow("Test1", fimg)
        # cv2.waitKey(0)
        mink_x, mink_y = np.where(img == 0)
        samples = list(
            zip(
                mink_x * self.occlusion.resol[0] + ws_low[0],
                mink_y * self.occlusion.resol[1] + ws_low[1],
                [z] * len(mink_x),
            )
        )
        # cv2.destroyAllWindows()
        return samples

    def get_dep_graph(self):
        occlusion_label, occupied_label, occluded_list = self.sense()
        return DepGraph(
            self.obj_poses, self.obj_colors, self.occlusion, occupied_label,
            occlusion_label
        )
