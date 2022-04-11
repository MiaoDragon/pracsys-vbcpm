import time
import copy
import random

import cv2
import open3d as o3d
import pybullet as p

import rospy
from planit.msg import PercievedObject

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
        obj_names=None,
        pid=0,
    ):
        self.robot = robot
        self.workspace = workspace
        self.camera = camera
        self.occlusion = occlusion
        self.obj_poses = obj_poses
        self.obj_pcds = obj_pcds
        self.obj_ids = obj_ids
        self.obj_colors = obj_colors
        self.obj_names = obj_names
        self.pid = pid
        self.prev_arm = None

        rospy.init_node("pipeline", anonymous=False)
        self.planner = Planner(self.robot, is_sim=True)
        perception_sub = rospy.Subscriber(
            '/perception', PercievedObject, self.planner.scene.updatePerception
        )

        try:
            iter(obj_names)
        except TypeError:
            self.name2ind = {oid: ind for ind, oid in enumerate(obj_ids)}
        else:
            self.name2ind = {name: ind for ind, name in enumerate(obj_names)}

    def sense(self, fake=True):
        width, height, rgb_img, depth_img, seg_img = p.getCameraImage(
            width=self.camera.info['img_size'],
            height=self.camera.info['img_size'],
            viewMatrix=self.camera.info['view_mat'],
            projectionMatrix=self.camera.info['proj_mat'],
            physicsClientId=self.pid,
        )

        depth_img = depth_img / self.camera.info['factor']
        far = self.camera.info['far']
        near = self.camera.info['near']
        depth_img = far * near / (far - (far - near) * depth_img)
        depth_img[depth_img >= far] = 0.
        depth_img[depth_img <= near] = 0.

        obj_poses = copy.deepcopy(self.obj_poses)
        if fake:
            seen_obj_ids = set(np.array(seg_img).astype(int).reshape(-1).tolist())
            for i, obj in enumerate(self.obj_ids):
                pos, rot = p.getBasePositionAndOrientation(obj, physicsClientId=self.pid)
                pose = np.zeros((4, 4))
                pose[:3, :3] = np.reshape(p.getMatrixFromQuaternion(rot), (3, 3))
                pose[:3, 3] = pos
                obj_poses[i] = pose
                if obj in seen_obj_ids:
                    self.obj_poses[i] = pose
                else:
                    self.obj_poses[i] = None
        else:
            rgb_img, depth_img, _tmp, self.obj_poses, target_obj_pose = camera.sense(
                self.obj_pcds[1:],
                self.obj_pcds[0],
                self.obj_ids[1:],
                self.obj_ids[0],
            )

        occluded = self.occlusion.scene_occlusion(
            depth_img, rgb_img, self.camera.info['extrinsics'],
            self.camera.info['intrinsics']
        )
        # print(obj_poses)
        occlusion_label, occupied_label, occluded_list = self.occlusion.label_scene_occlusion(
            occluded,
            self.camera.info['extrinsics'],
            self.camera.info['intrinsics'],
            obj_poses,
            self.obj_pcds,
            depth_nn=1
        )

        return occlusion_label, occupied_label, occluded_list

    def free_space_grid(self, obj_ind):
        obj_i = obj_ind + 1
        obj_id = self.obj_ids[obj_ind]

        ws_low = self.workspace.region_low
        ws_high = self.workspace.region_high

        # get z coord for object placement
        mins, maxs = p.getAABB(obj_id, physicsClientId=self.pid)
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
        img[0,:] = 255
        img[-1,:] = 255
        img[:,0] = 255
        img[:,-1] = 255
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
            self.obj_poses,
            self.obj_colors,
            self.obj_names,
            self.occlusion,
            occupied_label,
            occlusion_label,
        )

    def pick(self, obj_name):
        pre_disp_dist = 0.06
        grip_offset = 0.0
        obj_id = self.obj_ids[self.name2ind[obj_name]]

        # choose closest arm
        l_arm_pos = p.getLinkState(
            self.robot.robot_id,
            self.robot.left_gripper_id,
            physicsClientId=self.pid,
        )[0]
        r_arm_pos = p.getLinkState(
            self.robot.robot_id,
            self.robot.right_gripper_id,
            physicsClientId=self.pid,
        )[0]
        obj_pos, _rot = p.getBasePositionAndOrientation(obj_id, physicsClientId=self.pid)
        # print('Left:', np.linalg.norm(np.subtract(l_arm_pos, obj_pos)))
        # print('Right:', np.linalg.norm(np.subtract(r_arm_pos, obj_pos)))
        if np.linalg.norm(np.subtract(l_arm_pos, obj_pos)) <= np.linalg.norm(np.subtract(
                r_arm_pos, obj_pos)):
            arms = ['left', 'right']
        else:
            arms = ['right', 'left']

        for chirality in arms:
            gripper_id = self.robot.left_gripper_id if chirality == 'left' else self.robot.right_gripper_id
            t0 = time.time()
            poses = self.robot.getGrasps(
                obj_id, offset2=(0, 0, grip_offset - pre_disp_dist)
            )
            fposes = self.robot.filterGrasps(gripper_id, poses)
            # pick arm with best grasp pose:
            # chirality, fposes = min(
            #     ('left', filterPosesLeft), ('right', filterPosesRight),
            #     key=lambda x: (len(x[1][0]['collisions']), x[1][0]['dist'])
            #     if len(x[1]) > 0 else (np.inf, np.inf)
            # )
            t1 = time.time()
            print("Grasp-sampling Time: ", t1 - t0)

            eof_poses = [
                x['eof_pose_offset'] for x in fposes if len(x['collisions']) == 0
            ]

            if len(eof_poses) == 0:
                print(f"No valid grasps of '{obj_name} for {chirality} arm'!")
                continue

            if self.prev_arm and chirality != self.prev_arm:
                self.planner.go_to_rest_pose()
            self.prev_arm = chirality

            res = self.planner.pick(
                f'Obj_{obj_id}',
                grasps=eof_poses,
                grip_offset=grip_offset,
                pre_disp_dist=pre_disp_dist,
                v_scale=0.35,
                a_scale=1.0,
                grasping_group=chirality + "_hand",
                group_name=chirality + "_arm",
            )
            if res is True:
                break
            print(f"Failed to pick '{obj_name} with {chirality} hand'!")
        if res is not True:
            print(f"Failed to pick '{obj_name}'!")
            self.planner.go_to_rest_pose()
            return False
        return True

    def place(self, obj_name):
        obj_ind = self.name2ind[obj_name]
        obj_id = self.obj_ids[obj_ind]
        res_grid = self.free_space_grid(obj_ind)
        xyzposes = random.sample(res_grid, min(10, len(res_grid)))
        res = self.planner.place(
            f'Obj_{obj_id}',
            xyzposes,
            v_scale=0.35,
            a_scale=1.0,
            grasping_group=self.prev_arm + "_hand",
            group_name=self.prev_arm + "_arm",
        )
        if res is not True:
            print(f"Failed to place '{obj_name}'!")
            self.planner.go_to_rest_pose()
            return False
        return True

    def pick_and_place(self):
        print("Objects in the scene: ", self.obj_names)
        obj_name = input('Select an object to pick: ')
        while obj_name != 'q':
            if obj_name == 'reset':
                self.planner.go_to_rest_pose()
            if obj_name in self.name2ind:
                if self.pick(obj_name):
                    self.place(obj_name)
            obj_name = input('Select an object to pick: ')

    def retrieve(self):
        # TODO ask about target object
        target_obj_name = 'red'
        target_obj = 1
        occlusion_label, occupied_label, occluded_list = self.sense()
        target_vox = occupied_label == target_obj
        target_vol = target_vox.sum()
        print("Targets volume:", target_vol)

        while True:
            dg = self.get_dep_graph()
            dg.draw_graph(False)
            dg.draw_graph(True)
            # print(dg.graph.nodes(data='dname'))
            if target_obj in dg.graph.nodes:
                break
            suggestion = input("Where is the red object?\n")
            result = dg.update_target_confidence(target_obj_name, suggestion, 0)
            dg.draw_graph(False)
            dg.draw_graph(True)
            # result = dg.update_target_confidence(1, suggestion, target_vol)
            # print(result)
            if result == target_obj:
                break
            print(dg.pick_order(result))
            for obj_name in dg.pick_order(result)[:-1]:
                self.pick(obj_name)
                self.place(obj_name)

        print(dg.pick_order(target_obj))
        for obj_name in dg.pick_order(target_obj)[:-1]:
            self.pick(obj_name)
            self.place(obj_name)
        self.pick(target_obj_name)
        input(f"Picked Object: {target_obj_name}")

    def choose_retrieve(self):
        obj_lang_ind = []
        while len(obj_lang_ind) == 0:
            user_lang = input("What do you want me to pick up?\n")
            for oname in self.obj_names:
                try:
                    obj_lang_ind.append((user_lang.index(oname), oname))
                except ValueError:
                    pass
        objects = [y for x, y in sorted(obj_lang_ind)[:2]]
        if len(objects) == 2:
            target_obj_name, suggestion = objects
        else:
            target_obj_name = objects[0]
            suggestion = None

        target_obj = self.name2ind[target_obj_name] + 1
        while True:
            dg = self.get_dep_graph()
            # dg.draw_graph(False)
            # dg.draw_graph(True)
            # print(dg.graph.nodes(data='dname'))
            if target_obj in dg.graph.nodes:
                break
            if suggestion is None:
                user_lang = input(f"Where is the {target_obj_name} object?\n")
                for oname in self.obj_names:
                    if oname in user_lang:
                        suggestion = oname
                        break
            result = dg.update_target_confidence(target_obj_name, suggestion, 0)
            dg.draw_graph(False)
            # dg.draw_graph(True)
            # print(result)
            if result == target_obj:
                break
            print('Picking in order:', dg.pick_order(result)[:-1])
            for obj_name in dg.pick_order(result)[:-1]:
                self.pick(obj_name)
                self.place(obj_name)

        print('Picking in order:', dg.pick_order(target_obj))
        for obj_name in dg.pick_order(target_obj)[:-1]:
            self.pick(obj_name)
            self.place(obj_name)
        self.pick(target_obj_name)
        input(f"Retrieved: {target_obj_name}!")
