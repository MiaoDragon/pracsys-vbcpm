import sys
import copy
from math import pi, tau, dist, fabs, cos

import numpy as np

import rospy
from std_msgs.msg import Header
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Quaternion
from moveit_msgs.msg import OrientationConstraint, Constraints, RobotState
from baxter_core_msgs.msg import EndEffectorCommand

import moveit_commander
from moveit_commander.conversions import *
from planit import Planner
from planit.utils import *

import pybullet as p
from robot import Robot


class BaxterPlanner(Planner):

    class MoveGroup(moveit_commander.MoveGroupCommander):

        def __init__(
            self,
            name,
            robot_description="robot_description",
            ns="",
            wait_for_servers=5.0,
            rate=240
        ):
            super().__init__(name, robot_description, ns, wait_for_servers)
            self.gripper_left_cmd = 'open'
            self.gripper_right_cmd = 'open'
            self.pos_left = None
            self.pos_right = None
            self.force = 0.05
            self.attached_obj = None
            # self.cmode = p.POSITION_CONTROL
            self.robot = None
            self.rate = rospy.Rate(rate)

        def execute(self, plan_msg, wait=False):
            short_plan_msg = copy.deepcopy(plan_msg)
            short_plan_msg.joint_trajectory.points = [
                copy.deepcopy(plan_msg.joint_trajectory.points[0]),
                copy.deepcopy(plan_msg.joint_trajectory.points[-1]),
            ]
            short_duration = rospy.Duration.from_sec(0.01)
            short_plan_msg.joint_trajectory.points[-1].time_from_start = short_duration
            super().execute(short_plan_msg, wait)

            joint_names = plan_msg.joint_trajectory.joint_names
            indices = [self.robot.joint_name2ind[x] for x in joint_names]
            if len(joint_names) == 7:
                arm = joint_names[0].split('_')[0]
                group = arm + '_arm'
            elif len(joint_names) == 2:
                arm = 'left' if joint_names[0].split('_')[0] == 'l' else 'right'
                group = arm + '_hand'
            else:
                arm = 'both'
                group = arm + '_arms'
            if self.attached_obj is not None and arm != 'both':
                init_gripper_pose = self.robot.get_gripper_pose(arm)
                init_object_pose = p.getBasePositionAndOrientation(
                    self.attached_obj,
                    physicsClientId=self.robot.pybullet_id,
                )[0:2]
            # print(joint_names,len(joint_names))
            # print(indices)
            points = plan_msg.joint_trajectory.points

            # jump to goal state instanstly in moveit planning scene
            # goal = JointState()
            # goal.header = Header()
            # goal.header.stamp = rospy.Time.now()
            # goal.name = joint_names
            # goal.position = points[-1].positions
            # robot_state = RobotState()
            # robot_state.joint_state = goal
            # self.set_start_state(robot_state)

            p_secs = 0
            for point in points[1:]:
                secs = point.time_from_start.to_sec()
                poss = point.positions
                # print(secs)
                # print(poss)
                if secs == 0:
                    continue
                num_steps = int(240 * (secs - p_secs))
                p_secs = secs
                for i in range(num_steps):
                    p.setJointMotorControlArray(
                        bodyIndex=self.robot.robot_id,
                        jointIndices=indices,
                        controlMode=p.POSITION_CONTROL,
                        targetPositions=poss,
                        physicsClientId=self.robot.pybullet_id
                    )
                    self.robot.set_gripper(
                        'left',
                        state=self.gripper_left_cmd,
                        reset=False,
                        tgt_pos=self.pos_left,
                        force=self.force,
                        # mode=self.cmode,
                    )
                    self.robot.set_gripper(
                        'right',
                        state=self.gripper_right_cmd,
                        reset=False,
                        tgt_pos=self.pos_right,
                        force=self.force,
                        # mode=self.cmode,
                    )
                    # print(self.gripper_left_cmd)
                    # print(self.gripper_right_cmd)
                    p.stepSimulation()

                    # print("***ID2:", self.attached_obj)
                    if self.attached_obj is not None:
                        new_gripper_pose = self.robot.get_gripper_pose(arm)
                        rel_transform = p.multiplyTransforms(
                            *new_gripper_pose,
                            *p.invertTransform(*init_gripper_pose),
                        )
                        new_object_pose = p.multiplyTransforms(
                            *rel_transform,
                            *init_object_pose,
                        )
                        p.resetBasePositionAndOrientation(
                            self.attached_obj,
                            *new_object_pose,
                            physicsClientId=self.robot.pybullet_id,
                        )
                        init_object_pose = new_object_pose
                        init_gripper_pose = new_gripper_pose
                    self.rate.sleep()

    def __init__(
        self, robot, is_sim=False, commander_args=['joint_states:=/robot/joint_states']
    ):

        self.pb_robot = robot
        gripper_width = robot.right_flim[1] * 2  # gripper width
        # print(gripper_width)
        super().__init__(gripper_width, is_sim, commander_args)
        self.move_group_left = self.MoveGroup('left_arm')
        self.move_group_left.robot = self.pb_robot
        self.move_group_left.force = 1.05
        self.move_group_right = self.MoveGroup('right_arm')
        self.move_group_right.robot = self.pb_robot
        self.move_group_right.force = 1.05
        self.move_group_both = self.MoveGroup('both_arms')
        self.move_group_both.robot = self.pb_robot
        self.move_group_both.force = 1.05
        self.move_group_left_hand = self.MoveGroup('left_hand')
        self.move_group_left_hand.robot = self.pb_robot
        self.move_group_left_hand.force = 0.05
        # self.move_group_left_hand.cmode = p.VELOCITY_CONTROL
        self.move_group_right_hand = self.MoveGroup('right_hand')
        self.move_group_right_hand.robot = self.pb_robot
        self.move_group_right_hand.force = 0.05
        # self.move_group_right_hand.cmode = p.VELOCITY_CONTROL
        self.name2group = {
            'left_arm': self.move_group_left,
            'right_arm': self.move_group_right,
            'both_arms': self.move_group_both,
            'left_hand': self.move_group_left_hand,
            'right_hand': self.move_group_right_hand,
        }

        if not is_sim:
            self.eef_pub_left = rospy.Publisher(
                '/robot/end_effector/left_gripper/command',
                EndEffectorCommand,
                queue_size=5,
            )
            self.eef_pub_right = rospy.Publisher(
                '/robot/end_effector/right_gripper/command',
                EndEffectorCommand,
                queue_size=5,
            )

    @staticmethod
    def gen_gripper_constraint(quat, gripper):
        ocm = OrientationConstraint()
        ocm.link_name = gripper
        ocm.header.frame_id = "world"
        if type(quat) in (list, tuple, np.ndarray):
            ocm.orientation.x = quat[0]
            ocm.orientation.y = quat[1]
            ocm.orientation.z = quat[2]
            ocm.orientation.w = quat[3]
        elif type(quat) is Quaternion:
            ocm.orientation = quat
        else:
            ocm.orientation.w = 1.0
        ocm.orientation.w = 1.0
        ocm.absolute_x_axis_tolerance = 0.001
        ocm.absolute_y_axis_tolerance = 0.001
        ocm.absolute_z_axis_tolerance = 0.001
        ocm.weight = 1.0
        return ocm

    def do_end_effector(
        self,
        command,
        group_name="left_hand",
        # obj_id=None,
    ):
        if self.is_sim:
            # if obj_id is None:
            #     pos_off = 0.0
            # else:
            #     shape = p.getCollisionShapeData(obj_id, -1, self.pb_robot.pybullet_id)[0]
            #     if shape[2] == p.GEOM_BOX:
            #         sx, sy, sz = shape[3]
            #     elif shape[2] == p.GEOM_CYLINDER or shape[2] == p.GEOM_CAPSULE:
            #         h, r = shape[3][:2]
            #     elif shape[2] == p.GEOM_SPHERE:
            #         r = shape[3][0]
            #     pos_off = 0.0
            if group_name == "left_hand":
                self.move_group_left.gripper_left_cmd = command
                self.move_group_right.gripper_left_cmd = command
                self.move_group_left_hand.gripper_left_cmd = command
                self.move_group_right_hand.gripper_left_cmd = command
            if group_name == "right_hand":
                self.move_group_left.gripper_right_cmd = command
                self.move_group_right.gripper_right_cmd = command
                self.move_group_left_hand.gripper_right_cmd = command
                self.move_group_right_hand.gripper_right_cmd = command

            # make sim look nice
            # move_group = moveit_commander.MoveGroupCommander(group_name)
            move_group = self.name2group[group_name]
            joint_goal = move_group.get_current_joint_values()
            if command == 'open':
                joint_goal = [0.02, -0.02]
                self.move_group_left_hand.force = 10.0
                self.move_group_right_hand.force = 10.0
                self.move_group_left.attached_obj = None
                self.move_group_right.attached_obj = None
            elif command == 'close':
                joint_goal = [0.02, -0.02]
                self.move_group_left_hand.force = 0.04
                self.move_group_right_hand.force = 0.04
            # move_group.go(joint_goal, wait=True)
            move_group.set_joint_value_target(joint_goal)
            success, plan, planning_time, error_code = move_group.plan()
            # print(plan)
            if len(plan.joint_trajectory.points) > 0:
                last = copy.deepcopy(plan.joint_trajectory.points[-1])
                last.time_from_start = rospy.Duration.from_sec(0.5)
                plan.joint_trajectory.points.append(last)
            # print(plan)
            move_group.clear_pose_targets()
            move_group.execute(plan, wait=True)
            move_group.stop()

            if command == 'close':
                for i in range(p.getNumBodies(physicsClientId=self.pb_robot.pybullet_id)):
                    obj_pid = p.getBodyUniqueId(
                        i,
                        physicsClientId=self.pb_robot.pybullet_id,
                    )
                    if obj_pid in [0, 1]:
                        continue
                    contacts = p.getClosestPoints(
                        self.pb_robot.robot_id,
                        obj_pid,
                        distance=0.0,
                        physicsClientId=self.pb_robot.pybullet_id,
                    )
                    if len(contacts):
                        break
                    # print("***ID:", obj_pid)
                self.move_group_left.attached_obj = obj_pid
                self.move_group_right.attached_obj = obj_pid

            # print(last)
            if group_name == "left_hand":
                joints = p.getJointStates(
                    self.pb_robot.robot_id, self.pb_robot.left_fingers
                )
                # print(joints)
                # joints = [joints[0][0], -joints[0][0]]
                joints = [p[0] for p in joints]
                # print(self.pb_robot.left_flim)
                self.move_group_left.pos_left = joints
                self.move_group_right.pos_left = joints
            if group_name == "right_hand":
                joints = p.getJointStates(
                    self.pb_robot.robot_id, self.pb_robot.right_fingers
                )
                # print(joints)
                # joints = [joints[0][0], -joints[0][0]]
                joints = [p[0] for p in joints]
                # print(self.pb_robot.right_flim)
                self.move_group_left.pos_right = joints
                self.move_group_right.pos_right = joints
            print("Sustain joint positions: ", joints, file=sys.stderr)
        else:
            eef_cmd = EndEffectorCommand()
            eef_cmd.id = 65664
            if command == 'open':
                eef_cmd.command = 'go'
                eef_cmd.args = '{\"position\":100.0, \"force\":100}'
            elif command == 'close':
                eef_cmd.command = 'go'
                eef_cmd.args = '{\"position\":0.0, \"force\":100}'
            if group_name == 'left_hand':
                self.eef_pub_left.publish(eef_cmd)
            elif group_name == 'right_hand':
                self.eef_pub_right.publish(eef_cmd)
            else:
                print("Invalid group name!", file=sys.stderr)

    def pick(
        self,
        obj_name,
        constraints=None,
        grasps=None,
        grip_offset=0.0,
        pre_disp_dist=0.06,
        post_disp_dir=(0, 0, 1),
        post_disp_dist=0.06,
        eef_step=0.001,
        jump_threshold=5.0,
        v_scale=0.25,
        a_scale=1.0,
        grasping_group="left_hand",
        group_name="left_arm",
    ):
        # move_group = self.MoveGroup(group_name)
        # move_group.robot = self.pb_robot
        move_group = self.name2group[group_name]
        move_group.set_num_planning_attempts(25)
        move_group.set_planning_time(10.0)
        if constraints is not None:
            self.move_group.set_path_constraints(constraints)
        # move_group.set_support_surface_name(support)

        # open gripper
        self.do_end_effector('open', group_name=grasping_group)

        # plan to pre goal poses
        if grasps:
            poses = grasps
        else:
            poses = self.grasps.get_simple_grasps(
                obj_name, (0, 0, grip_offset - pre_disp_dist)
            )
        move_group.set_pose_targets(poses)
        success, raw_plan, planning_time, error_code = move_group.plan()
        move_group.clear_pose_targets()
        if not success:
            self.do_end_effector('close', group_name=grasping_group)
            return error_code
        else:
            print(
                "Planned pick for",
                obj_name,
                "in",
                planning_time,
                "seconds.",
                file=sys.stderr
            )

        # retime and execute trajectory
        plan = move_group.retime_trajectory(
            self.robot.get_current_state(),
            raw_plan,
            velocity_scaling_factor=v_scale,
            acceleration_scaling_factor=a_scale,
        )
        move_group.execute(plan, wait=True)
        move_group.stop()

        # slide to goal
        cpose = pose_msg2homogeneous(move_group.get_current_pose().pose)
        trans = translation_matrix((0, 0, pre_disp_dist))
        wpose = homogeneous2pose_msg(concatenate_matrices(cpose, trans))
        waypoints = [copy.deepcopy(wpose)]
        raw_plan, fraction = move_group.compute_cartesian_path(
            waypoints, eef_step, jump_threshold, avoid_collisions=False
        )
        print(
            "Planned approach",
            fraction * pre_disp_dist,
            "for",
            obj_name,
            ".",
            file=sys.stderr
        )
        if fraction < 0.5:
            return -fraction

        # retime and execute trajectory
        plan = move_group.retime_trajectory(
            self.robot.get_current_state(),
            raw_plan,
            velocity_scaling_factor=v_scale,
            acceleration_scaling_factor=a_scale,
        )
        move_group.execute(plan, wait=True)
        move_group.stop()
        print("Approached", obj_name, ".", file=sys.stderr)

        # close gripper
        self.do_end_effector('close', group_name=grasping_group)

        # attach to robot chain
        success = self.attach(
            obj_name, grasping_group=grasping_group, group_name=group_name
        )
        if success:
            print("Picked", obj_name, ".", file=sys.stderr)
        else:
            self.do_end_effector('open', group_name=grasping_group)
            return -1

        # displace
        scale = post_disp_dist / dist(post_disp_dir, (0, 0, 0))
        wpose = move_group.get_current_pose().pose
        wpose.position.x += scale * post_disp_dir[0]
        wpose.position.y += scale * post_disp_dir[1]
        wpose.position.z += scale * post_disp_dir[2]
        waypoints = [copy.deepcopy(wpose)]
        raw_plan, fraction = move_group.compute_cartesian_path(
            waypoints, eef_step, jump_threshold, avoid_collisions=False
        )
        print(
            "Planned displacement",
            fraction * post_disp_dist,
            "for",
            obj_name,
            ".",
            file=sys.stderr
        )
        if fraction < 0.5:
            return -fraction

        # retime and execute trajectory
        plan = move_group.retime_trajectory(
            self.robot.get_current_state(),
            raw_plan,
            velocity_scaling_factor=v_scale,
            acceleration_scaling_factor=a_scale,
        )
        move_group.execute(plan, wait=True)
        move_group.stop()
        move_group.clear_path_constraints()
        print("Displaced", obj_name, ".", file=sys.stderr)
        return True

    def place(
        self,
        obj_name,
        partial_poses,
        constraints=None,
        pre_disp_dir=(0, 0, 1),
        pre_disp_dist=0.06,
        post_disp_dir=(0, 0, 1),
        post_disp_dist=0.12,
        eef_step=0.005,
        jump_threshold=5.0,
        v_scale=0.25,
        a_scale=1.0,
        grasping_group="left_hand",
        group_name="left_arm",
    ):
        # move_group = moveit_commander.MoveGroupCommander(group_name)
        move_group = self.name2group[group_name]
        move_group.set_num_planning_attempts(25)
        move_group.set_planning_time(10.0)
        if constraints is not None:
            move_group.set_path_constraints(constraints)
        # move_group.set_support_surface_name(support)

        # plan preplacement
        displacement = [pre_disp_dist * x for x in pre_disp_dir]
        # poses = self.grasps.get_simple_placements(obj_name, partial_pose, displacement)
        poses = []
        for pose in partial_poses:
            # print(pose)
            epose = move_group.get_current_pose().pose
            epose.position.x = pose[0] - displacement[0]
            epose.position.y = pose[1] - displacement[1]
            # epose.position.z -= displacement[2]  #* 1.5
            # epose.position.z = pose[2] - displacement[2]
            epose.position.z -= pose[2] - displacement[2]
            poses.append(copy.deepcopy(epose))
            # print(epose.position)
        # poses = [copy.deepcopy(epose)]

        # gripper = "left_gripper" if grasping_group == "left_hand" else "right_gripper"
        # constrs = Constraints()
        # constrs.orientation_constraints = [
        #     self.gen_gripper_constraint(epose.orientation, gripper)
        # ]
        # move_group.set_path_constraints(constrs)

        success, raw_plan, planning_time, error_code = self.plan_ee_poses(
            poses, group_name=group_name
        )
        if not success:
            return error_code
        else:
            print(
                "Planned placement for",
                obj_name,
                "in",
                planning_time,
                "seconds.",
                file=sys.stderr
            )

        # retime and execute trajectory
        plan = move_group.retime_trajectory(
            self.robot.get_current_state(),
            raw_plan,
            velocity_scaling_factor=v_scale,
            acceleration_scaling_factor=a_scale,
        )
        move_group.execute(plan, wait=True)
        move_group.stop()
        print("Preplaced.", file=sys.stderr)

        # slide to placement
        scale = 0.95 * pre_disp_dist / dist(pre_disp_dir, (0, 0, 0))
        wpose = move_group.get_current_pose().pose
        wpose.position.x += scale * pre_disp_dir[0] * -1
        wpose.position.y += scale * pre_disp_dir[1] * -1
        wpose.position.z += scale * pre_disp_dir[2] * -1
        waypoints = [copy.deepcopy(wpose)]
        raw_plan, fraction = move_group.compute_cartesian_path(
            waypoints, eef_step, jump_threshold, avoid_collisions=True
        )
        print(
            "Planned placement approach",
            fraction * pre_disp_dist,
            "for",
            obj_name,
            ".",
            file=sys.stderr
        )
        if fraction < 0.5:
            # return -fraction
            print("Skipping Approach. Dropping Object instead!", file=sys.stderr)
        else:
            # retime and execute trajectory
            plan = move_group.retime_trajectory(
                self.robot.get_current_state(),
                raw_plan,
                velocity_scaling_factor=v_scale,
                acceleration_scaling_factor=a_scale,
            )
            move_group.execute(plan, wait=True)
            move_group.stop()
            print("Approached placement for", obj_name, ".", file=sys.stderr)

        # open gripper
        self.do_end_effector('open', group_name=grasping_group)

        # detach from robot chain
        success = self.detach(obj_name, group_name=group_name)
        if success:
            print("Placed", obj_name, ".", file=sys.stderr)
        else:
            return -1

        # displace
        scale = post_disp_dist / dist(post_disp_dir, (0, 0, 0))
        wpose = move_group.get_current_pose().pose
        wpose.position.x += scale * post_disp_dir[0]
        wpose.position.y += scale * post_disp_dir[1]
        wpose.position.z += scale * post_disp_dir[2]
        waypoints = [copy.deepcopy(wpose)]
        raw_plan, fraction = move_group.compute_cartesian_path(
            waypoints, eef_step, jump_threshold, avoid_collisions=False
        )
        print(
            "Planned displacement",
            fraction * post_disp_dist,
            "from",
            obj_name,
            ".",
            file=sys.stderr
        )
        if fraction < 0.5:
            return -fraction

        # retime and execute trajectory
        plan = move_group.retime_trajectory(
            self.robot.get_current_state(),
            raw_plan,
            velocity_scaling_factor=v_scale,
            acceleration_scaling_factor=a_scale,
        )
        move_group.execute(plan, wait=True)
        move_group.stop()
        move_group.clear_path_constraints()
        print("Displaced", obj_name, ".", file=sys.stderr)
        return True

    def go_to_rest_pose(
        self,
        constraints=None,
        v_scale=0.25,
        a_scale=1.0,
        group_name="both_arms",
    ):
        move_group = self.name2group[group_name]
        move_group.set_num_planning_attempts(25)
        move_group.set_planning_time(10.0)
        if constraints is not None:
            self.move_group.set_path_constraints(constraints)
        # move_group.set_support_surface_name(support)

        # open gripper
        self.do_end_effector('open', group_name='left_hand')
        self.do_end_effector('open', group_name='right_hand')
        # self.detach(None, group_name='left_hand')
        # self.detach(None, group_name='right_hand')
        self.scene.remove_attached_object()
        # move_group.detach_object()

        # plan to pre goal poses
        cur_joints = move_group.get_current_joint_values()
        move_group.set_joint_value_target([0] * len(cur_joints))
        success, raw_plan, planning_time, error_code = move_group.plan()
        move_group.clear_pose_targets()
        if not success:
            return error_code
        else:
            print(
                "Planned reset for",
                group_name,
                "in",
                planning_time,
                "seconds.",
                file=sys.stderr
            )

        # retime and execute trajectory
        plan = move_group.retime_trajectory(
            self.robot.get_current_state(),
            raw_plan,
            velocity_scaling_factor=v_scale,
            acceleration_scaling_factor=a_scale,
        )
        move_group.execute(plan, wait=True)
        move_group.stop()
        move_group.clear_path_constraints()
