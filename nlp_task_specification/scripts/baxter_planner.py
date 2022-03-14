import sys
import copy
from math import pi, tau, dist, fabs, cos

import rospy
# import moveit_msgs.msg
# import geometry_msgs.msg
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
            self.robot = None
            self.rate = rospy.Rate(rate)

        def execute(self, plan_msg, wait=False):
            super().execute(plan_msg, wait)
            joint_names = plan_msg.joint_trajectory.joint_names
            indices = [self.robot.joint_name2ind[x] for x in joint_names]
            # print(joint_names)
            # print(indices)
            points = plan_msg.joint_trajectory.points
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
                    self.robot.set_gripper('left', state=self.gripper_left_cmd)
                    self.robot.set_gripper('right', state=self.gripper_right_cmd)
                    # print(self.gripper_left_cmd)
                    # print(self.gripper_right_cmd)
                    p.stepSimulation()
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
        self.move_group_right = self.MoveGroup('right_arm')
        self.move_group_right.robot = self.pb_robot
        self.name2group = {
            'left_arm': self.move_group_left,
            'right_arm': self.move_group_right
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

    def do_end_effector(self, command, group_name="left_hand"):
        if self.is_sim:
            if group_name == "left_hand":
                self.move_group_left.gripper_left_cmd = command
                self.move_group_right.gripper_left_cmd = command
            if group_name == "right_hand":
                self.move_group_left.gripper_right_cmd = command
                self.move_group_right.gripper_right_cmd = command

            # make sim look nice
            move_group = moveit_commander.MoveGroupCommander(group_name)
            joint_goal = move_group.get_current_joint_values()
            if command == 'open':
                joint_goal = [0.02, -0.02]
            elif command == 'close':
                joint_goal = [0.018, -0.018]
            move_group.go(joint_goal, wait=True)
            move_group.stop()
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
        grip_offset=0.01,
        pre_disp_dist=0.05,
        post_disp_dir=(0, 0, 1),
        post_disp_dist=0.05,
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
            print("Planned pick for", obj_name, "in", planning_time, "seconds.")

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
        print("Planned approach", fraction * pre_disp_dist, "for", obj_name, ".")
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
        print("Approached", obj_name, ".")

        # close gripper
        self.do_end_effector('close', group_name=grasping_group)

        # attach to robot chain
        success = self.attach(
            obj_name, grasping_group=grasping_group, group_name=group_name
        )
        if success:
            print("Picked", obj_name, ".")
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
            waypoints, eef_step, jump_threshold, avoid_collisions=True
        )
        print("Planned displacement", fraction * post_disp_dist, "for", obj_name, ".")
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
        print("Displaced", obj_name, ".")
        return True

    def place(
        self,
        obj_name,
        partial_pose,
        constraints=None,
        pre_disp_dir=(0, 0, 1),
        pre_disp_dist=0.05,
        post_disp_dir=(0, 0, 1),
        post_disp_dist=0.1,
        eef_step=0.005,
        jump_threshold=0.0,
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
        for pose in partial_pose:
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

        success, raw_plan, planning_time, error_code = self.plan_ee_poses(
            poses, group_name=group_name
        )
        if not success:
            return error_code
        else:
            print("Planned placement for", obj_name, "in", planning_time, "seconds.")

        # retime and execute trajectory
        plan = move_group.retime_trajectory(
            self.robot.get_current_state(),
            raw_plan,
            velocity_scaling_factor=v_scale,
            acceleration_scaling_factor=a_scale,
        )
        move_group.execute(plan, wait=True)
        move_group.stop()
        print("Preplaced.")

        # slide to placement
        scale = pre_disp_dist / dist(pre_disp_dir, (0, 0, 0))
        wpose = move_group.get_current_pose().pose
        wpose.position.x += scale * pre_disp_dir[0] * -1
        wpose.position.y += scale * pre_disp_dir[1] * -1
        wpose.position.z += scale * pre_disp_dir[2] * -1
        waypoints = [copy.deepcopy(wpose)]
        raw_plan, fraction = move_group.compute_cartesian_path(
            waypoints, eef_step, jump_threshold, avoid_collisions=False
        )
        print(
            "Planned placement approach", fraction * pre_disp_dist, "for", obj_name, "."
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
        print("Approached placement for", obj_name, ".")

        # open gripper
        self.do_end_effector('open', group_name=grasping_group)

        # detach from robot chain
        success = self.detach(obj_name, group_name=group_name)
        if success:
            print("Placed", obj_name, ".")
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
        print("Planned displacement", fraction * post_disp_dist, "from", obj_name, ".")
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
        print("Displaced", obj_name, ".")
        return True
