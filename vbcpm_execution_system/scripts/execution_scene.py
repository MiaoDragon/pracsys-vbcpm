"""
This defines the setup in PyBullet for the Vision-Based Constrained Placement Task.
Taking input of a problem configuration (in the format of a json file), we load each
component into the PyBullet scene. We then set up the services and topics to communicate
with the planner, including a trajectory tracker service, a camera-related topic, and
a robot-related topic.
"""
import argparse
import utility
import pybullet as p
import rospy
import rospkg
import actionlib
import os
import time
import cv2
import numpy as np
from sensor_msgs.msg import JointState
import control_msgs.msg
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from vbcpm_execution_system.srv import RobotiqControl, RobotiqControlResponse
import geo_track_utility

import time
import matplotlib.pyplot as plt
from rosgraph_msgs.msg import Clock
class ExecutionScene():
    def __init__(self, args):
        rp = rospkg.RosPack()
        package_path = rp.get_path('vbcpm_execution_system')
        self.package_path = package_path
        f = args.prob_config  # problem configuration file (JSON)
        prob_config_dict = utility.prob_config_parser(f)
        # * load each component in the problem configuration into the scene
        """
        format: 
            {'robot': {'pose': pose, 'urdf': urdf},
            'table': {'pose': pose, 'urdf': urdf},  # (orientation: [x,y,z,w])
            'objects': [{'pose': pose, 'urdf': urdf}],
            'camera': {'pose': pose, 'urdf': urdf},
            'placement': [(x1,y1),(x2,y2),(x3,y3),(x4,y4)]}
        """
        self.physics_client = p.connect(p.GUI)
        # p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
        p.setGravity(0,0,-10)
        plane_id = p.loadURDF("../data/models/plane/plane.urdf")

        # robot
        robot_pos = prob_config_dict['robot']['pose']['pos']
        robot_ori = prob_config_dict['robot']['pose']['ori']
        print('robot_path: ', os.path.join(package_path, prob_config_dict['robot']['urdf']))
        robot_id = p.loadURDF(os.path.join(package_path, prob_config_dict['robot']['urdf']),robot_pos, robot_ori, useFixedBase=True)
        self.robot_id = robot_id
        # preprocess: loop over all joints and store them
        self.num_joints = p.getNumJoints(self.robot_id)
        self.joint_names = []
        self.joint_indices = []
        self.joint_name_ind_dict = {}
        self.last_tracked_vel = {}

        for i in range(self.num_joints):
            joint_info = p.getJointInfo(self.robot_id, i)
            if joint_info[2] != p.JOINT_FIXED:
                joint_index = joint_info[0]
                joint_name = joint_info[1].decode("utf-8")
                self.joint_names.append(joint_name)
                self.joint_indices.append(joint_index)
                self.joint_name_ind_dict[joint_name] = joint_index

                self.last_tracked_vel[joint_name] = 0.
        print('joint_names: ')
        print(self.joint_names)
        self.current_joint_pos = np.zeros(len(self.joint_indices))
        self.current_joint_vel = np.zeros(len(self.joint_indices))



        # table
        table_pos = prob_config_dict['table']['pose']['pos']
        table_ori = prob_config_dict['table']['pose']['ori']
        table_id = p.loadURDF(os.path.join(package_path, prob_config_dict['table']['urdf']),table_pos, table_ori, useFixedBase=True)
        self.table_id = table_id

        # camera

        view_mat = p.computeViewMatrix(
            cameraEyePosition=[0.35, 0, 1.25],
            cameraTargetPosition=[1.35, 0, 0.58],
            cameraUpVector=[1.25 - 0.58, 0, 1.35 - 0.35]
        )

        # https://stackoverflow.com/questions/60430958/understanding-the-view-and-projection-matrix-from-pybullet
        # https://stackoverflow.com/questions/39992968/how-to-calculate-field-of-view-of-the-camera-from-camera-intrinsic-matrix
        # https://github.com/bulletphysics/bullet3/blob/master/examples/SharedMemory/PhysicsClientC_API.cpp#L4372
        proj_mat = p.computeProjectionMatrixFOV(
            fov=90,
            aspect=1,
            nearVal=0.01,
            farVal=1.5
        )
        self.view_mat = view_mat
        self.proj_mat = proj_mat

        # objects
        objs = []
        for obj_dict in prob_config_dict['objects']:
            obj_i_c = p.createCollisionShape(shapeType=p.GEOM_MESH, meshScale=obj_dict['scale'], \
                                            fileName=os.path.join(package_path, obj_dict['collision_mesh']))
            obj_i_v = p.createVisualShape(shapeType=p.GEOM_MESH, meshScale=obj_dict['scale'], \
                                            fileName=os.path.join(package_path, obj_dict['visual_mesh']))
            obj_i = p.createMultiBody(baseCollisionShapeIndex=obj_i_c, baseVisualShapeIndex=obj_i_v, \
                            basePosition=obj_dict['pose']['pos'], baseOrientation=obj_dict['pose']['ori'],
                            baseMass=obj_dict['mass'])
            objs.append(obj_i)
        self.objs = objs


        # * start running the ROS service for the robot topic
        self.joint_state_pub = rospy.Publisher('joint_states', JointState, queue_size=10)
        # * start running the ROS service for the camera

        # * start running the ROS service for trajectory tracker
        # Geometric Waypoints -> Trajectory Tracker
        self.vel_limit = [-15*np.pi/180,15*np.pi/180]
        self.acc_limit = [-500.*np.pi/180,500.*np.pi/180]
        self.geo_track_as = actionlib.SimpleActionServer("geometric_trajectory_server", control_msgs.msg.FollowJointTrajectoryAction,
                            execute_cb=self.geo_track_cb, auto_start = False)
        # Geometric & Kinematic Waypoints -> Trajectory Tracker
        self.traj_track_as = actionlib.SimpleActionServer("trajectory_server", control_msgs.msg.FollowJointTrajectoryAction,
                            execute_cb=self.traj_track_cb, auto_start = False)
        self.traj_track_as.start()
        self.geo_track_as.start()

        self.traj_client = actionlib.SimpleActionClient('trajectory_server', control_msgs.msg.FollowJointTrajectoryAction)

        # Gripper
        self.gripper_srv = rospy.Service('robotiq_controller', RobotiqControl, self.gripper_cb)


        self.vision = 0
        self.exec_lock = 0  # a lock to make sure when we execute traj, we don't simulate twice
        
        self.sim_time = 1/240.

        # for main step loop, to take care of tracking service
        self.traj_track_flag = 0  # idle
        self.gripper_track_flag = 0

        # for i in range(p.getNumJoints(self.robot_id)):
        #     dynamics = p.getDynamicsInfo(self.robot_id, i)
        #     if 'arm' in p.getJointInfo(self.robot_id, i)[12].decode("utf-8"):
        #         # change arm mass to higher
        #         p.changeDynamics(self.robot_id, i, mass=50.)


        # clock for ROS time
        self.clock_pub = rospy.Publisher('clock', Clock, queue_size=10)
        self.cur_time = rospy.get_time()


    def joint_state_publish(self):
        """
        Running outside is a ROS node: robot_state_publisher, which publishes the TF info for each link
        """
        # get current joint states
        res = p.getJointStates(self.robot_id, self.joint_indices)
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = self.joint_names
        joint_pos = []
        joint_vel = []
        # joint_effort = []  # in Motoman we don't have these
        for i in range(len(res)):
            joint_pos.append(res[i][0])
            joint_vel.append(res[i][1])
        self.current_joint_pos = np.array(joint_pos)
        self.current_joint_vel = np.array(joint_vel)
        msg.position = joint_pos
        self.joint_state_pub.publish(msg)

    
    def geo_track_cb(self, goal):
        """
        Given Geometric waypoints, (compute kinematics waypoints and) track the trajectory
        """
        geo_traj = goal.trajectory

        names = geo_traj.joint_names
        pos_traj = []
        for point in geo_traj.points:
            pos_traj.append(point.positions)
        # init velocity using the previous velocity value
        init_vel = []
        for i in range(len(names)):
            # last_tracked_vel: name -> vel
            if names[i] in self.last_tracked_vel:
                init_vel.append(self.last_tracked_vel[names[i]])
            else:
                init_vel.append(0.)

        pos_traj = np.array(pos_traj)
        init_vel = np.array(init_vel)
        print('pos_traj: ')
        print(pos_traj.shape)
        print(pos_traj[0])
        print('init_vel: ')
        print(init_vel.shape)
        print(init_vel)

        vel_lls, vel_uls, vel_half_lls, vel_half_uls = geo_track_utility.compute_vel_limit(pos_traj, init_vel, self.vel_limit, self.acc_limit)
        new_pos_traj, vel_traj, time_traj = geo_track_utility.compute_vel_time_qcqp(pos_traj, init_vel, vel_lls, vel_uls, vel_half_lls, vel_half_uls, self.vel_limit, self.acc_limit)
        # set the trajectory
        vel_traj = np.array(vel_traj)
        time_traj = np.array(time_traj)
        #new_pos_traj = np.array(pos_traj)
        print('after computing vel time qcqp')
        rospy.loginfo('computed velocity and time.')
        ori_pos_traj = pos_traj
        pos_traj = new_pos_traj
        # set trajectory to follow
        traj = JointTrajectory()
        traj.joint_names = geo_traj.joint_names
        points = []

        # Note: Below uses the original waypoint for tracking
        # TODO: use the actual computed split for tracking
        # below is for QCQP approach
        # for i in range(len(pos_traj[::3])):
        #     point = JointTrajectoryPoint()
        #     point.positions = pos_traj[::3][i]
        #     point.velocities = vel_traj[::3][i]
        #     point.time_from_start = rospy.Duration(time_traj[::3][i])
        #     points.append(point)
        for i in range(len(pos_traj)):
            point = JointTrajectoryPoint()
            point.positions = pos_traj[i]
            point.velocities = vel_traj[i]
            point.time_from_start = rospy.Duration(time_traj[i])
            points.append(point)


        # last point we want the vel to be zero
        point = JointTrajectoryPoint()
        point.positions = pos_traj[-1]
        point.velocities = np.zeros(7)
        point.time_from_start = rospy.Duration(time_traj[-1]+5.)  # allow a large enoguh time for stablizing
        points.append(point)

        traj.points = points
        result, tracked_traj, tracked_time_traj = self.traj_track(traj)

        for i in range(len(names)):
            # fig = plt.figure()
            fig = plt.figure()

            plt.plot(time_traj, new_pos_traj[:,i], label='calculated')

            # QCQP plot every 3
            plt.scatter(time_traj[::3], ori_pos_traj[:,i], label='calculated', s=1, c='green')
            # plt.scatter(time_traj, ori_pos_traj[:,i], label='calculated')
            
            # plt.plot(ts_total, pos_total, label='calculated')
            # plt.scatter(time_traj, pos_traj[:,i])
            # plt.scatter([time_traj[i]],[pos_traj[i][j]], c=colors[j])


            # fig.savefig('calculated_position_%s.pdf' % (arm_joint.name[i]))

            # obtain the time
            tracked_joint_traj = []
            for j in range(len(tracked_traj)):
                tracked_joint_traj.append(tracked_traj[j][names[i]])
            tracked_time_traj = np.array(tracked_time_traj)
            tracked_time_traj = tracked_time_traj - tracked_time_traj[0]
            plt.plot(tracked_time_traj, tracked_joint_traj, label='tracked')
            plt.legend()
            fig.savefig('position_%s.pdf' % (names[i]))       



        self.geo_track_as.set_succeeded(result)
        print('geo_track_cb: end of tracking.')
    def traj_track(self, traj):
        print('traj to track: ')
        print(traj.joint_names)
        print('first 5: ')
        print(traj.points[:5])
        print('last 5: ')
        print(traj.points[-5:])
        # Waits until the action server has started up and started
        # listening for goals.
        client = self.traj_client
        client.wait_for_server()

        action_goal = FollowJointTrajectoryGoal()
        #goal.trajectory = left_joint_trajectory
        action_goal.trajectory = traj
        feedbacks = []
        self.done = False
        def feedback_cb(feedback):
            rospy.loginfo('receiving feedback...')
            feedbacks.append(feedback)
        def done_cb(state, result):
            rospy.loginfo("Action server is done. State: %s, result: %s" % (str(state), str(result)))
            self.done = True
        
        # Sends the goal to the action server.
        client.send_goal(action_goal, feedback_cb=feedback_cb, done_cb=done_cb)
        print('goal is sent.')
        # # Waits for the server to finish performing the action.
        #client.wait_for_result()
        joint_names = traj.joint_names
        tracked_traj = []
        tracked_time_traj = []
        while not self.done:
            # obtain the joint trajectory during tracking
            arm_joint = rospy.wait_for_message("joint_states", JointState)
            total_joints = {}
            for i in range(len(arm_joint.name)):
                total_joints[arm_joint.name[i]] = arm_joint.position[i]
            tracked_traj_i = {}
            for i in range(len(traj.joint_names)):
                tracked_traj_i[traj.joint_names[i]] = total_joints[traj.joint_names[i]]

            tracked_traj.append(tracked_traj_i)
            tracked_time_traj.append(arm_joint.header.stamp.secs)
            #print('done: ')
            #print(done)
            time.sleep(0.1)

        # # Prints out the result of executing the action
        result = client.get_result()
        print('result:')
        print(result)
        
        return result, tracked_traj, tracked_time_traj       

    def traj_track_cb(self, goal):
        """
        Given kinematics waypoints, track the trajectory
        """
        # * extract the goal trajectory
        traj = goal.trajectory

        names = traj.joint_names
        pos_traj = []
        vel_traj = []
        time_traj = []
        for point in traj.points:
            pos_traj.append(point.positions)
            vel_traj.append(point.velocities)
            time_traj.append(point.time_from_start.secs)

        pos_traj = np.array(pos_traj)
        vel_traj = np.array(vel_traj)
        pos_traj, vel_traj, time_traj = self.interpolate_traj(pos_traj, vel_traj, time_traj)
        acc_traj = self.compute_acc(pos_traj, vel_traj, time_traj)
        time_traj.insert(0, 0.)
        time_traj = np.array(time_traj)
        print(time_traj)
        time_traj = time_traj[1:] - time_traj[:-1]


        joint_ids = []
        for name in names:
            joint_id = self.joint_name_ind_dict[name]
            joint_ids.append(joint_id)
        
        # * track the goal trajectory using simulator (by communicating with the main thread)
        while self.traj_track_flag:
            time.sleep(0.5)
        # set the tracking data and flag
        self.traj_track_joint_ids = joint_ids
        self.traj_track_pos_traj = pos_traj
        self.traj_track_vel_traj = vel_traj
        self.traj_track_time_traj = time_traj
        self.traj_track_acc_traj = acc_traj
        self.traj_track_step = 0
        self.traj_track_idx = 0  # which point are we tracking
        self.traj_track_flag = 1
        # wait for the tracking to finish
        while self.traj_track_flag:
            time.sleep(1.)

        # get the last velocity value
        self.last_tracked_vel = {}
        for i in range(len(names)):
            self.last_tracked_vel[names[i]] = vel_traj[-1][i]

        # return status
        result = control_msgs.msg.FollowJointTrajectoryResult()
        self.traj_track_as.set_succeeded(result)

    def gripper_cb(self, req):
        print('inside gripper_cb...')
        names = req.finger_joint_names
        indices = []
        for i in range(len(names)):
            indices.append(self.joint_name_ind_dict[names[i]])
        position = req.position


        # * track the goal trajectory using simulator (by communicating with the main thread)
        while self.gripper_track_flag:
            time.sleep(0.5)
        print('gripper_cb: pass to main thread')
        # set the tracking data and flag
        self.gripper_track_joint_ids = indices
        self.gripper_track_pos = position
        self.gripper_track_time = req.time
        self.gripper_track_step = 0
        self.gripper_track_flag = 1
        # wait for the tracking to finish
        while self.gripper_track_flag:
            time.sleep(1.)

        print('gripper_cb: finished tracking gripper.')
        res = RobotiqControlResponse()
        res.success = True
        return res




    def compute_acc(self, pos_traj, vel_traj, time_traj):
        # assume linear accleration
        acc_traj = []
        for i in range(len(pos_traj)-1):
            x1 = pos_traj[i]
            x2 = pos_traj[i+1]
            t1 = time_traj[i]
            t2 = time_traj[i+1]
            v1 = vel_traj[i]
            v2 = vel_traj[i+1]
            pos_dif = x2 - x1
            # if pos_dif > np.pi:
            #     pos_dif = pos_dif - 2*np.pi
            # if pos_dif < -np.pi:
            #     pos_dif = pos_dif + 2*np.pi
            a2 = -12*pos_dif/((t2-t1)**3) + 6*(v2+v1)/((t2-t1)**2)
            a1 = 6*pos_dif/((t2-t1)**2)-2*(v2+2*v1)/(t2-t1)
            acc_traj.append(a1)
        return acc_traj

    def interpolate_traj(self, pos_traj, vel_traj, time_traj):
        # interpolate each segment using sim_time, and compute the interpolated velocity and acceleration
        new_vel_traj = [pos_traj[0]]
        new_pos_traj = [vel_traj[0]]
        new_time_traj = [time_traj[0]]
        for i in range(len(time_traj)-1):
            # pos_traj: from i to i+1
            num_segs = int(np.ceil((time_traj[i+1]-time_traj[i]) / self.sim_time))
            if num_segs == 0:
                # with zero time, we don't care
                continue
            # compute interpolated vel
            start_pos = pos_traj[i]
            start_vel = vel_traj[i]
            start_time = time_traj[i]
            d_vel = (vel_traj[i+1] - vel_traj[i]) / num_segs
            for j in range(1,num_segs+1):
                new_vel_traj_i = start_vel + d_vel * j
                new_pos_traj_i = (start_vel + new_vel_traj_i) / 2 * j * self.sim_time + start_pos
                new_time_traj_i = start_time + j * self.sim_time
                new_pos_traj.append(new_pos_traj_i)
                new_vel_traj.append(new_vel_traj_i)
                new_time_traj.append(new_time_traj_i)
        new_pos_traj = np.array(new_pos_traj)
        new_vel_traj = np.array(new_vel_traj)
        return new_pos_traj, new_vel_traj, new_time_traj
            



    def step_traj_track(self):
        # check if we need to update the point idx
        # print('traj_track_time: ')
        # print(self.traj_track_time_traj[self.traj_track_idx])
        # print()
        if self.traj_track_step * self.sim_time > self.traj_track_time_traj[self.traj_track_idx]:
            self.traj_track_idx += 1
        if self.traj_track_idx >= len(self.traj_track_time_traj):
            # tracking is done
            # TODO: reset the motor
            self.traj_track_flag = 0
            return
        #TODO: try force control. First calculate accleration, then use that to compute force
        p.setJointMotorControlArray(self.robot_id, jointIndices=self.traj_track_joint_ids, controlMode=p.POSITION_CONTROL,
                                    targetPositions=self.traj_track_pos_traj[self.traj_track_idx], 
                                    targetVelocities=self.traj_track_vel_traj[self.traj_track_idx])
        
        # mathematically validate if the acc is correct
        # print('target acc: ', self.traj_track_acc_traj[self.traj_track_idx])
        # print('calculated acc: ', s)

        # Note: acceleration somehow doesn't work as expected
        # target_acc = np.zeros(len(self.joint_indices))
        # for i in range(len(self.traj_track_joint_ids)):
        #     target_acc[i] = self.traj_track_acc_traj[self.traj_track_idx][i]
        # forces = p.calculateInverseDynamics(self.robot_id, objPositions=self.current_joint_pos.tolist(), objVelocities=self.current_joint_vel.tolist(),
        #                             objAccelerations=target_acc.tolist())
        # p.setJointMotorControlArray(self.robot_id, jointIndices=self.joint_indices, controlMode=p.TORQUE_CONTROL,
        #                             forces=forces)

        self.traj_track_step += 1  # step once

    def step_gripper_track(self):

        # check if we need to update the point idx
        if self.gripper_track_step * self.sim_time >= self.gripper_track_time:
            self.gripper_track_flag = 0
            return
        p.setJointMotorControlArray(self.robot_id, jointIndices=self.gripper_track_joint_ids, controlMode=p.POSITION_CONTROL,
                                    targetPositions=self.gripper_track_pos+np.zeros(len(self.gripper_track_joint_ids)))
        self.gripper_track_step += 1  # step once




    def step(self, vision=False):
        if vision:
            # * publish camera info
            width, height, rgb_img, depth_img, seg_img = p.getCameraImage(
                width=640,
                height=640,
                viewMatrix=self.view_mat,
                projectionMatrix=self.proj_mat)
            cv2.imshow('camera_rgb', rgb_img)

        # * publish current joint state
        self.joint_state_publish()

        # set stablizing motor by default
        force = 1e100
        p.setJointMotorControlArray(self.robot_id, jointIndices=self.joint_indices, controlMode=p.VELOCITY_CONTROL,
                                    targetVelocities=np.zeros(len(self.joint_indices)), forces=np.zeros(len(self.joint_indices))+force)

        if (not self.traj_track_flag) and (not self.gripper_track_flag):
            forces = p.calculateInverseDynamics(self.robot_id, objPositions=self.current_joint_pos.tolist(), objVelocities=np.zeros(len(self.joint_indices)).tolist(),
                                       objAccelerations=np.zeros(len(self.joint_indices)).tolist())
            p.setJointMotorControlArray(self.robot_id, jointIndices=self.joint_indices, controlMode=p.TORQUE_CONTROL,
                                        forces=forces)

        # track traj
        if self.traj_track_flag:
            self.step_traj_track()

        # gripper traj
        if self.gripper_track_flag:
            self.step_gripper_track()
        
            

        p.stepSimulation()
        msg = Clock()
        msg.clock = rospy.Time.from_sec(self.cur_time + self.sim_time)
        self.cur_time = self.cur_time + self.sim_time
        self.clock_pub.publish(msg)
        time.sleep(1/240)



def main(args):
    rospy.init_node('pybullet_execution_scene')
    exec_scene = ExecutionScene(args)
    while not rospy.is_shutdown():
        exec_scene.step()

    p.disconnect()


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--prob_config',type=str, required=True, help='the path to the problem configuration file (JSON).')

args = parser.parse_args()
main(args)