"""
motion planner using moveit
"""
from moveit_commander import move_group
import rospy
import moveit_commander
from moveit_msgs.srv import GetStateValidityRequest, GetStateValidity
import numpy as np
from sensor_msgs.msg import PointCloud2, Image
import sensor_msgs

import sensor_msgs.point_cloud2 as pcl2
from moveit_msgs.msg import PlanningScene, CollisionObject, AttachedCollisionObject

from shape_msgs.msg import SolidPrimitive, Plane, Mesh, MeshTriangle

import std_msgs
from geometry_msgs.msg import PoseStamped, Point
import transformations as tf
from std_srvs.srv import Empty

from moveit_msgs.msg import RobotState, DisplayRobotState
from sensor_msgs.msg import JointState
import sys

from visual_utilities import *
import open3d as o3d

from geometry_msgs.msg import Pose, Point
from shape_msgs.msg import SolidPrimitive, Plane, Mesh, MeshTriangle

from memory_profiler import profile
import os
import psutil

import objgraph
import gc
class MotionPlanner():
    def __init__(self, robot, workspace, commander_args=[]):
        # set up the scene
        moveit_commander.roscpp_initialize(commander_args)
        self.robot_commander = moveit_commander.RobotCommander()
        self.group_arm_name = "left_arm"
        self.move_group = moveit_commander.MoveGroupCommander(self.group_arm_name)
        self.scene_interface = moveit_commander.PlanningSceneInterface()

        self.scene_interface.remove_world_object()
        self.scene_interface.remove_world_object('suction_object')
        rospy.sleep(1.0)
        self.clear_octomap()

        self.robot = robot
        self.workspace = workspace

        self.pcd_topic = '/perception/points'
        self.pcd_pub = rospy.Publisher(self.pcd_topic, PointCloud2, queue_size=3, latch=True)

        self.co_pub = rospy.Publisher('/collision_object', CollisionObject, queue_size=3, latch=True)

        self.rs_pub = rospy.Publisher('/display_robot_state', DisplayRobotState, queue_size=3, latch=True)
        # set up workspace collision
        components = workspace.components
        for component_name, component in components.items():
            shape = component['shape']
            shape = np.array(shape)
            pos = component['pose']['pos']
            ori = component['pose']['ori']  # x y z w
            # transform from workspace to robot
            mat = tf.quaternion_matrix([ori[3], ori[0], ori[1], ori[2]]) # tf: w x y z
            mat[:3,3] = pos
            mat = robot.world_in_robot.dot(mat)
            self.add_box(component_name, mat, [shape[0],shape[1],shape[2]])
            rospy.sleep(1.0)

            # self.scene_interface.add_box(name=component_name, pose=)

    def get_state_validity(self, state, group_name="robot_arm"):
        rospy.wait_for_service('/check_state_validity')
        sv_srv = rospy.ServiceProxy('/check_state_validity', GetStateValidity)
        gsvr = GetStateValidityRequest()
        gsvr.robot_state = state
        gsvr.group_name = group_name
        result = sv_srv.call(gsvr)
        # if not result.valid:
        #     print('Collision Checker failed.')
        #     print('contact: ')
        #     for i in range(len(result.contacts)):
        #         print('contact_body_1: %s, type: %d' % (result.contacts[i].contact_body_1, result.contacts[i].body_type_1))
        #         print('contact_body_2: %s, type: %d' % (result.contacts[i].contact_body_2, result.contacts[i].body_type_2))   
        return result

    def clear_octomap(self):
        # update octomap by clearing existing ones
        rospy.loginfo("calling clear_octomap...")
        rospy.wait_for_service('clear_octomap')
        # generate message
        try:
            ros_srv = rospy.ServiceProxy('clear_octomap', Empty)
            resp1 = ros_srv()

            del ros_srv
            del resp1

        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)
            sys.exit(1)

        gc.collect()
        rospy.sleep(.5)

    def wait(self, time):
        rospy.sleep(time)

    def set_collision_env_with_filter(self, occlusion, col_filter):
        # pcd -> octomap
        # clear environment first
        self.clear_octomap()
        # rospy.sleep(2.0)

        # collision_voxel = visualize_voxel(occlusion.voxel_x, occlusion.voxel_y, occlusion.voxel_z,
        #                                  occluded | occupied, [1,0,0])
        # o3d.visualization.draw_geometries([collision_voxel])
        # * generate point cloud for occlusion space
        total_pcd = occlusion.sample_pcd(col_filter)
        # occupied_pcd = occlusion.sample_pcd(occupied)
        total_pcd = occlusion.transform[:3,:3].dot(total_pcd.T).T + occlusion.transform[:3,3]
        total_pcd = self.robot.world_in_robot[:3,:3].dot(total_pcd.T).T + self.robot.world_in_robot[:3,3]
        # publish the pcd to the rostopic of sensor
        header = std_msgs.msg.Header()
        header.frame_id = 'base'  # use robot base as frame
        pcd_msg = pcl2.create_cloud_xyz32(header, total_pcd)
        # while True:
        self.pcd_pub.publish(pcd_msg)
        rospy.sleep(0.5)
        # input('after publish...')
        del total_pcd
        del pcd_msg

        gc.collect()


    def set_collision_env(self, occlusion, occluded, occupied):
        # pcd -> octomap
        # clear environment first
        self.clear_octomap()
        # rospy.sleep(2.0)
        # collision_voxel = visualize_voxel(occlusion.voxel_x, occlusion.voxel_y, occlusion.voxel_z,
        #                                  occluded | occupied, [1,0,0])
        # o3d.visualization.draw_geometries([collision_voxel])


        # * generate point cloud for occlusion space

        occluded_pcd = occlusion.sample_pcd(occluded)
        occupied_pcd = occlusion.sample_pcd(occupied)
        total_pcd = np.concatenate([occluded_pcd, occupied_pcd], axis=0)
        total_pcd = occlusion.transform[:3,:3].dot(total_pcd.T).T + occlusion.transform[:3,3]
        total_pcd = self.robot.world_in_robot[:3,:3].dot(total_pcd.T).T + self.robot.world_in_robot[:3,3]
        # publish the pcd to the rostopic of sensor
        header = std_msgs.msg.Header()
        header.frame_id = 'base'  # use robot base as frame
        pcd_msg = pcl2.create_cloud_xyz32(header, total_pcd)
        # while True:
        self.pcd_pub.publish(pcd_msg)
        rospy.sleep(0.1)
        # input('after publish...')
        del total_pcd
        del occupied_pcd
        del occluded_pcd
        del pcd_msg

        gc.collect()
        # objgraph.show_most_common_types(limit=20)
        # objgraph.show_growth(limit=3)

    def get_robot_state_from_joint_dict(self, joint_dict, attached_acos=[]):
        joint_state = JointState()
        # joint_state.header = Header()
        # joint_state.header.stamp = rospy.Time.now()
        names = list(joint_dict.keys())
        vals = []
        for i in range(len(names)):
            vals.append(joint_dict[names[i]])
        joint_state.name = names
        joint_state.position = vals
        moveit_robot_state = RobotState()
        moveit_robot_state.joint_state = joint_state
        moveit_robot_state.attached_collision_objects = attached_acos
        moveit_robot_state.is_diff = True
        return moveit_robot_state

    def extract_plan_to_joint_list(self, plan):
        """
        given a motion plan generated by MoveIt!, extract the list of joints        

        (success flag : boolean, trajectory message : RobotTrajectory,
         planning time : float, error code : MoveitErrorCodes)        
        """
        traj = plan[1]
        traj = traj.joint_trajectory
        joint_names = traj.joint_names
        points = traj.points
        positions = []
        # velocities = []
        # accelerations = []
        time_from_starts = []
        for point in points:
            positions.append(point.positions)
            time_from_starts.append(point.time_from_start)
        if plan[0]:
            return joint_names, positions, time_from_starts
        else:
            return joint_names, [], []

    
    def format_joint_name_val_dict(self, joint_names, joint_vals):
        """
        given a list of joint_names and a list of joint_vals
        construct a list of dict: [{joint_name: joint_val}]
        """
        joint_dict_list = []
        for i in range(len(joint_vals)):
            joint_dict = {}
            for j in range(len(joint_names)):
                joint_dict[joint_names[j]] = joint_vals[i][j]
            joint_dict_list.append(joint_dict)
        return joint_dict_list

    def suction_plan(self, start_joint_dict, suction_pose, suction_joint, robot, workspace, display=False):
        """
        return a list of dict: [{joint_name -> joint_vals}]
        """
        # * generate pre_suction_pose from suction_pose by moving the gripper back
        self.scene_interface.remove_world_object('suction_object')
        # rospy.sleep(1.0)
        pre_pose_dist = 0.10
        suction_pos = suction_pose[:3,3]
        retreat_vec = -suction_pose[:3,2]  # z vec
        pre_suction_pos = suction_pose[:3,3] + retreat_vec * pre_pose_dist
        step = 0.01
        n_step = int(np.ceil(pre_pose_dist / step))
        step = pre_pose_dist / n_step
        # * first find a straight-line path from suction_pose to pre_suction_pose
        joint_vals = []

        quat = tf.quaternion_from_matrix(suction_pose)  # w x y z
        prev_suction_joint = suction_joint
        joint_vals.append(prev_suction_joint)
        for i in range(1,n_step):
            if i == 0:
                collision_check = False
            else:
                collision_check = True
            suction_pos_step = suction_pos + retreat_vec * i * step
            # get joints (through ik) for current step
            valid, suction_joint_step = robot.get_ik(robot.tip_link_name, suction_pos_step, 
                                                    [quat[1],quat[2],quat[3],quat[0]], prev_suction_joint,
                                                    collision_check=collision_check, workspace=workspace)
            if valid:
                prev_suction_joint = suction_joint_step
                joint_vals.append(prev_suction_joint)
                if display:
                    robot.set_joints_without_memorize(suction_joint_step)
                    input('next straight-line in suction_plan...')
                    robot.set_joints_without_memorize(robot.joint_vals)

        joint_vals = joint_vals[::-1]  # reverse: from pre_suction_pos to suction_pos
        suction_joint_dict_list = self.format_joint_name_val_dict(robot.joint_names, joint_vals)
        

        # * generate a plan from start_joint to pre_suction_joint
        pre_suction_joint_dict = suction_joint_dict_list[0]
        # print('suction_plan calling motion_plan_joint..')
        plan = self.motion_plan_joint(start_joint_dict, pre_suction_joint_dict, robot)
        joint_names, positions, _ = self.extract_plan_to_joint_list(plan)
        pre_suction_joint_dict_list = self.format_joint_name_val_dict(joint_names, positions)

        # input('after suction_plan, length of trajectory: %d' % (len(pre_suction_joint_dict_list)))
        if len(pre_suction_joint_dict_list) == 0:
            return []
        # concatenate the two list to return
        return pre_suction_joint_dict_list + suction_joint_dict_list

    def suction_with_obj_plan(self, start_joint_dict, tip_pose_in_obj, target_joint_val, robot, 
                                obj):
        """
        suction with the object attached to the robot
        create the object mesh from its point cloud or voxel info
        attach the mesh to the robot when doing motion planning
        """

        # get the object pose at start
        start_tip_pose = robot.get_tip_link_pose(start_joint_dict)
        obj_transform = start_tip_pose.dot(np.linalg.inv(tip_pose_in_obj))

        self.scene_interface.remove_world_object('suction_object')
        # rospy.sleep(1.0)
        mesh_vertices, mesh_faces = self.create_mesh_from_voxel(obj.get_conservative_model(), obj)
        co = self.add_mesh_from_vertices_and_faces(mesh_vertices, mesh_faces, obj_transform, 'suction_object')
        # attach the added mesh to robot link
        touch_links = ['motoman_left_ee', 'arm_left_link_tool0', 'motoman_left_hand']
        aco = self.attach_object(co, robot.tip_link_name, touch_links)
        # self.move_group.attach_object('suction_object', robot.tip_link_name)
        # rospy.sleep(1.0)


        # plan a trajectory from start joint to target joint

        goal_joint_dict_list = self.format_joint_name_val_dict(robot.joint_names, [target_joint_val])
        goal_joint_dict = goal_joint_dict_list[0]
        new_goal_joint_dict = {}
        for name, val in goal_joint_dict.items():
            if 'right' in name:
                continue
            new_goal_joint_dict[name] = val
        goal_joint_dict = new_goal_joint_dict
        
        start_time = time.time()
        plan = self.motion_plan_joint(start_joint_dict, goal_joint_dict, robot, [aco])
        print('motion_plan_joint takes time: ', time.time() - start_time)
        joint_names, positions, _ = self.extract_plan_to_joint_list(plan)
        suction_joint_dict_list = self.format_joint_name_val_dict(joint_names, positions)

        del co
        del aco
        del mesh_vertices
        del mesh_faces

        print('after suction_with_obj_plan..')

        return suction_joint_dict_list

    def joint_dict_motion_plan(self, start_joint_dict, goal_joint_dict, robot):
        plan = self.motion_plan_joint(start_joint_dict, goal_joint_dict, robot, [])
        joint_names, positions, _ = self.extract_plan_to_joint_list(plan)
        joint_dict_list = self.format_joint_name_val_dict(joint_names, positions)
        return joint_dict_list

    def motion_plan_joint(self, start_joint_dict, goal_joint_dict, robot, attached_acos=[]):

        joint_state = JointState()
        # joint_state.header = Header()
        # joint_state.header.stamp = rospy.Time.now()
        names = list(start_joint_dict.keys())
        vals = []
        for i in range(len(names)):
            vals.append(start_joint_dict[names[i]])
        joint_state.name = names
        joint_state.position = vals
        moveit_robot_state = RobotState()
        moveit_robot_state.joint_state = joint_state
        moveit_robot_state.attached_collision_objects = attached_acos
        moveit_robot_state.is_diff = True
        # make sure the goal is only the ones in the group
        new_goal_joint_dict = {}
        for name, val in goal_joint_dict.items():
            if 'right' in name:
                continue
            new_goal_joint_dict[name] = val
        goal_joint_dict = new_goal_joint_dict
        
        # self.move_group.set_planner_id('PersistentLazyPRM')

        self.move_group.set_planner_id('BiTRRT')
        self.move_group.set_start_state(moveit_robot_state)
        
        self.move_group.set_joint_value_target(goal_joint_dict)

        self.move_group.set_planning_time(14)
        self.move_group.set_num_planning_attempts(5)
        self.move_group.allow_replanning(False)


        plan = self.move_group.plan()  # returned value: tuple (flag, RobotTrajectory, planning_time, error_code)

        # input("next...")
        del moveit_robot_state
        del new_goal_joint_dict

        
        return plan

    def straight_line_motion(self, start_joint_dict, start_tip_pose, relative_tip_pose, robot, workspace, 
                            collision_check=False, display=False):
        # given the start joint values and the relative tip pose to move, move in straight line

        start_pos = start_tip_pose[:3,3]
        move_vec = relative_tip_pose[:3,3]
        pose_dist = np.linalg.norm(move_vec)
        move_vec = move_vec / pose_dist
        step = 0.01
        n_step = int(np.ceil(pose_dist / step))
        step = pose_dist / n_step

        tip_pos = start_pos
        quat = tf.quaternion_from_matrix(start_tip_pose)  # w x y z
        # convert from joint dict to joint list
        start_joint = []
        for i in range(len(robot.joint_names)):
            start_joint.append(start_joint_dict[robot.joint_names[i]])

        prev_joint = start_joint

        joint_vals = [prev_joint]

        for i in range(1,n_step):
            tip_pos_step = tip_pos + move_vec * i * step
            # get joints (through ik) for current step
            if i == 0:
                ik_collision_check = False
            else:
                ik_collision_check = collision_check
            valid, joint_step = robot.get_ik(robot.tip_link_name, tip_pos_step, 
                                                    [quat[1],quat[2],quat[3],quat[0]], prev_joint,
                                                    ik_collision_check, workspace)
            if display:
                input('step %d/%d..., valid: %d' % (i, n_step, valid))
                joint_dict = self.format_joint_name_val_dict(robot.joint_names, [joint_step])[0]
                rs = self.get_robot_state_from_joint_dict(joint_dict)
                self.display_robot_state(rs)

                
            if valid:
                prev_joint = joint_step
                joint_vals.append(prev_joint)


        joint_dict_list = self.format_joint_name_val_dict(robot.joint_names, joint_vals)   

        if display:
            del rs
            del joint_dict

        return joint_dict_list

    def display_robot_state(self, state, group_name='robot_arm'):
        msg = DisplayRobotState()
        msg.state = state
        self.rs_pub.publish(msg)
        rospy.sleep(1.0)


    def get_state_validity(self, state, group_name="robot_arm"):
        rospy.wait_for_service('/check_state_validity')
        sv_srv = rospy.ServiceProxy('/check_state_validity', GetStateValidity)
        gsvr = GetStateValidityRequest()
        gsvr.robot_state = state
        gsvr.group_name = group_name
        result = sv_srv.call(gsvr)
        # if not result.valid:
        #     print('Collision Checker failed.')
        #     print('contact: ')
        #     for i in range(len(result.contacts)):
        #         print('contact_body_1: %s, type: %d' % (result.contacts[i].contact_body_1, result.contacts[i].body_type_1))
        #         print('contact_body_2: %s, type: %d' % (result.contacts[i].contact_body_2, result.contacts[i].body_type_2))   
        return result


    def collision_check(self, state, model_name, group_name="robot_arm"):
        res = self.get_state_validity(state)
        if res.valid:
            return True
        else:
            return False
        for i in range(len(res.contacts)):
            if not (res.contacts[i].contact_body_1 == model_name and res.contacts[i].contact_body_2 == model_name):
                # other cases, failure
                return False
        return True    

    def create_mesh_from_voxel(self, filter, obj):
        voxel_x = obj.voxel_x
        voxel_y = obj.voxel_y
        voxel_z = obj.voxel_z
        voxel_pts = np.array([voxel_x, voxel_y, voxel_z]).transpose([1,2,3,0])[filter].reshape(-1,3)
        voxel_pts_vertices = []
        vertex_items = []
        for dx in [0,1]:
            for dy in [0,1]:
                for dz in [0,1]:
                    voxel_pts_1 = voxel_pts + np.array([dx,dy,dz]).reshape(-1,3)
                    voxel_pts_vertices.append(voxel_pts_1)
                    vertex_items.append(np.array([dx, dy, dz]))
        face_items = []
        for i in range(len(vertex_items)):
            for j in range(i+1, len(vertex_items)):
                for k in range(j+1, len(vertex_items)):
                    # the three vertices should lie on one plane (have one value identical)
                    if ((vertex_items[i]==vertex_items[j]) & (vertex_items[j]==vertex_items[k])).sum()>0:
                        face_items.append([i,j,k])

        voxel_pts_total = np.concatenate(voxel_pts_vertices, axis=0)
        voxel_pt_indices = np.array(list(range(len(voxel_pts))))
        voxel_face_total = []
        for i in range(len(face_items)):
            face1 = face_items[i][0]
            face1 = vertex_items[face1]
            face1_idx = face1[0]*4+face1[1]*2+face1[2]
            face2 = face_items[i][1]
            face2 = vertex_items[face2]
            face2_idx = face2[0]*4+face2[1]*2+face2[2]
            face3 = face_items[i][2]
            face3 = vertex_items[face3]
            face3_idx = face3[0]*4+face3[1]*2+face3[2]

            face = [voxel_pt_indices+face1_idx*len(voxel_pts), 
                    voxel_pt_indices+face2_idx*len(voxel_pts),
                    voxel_pt_indices+face3_idx*len(voxel_pts)]
            face = np.array(face).T
            voxel_face_total.append(face)
        voxel_face_total = np.concatenate(voxel_face_total, axis=0)
        voxel_pts_total = voxel_pts_total * obj.resol


        # voxel_pts_total_in_world = obj.transform[:3,:3].dot(voxel_pts_total.T).T + obj.transform[:3,3]

        # voxel_pts_mesh = o3d.utility.Vector3dVector(voxel_pts_total)
        # voxel_face_mesh = o3d.utility.Vector3iVector(voxel_face_total)

        # mesh = o3d.geometry.TriangleMesh(voxel_pts_mesh, voxel_face_mesh)
        # pcd = obj.sample_conservative_pcd()# * obj.resol
        # # pcd = obj.transform[:3,:3].dot(pcd.T).T + obj.transform[:3,3]
        # pcd = visualize_pcd(pcd, [1,0,0])

        # cvx_hull = pcd.compute_convex_hull()[0]
        # o3d.visualization.draw_geometries([mesh, pcd, cvx_hull])

        # voxel_pts_total = np.asarray(cvx_hull.vertices)
        # voxel_face_total = np.asarray(cvx_hull.triangles)

        
        del voxel_pt_indices
        del vertex_items
        del face_items
        del voxel_pts_vertices
        del voxel_pts
        del face

        return voxel_pts_total, voxel_face_total

    def add_mesh_from_vertices_and_faces(self, vertices, faces, transform, name):
        co = CollisionObject()
        co.operation = CollisionObject.ADD
        co.id = 'suction_object'
        co.header.frame_id = 'base'
        # co.header = pose.header

        mesh = Mesh()
        for i in range(len(faces)):
            triangle = MeshTriangle()
            triangle.vertex_indices = [faces[i,0], faces[i,1], faces[i,2]] #[face[0], face[1], face[2]]
            mesh.triangles.append(triangle)        

        for i in range(len(vertices)):
            point = Point()
            point.x = vertices[i,0]
            point.y = vertices[i,1]
            point.z = vertices[i,2]
            mesh.vertices.append(point)

        co.meshes = [mesh]

        pose = Pose()
        quat = tf.quaternion_from_matrix(transform)  # w x y z
        pose.position.x = transform[0,3]
        pose.position.y = transform[1,3]
        pose.position.z = transform[2,3]
        pose.orientation.w = quat[0]
        pose.orientation.x = quat[1]
        pose.orientation.y = quat[2]
        pose.orientation.z = quat[3]


        co.mesh_poses = [pose]

        del mesh

        return co

    def attach_object(self, collision_object, link_name, touch_links):
        aco = AttachedCollisionObject()
        aco.link_name = link_name
        aco.object = collision_object
        aco.touch_links = touch_links
        return aco

    def add_mesh(self, model_name, scale, obj_pose):
        return True
        collision_obj = self.obtain_collision_object_mesh(model_name, scale, obj_pose)
        co_pub.publish(collision_obj)
        #rospy.sleep(1)

    def add_box(self, name, pose, size):
        # print('adding box to the planning scene...')
        collision_obj = self.obtain_collision_object_box(name, pose, size)
        # print('message: ')
        # print(collision_obj)
        self.co_pub.publish(collision_obj)

    def obtain_collision_object_box(self, name, pose, size):
        # given pose: specified by a transformation matrix
        co = CollisionObject()
        co.operation = CollisionObject.ADD
        co.id = name
        # co.pose.position = [0,0,0]
        co.pose.position.x = 0
        co.pose.position.y = 0
        co.pose.position.z = 0
        co.pose.orientation.x = 0
        co.pose.orientation.y = 0
        co.pose.orientation.z = 0
        co.pose.orientation.w = 1

        pose_stamped = PoseStamped()
        pose_stamped.pose.position.x = pose[0,3]
        pose_stamped.pose.position.y = pose[1,3]
        pose_stamped.pose.position.z = pose[2,3]

        quat = tf.quaternion_from_matrix(pose)  # w x y z
        pose_stamped.pose.orientation.x = quat[1]
        pose_stamped.pose.orientation.y = quat[2]
        pose_stamped.pose.orientation.z = quat[3]
        pose_stamped.pose.orientation.w = quat[0]
        pose_stamped.header.frame_id = 'base'
        co.header = pose_stamped.header
        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = list(size)
        co.primitives = [box]
        co.primitive_poses = [pose_stamped.pose]
        return co

    def obtain_collision_object_mesh(self, model_name, scale, obj_pose):
        return
        collision_obj = CollisionObject()
        collision_obj.id = model_name
        mesh = Mesh()
        # load mesh file
        mesh_file_name = "/root/ocrtoc_materials/models/"+model_name+"/collision_meshes/collision.obj"

        # use trimesh
        import trimesh
        trimesh_mesh = trimesh.load(mesh_file_name)
        for face in trimesh_mesh.faces:
            triangle = MeshTriangle()
            triangle.vertex_indices = [face[0], face[1], face[2]]
            mesh.triangles.append(triangle)
        for vertex in trimesh_mesh.vertices:
            point = Point()
            point.x = vertex[0]*scale[0]
            point.y = vertex[1]*scale[1]
            point.z = vertex[2]*scale[2]
            mesh.vertices.append(point)

        collision_obj.meshes = [mesh]
        collision_obj.mesh_poses = [obj_pose]
        collision_obj.operation = CollisionObject.ADD  # will replace the object if it existed before
        collision_obj.header = self.move_sgroup.get_current_pose().header
        return collision_obj

    def obtain_attach_object(self, model_name, scale, obj_pose):
        return
        # achieve this by modifying the start robot state
        # MoveIt also uses this from http://docs.ros.org/api/moveit_commander/html/planning__scene__interface_8py_source.html
        attached_obj = AttachedCollisionObject()
        attached_obj.link_name = "robotiq_2f_85_left_pad"
        touch_links = ["robotiq_2f_85_left_pad", "robotiq_2f_85_right_pad", \
                    "robotiq_2f_85_left_spring_link", "robotiq_2f_85_right_spring_link", \
                    "robotiq_2f_85_left_follower", "robotiq_2f_85_right_follower", \
                    "robotiq_2f_85_left_driver", "robotiq_2f_85_right_driver", \
                    "robotiq_2f_85_left_coupler", "robotiq_2f_85_right_coupler", \
                    "robotiq_arg2f_base_link", "robotiq_2f_85_base", \
                    "robotiq_ur_coupler", "tool0", \
                    "robotiq_arg2f_base_link", "realsense_camera_link", \
                    "table"]
        attached_obj.touch_links = touch_links
        # specify the attached object shape
        attached_obj_shape = CollisionObject()
        attached_obj_shape.id = model_name
        mesh = Mesh()
        # load mesh file
        mesh_file_name = "/root/ocrtoc_materials/models/"+model_name+"/collision_meshes/collision.obj"

        # use trimesh
        import trimesh
        trimesh_mesh = trimesh.load(mesh_file_name)
        for face in trimesh_mesh.faces:
            triangle = MeshTriangle()
            triangle.vertex_indices = [face[0], face[1], face[2]]
            mesh.triangles.append(triangle)
        for vertex in trimesh_mesh.vertices:
            point = Point()
            point.x = vertex[0]*scale[0]
            point.y = vertex[1]*scale[1]
            point.z = vertex[2]*scale[2]
            mesh.vertices.append(point)

        # for box filtering (notice that this is unscaled)
        min_xyz = np.array(trimesh_mesh.vertices).min(axis=0)
        max_xyz = np.array(trimesh_mesh.vertices).max(axis=0)
        attached_obj_shape.meshes = [mesh]
        attached_obj_shape.mesh_poses = [obj_pose]
        attached_obj_shape.operation = CollisionObject.ADD  # will replace the object if it existed before
        attached_obj_shape.header = group.get_current_pose().header

        attached_obj.object = attached_obj_shape
        return attached_obj
    def attach_object_to_gripper(self, model_name, scale, obj_pose, grasp_state):
        return
        try:
            attached_obj = obtain_attach_object(model_name, scale, obj_pose)
            grasp_state.attached_collision_objects.append(attached_obj)
        except:
            rospy.logerr("Motion Planning: error loading mesh file. must be new object.")

        grasp_state.is_diff = True  # is different from others since we attached the object

        rospy.sleep(1.0) # allow publisher to initialize
        return grasp_state
