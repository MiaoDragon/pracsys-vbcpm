"""
Provide a ROS node that does the following:

- publish:
  - joint state of robot
  - camera images (through ROS service)

- subscribe:
  - joint trajectory for tracking

PyBullet:
- usage:
  randomly generate a problem, 
  or load a previously generated problem instance
- load robot, and objects in the scene. Generate a scene graph to represent the problem
"""
import object_retrieval_prob_generation as prob_gen
from visual_utilities import *
from vbcpm_integrated_system.srv import AttachObject, ExecuteTrajectory, \
                                    AttachObjectResponse, ExecuteTrajectoryResponse

import rospy
from sensor_msgs.msg import Image, JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Int32
from cv_bridge import CvBridge
import transformations as tf
import pickle
import json
import sys

class ExecutionSystem():
    def __init__(self, load=None, scene_name='scene1'):
        # if load is None, then randomly generate a problem instance
        # otherwise use the load as a filename to load the previously generated problem
        # obj_type = input('using simple geometry? y or n: ').strip()
        obj_type = sys.argv[3].strip()

        if load is not None:
            print('loading file: ', load)
            # load previously generated object
            f = open(load, 'rb')
            data = pickle.load(f)
            f.close()
            if obj_type == 'y':
                scene_f, obj_poses, obj_pcds, obj_shapes, obj_sizes, \
                    target_pose, target_pcd, target_obj_shape, target_obj_size = data
                data = prob_gen.load_problem_level(scene_f, obj_poses, obj_pcds, obj_shapes, obj_sizes, 
                                    target_pose, target_pcd, target_obj_shape, target_obj_size)
                pid, scene_f, robot, workspace, camera, \
                    obj_poses, obj_pcds, obj_ids, \
                    target_pose, target_pcd, target_obj_id = data
                f = open(scene_f, 'r')
                scene_dict = json.load(f)                 
                print('loaded')
            else:
                scene_f, obj_poses, obj_shapes, \
                    target_pose, target_obj_shape = data
                data = prob_gen.load_problem_ycb(scene_f, obj_poses, obj_shapes, 
                                    target_pose, target_obj_shape)
                pid, scene_f, robot, workspace, camera, \
                    obj_poses, obj_ids, \
                    target_pose, target_obj_id = data
                f = open(scene_f, 'r')
                scene_dict = json.load(f)                 
                print('loaded')                
                obj_pcds = None
                obj_sizes = None
                target_pcd = None
                target_obj_size = None
        else:
            scene_f = scene_name+'.json'
            level = input('Input the difficulty level: 1 - simple, 2 - medium, 3 - hard...')
            level = int(level)
            num_objs = input('Input the number of objects: ')
            num_objs = int(num_objs)
            if obj_type == 'y':
                data = prob_gen.random_one_problem_level(scene=scene_f, level=level, num_objs=num_objs, num_hiding_objs=1)
                pid, scene_f, robot, workspace, camera, obj_poses, obj_pcds, obj_ids, obj_shapes, obj_sizes, \
                    target_pose, target_pcd, target_obj_id, target_obj_shape, target_obj_size = data
                data = (scene_f, obj_poses, obj_pcds, obj_shapes, obj_sizes, target_pose, target_pcd, target_obj_shape, target_obj_size)
                save = input('save current scene? 0 - no, 1 - yes...')
                f = open(scene_f, 'r')
                scene_dict = json.load(f)
                f.close()
                if int(save) == 1:
                    save_f = input('save name: ').strip()
                    f = open(save_f + '.pkl', 'wb')
                    pickle.dump(data, f)
                    f.close()
                    print('saved')
            else:
                data = prob_gen.random_one_problem_ycb(scene=scene_f, level=level, num_objs=num_objs, num_hiding_objs=1)
                pid, scene_f, robot, workspace, camera, obj_poses, obj_ids, obj_shapes, \
                    target_pose, target_obj_id, target_obj_shape = data
                data = (scene_f, obj_poses, obj_shapes, target_pose, target_obj_shape)
                save = input('save current scene? 0 - no, 1 - yes...')
                f = open(scene_f, 'r')
                scene_dict = json.load(f)
                f.close()
                if int(save) == 1:
                    save_f = input('save name: ').strip()
                    f = open(save_f + '.pkl', 'wb')
                    pickle.dump(data, f)
                    f.close()
                    print('saved')
                obj_pcds = None
                obj_sizes = None
                target_pcd = None
                target_obj_size = None

        self.pid = pid
        self.scene_dict = scene_dict
        self.robot = robot
        self.workspace = workspace
        self.camera = camera
        self.obj_poses = obj_poses
        self.obj_pcds = obj_pcds
        self.obj_ids = obj_ids
        self.obj_shapes = obj_shapes
        self.obj_sizes = obj_sizes
        self.target_pose = target_pose
        self.target_pcd = target_pcd
        self.target_obj_id = target_obj_id
        self.target_obj_shape = target_obj_shape
        self.target_obj_size = target_obj_size

        self.total_obj_ids = self.obj_ids + [self.target_obj_id]
        self.bridge = CvBridge()

        self.attached_obj_id = None
        # * initialize ROS services
        # - robot trajectory tracker
        rospy.Service("execute_trajectory", ExecuteTrajectory, self.execute_trajectory)
        rospy.Service("attach_object", AttachObject, self.attach_object)

        # * initialize ROS pubs and subs
        # - camera
        # - robot_state_publisher
        self.rgb_cam_pub = rospy.Publisher('rgb_image', Image)
        self.depth_cam_pub = rospy.Publisher('depth_image', Image)
        self.seg_cam_pub = rospy.Publisher('seg_image', Image)
        self.rs_pub = rospy.Publisher('robot_state_publisher', JointState)

        # self.done_sub = rospy.Subscriber('done_msg', Int32, self.done_callback)

        # construct_occlusion_graph(obj_ids, obj_poses, camera, pid)

    # def done_callback(self, msg):
    #     if msg.data == 1:
    #         # finished
    #         exit(1)

    def check_collision(self, ignored_obj_id):
        """
        check collision between objects, workspace and robots
        """
        # check collision between objects and robot
        for obj_id in self.obj_ids:
            if self.attached_obj_id is not None and obj_id == self.attached_obj_id:
                continue
            if obj_id == ignored_obj_id:
                continue
            distance = -0.02  # seems the PyBullet collision geometry is very conservative. The moveit planned path
                                # sometimes collide with the environment
            contacts = p.getClosestPoints(obj_id, self.robot.robot_id, distance=distance,physicsClientId=self.pid)
            if len(contacts) > 0:
                print('collision happened between robot and object ', obj_id)
                # input('press Enter')
                return True
        for c_name, cid in self.workspace.component_id_dict.items():
            distance = -0.02  # seems the PyBullet collision geometry is very conservative. The moveit planned path
                                # sometimes collide with the environment            
            contacts = p.getClosestPoints(cid, self.robot.robot_id, distance=distance,physicsClientId=self.pid)
            if len(contacts) > 0:
                print('collision happened between robot and ', c_name)
                # input('press Enter')
                return True
        # check for attached object
        if self.attached_obj_id is not None:
            for obj_id in self.obj_ids:
                if obj_id == self.attached_obj_id:
                    continue
                if obj_id == ignored_obj_id:
                    continue
                distance = -0.02  # seems the PyBullet collision geometry is very conservative. The moveit planned path
                                    # sometimes collide with the environment
                contacts = p.getClosestPoints(obj_id, self.attached_obj_id, distance=distance,physicsClientId=self.pid)
                if len(contacts) > 0:
                    print('collision happened between attached object and object ', obj_id)
                    # input('press Enter')
                    return True
            for c_name, cid in self.workspace.component_id_dict.items():
                if c_name == 'shelf_bottom':
                    distance = -0.02
                else:
                    distance = -0.02  # seems the PyBullet collision geometry is very conservative. The moveit planned path
                                      # sometimes collide with the environment
                contacts = p.getClosestPoints(cid, self.attached_obj_id, distance=distance,physicsClientId=self.pid)
                if len(contacts) > 0:
                    print('collision happened between attached object and ', c_name)
                    # input('press Enter')
                    return True
        return False

    def execute_trajectory(self, req):
        """
        PyBullet:
        if object attached, move arm and object at the same time
        """
        traj = req.trajectory # sensor_msgs/JointTrajectory
        ignored_obj_id = req.ignored_obj_id
        # joint_dict_list = []
        joint_names = traj.joint_names
        points = traj.points
        num_collision = 0
        if self.attached_obj_id is not None:
            # compute initial transformation of object and robot link
            transform = self.robot.get_tip_link_pose()

            link_state = p.getBasePositionAndOrientation(self.attached_obj_id, physicsClientId=self.robot.pybullet_id)
            pos = link_state[0]
            ori = link_state[1]
            obj_transform = tf.quaternion_matrix([ori[3],ori[0],ori[1],ori[2]])
            obj_transform[:3,3] = pos

        for i in range(len(points)):
            pos = points[i].positions
            time_from_start = points[i].time_from_start
            joint_dict = {joint_names[j]: pos[j] for j in range(len(pos))}
            self.robot.set_joint_from_dict(joint_dict)

            if self.attached_obj_id is not None:
                new_transform = self.robot.get_tip_link_pose()
                rel_transform = new_transform.dot(np.linalg.inv(transform))
                new_obj_transform = rel_transform.dot(obj_transform)
                quat = tf.quaternion_from_matrix(new_obj_transform) # w x y z
                p.resetBasePositionAndOrientation(self.attached_obj_id, new_obj_transform[:3,3], [quat[1],quat[2],quat[3],quat[0]], 
                                                    physicsClientId=self.robot.pybullet_id)
                transform = new_transform
                obj_transform = new_obj_transform
                if self.check_collision(ignored_obj_id):
                    num_collision += 1
                    print('collision happened.')
        input('waiting...')
            # rospy.sleep(0.03)
        return ExecuteTrajectoryResponse(num_collision, True)

    def attach_object(self, req):
        """
        attach the object closest to the robot
        """
        if req.attach == True:
            self.attached_obj_id = req.obj_id
        else:
            self.attached_obj_id = None

        return AttachObjectResponse(True)

    def publish_image(self):
        """
        obtain image from PyBullet and publish
        """
        rgb_img, depth_img, seg_img = self.camera.sense()
        msg = self.bridge.cv2_to_imgmsg(rgb_img, 'passthrough')
        self.rgb_cam_pub.publish(msg)

        msg = self.bridge.cv2_to_imgmsg(depth_img, 'passthrough')
        self.depth_cam_pub.publish(msg)

        msg = self.bridge.cv2_to_imgmsg(seg_img, 'passthrough')
        self.seg_cam_pub.publish(msg)
        
    def publish_robot_state(self):
        """
        obtain joint state from PyBullet and publish
        """
        msg = JointState()
        for name, val in self.robot.joint_dict.items():
            msg.name.append(name)
            msg.position.append(val)
        self.rs_pub.publish(msg)

    def run(self):
        """
        keep spinning and publishing to the ROS topics
        """
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            self.publish_robot_state()
            self.publish_image()
            rate.sleep()



def main():
    rospy.init_node("execution_system")
    rospy.sleep(1.0)
    scene_name = 'scene2'

    if int(sys.argv[1]) > 0:
        load = True
        # load = input("enter the problem name for loading: ").strip()
        load = sys.argv[2].strip()

        load = load + '.pkl'
    else:
        load = None
    execution_system = ExecutionSystem(load, scene_name)
    execution_system.run()

if __name__ == "__main__":
    main()