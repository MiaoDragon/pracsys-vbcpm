"""
This script defines a robot object
"""
import pybullet as p
class Robot():
    def __init__(self, urdf, pos, ori, pybullet_id):
        """
        given the URDF file, the pose of the robot base, and the joint angles,
        initialize a robot model in the scene
        """
        robot_id = p.loadURDF(urdf, pos, ori, useFixedBase=True, physicsClientId=pybullet_id)
        # get the number of active joints
        num_joints = p.getNumJoints(robot_id, pybullet_id)
        # joint_dict = {}
        joint_names = []
        joint_indices = []
        joint_name_ind_dict = {}
        for i in range(num_joints):
            info = p.getJointInfo(robot_id, i, pybullet_id)
            name = info[1].decode("utf-8")
            joint_type = info[2]
            if joint_type == p.JOINT_FIXED:
                continue
            # joint_dict[name] = 
            joint_names.append(name)
            joint_indices.append(i)
            joint_name_ind_dict[name] = i


        self.robot_id = robot_id
        self.num_joints = num_joints
        self.joint_names = joint_names
        self.pybullet_id = pybullet_id
        pass
    def set_joints(self, joints):
        for i in range(self.num_joints):
            p.resetJointState(self.robot_id, i, joints[i], 0., self.pybullet_id)
        # reset joints
        self.joint_vals = joints