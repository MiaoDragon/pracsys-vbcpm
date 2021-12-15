"""
This script defines a robot object
"""
import numpy as np
import pybullet as p


class Robot():
    def __init__(self, urdf, pos, ori, pybullet_id):
        """
        given the URDF file, the pose of the robot base, and the joint angles,
        initialize a robot model in the scene
        """
        robot_id = p.loadURDF(
            urdf, pos, ori, useFixedBase=True, physicsClientId=pybullet_id
        )
        # get the number of active joints
        num_joints = p.getNumJoints(robot_id, physicsClientId=pybullet_id)
        # joint_dict = {}
        joint_names = []
        joint_indices = []
        joint_name_ind_dict = {}
        for i in range(num_joints):
            info = p.getJointInfo(robot_id, i, physicsClientId=pybullet_id)
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

        self.right_gripper_id = 26
        self.left_gripper_id = 48

    def set_joints(self, joints):
        for i in range(self.num_joints):
            p.resetJointState(
                self.robot_id, i, joints[i], 0., physicsClientId=self.pybullet_id
            )
        # reset joints
        self.joint_vals = joints

    def getJointRanges(self, includeFixed=False):
        """
        Parameters
        ----------
        includeFixed : bool

        Returns
        -------
        lowerLimits : [ float ] * numDofs
        upperLimits : [ float ] * numDofs
        jointRanges : [ float ] * numDofs
        restPoses : [ float ] * numDofs
        """

        lowerLimits, upperLimits, jointRanges, restPoses = [], [], [], []

        numJoints = p.getNumJoints(self.robot_id, physicsClientId=self.pybullet_id)

        for i in range(numJoints):
            jointInfo = p.getJointInfo(self.robot_id, i, physicsClientId=self.pybullet_id)

            if includeFixed or jointInfo[3] > -1:

                ll, ul = jointInfo[8:10]
                jr = ul - ll

                # For simplicity, assume resting state == initial state
                rp = p.getJointState(
                    self.robot_id, i, physicsClientId=self.pybullet_id
                )[0]

                lowerLimits.append(-2)
                upperLimits.append(2)
                jointRanges.append(2)
                restPoses.append(rp)

        return lowerLimits, upperLimits, jointRanges, restPoses

    def accurateIK(
        self,
        endEffectorId,
        targetPosition,
        lowerLimits,
        upperLimits,
        jointRanges,
        restPoses,
        useNullSpace=False,
        maxIter=10,
        threshold=1e-4
    ):
        """
        Parameters
        ----------
        endEffectorId : int
        targetPosition : [float, float, float]
        lowerLimits : [float] 
        upperLimits : [float] 
        jointRanges : [float] 
        restPoses : [float]
        useNullSpace : bool
        maxIter : int
        threshold : float

        Returns
        -------
        jointPoses : [float] * numDofs
        """
        closeEnough = False
        iter = 0
        dist2 = 1e30

        numJoints = p.getNumJoints(self.robot_id, physicsClientId=self.pybullet_id)

        while (not closeEnough and iter < maxIter):
            if useNullSpace:
                jointPoses = p.calculateInverseKinematics(
                    self.robot_id,
                    endEffectorId,
                    targetPosition,
                    lowerLimits=lowerLimits,
                    upperLimits=upperLimits,
                    jointRanges=jointRanges,
                    restPoses=restPoses,
                    physicsClientId=self.pybullet_id
                )
            else:
                jointPoses = p.calculateInverseKinematics(
                    self.robot_id,
                    endEffectorId,
                    targetPosition,
                    physicsClientId=self.pybullet_id
                )

            for i in range(numJoints):
                jointInfo = p.getJointInfo(
                    self.robot_id, i, physicsClientId=self.pybullet_id
                )
                qIndex = jointInfo[3]
                if qIndex > -1:
                    p.resetJointState(
                        self.robot_id,
                        i,
                        jointPoses[qIndex - 7],
                        physicsClientId=self.pybullet_id
                    )
            ls = p.getLinkState(
                self.robot_id, endEffectorId, physicsClientId=self.pybullet_id
            )
            newPos = ls[4]
            diff = [
                targetPosition[0] - newPos[0],
                targetPosition[1] - newPos[1],
                targetPosition[2] - newPos[2],
            ]
            dist2 = np.sqrt((diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]))
            # print("dist2=", dist2)
            closeEnough = (dist2 < threshold)
            iter = iter + 1
        # print("iter=", iter)
        return jointPoses

    def setMotors(self, jointPoses):
        """
        Parameters
        ----------
        jointPoses : [float] * numDofs
        """
        numJoints = p.getNumJoints(self.robot_id, physicsClientId=self.pybullet_id)

        for i in range(numJoints):
            jointInfo = p.getJointInfo(self.robot_id, i, physicsClientId=self.pybullet_id)
            # print(jointInfo)
            qIndex = jointInfo[3]
            if qIndex > -1:
                p.setJointMotorControl2(
                    bodyIndex=self.robot_id,
                    jointIndex=i,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=jointPoses[qIndex - 7],
                    physicsClientId=self.pybullet_id
                )
