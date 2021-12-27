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

        self.lowerLimits, self.upperLimits, self.jointRanges, self.restPoses = self.getJointRanges(
            includeFixed=False
        )

    def set_joints(self, joints):
        for i in range(self.num_joints):
            p.resetJointState(
                self.robot_id, i, joints[i], 0, physicsClientId=self.pybullet_id
            )

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
        targetOrientation,
        lowerLimits=None,
        upperLimits=None,
        jointRanges=None,
        restPoses=None,
        maxIter=500,
        threshold=1e-3
    ):
        """
        Parameters
        ----------
        endEffectorId : int
        targetPosition : [float, float, float]
        targetOrientation : [float, float, float, float]
        lowerLimits : [float]
        upperLimits : [float]
        jointRanges : [float]
        restPoses : [float]
        maxIter : int
        threshold : float

        Returns
        -------
        jointPoses : [float] * numDofs
        inthresh: [bool]
        """

        if lowerLimits is None:
            lowerLimits = self.lowerLimits
        if upperLimits is None:
            upperLimits = self.upperLimits
        if jointRanges is None:
            jointRanges = self.jointRanges
        if restPoses is None:
            restPoses = self.restPoses

        inthresh = False
        dist2 = 1e30
        # niter = 0

        numJoints = p.getNumJoints(self.robot_id, physicsClientId=self.pybullet_id)

        # while (not inthresh and niter < maxIter):
        jointPoses = p.calculateInverseKinematics(
            self.robot_id,
            endEffectorId,
            targetPosition,
            targetOrientation,
            lowerLimits,
            upperLimits,
            jointRanges,
            restPoses,
            maxNumIterations=maxIter,
            residualThreshold=threshold,
            physicsClientId=self.pybullet_id
        )

        for i in range(numJoints):
            jointInfo = p.getJointInfo(self.robot_id, i, physicsClientId=self.pybullet_id)
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
        # inthresh = (dist2 < threshold)
        # niter = niter + 1
        return jointPoses, dist2

    def setMotors(self, jointPoses):
        """
        Parameters
        ----------
        jointPoses : [float] * numDofs
        """

        for i in range(self.num_joints):
            jointInfo = p.getJointInfo(self.robot_id, i, physicsClientId=self.pybullet_id)
            print(jointInfo)
            qIndex = jointInfo[3]
            if qIndex > -1:
                p.setJointMotorControl2(
                    bodyIndex=self.robot_id,
                    jointIndex=i,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=jointPoses[qIndex - 7],
                    physicsClientId=self.pybullet_id
                )

    def getGrasps(self, object_id, resolution=9):
        """
        resolution must be odd
        """

        # grasp orientations
        # vertical
        vert = []
        x = 0
        y = np.pi
        # rotate along gripper axis:
        for z in np.linspace(-np.pi, np.pi, resolution):
            vert.append(p.getQuaternionFromEuler((x, y, z)))

        # horizontal
        horz = []
        y = np.pi / 2
        # rotate along gripper axis:
        for x in np.linspace(-np.pi, np.pi, resolution):
            # rotate along horizontal axis:
            for z in np.linspace(-np.pi / 2, np.pi / 2, resolution):
                horz.append(p.getQuaternionFromEuler((x, y, z)))

        # object position and orientation
        obj_pos, obj_rot = p.getBasePositionAndOrientation(object_id, self.pybullet_id)

        # positions along shape
        grasps = []
        shape = p.getCollisionShapeData(object_id, -1, self.pybullet_id)[0]
        if shape[2] == p.GEOM_BOX:
            sx, sy, sz = shape[3]
            top = [0, 0, sz / 2]
            left = [0, -sy / 2, 0]
            right = [0, sy / 2, 0]
            # front = [-sx/2,0,0]
            grasps = [
                [left, horz[0]], [right, horz[-1]], [top, vert[(resolution - 1) // 2]]
            ]
        elif shape[2] == p.GEOM_CYLINDER or shape[2] == p.GEOM_CAPSULE:
            h, r = shape[3][:2]
            grasps = [[(0, 0, 0), o] for o in vert + horz]
        elif shape[2] == p.GEOM_SPHERE:
            r = shape[3][0]
            grasps = [[(0, 0, 0), o] for o in vert + horz]
        # elif shape[2] == p.GEOM_MESH:
        # elif shape[2] == p.GEOM_PLANE:

        poses = []
        for pos, rot in grasps:
            tpos, trot = p.multiplyTransforms(
                (0, 0, 0), rot, (0, 0, 0), obj_rot, self.pybullet_id
            )
            pose = [np.add(obj_pos, pos), trot]
            poses.append(pose)

        return poses

    def filterGrasps(
        self,
        endEffectorId,
        grasps,
        stopThreshold=1e-3,
        filterThreshold=1e-02,
    ):

        init_states = [
            x[0] for x in p.getJointStates(self.robot_id, range(self.num_joints))
        ]

        filteredJointPoses = []
        for pos, rot in grasps:
            jointPoses, dist = self.accurateIK(
                endEffectorId,
                pos,
                rot,
                threshold=stopThreshold,
            )
            input(dist)
            if dist < filterThreshold:
                joint_states = [
                    x[0] for x in p.getJointStates(self.robot_id, range(self.num_joints))
                ]
                # filteredJointPoses.append(jointPoses)
                filteredJointPoses.append(joint_states)

        self.set_joints(init_states)

        return filteredJointPoses
