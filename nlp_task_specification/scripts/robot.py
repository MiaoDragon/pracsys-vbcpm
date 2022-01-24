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
        self.right_fingers = [27, 29]
        self.right_flim = p.getJointInfo(self.robot_id, self.right_fingers[0])[8:10]
        self.left_gripper_id = 48
        self.left_fingers = [49, 51]
        self.left_flim = p.getJointInfo(self.robot_id, self.left_fingers[0])[8:10]

        self.lowerLimits, self.upperLimits, self.jointRanges, self.restPoses = self.getJointRanges(
            includeFixed=False
        )

    def set_joints(self, joints):
        for i in range(self.num_joints):
            p.resetJointState(
                self.robot_id, i, joints[i], 0, physicsClientId=self.pybullet_id
            )

    def set_gripper(self, gripper, state='open'):
        ind = 0 if state == 'closed' else 1
        if gripper == 'left' or gripper == self.left_gripper_id:
            p.resetJointState(
                self.robot_id,
                self.left_fingers[0],
                self.left_flim[ind],
                0,
                physicsClientId=self.pybullet_id
            )
            p.resetJointState(
                self.robot_id,
                self.left_fingers[1],
                -self.left_flim[ind],
                0,
                physicsClientId=self.pybullet_id
            )
        elif gripper == 'right' or gripper == self.right_gripper_id:
            p.resetJointState(
                self.robot_id,
                self.right_fingers[0],
                self.right_flim[ind],
                0,
                physicsClientId=self.pybullet_id
            )
            p.resetJointState(
                self.robot_id,
                self.right_fingers[1],
                -self.right_flim[ind],
                0,
                physicsClientId=self.pybullet_id
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

    def getGrasps(self, object_id, offset=(0, 0, 0.021), resolution=8):
        """
        resolution must be even
        """
        shape = p.getCollisionShapeData(object_id, -1, self.pybullet_id)[0]
        res = 5 if shape[2] == p.GEOM_BOX else resolution + 1
        hres = (res // 2) + 1

        # grasp orientations
        # vertical
        vert = []
        x = 0
        y = np.pi
        # rotate along gripper axis:
        for z in np.linspace(-np.pi, np.pi, res):
            vert.append(p.getQuaternionFromEuler((x, y, z)))

        # horizontal
        horz = []
        x = -np.pi / 2
        # rotate along gripper axis:
        for y in np.linspace(-np.pi, np.pi, res):
            # rotate along horizontal axis:
            for z in np.linspace(-np.pi, 0, hres):
                horz.append(p.getQuaternionFromEuler((x, y, z)))

        # object position and orientation
        obj_pos, obj_rot = p.getBasePositionAndOrientation(object_id, self.pybullet_id)

        # positions along shape
        grasps = []
        if shape[2] == p.GEOM_BOX:
            sx, sy, sz = shape[3]

            gw = self.right_flim[1] * 2  # gripper width

            def nearOdd(n):
                return round((n - 1) / 2) * 2 + 1

            # top = [0, 0, sz / 2]
            # left = [0, -sy / 2, 0]
            # right = [0, sy / 2, 0]
            # front = [-sx / 2, 0, 0]
            if sx < gw:
                noz = nearOdd(sz / gw)
                for z in np.linspace(-(noz - 1) / (2 * noz), (noz - 1) / (2 * noz), noz):
                    grasps.append([[0, sy / 2, z * sz], horz[3]])  # right
                    grasps.append([[0, sy / 2, z * sz], horz[9]])  # right
                    grasps.append([[0, -sy / 2, z * sz], horz[5]])  # left
                    grasps.append([[0, -sy / 2, z * sz], horz[11]])  # left
                noy = nearOdd(sy / gw)
                for y in np.linspace(-(noy - 1) / (2 * noy), (noy - 1) / (2 * noy), noy):
                    grasps.append([[0, y * sy, sz / 2], vert[1]])  # top
                    grasps.append([[0, y * sy, sz / 2], vert[3]])  # top
            if sy < gw:
                noz = nearOdd(sz / gw)
                for z in np.linspace(-(noz - 1) / (2 * noz), (noz - 1) / (2 * noz), noz):
                    grasps.append([[-sx / 2, 0, z * sz], horz[4]])  # front
                    grasps.append([[-sx / 2, 0, z * sz], horz[10]])  # front
                nox = nearOdd(sx / gw)
                for x in np.linspace(-(nox - 1) / (2 * nox), (nox - 1) / (2 * nox), nox):
                    grasps.append([[x * sx, 0, sz / 2], vert[0]])  # top
                    grasps.append([[x * sx, 0, sz / 2], vert[2]])  # top
        elif shape[2] == p.GEOM_CYLINDER or shape[2] == p.GEOM_CAPSULE:
            h, r = shape[3][:2]
            grasps += [[(0, 0, 0), o] for o in vert]
            grasps += [
                [(0, 0, 0), o]
                for o in horz[hres * ((res - 1) // 4):hres * ((res + 3) // 4)]
            ]
            grasps += [
                [(0, 0, 0), o]
                for o in horz[hres * ((-res - 1) // 4):hres * ((-res + 3) // 4)]
            ]
        elif shape[2] == p.GEOM_SPHERE:
            r = shape[3][0]
            grasps = [[(0, 0, 0), o] for o in vert + horz]
        # elif shape[2] == p.GEOM_MESH:
        # elif shape[2] == p.GEOM_PLANE:

        poses = []
        for pos, rot in grasps:
            tpos, trot = p.multiplyTransforms(
                (0, 0, 0), rot, offset, obj_rot, self.pybullet_id
            )
            pose = [tpos + np.add(obj_pos, pos), trot]
            poses.append(pose)

        return poses

    def filterGrasps(
        self,
        endEffectorId,
        grasps,
        collision_ignored=[],
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
            # input(rot)
            if dist < filterThreshold:  # filter by succesful IK
                self.set_gripper(endEffectorId, 'open')
                joint_states = [
                    x[0] for x in p.getJointStates(self.robot_id, range(self.num_joints))
                ]
                # filteredJointPoses.append(jointPoses)
                ignore_ids = collision_ignored + [self.robot_id]
                collisions = set()
                for i in range(p.getNumBodies(physicsClientId=self.pybullet_id)):
                    obj_pid = p.getBodyUniqueId(i, physicsClientId=self.pybullet_id)
                    if obj_pid in ignore_ids:
                        continue
                    contacts = p.getClosestPoints(
                        self.robot_id,
                        obj_pid,
                        distance=0.,
                        physicsClientId=self.pybullet_id,
                    )
                    if len(contacts):
                        collisions.add(obj_pid)
                if 1 not in collisions:  # dont add if collides with table
                    filteredJointPoses.append((joint_states, collisions, dist))

        self.set_joints(init_states)

        # return pose and collisions in sorted order
        filteredJointPoses = sorted(filteredJointPoses, key=lambda x: (len(x[1]), x[2]))
        return [x[0:2] for x in filteredJointPoses]
