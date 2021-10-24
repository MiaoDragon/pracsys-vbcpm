from robot import Robot
from workspace import Workspace
import pybullet as p
class Collision():
    def __init__(self, pybullet_id):
        self.pybullet_id = pybullet_id

    def collide(self, robot: Robot, joints, shelf: Workspace):
        # check collision of the robot at joint angle with the shelf
        # set robot to the joint
        prev_joints = robot.joint_vals
        robot.set_joints(joints)
        for name, component_id in shelf.components:
            contact_points = p.getClosestPoints(robot.robot_id, component_id, distance=0.01)
        robot.set_joints(prev_joints)  # reset joints
        return len(contact_points) == 0

    def collide_traj(self, robot: Robot, sequence, shelf: Workspace):
        # check for the trajectory if collision happens
        for i in range(len(sequence)):
            tag, seq_i = sequence[i]
            # TODO: when the tag is grasping, should also check collision for the grasped object
            # sequence can be a trajectory of points, or a trajectory that has velocities too
            # for now we assume we only care about the point information
            for joints in seq_i:
                if self.collide(robot, joints, shelf):
                    return True
        return False
