
"""
elements: robot, [object_i], target_object, shelf
problem:
    given:
        object_i at pose_i for all i
        target_object at target_pose
    find:
        robot sequences tau (including actions grasp and push)
    such that:
        object[i].propagate(tau) is stable and within shelf, for all i
        target_object.propagate(tau).endpoint is in robot hand
        sweep(robot, tau) does not collide with the shelf
"""
from collision import Collision
class FullyObsProblemDefinition():
    def __init__(self, objects, target_object, robot, shelf, collision: Collision):
        self.objects = objects
        self.target_object = target_object
        self.robot = robot
        self.shelf = shelf
        self.collision = collision
    def evaluate(self, robot_sequence):
        """
        evaluate an open-loop trajectory
        """
        # * check if the robot collides with the environment
        if self.collision.robot_collide_traj(self.robot, robot_sequence, self.shelf):
            return False
        
        # * check if each object is stable and within shelf & if target_object is already in hand
        if not self.objects.propagate_and_target_eval(self.target_object, robot_sequence):
            return False  # invalid robot sequence
        return True  # evaluation is successfuls