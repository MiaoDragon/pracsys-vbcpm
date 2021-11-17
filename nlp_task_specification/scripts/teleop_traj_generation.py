"""
a teleop interface for generating a trajectory. Trajetory format is:

Action Tag | Trajectory Points

We assume the trajectory results in stable motion of the object, i.e.
the object will remain in contact with the robot when pushed, and will
suddenly stop at the current location when robot moves away.
"""
class TeleopTrajGeneration():
    def __init__(self):
        pass