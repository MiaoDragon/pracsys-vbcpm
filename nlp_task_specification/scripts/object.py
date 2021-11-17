"""
This script defines a single object, and multiple objects
"""
import pybullet as p
import open3d as o3d
import numpy as np
class MoveableObject():
    def __init__(self, geo_info, pos, ori, pybullet_id):
        """
        geo_info:
            - visual_mesh
            - collision_mesh
            - scale
            - pcd
            - mass  (TODO: might be a dynamic property instead)
        """
        obj_i_c = p.createCollisionShape(shapeType=p.GEOM_MESH, meshScale=geo_info['scale'], \
                                        fileName=geo_info['collision_mesh'], \
                                        physicsClientId=pybullet_id)
        obj_i_v = p.createVisualShape(shapeType=p.GEOM_MESH, meshScale=geo_info['scale'], \
                                        fileName=geo_info['visual_mesh'], \
                                        physicsClientId=pybullet_id)
        obj_i = p.createMultiBody(baseCollisionShapeIndex=obj_i_c, baseVisualShapeIndex=obj_i_v, \
                        basePosition=pos, baseOrientation=ori,
                        baseMass=geo_info['mass'], physicsClientId=pybullet_id)
        self.id = obj_i
        pcd = o3d.io.read_point_cloud(geo_info['pcd'])
        pcd = np.asarray(pcd.points)*np.array(geo_info['scale'])  # unless we need other info like normals
        obj_height = np.max(pcd[:,2]) - np.min(pcd[:,2])




"""
Define a collect of moveable objects. This data structure should take care of object-object interactions
to keep track of when they're in contact.
A contact graph is built to monitor the behavior, where the robot is also part of.
"""
class MoveableObjectCollection():
    def __init__(self, objects, robot):
        # given a list of objects, and the robot, build a collect (contact graph)
        self.objects = objects
        self.robot = robot