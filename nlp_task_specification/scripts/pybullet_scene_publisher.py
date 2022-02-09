#!/usr/bin/env python
from __future__ import print_function

import rospy
import pybullet as p
from geometry_msgs.msg import Point
from moveit_commander.conversions import *
from shape_msgs.msg import SolidPrimitive, Mesh, MeshTriangle

from planit.msg import PercievedObject


class PybulletScenePublisher:

    def __init__(self, ignore_ids={0}):
        rospy.init_node('PybulletScenePublisher')
        self.publisher = rospy.Publisher('/perception', PercievedObject, queue_size=10)
        self.pybullet_id = p.connect(p.SHARED_MEMORY)
        self.ignore_ids = ignore_ids

    def publish(self):
        for i in range(p.getNumBodies(physicsClientId=self.pybullet_id)):
            obj_pid = p.getBodyUniqueId(i, physicsClientId=self.pybullet_id)
            if obj_pid in self.ignore_ids:
                continue
            obj_msg = self.obj2msg(obj_pid, self.pybullet_id)
            self.publisher.publish(obj_msg)
            # print(obj_msg)

    def obj2msg(self, object_id, use_collision=True):
        obj_msg = PercievedObject()
        obj_msg.header.frame_id = 'world'
        obj_msg.header.stamp = rospy.get_rostime()
        obj_msg.name = f'Obj_{object_id}'
        pos, rot = p.getBasePositionAndOrientation(object_id, self.pybullet_id)
        obj_msg.pose = Pose()
        obj_msg.pose.position.x = pos[0]
        obj_msg.pose.position.y = pos[1]
        obj_msg.pose.position.z = pos[2]
        obj_msg.pose.orientation.x = rot[0]
        obj_msg.pose.orientation.y = rot[1]
        obj_msg.pose.orientation.z = rot[2]
        obj_msg.pose.orientation.w = rot[3]
        # obj_msg.mesh = Mesh()
        obj_msg.solid = SolidPrimitive()
        obj_msg.solid.dimensions = [0]

        if use_collision:
            shape = p.getCollisionShapeData(object_id, -1, self.pybullet_id)[0]
        else:
            # TODO: use visual data
            shape = p.getCollisionShapeData(object_id, -1, self.pybullet_id)[0]

        if shape[2] == p.GEOM_MESH:
            print(
                "Element %s with geometry type %s not supported. Ignored." %
                (object_id, shape[2])
            )
            return None
        else:
            obj_msg.type = PercievedObject.SOLID_PRIMITIVE
            if shape[2] == p.GEOM_BOX:
                obj_msg.solid.type = SolidPrimitive.BOX
                obj_msg.solid.dimensions = list(shape[3])
            elif shape[2] == p.GEOM_CYLINDER:
                obj_msg.solid.type = SolidPrimitive.CYLINDER
                obj_msg.solid.dimensions = list(shape[3])
            elif shape[2] == p.GEOM_SPHERE:
                obj_msg.solid.type = SolidPrimitive.SPHERE
                obj_msg.solid.dimensions = list(shape[3])
            elif shape[2] == p.GEOM_CAPSULE:
                print(
                    "Element %s with geometry type %s not supported. Ignored." %
                    (object_id, shape[2])
                )
                return None
            elif shape[2] == p.GEOM_PLANE:
                print(
                    "Element %s with geometry type %s not supported. Ignored." %
                    (object_id, shape[2])
                )
                return None

        return obj_msg


if __name__ == '__main__':
    pybullet_scene_pub = PybulletScenePublisher()
    rate = rospy.Rate(2)
    while not rospy.is_shutdown():
        pybullet_scene_pub.publish()
        rate.sleep()

    rospy.loginfo('Spinning...')
    rospy.spin()
