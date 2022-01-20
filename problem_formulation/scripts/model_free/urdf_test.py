"""
test the urdfpy package
"""
import json
from numpy.lib.shape_base import column_stack
import rospkg
import os

from urdfpy import URDF, Link, Joint, Transmission, Material
import numpy as np
import open3d as o3d
from visual_utilities import *
import pybullet as p
from robot import Robot

scene = 'scene1.json'
f = open(scene, 'r')
scene_dict = json.load(f)

rp = rospkg.RosPack()
package_path = rp.get_path('vbcpm_execution_system')
urdf_path = os.path.join(package_path,scene_dict['robot']['urdf'])

robot = URDF.load(urdf_path)
print('links: ')
for link in robot.links:
    print(link.name)

for joint in robot.joints:
    print('joint name: ')
    print(joint.name)
    print('parent: ')
    print(joint.parent)
    print('child: ')
    print(joint.child)

# check the collision mesh
link_pose_dict = robot.link_fk()

print('############checking collision mesh#############3')
pcd_dict = {}
pcd_list = []
mesh_list = []
for link in robot.links:
    print('link name: ', link.name)
    collisions = link.collisions
    for collision in collisions:
        origin = collision.origin  # this is the relative transform to get the pose of the geometry (mesh)
        # pose of the trimesh:
        # pose of link * origin * scale * trimesh_obj
        geometry = collision.geometry.geometry
        print('geometry: ', geometry)
        print('tag: ', geometry._TAG)
        # geometry.scale: give us the scale for the mesh
        
        # trimesh pose
        transform = link_pose_dict[link]

        meshes = geometry.meshes
        for mesh in meshes:
            # print('mesh vertices: ')
            # print(mesh.vertices)
            # mesh.sample()
            vertices = np.asarray(mesh.vertices)
            triangles = np.asarray(mesh.faces)
            if collision.geometry.mesh is not None:
                if collision.geometry.mesh.scale is not None:
                    vertices = vertices * collision.geometry.mesh.scale  # scale
            vertices = origin[:3,:3].dot(vertices.T).T + origin[:3,3]
            vertices = transform[:3,:3].dot(vertices.T).T + transform[:3,3]
            # get link pose
            o3d_mesh = visualize_mesh(vertices, triangles)
            mesh_list.append(o3d_mesh)


            pcd = mesh.sample(len(vertices)*5)
            if collision.geometry.mesh is not None:
                if collision.geometry.mesh.scale is not None:
                    pcd = pcd * collision.geometry.mesh.scale
            pcd = origin[:3,:3].dot(pcd.T).T + origin[:3,3]
            pcd = transform[:3,:3].dot(pcd.T).T + transform[:3,3]
            
            o3d_pcd = visualize_pcd(pcd, color=[1,0,0])
            pcd_list.append(o3d_pcd)
            print('len(pcd): ', len(pcd))

o3d.visualization.draw_geometries(mesh_list)            
o3d.visualization.draw_geometries(pcd_list)            


input('next...')
pid = p.connect(p.GUI)


joints = [0.] * 16

ll = [-1.58, \
    -3.13, -1.90, -2.95, -2.36, -3.13, -1.90, -3.13, \
    -3.13, -1.90, -2.95, -2.36, -3.13, -1.90, -3.13] +  \
    [0.0, -0.8757, 0.0, 0.0, -0.8757, 0.0]
### upper limits for null space
ul = [1.58, \
    3.13, 1.90, 2.95, 2.36, 3.13, 1.90, 3.13, \
    3.13, 1.90, 2.95, 2.36, 3.13, 1.90, 3.13] + \
    [0.8, 0.0, 0.8757, 0.81, 0.0, 0.8757]
### joint ranges for null space
jr = [1.58*2, \
    6.26, 3.80, 5.90, 4.72, 6.26, 3.80, 6.26, \
    6.26, 3.80, 5.90, 4.72, 6.26, 3.80, 6.26] + \
    [0.8, 0.8757, 0.8757, 0.81, 0.8757, 0.8757]

joints = [0,
          1.75, 0.8, 0.0, -0.66, 0.0, 0.0 ,0.0,
          0.0, 0.0, 0.0, -2.08, 0.0, 0.0, 0.0]

robot = Robot(urdf_path, scene_dict['robot']['pose']['pos'], scene_dict['robot']['pose']['ori'], 
                ll, ul, jr, 'motoman_left_ee', 0.3015, pid)

pcd = robot.get_pcd_at_joints(joints)
o3d_pcd = visualize_pcd(pcd, [1,0,0])
o3d.visualization.draw_geometries([o3d_pcd])            
