import open3d as o3d
import numpy as np
from psutil import Process
import pygraphviz as pgv


def visualize_pcd(pcd, color):
    pcd_pcd = o3d.geometry.PointCloud()
    pcd_pcd.points = o3d.utility.Vector3dVector(pcd)
    colors = np.zeros(pcd.shape)
    colors[:,0] = color[0]
    colors[:,1] = color[1]
    colors[:,2] = color[2]
    pcd_pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd_pcd

def visualize_voxel(voxel_x, voxel_y, voxel_z, filter, color):
    pcd = o3d.geometry.PointCloud()
    voxel_x = voxel_x[filter].reshape(-1,1)
    voxel_y = voxel_y[filter].reshape(-1,1)
    voxel_z = voxel_z[filter].reshape(-1,1)
    pcd_points = np.concatenate([voxel_x+0.5, voxel_y+0.5, voxel_z+0.5], axis=1)
    pcd.points = o3d.utility.Vector3dVector(pcd_points)
    colors = np.zeros(pcd_points.shape)
    colors[:,0] = color[0]
    colors[:,1] = color[1]
    colors[:,2] = color[2]
    pcd.colors = o3d.utility.Vector3dVector(colors)

    min_bound = [voxel_x.min(),voxel_y.min(),voxel_z.min()]
    min_bound = np.array(min_bound)
    max_bound = [voxel_x.max(), voxel_y.max(), voxel_z.max()]
    max_bound = np.array(max_bound)+1.0
    voxel = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(pcd, 1., min_bound, max_bound)


    # bbox = voxel.get_axis_aligned
    return voxel

def visualize_bbox(voxel_x, voxel_y, voxel_z):
    min_bound = [voxel_x.min(),voxel_y.min(),voxel_z.min()]
    min_bound = np.array(min_bound)
    max_bound = [voxel_x.max(), voxel_y.max(), voxel_z.max()]
    max_bound = np.array(max_bound)+1.0

    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    return bbox

def visualize_coordinate_frame_centered(size=1.0, transform=np.eye(4)):
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size).transform(transform)
    return frame

def visualize_arrow(scale=1.0, translation=np.zeros(3), direction=np.array([0,0,1.0]), color=[1,0,0]):
    arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=1*scale, cone_radius=1.5*scale, cylinder_height=5*scale, cone_height=4*scale)
    z_axis = direction
    x_axis = np.array([-z_axis[2], 0, z_axis[0]])
    y_axis = np.cross(z_axis, x_axis)
    rotation = np.array([x_axis, y_axis, z_axis]).T
    transform = np.eye(4)
    transform[:3,:3] = rotation
    transform[:3,3] = translation
    arrow.transform(transform)
    arrow.paint_uniform_color(color)
    return arrow
    
def visualize_mesh(vertices, triangles, color=[1,0,0]):
    vert = o3d.utility.Vector3dVector(vertices)
    tria = o3d.utility.Vector3iVector(triangles)
    mesh = o3d.geometry.TriangleMesh(vert, tria)
    mesh.paint_uniform_color(color)
    return mesh

def get_color_picks():
    color_pick = np.zeros((8,3))
    color_pick[0] = np.array([1., 0., 0.])
    color_pick[1] = np.array([0., 1.0, 0.])
    color_pick[2] = np.array([0., 0., 1.])
    color_pick[3] = np.array([252/255, 169/255, 3/255])
    color_pick[4] = np.array([252/255, 3/255, 252/255])
    color_pick[5] = np.array([20/255, 73/255, 82/255])
    color_pick[6] = np.array([22/255, 20/255, 82/255])
    color_pick[7] = np.array([60/255, 73/255, 10/255])
    return color_pick


import copy
import pybullet as p
import transformations as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time

def transform_pybullet_obj(obj_id, obj_pybullet_pose, pid):
    quat = tf.quaternion_from_matrix(obj_pybullet_pose) # w x y z
    p.resetBasePositionAndOrientation(obj_id, obj_pybullet_pose[:3,3], [quat[1],quat[2],quat[3],quat[0]], physicsClientId=pid)


def construct_occlusion_graph(obj_ids, obj_pybullet_poses, camera, pid):
    plt.ion()
    dot = pgv.AGraph(directed=True)
    # move each object to faraway and capture depth image, seg image, then move back
    translation_z = 10.0
    transform = np.eye(4)
    transform[2,3] = translation_z
    rev_transform = np.array(transform)
    rev_transform[2,3] = -translation_z

    obj_pybullet_poses = copy.deepcopy(obj_pybullet_poses)

    # move object away
    for obj_i in range(len(obj_ids)):
        # obj_i.update_transform_from_relative(transform)
        obj_pybullet_poses[obj_i][2,3] += translation_z
        transform_pybullet_obj(obj_ids[obj_i], obj_pybullet_poses[obj_i], pid)


    # take depth image
    depth_img = np.zeros(camera.info['img_shape'])
    seg_img = np.zeros(camera.info['img_shape']).astype(int)-1

    depth_imgs = []
    seg_imgs = []
    for obj_i in range(len(obj_ids)):
        obj_pybullet_poses[obj_i][2,3] -= translation_z
        transform_pybullet_obj(obj_ids[obj_i], obj_pybullet_poses[obj_i], pid)

        rgb_img_i, depth_img_i, seg_img_i = camera.sense()

        depth_imgs.append(depth_img_i)
        seg_img = (seg_img_i==obj_ids[obj_i])
        seg_imgs.append(seg_img)

        obj_pybullet_poses[obj_i][2,3] += translation_z
        transform_pybullet_obj(obj_ids[obj_i], obj_pybullet_poses[obj_i], pid)

    # reset
    for obj_i in range(len(obj_ids)):
        obj_pybullet_poses[obj_i][2,3] -= translation_z
        transform_pybullet_obj(obj_ids[obj_i], obj_pybullet_poses[obj_i], pid)

    visible_nodes = []
    nodes = [obj_i for obj_i in obj_ids]

    # connect edge from obj to camera
    rgb_img_i, depth_img_i, seg_img_i = camera.sense()
    visible_set = set(seg_img_i.reshape(-1).tolist())
    node_set = set(nodes)
    visible_set = node_set.intersection(visible_set)
    visible_set = list(visible_set)



    # for each object identify the object that it directly hides
    edges = []
    for obj_i in range(len(obj_ids)):
        min_depth = np.zeros(depth_imgs[0].shape) + 10e3
        min_id_img = np.zeros(depth_imgs[0].shape).astype(int)-1
        for obj_j in range(len(obj_ids)):
            if obj_i == obj_j:
                continue
            # j is directly hidden by i: depth of i < depth of j
            mask = seg_imgs[obj_i] & seg_imgs[obj_j]
            mask = mask & (depth_imgs[obj_i] < depth_imgs[obj_j])
            if mask.sum() == 0:
                continue
            min_id_img[mask & (depth_imgs[obj_j]<min_depth)] = obj_ids[obj_j]
            min_depth[mask & (depth_imgs[obj_j]<min_depth)] = depth_imgs[obj_j][mask & (depth_imgs[obj_j]<min_depth)]
        min_id_list = list(set(min_id_img.reshape(-1).tolist()))
        if len(min_id_list) == 1:
            # no direct occlusion
            continue
        for hide_id in min_id_list:
            if hide_id == -1:
                continue
            edges.append((hide_id, obj_ids[obj_i]))

    # given the edges, construct the graph
    for i in range(len(nodes)):
        dot.add_node(nodes[i], pos='%f,%f!' % (-10*obj_pybullet_poses[i][1,3],10*obj_pybullet_poses[i][0,3]))
    for i in range(len(edges)):
        dot.add_edge(edges[i][0], edges[i][1])
    
    dot.add_node('cam', pos='%f,%f!' % (-10*camera.info['extrinsics'][1,3],10*camera.info['extrinsics'][0,3]))    
    for node_i in visible_set:
        dot.add_edge(node_i, 'cam')

    dot.layout()
    dot.draw("graph.png", format="png")
    img = mpimg.imread('graph.png')
    plt.clf()
    plt.imshow(img)
    plt.show()
    plt.pause(0.0001)

def update_occlusion_graph(obj_ids, obj_pybullet_poses, moved_obj_ids, valid_obj_ids, move_obj_id, camera, pid):
    plt.ion()

    dot = pgv.AGraph(directed=True)
    # move each object to faraway and capture depth image, seg image, then move back
    translation_z = 10.0
    transform = np.eye(4)
    transform[2,3] = translation_z
    rev_transform = np.array(transform)
    rev_transform[2,3] = -translation_z

    obj_pybullet_poses = copy.deepcopy(obj_pybullet_poses)

    # move object away
    for obj_i in range(len(obj_ids)):
        # obj_i.update_transform_from_relative(transform)
        obj_pybullet_poses[obj_i][2,3] += translation_z
        transform_pybullet_obj(obj_ids[obj_i], obj_pybullet_poses[obj_i], pid)


    # take depth image
    depth_img = np.zeros(camera.info['img_shape'])
    seg_img = np.zeros(camera.info['img_shape']).astype(int)-1

    depth_imgs = []
    seg_imgs = []
    for obj_i in range(len(obj_ids)):
        obj_pybullet_poses[obj_i][2,3] -= translation_z
        transform_pybullet_obj(obj_ids[obj_i], obj_pybullet_poses[obj_i], pid)

        rgb_img_i, depth_img_i, seg_img_i = camera.sense()

        depth_imgs.append(depth_img_i)
        seg_img = (seg_img_i==obj_ids[obj_i])
        seg_imgs.append(seg_img)

        obj_pybullet_poses[obj_i][2,3] += translation_z
        transform_pybullet_obj(obj_ids[obj_i], obj_pybullet_poses[obj_i], pid)

    # reset
    for obj_i in range(len(obj_ids)):
        obj_pybullet_poses[obj_i][2,3] -= translation_z
        transform_pybullet_obj(obj_ids[obj_i], obj_pybullet_poses[obj_i], pid)

    visible_nodes = []
    nodes = [obj_i for obj_i in obj_ids]

    # connect edge from obj to camera
    rgb_img_i, depth_img_i, seg_img_i = camera.sense()
    visible_set = set(seg_img_i.reshape(-1).tolist())
    node_set = set(nodes)
    visible_set = node_set.intersection(visible_set)
    visible_set = list(visible_set)



    # for each object identify the object that it directly hides
    edges = []
    for obj_i in range(len(obj_ids)):
        min_depth = np.zeros(depth_imgs[0].shape) + 10e3
        min_id_img = np.zeros(depth_imgs[0].shape).astype(int)-1
        for obj_j in range(len(obj_ids)):
            if obj_i == obj_j:
                continue
            # if the object has already been move and sensed, then skip
            if obj_ids[obj_j] in moved_obj_ids:
                continue

            # j is directly hidden by i: depth of i < depth of j
            mask = seg_imgs[obj_i] & seg_imgs[obj_j]
            mask = mask & (depth_imgs[obj_i] < depth_imgs[obj_j])
            if mask.sum() == 0:
                continue
            min_id_img[mask & (depth_imgs[obj_j]<min_depth)] = obj_ids[obj_j]
            min_depth[mask & (depth_imgs[obj_j]<min_depth)] = depth_imgs[obj_j][mask & (depth_imgs[obj_j]<min_depth)]
        min_id_list = list(set(min_id_img.reshape(-1).tolist()))
        if len(min_id_list) == 1:
            # no direct occlusion
            continue
        for hide_id in min_id_list:
            if hide_id == -1:
                continue
            edges.append((hide_id, obj_ids[obj_i]))

    # given the edges, construct the graph
    for i in range(len(nodes)):
        color = 'white'
        if nodes[i] in valid_obj_ids:
            color = 'skyblue'
        if nodes[i] in moved_obj_ids:
            color = 'lightpink'
        if nodes[i] == move_obj_id:
            color = 'seagreen'

        dot.add_node(nodes[i], pos='%f,%f!' % (-10*obj_pybullet_poses[i][1,3],10*obj_pybullet_poses[i][0,3]),
                     fillcolor=color, style='filled')
    for i in range(len(edges)):
        dot.add_edge(edges[i][0], edges[i][1])

    dot.add_node('cam', pos='%f,%f!' % (-25*camera.info['extrinsics'][1,3],25*camera.info['extrinsics'][0,3]))    


    # dot.add_node('cam', pos='%f,%f!' % (-10*camera.info['extrinsics'][1,3],10*camera.info['extrinsics'][0,3]))    
    for node_i in visible_set:
        dot.add_edge(node_i, 'cam')

    dot.layout()
    dot.draw("graph.png", format="png")
    dot.draw("graph.pdf", format="pdf")

    img = mpimg.imread('graph.png')
    plt.clf()
    plt.imshow(img)
    plt.show()
    plt.pause(0.1)


