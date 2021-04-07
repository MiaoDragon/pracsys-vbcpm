"""
This defines the setup in PyBullet for the Vision-Based Constrained Placement Task.
Taking input of a problem configuration (in the format of a json file), we load each
component into the PyBullet scene. We then set up the services and topics to communicate
with the planner, including a trajectory tracker service, a camera-related topic, and
a robot-related topic.
"""
import argparse
import utility
import pybullet as p
import rospy
import rospkg
import os
import time
import cv2
def main(args):
    rp = rospkg.RosPack()
    package_path = rp.get_path('vbcpm_execution_system')

    f = args.prob_config  # problem configuration file (JSON)
    prob_config_dict = utility.prob_config_parser(f)
    # * load each component in the problem configuration into the scene
    """
    format: 
        {'robot': {'pose': pose, 'urdf': urdf},
         'table': {'pose': pose, 'urdf': urdf},  # (orientation: [x,y,z,w])
         'objects': [{'pose': pose, 'urdf': urdf}],
         'camera': {'pose': pose, 'urdf': urdf},
         'placement': [(x1,y1),(x2,y2),(x3,y3),(x4,y4)]}
    """
    physicsClient = p.connect(p.GUI)
    # p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
    p.setGravity(0,0,-10)
    planeId = p.loadURDF("../data/models/plane/plane.urdf")

    # robot
    robot_pos = prob_config_dict['robot']['pose']['pos']
    robot_ori = prob_config_dict['robot']['pose']['ori']
    print('robot_path: ', os.path.join(package_path, prob_config_dict['robot']['urdf']))
    robot_id = p.loadURDF(os.path.join(package_path, prob_config_dict['robot']['urdf']),robot_pos, robot_ori, useFixedBase=True)

    # table
    table_pos = prob_config_dict['table']['pose']['pos']
    table_ori = prob_config_dict['table']['pose']['ori']
    robot_id = p.loadURDF(os.path.join(package_path, prob_config_dict['table']['urdf']),table_pos, table_ori, useFixedBase=True)


    # camera

    view_mat = p.computeViewMatrix(
        cameraEyePosition=[0.35, 0, 1.25],
        cameraTargetPosition=[1.35, 0, 0.58],
        cameraUpVector=[1.25 - 0.58, 0, 1.35 - 0.35]
    )

    # https://stackoverflow.com/questions/60430958/understanding-the-view-and-projection-matrix-from-pybullet
    # https://stackoverflow.com/questions/39992968/how-to-calculate-field-of-view-of-the-camera-from-camera-intrinsic-matrix
    # https://github.com/bulletphysics/bullet3/blob/master/examples/SharedMemory/PhysicsClientC_API.cpp#L4372
    proj_mat = p.computeProjectionMatrixFOV(
        fov=90,
        aspect=1,
        nearVal=0.01,
        farVal=1.5
    )

    # objects
    objs = []
    for obj_dict in prob_config_dict['objects']:
        obj_i_c = p.createCollisionShape(shapeType=p.GEOM_MESH, meshScale=obj_dict['scale'], \
                                        fileName=os.path.join(package_path, obj_dict['collision_mesh']))
        obj_i_v = p.createVisualShape(shapeType=p.GEOM_MESH, meshScale=obj_dict['scale'], \
                                        fileName=os.path.join(package_path, obj_dict['visual_mesh']))
        obj_i = p.createMultiBody(baseCollisionShapeIndex=obj_i_c, baseVisualShapeIndex=obj_i_v, \
                          basePosition=obj_dict['pose']['pos'], baseOrientation=obj_dict['pose']['ori'],
                          baseMass=obj_dict['mass'])
        objs.append(obj_i)
    for i in range (10000):
        # width, height, rgb_img, depth_img, seg_img = p.getCameraImage(
        #     width=640,
        #     height=640,
        #     viewMatrix=view_mat,
        #     projectionMatrix=proj_mat)
        # cv2.imshow('camera_rgb', rgb_img)
        p.stepSimulation()
        time.sleep(1/240)

    # * start running the ROS service for trajectory tracker
    # * start running the ROS service for the camera
    # * start running the ROS service for the robot topic

    p.disconnect()


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--prob_config',type=str, required=True, help='the path to the problem configuration file (JSON).')

args = parser.parse_args()
main(args)