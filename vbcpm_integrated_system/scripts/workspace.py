"""
This script defines the workspace object that remains static, and should be avoided
during planning for collision.
"""
import pybullet as p
import numpy as np
import transformations as tf
class Workspace():
    def __init__(self, base_pos, base_ori, components, workspace_low, workspace_high, padding, pybullet_id):
        # components is a list of geometric parts
        self.components = {}
        self.component_id_dict = {}
        self.component_ids = []
        bbox_lls = {}
        bbox_uls = {}
        transforms = {}
        for component_name, component in components.items():
            print('component name: ')
            print(component_name)
            shape = component['shape']
            shape = np.array(shape)
            # pos = np.array(component['pose']['pos'])
            component['pose']['pos'] = np.array(component['pose']['pos']) + np.array(base_pos)
            pos = np.array(component['pose']['pos'])
            ori = component['pose']['ori']  # x y z w

            col_id = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=shape/2, physicsClientId=pybullet_id)
            vis_id = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=shape/2, rgbaColor=[160/255, 107/255, 84/255, 1.0], physicsClientId=pybullet_id)
            comp_id = p.createMultiBody(baseCollisionShapeIndex=col_id, baseVisualShapeIndex=vis_id,
                                        basePosition=pos, baseOrientation=ori, physicsClientId=pybullet_id)
            self.component_id_dict[component_name] = comp_id
            self.component_ids.append(comp_id)
            self.components[component_name] = component


            rot_mat = tf.quaternion_matrix([ori[3], ori[0], ori[1], ori[2]])
            rot_mat[:3,3] = pos
            transforms[component_name] = rot_mat
            bbox_lls[component_name] = -shape/2
            bbox_uls[component_name] = shape/2

        self.pos = pos
        self.ori = ori

        # modify the workspace_low and high by using the padding
        workspace_low = np.array(workspace_low)
        workspace_high = np.array(workspace_high)

        workspace_low[1] = workspace_low[1] + padding[1]
        workspace_high[0] = workspace_high[0] - padding[0]
        workspace_high[1] = workspace_high[1] - padding[1]

        self.region_low = np.array(workspace_low) + np.array(base_pos)  # the bounding box of the valid regions in the workspace
        self.region_high = np.array(workspace_high) + np.array(base_pos)
        
        self.bbox_lls = bbox_lls
        self.bbox_uls = bbox_uls
        self.transforms = transforms