"""
This script defines the workspace object that remains static, and should be avoided
during planning for collision.
"""
import pybullet as p
import numpy as np
class Workspace():
    def __init__(self, pos, ori, components, workspace_low, workspace_high, pybullet_id):
        # components is a list of geometric parts
        self.components = {}
        self.component_ids = []
        for component_name, component in components.items():
            print('component name: ')
            print(component_name)
            shape = component['shape']
            shape = np.array(shape)
            pos = component['pose']['pos']
            ori = component['pose']['ori']

            col_id = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=shape/2, physicsClientId=pybullet_id)
            vis_id = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=shape/2, physicsClientId=pybullet_id)
            comp_id = p.createMultiBody(baseCollisionShapeIndex=col_id, baseVisualShapeIndex=vis_id,
                                        basePosition=pos, baseOrientation=ori, physicsClientId=pybullet_id)
            self.components[component_name] = comp_id
            self.component_ids.append(comp_id)
        self.pos = pos
        self.ori = ori
        self.region_low = workspace_low  # the bounding box of the valid regions in the workspace
        self.region_high = workspace_high
    