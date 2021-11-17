"""
Implement the graph representing the "sharing" relationship of occlusions bewteen different objects
"""
import numpy as np
class OcclusionShareGraph():
    def __init__(self, obj_poses, obj_occlusions):
        # given a scene information, generate an occlusion-share graph
        # an edge is between two nodes if the two nodes share the same occlusion region
        self.nodes = np.arange(0,len(obj_poses),1).astype(int)
        self.connected = np.zeros((len(obj_poses), len(obj_poses))).astype(bool)
        # won't be too many nodes, so we can use adjacent matrix for the edge

        for i in range(len(obj_poses)):
            if obj_poses[i] is None:
                continue
            for j in range(i+1,len(obj_poses)):
                if obj_poses[j] is None:
                    continue
                if (obj_occlusions[i] & obj_occlusions[j]).sum() > 0:
                    self.connected[i,j] = 1
                    self.connected[j,i] = 1
        