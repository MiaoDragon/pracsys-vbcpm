"""
associate the segmented image with perceived model.
If not found in the model list, then create a new model
{seg_id: obj_id}
if new object found, add one to the obj_id
"""
import numpy as np
class GroundTruthDataAssociation():
    def __init__(self):
        self.objects = []  # record recognized objects
        self.obj_ids = {}  # pybullet id to object id
        self.obj_ids_reverse = {}
        self.num_objs = 0
        self.current_id = -1
    def set_recognized_objects(self, objects):
        self.objects = objects
    def data_association(self, seg_img, robot_ids, workspace_ids):
        # ground truth reorders the pybullet object id
        # so that they're seen from 0
        seg_ids = list(set(seg_img.reshape(-1).tolist()))
        assoc = {}
        sensed_obj_ids = []
        new_seg_img = np.array(seg_img).astype(int)
        if self.current_id == -1:
            max_id = max(robot_ids + workspace_ids)
            max_id += 1
            self.current_id = max_id
        for i in range(len(seg_ids)):
            if seg_ids[i] == -1:
                continue
            if seg_ids[i] in robot_ids:
                continue
            if seg_ids[i] in workspace_ids:
                continue
            # print('seg id: ', seg_ids[i])
            if seg_ids[i] in self.obj_ids.keys():
                assoc[seg_ids[i]] = self.obj_ids[seg_ids[i]]

            else:
                # create a new entry
                self.obj_ids[seg_ids[i]] = self.current_id
                self.current_id += 1
                assoc[seg_ids[i]] = self.obj_ids[seg_ids[i]]
                self.obj_ids_reverse[self.obj_ids[seg_ids[i]]] = seg_ids[i]
            new_seg_img[seg_img==seg_ids[i]] = self.obj_ids[seg_ids[i]]
            sensed_obj_ids.append(self.obj_ids[seg_ids[i]])
        return assoc, new_seg_img, sensed_obj_ids



