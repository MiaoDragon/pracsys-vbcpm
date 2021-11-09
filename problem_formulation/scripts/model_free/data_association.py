"""
associate the segmented image with perceived model.
If not found in the model list, then create a new model
{seg_id: obj_id}
if new object found, add one to the obj_id
"""
class GroundTruthDataAssociation():
    def __init__(self):
        self.objects = []  # record recognized objects
        self.obj_ids = {}  # pybullet id to object id
        self.num_objs = 0
    def set_recognized_objects(self, objects):
        self.objects = objects
    def data_association(self, seg_img, robot_ids, workspace_ids):
        # ground truth reorders the pybullet object id
        # so that they're seen from 0
        seg_ids = list(set(seg_img.reshape(-1).tolist()))
        assoc = {}
        for i in range(len(seg_ids)):
            if seg_ids[i] == -1:
                continue
            if seg_ids[i] in robot_ids:
                continue
            if seg_ids[i] in workspace_ids:
                continue
            print('seg id: ', seg_ids[i])
            if seg_ids[i] in self.obj_ids.keys():
                assoc[seg_ids[i]] = self.obj_ids[seg_ids[i]]
            else:
                # create a new entry
                self.obj_ids[seg_ids[i]] = self.num_objs
                self.num_objs += 1
                assoc[seg_ids[i]] = self.obj_ids[seg_ids[i]]
        return assoc



