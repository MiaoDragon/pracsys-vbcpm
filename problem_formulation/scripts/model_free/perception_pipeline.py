"""
Perception system that uses Segmentation, Data Association and SLAM
"""
from slam import SLAMPerception
from data_association import GroundTruthDataAssociation
from segmentation import GroundTruthSegmentation
from target_recognition import GroundTruthTargetRecognition
import numpy as np
import cv2

class PerceptionPipeline():
    def __init__(self, occlusion_params, object_params, perception_params):
        self.slam_system = SLAMPerception(occlusion_params, object_params)
        self.data_assoc = GroundTruthDataAssociation()
        self.segmentation = GroundTruthSegmentation()
        self.target_recognition = GroundTruthTargetRecognition(perception_params['target_pybullet_id'])
        self.last_assoc = {}
        self.depth_img = None
        self.color_img = None
        self.seg_img = None
        self.target_seg_img = None

    def pipeline_sim(self, camera, robot_ids, workspace_ids):
        """
        given the camera input, segment the image, and data association
        """    
        color_img, depth_img, seg_img = camera.sense()

        self.segmentation.set_ground_truth_seg_img(seg_img)
        seg_img = self.segmentation.segment_img(color_img, depth_img)

        self.target_recognition.set_ground_truth_seg_img(seg_img)
        target_seg_img = self.target_recognition.recognize(color_img, depth_img)

        self.depth_img = depth_img
        self.color_img = color_img
        self.seg_img = seg_img
        self.target_seg_img = target_seg_img

        assoc = self.data_assoc.data_association(seg_img, robot_ids, workspace_ids)
        self.last_assoc = assoc

        # objects that have been revealed will stay revealed
        valid_objects = self.obtain_unhidden_objects(robot_ids, workspace_ids)

        object_hide_set = self.obtain_object_hide_set(robot_ids, workspace_ids)
        self.current_hide_set = object_hide_set
            
        self.slam_system.perceive(depth_img, color_img, seg_img, 
                                    assoc, self.data_assoc.obj_ids_reverse, object_hide_set, 
                                    camera.info['extrinsics'], camera.info['intrinsics'], camera.info['far'], 
                                    robot_ids, workspace_ids)

        # update each object's hide set
        for obj_i, obj_hide_list in object_hide_set.items():
            self.slam_system.objects[obj_i].update_obj_hide_set(obj_hide_list)


        for obj_id in valid_objects:
            self.slam_system.objects[obj_id].set_active()
            

    def sense_object(self, obj_id, camera, robot_ids, workspace_ids):
        color_img, depth_img, seg_img = camera.sense()
        self.segmentation.set_ground_truth_seg_img(seg_img)
        seg_img = self.segmentation.segment_img(color_img, depth_img)
        assoc = self.data_assoc.data_association(seg_img, robot_ids, workspace_ids)

        self.slam_system.update_obj_model(obj_id, depth_img, color_img, seg_img, self.data_assoc.obj_ids_reverse, 
                                          camera.info['extrinsics'], camera.info['intrinsics'], camera.info['far'], robot_ids, workspace_ids)
    

    def obtain_object_hide_set(self, robot_ids, workspace_ids):
        depth_img = self.depth_img
        seg_img = self.seg_img
        assoc = self.last_assoc
        # determine hiding relation: the target object shouldn't be hidden and inactive
        # hidden: at least one depth value is larger than a neighboring object depth value

        # determine where there are objects in the segmented img

        # UPDATE: we want to consider robot hiding as well
        obj_seg_filter = np.ones(seg_img.shape).astype(bool)
        for wid in workspace_ids:
            obj_seg_filter[seg_img==wid] = 0
        obj_seg_filter[seg_img==-1] = 0
        for rid in robot_ids:
            obj_seg_filter[seg_img==rid] = 0
        # for seg_id, obj_id in assoc.items():
        #     obj_seg_filter[seg_img==seg_id] = 1

        valid_objects = []  # only return unhidden objects.

        seen_objs = []
        hiding_objs = {}  # obj_id -> objects that are hiding it
        for seg_id, obj_id in assoc.items():
            seen_objs.append(obj_id)
            hiding_set = set()

            seged_depth_img = np.zeros(depth_img.shape)
            seged_depth_img[seg_img==seg_id] = depth_img[seg_img==seg_id]
            # cv2.imshow("seen_obj", seged_depth_img)
            # cv2.waitKey(0)
            # obtain indices of the segmented object
            img_i, img_j = np.indices(seg_img.shape)
            # check if the neighbor object is hiding this object
            valid_1 = (img_i-1>=0) & (seg_img==seg_id)
            # the neighbor object should be 
            # 1. an object (can be robot)
            # 2. not the current object
            filter1 = obj_seg_filter[img_i[valid_1]-1,img_j[valid_1]]
            filter2 = (seg_img[img_i[valid_1]-1,img_j[valid_1]] != seg_id)

            # filter1_img = np.zeros(depth_img.shape)
            # filter1_img[img_i[valid_1]-1, img_j[valid_1]] = (filter1&filter2)
            # cv2.imshow('filter obj', filter1_img)
            # cv2.waitKey(0)


            depth_filtered = depth_img[img_i[valid_1]-1,img_j[valid_1]][filter1&filter2]
            seg_obj_depth_filtered = depth_img[img_i[valid_1],img_j[valid_1]][filter1&filter2]
            seg_obj_filtered = seg_img[img_i[valid_1]-1,img_j[valid_1]][filter1&filter2]

            hiding_seg_obj_filtered = seg_obj_filtered[depth_filtered<seg_obj_depth_filtered]
            hiding_set = hiding_set.union(set(hiding_seg_obj_filtered.tolist()))


            valid_1 = (img_i+1<seg_img.shape[0]) & (seg_img==seg_id)
            filter1 = obj_seg_filter[img_i[valid_1]+1,img_j[valid_1]]
            filter2 = (seg_img[img_i[valid_1]+1,img_j[valid_1]] != seg_id)
            depth_filtered = depth_img[img_i[valid_1]+1,img_j[valid_1]][filter1&filter2]
            seg_obj_depth_filtered = depth_img[img_i[valid_1],img_j[valid_1]][filter1&filter2]
            seg_obj_filtered = seg_img[img_i[valid_1]+1,img_j[valid_1]][filter1&filter2]

            hiding_seg_obj_filtered = seg_obj_filtered[depth_filtered<seg_obj_depth_filtered]
            hiding_set = hiding_set.union(set(hiding_seg_obj_filtered.tolist()))




            valid_1 = (img_j-1>=0) & (seg_img==seg_id)
            filter1 = obj_seg_filter[img_i[valid_1],img_j[valid_1]-1]
            filter2 = (seg_img[img_i[valid_1],img_j[valid_1]-1] != seg_id)
            depth_filtered = depth_img[img_i[valid_1],img_j[valid_1]-1][filter1&filter2]
            seg_obj_depth_filtered = depth_img[img_i[valid_1],img_j[valid_1]][filter1&filter2]
            seg_obj_filtered = seg_img[img_i[valid_1],img_j[valid_1]-1][filter1&filter2]

            hiding_seg_obj_filtered = seg_obj_filtered[depth_filtered<seg_obj_depth_filtered]
            hiding_set = hiding_set.union(set(hiding_seg_obj_filtered.tolist()))




            valid_1 = (img_j+1<seg_img.shape[1]) & (seg_img==seg_id)
            filter1 = obj_seg_filter[img_i[valid_1],img_j[valid_1]+1]
            filter2 = (seg_img[img_i[valid_1],img_j[valid_1]+1] != seg_id)
            depth_filtered = depth_img[img_i[valid_1],img_j[valid_1]+1][filter1&filter2]
            seg_obj_depth_filtered = depth_img[img_i[valid_1],img_j[valid_1]][filter1&filter2]
            seg_obj_filtered = seg_img[img_i[valid_1],img_j[valid_1]+1][filter1&filter2]

            hiding_seg_obj_filtered = seg_obj_filtered[depth_filtered<seg_obj_depth_filtered]
            hiding_set = hiding_set.union(set(hiding_seg_obj_filtered.tolist()))


            # NOTE: hiding_set stores seg_ids, which are pybullet ids instead of obj_id
            # we need to convert them
            hiding_set = list(hiding_set)
            hiding_set = [assoc[sid] for sid in hiding_set]
            hiding_objs[obj_id] = hiding_set
        
        # seen_objs = set(seen_objs)
        # total_obj_ids = set(list(self.slam_system.objects.keys()))
        # unseen_objs = total_obj_ids - seen_objs

        # for obj_id in unseen_objs:
        #     if (obj_id in self.slam_system.objects) and self.slam_system.objects[obj_id].active:
        #         valid_objects.append(obj_id)

        return hiding_objs
            


    def obtain_unhidden_objects(self, robot_ids, workspace_ids):

        depth_img = self.depth_img
        seg_img = self.seg_img
        assoc = self.last_assoc
        # determine hiding relation: the target object shouldn't be hidden and inactive
        # hidden: at least one depth value is larger than a neighboring object depth value

        # determine where there are objects in the segmented img

        # UPDATE: we want to consider robot hiding as well
        obj_seg_filter = np.ones(seg_img.shape).astype(bool)
        for wid in workspace_ids:
            obj_seg_filter[seg_img==wid] = 0
        obj_seg_filter[seg_img==-1] = 0
        # for seg_id, obj_id in assoc.items():
        #     obj_seg_filter[seg_img==seg_id] = 1

        valid_objects = []  # only return unhidden objects.

        seen_objs = []
        for seg_id, obj_id in assoc.items():
            seen_objs.append(obj_id)

            seged_depth_img = np.zeros(depth_img.shape)
            seged_depth_img[seg_img==seg_id] = depth_img[seg_img==seg_id]
            # cv2.imshow("seen_obj", seged_depth_img)
            # cv2.waitKey(0)
            # obtain indices of the segmented object
            img_i, img_j = np.indices(seg_img.shape)
            # check if the neighbor object is hiding this object
            valid_1 = (img_i-1>=0) & (seg_img==seg_id)
            # the neighbor object should be 
            # 1. an object (can be robot)
            # 2. not the current object
            filter1 = obj_seg_filter[img_i[valid_1]-1,img_j[valid_1]]
            filter2 = (seg_img[img_i[valid_1]-1,img_j[valid_1]] != seg_id)

            # filter1_img = np.zeros(depth_img.shape)
            # filter1_img[img_i[valid_1]-1, img_j[valid_1]] = (filter1&filter2)
            # cv2.imshow('filter obj', filter1_img)
            # cv2.waitKey(0)


            depth_filtered = depth_img[img_i[valid_1]-1,img_j[valid_1]][filter1&filter2]
            seg_obj_depth_filtered = depth_img[img_i[valid_1],img_j[valid_1]][filter1&filter2]
            if (depth_filtered < seg_obj_depth_filtered).sum() > 0:
                # this object is hidden
                continue
                # if (not obj_id in self.slam_system.objects):# or (not self.slam_system.objects[obj_id].active):
                #     continue

            valid_1 = (img_i+1<seg_img.shape[0]) & (seg_img==seg_id)
            filter1 = obj_seg_filter[img_i[valid_1]+1,img_j[valid_1]]
            filter2 = (seg_img[img_i[valid_1]+1,img_j[valid_1]] != seg_id)
            depth_filtered = depth_img[img_i[valid_1]+1,img_j[valid_1]][filter1&filter2]
            seg_obj_depth_filtered = depth_img[img_i[valid_1],img_j[valid_1]][filter1&filter2]
            if (depth_filtered < seg_obj_depth_filtered).sum() > 0:
                # this object is hidden
                continue
                # if (not obj_id in self.slam_system.objects):# or (not self.slam_system.objects[obj_id].active):
                #     continue

            valid_1 = (img_j-1>=0) & (seg_img==seg_id)
            filter1 = obj_seg_filter[img_i[valid_1],img_j[valid_1]-1]
            filter2 = (seg_img[img_i[valid_1],img_j[valid_1]-1] != seg_id)
            depth_filtered = depth_img[img_i[valid_1],img_j[valid_1]-1][filter1&filter2]
            seg_obj_depth_filtered = depth_img[img_i[valid_1],img_j[valid_1]][filter1&filter2]
            if (depth_filtered < seg_obj_depth_filtered).sum() > 0:
                # this object is hidden
                continue
                # if (not obj_id in self.slam_system.objects):# or (not self.slam_system.objects[obj_id].active):
                #     continue

            valid_1 = (img_j+1<seg_img.shape[1]) & (seg_img==seg_id)
            filter1 = obj_seg_filter[img_i[valid_1],img_j[valid_1]+1]
            filter2 = (seg_img[img_i[valid_1],img_j[valid_1]+1] != seg_id)
            depth_filtered = depth_img[img_i[valid_1],img_j[valid_1]+1][filter1&filter2]
            seg_obj_depth_filtered = depth_img[img_i[valid_1],img_j[valid_1]][filter1&filter2]
            if (depth_filtered < seg_obj_depth_filtered).sum() > 0:
                # this object is hidden
                continue
                # if (not obj_id in self.slam_system.objects) or (not self.slam_system.objects[obj_id].active):
                #     continue
            
            # print('object %d is valid' % (obj_id))
            valid_objects.append(obj_id)


        # seen_objs = set(seen_objs)
        # total_obj_ids = set(list(self.slam_system.objects.keys()))
        # unseen_objs = total_obj_ids - seen_objs

        # for obj_id in unseen_objs:
        #     if (obj_id in self.slam_system.objects) and self.slam_system.objects[obj_id].active:
        #         valid_objects.append(obj_id)

        return valid_objects
    
    def label_obj_seg_img(self):
        """
        label the object segmented image so that 
        """
        pass