"""
Perception system that uses Segmentation, Data Association and SLAM
"""
from slam import SLAMPerception
from data_association import GroundTruthDataAssociation
from segmentation import GroundTruthSegmentation


class PerceptionPipeline():
    def __init__(self, occlusion_params, object_params):
        self.slam_system = SLAMPerception(occlusion_params, object_params)
        self.data_assoc = GroundTruthDataAssociation()
        self.segmentation = GroundTruthSegmentation()
        self.last_assoc = {}
    def pipeline_sim(self, camera, robot_ids, workspace_ids):
        """
        given the camera input, segment the image, and data association
        """    
        color_img, depth_img, seg_img = camera.sense()
        self.segmentation.set_ground_truth_seg_img(seg_img)
        seg_img = self.segmentation.segment_img(color_img, depth_img)
        assoc = self.data_assoc.data_association(seg_img, robot_ids, workspace_ids)
        self.last_assoc = assoc
        self.slam_system.perceive(depth_img, color_img, seg_img, assoc, camera.info['extrinsics'], camera.info['intrinsics'])
    

