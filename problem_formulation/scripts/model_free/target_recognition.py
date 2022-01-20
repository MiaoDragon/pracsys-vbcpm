"""
ground truth target recognition from image.
Segment the target objects afterwards.
"""
class GroundTruthTargetRecognition():
    def __init__(self, target_seg_id):
        self.seg_img = None
        self.target_seg_id = target_seg_id
    def set_ground_truth_seg_img(self, seg_img):
        self.seg_img = seg_img
    def recognize(self, rgb_img, depth_img):
        return self.seg_img == self.target_seg_id

    