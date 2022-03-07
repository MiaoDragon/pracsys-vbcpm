"""
obtain segmentation of the scene

TODO: the segmentation may be split
"""
class GroundTruthSegmentation():
    def __init__(self):
        self.seg_img = None
    def set_ground_truth_seg_img(self, seg_img):
        self.seg_img = seg_img
    def segment_img(self, rgb_img, depth_img):
        return self.seg_img

    