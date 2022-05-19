"""
generate the video
"""

import cv2
import numpy as np
import glob
import os
from tqdm import trange
img_array = []
prob = 'ycb-prob-5-1-1'
root = 'demo-videos/%s/' % (prob)
folder = 'demo-videos/%s/images/' % (prob)
dir_list = os.listdir(folder)
indices = []
for i in range(len(dir_list)):
    fname = dir_list[i].split('.png')[0]
    traj_id, pt_id = fname.split('-')
    traj_id = int(traj_id)
    pt_id = int(pt_id)
    indices.append((traj_id, pt_id))
indices = np.array(indices)
sorted_indices = np.lexsort(np.rot90(indices))


img = cv2.imread(os.path.join(folder,dir_list[0]))
height, width, layers = img.shape
size = (width,height)

out = cv2.VideoWriter(root+'video.avi',cv2.VideoWriter_fourcc(*'DIVX'), 30, size)
for i in trange(len(sorted_indices)):
    idx = sorted_indices[i]
    img = cv2.imread(os.path.join(folder,dir_list[idx]))
    out.write(img)

 
# for i in trange(len(img_array)):
out.release()