"""
Implement the graph representing relationships between objects or regions
"""
import numpy as np
from matplotlib.pyplot import show

import open3d as o3d
import networkx as nx

from visual_utilities import *


class DepGraph():
    def __init__(self, obj_poses, obj_colors, occlusion, occupied_label, occlusion_label):

        self.colors = obj_colors
        self.poses = obj_poses
        self.graph = nx.DiGraph()

        self.upmasks = []
        self.bhmasks = []
        for i in range(len(obj_poses)):
            obj_i = i + 1
            self.graph.add_node(obj_i, name=obj_i, color=obj_colors[i])
            up_i = occupied_label == obj_i
            bh_i = occupied_label == obj_i

            for x in range(occupied_label.shape[0]):
                for y in range(occupied_label.shape[1]):
                    up_i[x, y, up_i[x, y, :].argmax():] = up_i[x, y, :].any()
            self.upmasks.append(up_i)

            for y in range(occupied_label.shape[1]):
                for z in range(occupied_label.shape[2]):
                    bh_i[bh_i[:, y, z].argmax():, y, z] = bh_i[:, y, z].any()
            self.bhmasks.append(bh_i)

        for i in range(len(obj_poses)):
            obj_i = i + 1
            obj_i_vox = occupied_label == obj_i
            obj_i_area = obj_i_vox.sum()
            for j in range(len(obj_poses)):
                if i == j:
                    continue
                obj_j = j + 1
                up_j_vox = self.upmasks[j]
                bh_j_vox = self.bhmasks[j]

                # test if above
                if (obj_i_vox & up_j_vox).sum() / obj_i_area > 0.5:
                    self.graph.add_edge(obj_i, obj_j, etype="above")

                # test if behind
                if (obj_i_vox & bh_j_vox).sum() / obj_i_area > 0.5:
                    self.graph.add_edge(obj_i, obj_j, etype="behind")

    def draw_graph(self, label="name"):
        # pos = nx.spring_layout(self.graph)
        # print(self.poses)
        pos = {k + 1: p[:2, 3] + 0.1 * np.random.rand(2) for k, p in enumerate(self.poses)}
        nx.draw(self.graph, pos, node_color=self.colors)
        nx.draw_networkx_labels(self.graph, pos, dict(self.graph.nodes(data=label)))
        nx.draw_networkx_edge_labels(self.graph, pos, {(i, j): k for i, j, k in self.graph.edges(data="etype")})
        show()
