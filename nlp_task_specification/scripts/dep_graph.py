"""
Implement the graph representing relationships between objects or regions
"""
import numpy as np
from matplotlib.pyplot import show

import open3d as o3d
import networkx as nx

from visual_utilities import *


def perturbation(rmin, rmax, amin=0, amax=2 * np.pi):
    rad = np.random.uniform(rmin, rmax)
    ang = np.random.uniform(amin, amax)
    return np.array([rad * np.cos(ang), rad * np.sin(ang)])


class DepGraph():
    def __init__(self, obj_poses, obj_colors, occlusion, occupied_label, occlusion_label):

        self.poses = obj_poses
        self.colors = obj_colors
        self.occupied_label = occupied_label
        self.occlusion_label = occlusion_label

        self.gt_graph = nx.DiGraph()
        self.graph = nx.DiGraph()

        self.upmasks = [None] * len(obj_poses)
        self.bhmasks = [None] * len(obj_poses)
        for i in range(len(obj_poses)):
            if obj_poses[i] is None:
                continue

            obj_i = i + 1
            self.graph.add_node(obj_i, name=obj_i, color=obj_colors[i])
            up_i = occupied_label == obj_i
            bh_i = occupied_label == obj_i
            self.graph.add_node(-obj_i, name=-obj_i, vol=bh_i.sum(), color=obj_colors[i])
            self.graph.add_edge(-obj_i, obj_i, etype="hidden")

            for x in range(occupied_label.shape[0]):
                for y in range(occupied_label.shape[1]):
                    up_i[x, y, up_i[x, y, :].argmax():] = up_i[x, y, :].any()
            self.upmasks[i] = up_i

            for y in range(occupied_label.shape[1]):
                for z in range(occupied_label.shape[2]):
                    bh_i[bh_i[:, y, z].argmax():, y, z] = bh_i[:, y, z].any()
            self.bhmasks[i] = bh_i

        for i in range(len(obj_poses)):
            if obj_poses[i] is None:
                continue

            obj_i = i + 1
            obj_i_vox = occupied_label == obj_i
            obj_i_vol = obj_i_vox.sum()
            for j in range(len(obj_poses)):
                if obj_poses[j] is None:
                    continue
                if i == j:
                    continue

                obj_j = j + 1
                up_j_vox = self.upmasks[j]
                bh_j_vox = self.bhmasks[j]
                occ_j_vox = (occlusion_label == obj_j) | (occupied_label == obj_j)

                # test if above
                if (obj_i_vox & up_j_vox).sum() / obj_i_vol > 0.5:
                    self.graph.add_edge(obj_j, obj_i, etype="below")

                # test if behind
                # if (obj_i_vox & bh_j_vox).sum() / obj_i_vol > 0.5:
                #     self.graph.add_edge(obj_i, obj_j, etype="behind")

                # test if occluded
                # if (obj_i_vox & occ_j_vox).sum() / obj_i_vol > 0.9:
                #     self.graph.add_edge(obj_i, obj_j, etype="hidden")

    def update_target_confidence(self, target, suggestion, estimated_volume):
        to_return = False
        for v, n in list(self.graph.nodes(data="name")):
            if v > 0 and n == target:
                print("Target Visible!")
                to_return = True
            if v < 0 and n == target:
                self.graph.remove_node(v)
        if to_return: return

        for v, n in list(self.graph.nodes(data="name")):
            if v < 0:
                new_id = min(self.graph.nodes) - 1
                self.graph.add_node(new_id, name=target, color=[1, 0, 0])
                weight = 1
                if n == suggestion:
                    weight += 1
                print(estimated_volume, (self.occlusion_label == np.abs(v)).sum())
                if estimated_volume < (self.occlusion_label == np.abs(v)).sum():
                    weight += 1
                self.graph.add_edge(new_id, v, etype="in", w=weight)

    def draw_graph(self, label="name"):
        # print(self.poses, self.graph.nodes)
        pos = {
            # v: self.poses[np.abs(v) - 1][:2, 3] + perturbation(0.005, 0.006)
            v: self.poses[v - 1][:2, 3] if 0 < v - 1 < len(self.poses) else [0, 0]
            for v in self.graph.nodes
        }
        pos = nx.spring_layout(
            self.graph,
            k=1 / len(self.graph.nodes),
            pos=pos,
            fixed=[v for v, d in self.graph.out_degree if v > 0 and d == 0],
            iterations=100
        )
        nx.draw(
            self.graph,
            pos,
            node_color=list(dict(self.graph.nodes(data="color")).values())
        )
        nx.draw_networkx_labels(self.graph, pos, dict(self.graph.nodes(data=label)))
        nx.draw_networkx_edge_labels(
            self.graph, pos, {(i, j): k
                              for i, j, k in self.graph.edges(data="etype")}
        )
        nx.draw_networkx_edge_labels(
            self.graph, pos,
            {(i, j): "" if k is None else k
             for i, j, k in self.graph.edges(data="w")}
        )
        show()
