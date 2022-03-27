"""
Implement the graph representing relationships between objects or regions
"""
import copy
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

    def __init__(
        self,
        obj_poses,
        obj_colors,
        obj_names,
        occlusion,
        occupied_label,
        occlusion_label,
    ):

        self.poses = obj_poses
        self.colors = obj_colors
        self.occupied_label = occupied_label
        self.occlusion_label = occlusion_label

        self.gt_graph = nx.DiGraph()
        self.graph = nx.DiGraph()

        self.upmasks = [None] * len(obj_poses)
        self.bhmasks = [None] * len(obj_poses)
        for i in range(len(obj_poses)):
            obj_i = i + 1
            oname = obj_names[i]
            self.gt_graph.add_node(obj_i, dname=oname, color=obj_colors[i])
            if obj_poses[i] is not None:
                self.graph.add_node(obj_i, dname=oname, color=obj_colors[i])
            up_i = occupied_label == obj_i
            bh_i = occupied_label == obj_i
            # self.graph.add_node(-obj_i, dname=-obj_i, vol=bh_i.sum(), color=obj_colors[i])
            # self.graph.add_edge(-obj_i, obj_i, etype="hidden")

            for x in range(occupied_label.shape[0]):
                for y in range(occupied_label.shape[1]):
                    up_i[x, y, up_i[x, y, :].argmax():] = up_i[x, y, :].any()
            self.upmasks[i] = up_i

            for y in range(occupied_label.shape[1]):
                for z in range(occupied_label.shape[2]):
                    bh_i[bh_i[:, y, z].argmax():, y, z] = bh_i[:, y, z].any()
            self.bhmasks[i] = bh_i

        for i in range(len(obj_poses)):
            obj_i = i + 1
            obj_i_vox = occupied_label == obj_i
            obj_i_vol = obj_i_vox.sum()
            for j in range(len(obj_poses)):
                if i == j:
                    continue

                obj_j = j + 1
                up_j_vox = self.upmasks[j]
                bh_j_vox = self.bhmasks[j]
                occ_j_vox = (occlusion_label == obj_j) | (occupied_label == obj_j)

                # test if above
                if (obj_i_vox & up_j_vox).sum() / obj_i_vol > 0.5:
                    if obj_poses[i] is not None and obj_poses[j] is not None:
                        self.graph.add_edge(obj_j, obj_i, etype="below")
                    self.gt_graph.add_edge(obj_j, obj_i, etype="below")

                # test if behind
                # if (obj_i_vox & bh_j_vox).sum() / obj_i_vol > 0.5:
                #     self.graph.add_edge(obj_i, obj_j, etype="behind")

                # test if occluded
                if (obj_i_vox & occ_j_vox).sum() / obj_i_vol > 0.4:
                    self.gt_graph.add_edge(obj_i, obj_j, etype="hidden_by")

    def update_target_confidence(self, target, suggestion, estimated_volume):
        to_return = False
        for v, n in list(self.graph.nodes(data="dname")):
            if v <= len(self.poses) and n == target:
                print("Target Visible!")
                to_return = v
            if v > len(self.poses) and n == target:
                self.graph.remove_node(v)
        if to_return:
            return to_return

        for v, n in list(self.graph.nodes(data="dname")):
            new_id = max(self.graph.nodes) + 1
            weight = 1
            if n == suggestion:
                weight += 1
            # print(estimated_volume, (self.occlusion_label == v).sum())
            if estimated_volume < (self.occlusion_label == v).sum():
                weight += 1

            # ignore weights and assume suggestion is perfect but only if it fits
            if n == suggestion and estimated_volume < (self.occlusion_label == v).sum():
                self.graph.add_node(new_id, dname=target, color=[1.0, 0.0, 0.0])
                self.graph.add_edge(new_id, v, etype="hidden_by", w=1)
                return new_id

            # uncomment to include weight
            # self.graph.add_node(new_id, dname=target, color=[1, 0, 0])
            # self.graph.add_edge(new_id, v, etype="in", w=weight)

        return False

    def pick_order(self, pick_node):
        order = nx.dfs_postorder_nodes(self.graph, source=pick_node)
        ind2name = dict(self.graph.nodes(data="dname"))
        return [ind2name[v] for v in order]

    def draw_graph(self, ground_truth=False, label="dname"):
        if ground_truth:
            graph = self.gt_graph
        else:
            graph = self.graph
        # print(self.poses, graph.nodes)
        # pos = {
        #     # v: self.poses[np.abs(v) - 1][:2, 3] + perturbation(0.005, 0.006)
        #     v: self.poses[v - 1][:2, 3] if 0 < v - 1 < len(self.poses) else perturbation(0.005,0.01)
        #     for v in graph.nodes
        # }
        # pos = nx.planar_layout(
        pos = nx.nx_pydot.graphviz_layout(
            graph,
            # 'fdp',
            # k=1 / len(graph.nodes),
            # pos=pos,
            # fixed=[v for v, d in graph.out_degree if v > 0 and d == 0],
            # iterations=100
        )
        colors = [
            color if color is not None else [1.0, 1.0, 1.0, 1.0]
            for color in dict(graph.nodes(data="color")).values()
        ]
        # print(colors)
        nx.draw(graph, pos, node_color=colors)
        nx.draw_networkx_labels(graph, pos, dict(graph.nodes(data=label)))
        nx.draw_networkx_edge_labels(
            graph,
            pos,
            {(i, j): k
             for i, j, k in graph.edges(data="etype")},
        )
        nx.draw_networkx_edge_labels(
            graph,
            pos,
            {(i, j): "" if k is None else k
             for i, j, k in graph.edges(data="w")},
        )
        show()
