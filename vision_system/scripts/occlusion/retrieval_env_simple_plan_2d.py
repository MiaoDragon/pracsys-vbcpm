"""
planner for the 2d retrieval task where we only allow
straight paths
"""
import copy
from collections import deque
import heapq

from retrieval_env_simple_2d import RetrievalPickPlace2D, RetrievalPickPlace2DVisual
import pygame
from pygame import surface
from pygame.locals import (QUIT, KEYDOWN, K_ESCAPE)

import numpy as np
class ConstraintGraphNode():
    def __init__(self, data):
        self.data = data
    def set_id(self, id):
        self.id = id  # for use in the graph
class ConstraintGraph():
    def __init__(self):
        self.nodes = {}  # each node we give them an id
        self.edges = {}  # we use adjacent list for edge implementation
        self.idle_id = 0
    def add_node(self, new_node):
        id = self.idle_id
        new_node = copy.deepcopy(new_node)
        new_node.set_id(id)  # add an ID to the node
        self.nodes[id] = new_node
        self.edges[id] = set()
        self.idle_id = self.idle_id + 1
        return new_node  
    def add_edge(self, node1, node2):
        self.edges[node1.id].add(node2.id)
        self.edges[node2.id].add(node1.id)
    def delete_node(self, node):
        # remove the edge
        edges = list(self.edges[node.id])
        for i in range(edges):
            node_id = edges[i]
            self.edges[node_id].remove(node.id)
        self.edges.pop(node.id, None)
        self.nodes.pop(node.id, None)
    def get_subgraph(self, nodes):
        # get a subgraph formed by nodes
        new_graph = ConstraintGraph()
        new_nodes = {}
        for i in range(len(nodes)):
            new_node = new_graph.add_node(nodes[i])
            new_nodes[nodes[i].id] = new_node
        for i in range(len(nodes)):
            edges = list(self.edges[nodes[i].id])
            for j in range(len(edges)):
                node1 = new_nodes[nodes[i].id]
                node2 = new_nodes[edges[j]]  # hash to the new node using previous node id
                new_graph.add_edge(node1, node2)
        return new_graph
    def bfs(self, node):
        # do a bfs on the entire graph starting from node
        # return a list of ids for which nodes are visited
        q = deque()
        visited_q = set()
        q.append(node.id)
        while len(q) > 0:
            node_id = q.popleft()
            visited_q.add(node_id)
            edges = list(self.edges[node_id])
            for new_node_id in edges:
                if new_node_id in visited_q:
                    continue
                q.append(new_node_id)
        return visited_q
    def get_connected_components(self):
        # return all connected components as subgraphs
        # initialize the cc number
        cc_list = []
        cc_dict = {}
        for id, node in self.nodes.items():
            cc_dict[id] = -1
        cc_num = 0
        for id, node in self.nodes.items():
            if cc_dict[id] == -1:
                # not yet explored
                visited_ids = self.bfs(node)
                visited_nodes = []
                for visited_id in visited_ids:
                    cc_dict[visited_id] = cc_num
                    visited_nodes.append(self.nodes[visited_id])
                cc_list.append(visited_nodes)
                cc_num += 1
        # obtain the subgraph
        subgraphs = []
        for i in range(len(cc_list)):
            graph_i = self.get_subgraph(cc_list[i])
            subgraphs.append(graph_i)
        return subgraphs

    
def a_star_arrangement(scene, graph):
    """
    compute an arrangement for the given constraint graph
    assume the given graph is already a connected component
    """
    queue = []
    obj_ids = []
    for node_id, node in graph.nodes.items():
        obj_ids.append(node.data['obj_id'])
    obj_ids = set(obj_ids)
    for node_id, node in graph.nodes.items():
        if scene.accessibility_check_all(node.data['obj_id'], obj_ids):
            arrangement = [node.data['obj_id']]
            cost_to_come, total_time = scene.cost_to_come(arrangement, scene.visible_obj_ids)
            cost_to_go = scene.cost_to_go(arrangement, scene.visible_obj_ids)
            remaining_obj_ids = set(obj_ids)
            remaining_obj_ids.remove(node.data['obj_id'])
            heapq.heappush(queue, \
                (cost_to_come+cost_to_go, cost_to_come, cost_to_go, total_time, node.data['obj_id'], arrangement, remaining_obj_ids))
    while len(queue)>0:
        total_cost, cost_to_come, cost_to_go, total_time, obj_id, arrangement, remaining_obj_ids = heapq.heappop(queue)
        print('remaining_obj_ids: ', remaining_obj_ids)
        print('arrangement: ', arrangement)
        if len(remaining_obj_ids) == 0:
            return arrangement
        remaining_obj_ids = list(remaining_obj_ids)
        for new_obj_id in remaining_obj_ids:
            if scene.accessibility_check_all(new_obj_id, remaining_obj_ids):
                new_arrangement = arrangement + [new_obj_id]
                new_remaining_obj_ids = set(remaining_obj_ids)
                new_remaining_obj_ids.remove(new_obj_id)
                new_cost_to_come, new_total_time = scene.cost_to_come_acc(new_obj_id, remaining_obj_ids, cost_to_come, total_time)
                new_cost_to_go = scene.cost_to_go_acc(new_obj_id, remaining_obj_ids, total_time)
                new_total_cost = new_cost_to_come + new_cost_to_go
                heapq.heappush(queue, \
                    (new_cost_to_come+new_cost_to_go, new_cost_to_come, new_cost_to_go, \
                        new_total_time, new_obj_id, new_arrangement, new_remaining_obj_ids))
        # until we find the first path


def arrangement_plan(scene, graph):
    # break graph into subgraphs
    subgraphs = graph.get_connected_components()
    print('number of subgraphs: ', len(subgraphs))
    arrangements = []
    for subgraph in subgraphs:
        arrangements.append(a_star_arrangement(scene, subgraph))        
    print('arrangements: ')
    print(arrangements)
    total_arrangement = []
    visible_obj_ids = scene.visible_obj_ids
    print('visible obj: ')
    print(scene.visible_obj_ids)
    while len(total_arrangement) < len(scene.visible_obj_ids):
        # add arrangement into the bag
        bag = []
        for i in range(len(subgraphs)):
            for j in range(1,len(arrangements[i])+1):
                heapq.heappush(bag, (-scene.collective_utility(arrangements[i][:j], visible_obj_ids), arrangements[i][:j], i, j))
                # print('checking collective utility: ', scene.occlusion_obj_masks[arrangements[i][:j]].astype(float).sum())
        print('bag: ')
        print(bag)
        utility, arrangement, i, j = heapq.heappop(bag)
        utility = -utility
        total_arrangement += arrangement
        arrangements[i] = arrangements[i][j:]
    return total_arrangement




if __name__ == '__main__':
    # testing graph
    graph = ConstraintGraph()
    nodes = []
    for i in range(7):
        node = ConstraintGraphNode(i)
        node = graph.add_node(node)
        nodes.append(node)
    graph.add_edge(nodes[0], nodes[1])
    graph.add_edge(nodes[1], nodes[2])
    graph.add_edge(nodes[0], nodes[2])

    graph.add_edge(nodes[3], nodes[5])
    graph.add_edge(nodes[4], nodes[5])
    graph.add_edge(nodes[6], nodes[5])

    print(graph.bfs(nodes[0]))

    subgraphs = graph.get_connected_components()
    print('there are %d subgraphs' % (len(subgraphs)))
    for graph_i in range(len(subgraphs)):
        graph = subgraphs[graph_i]
        print('graph %d...' % (graph_i))
        for id, node in graph.nodes.items():
            print('node data: %d' % (node.data))



    # testing planning code
    world = RetrievalPickPlace2D()
    world.setup()
    occlusion, cone_mask, _ = world.get_occlusion(world.objects)
    print('occlusion: ')
    print(occlusion)
    sweep = world.obtain_sweep(world.objects[0])

    vis = RetrievalPickPlace2DVisual()
    world.sense()
    # world.move_times = np.array([1., 1., 1.0])
    cost = world.cost_to_come([1,0], [0,1])
    print('world cost: ', cost)
    cost_to_go = world.cost_to_go([1], [0,1])
    print('cost to go: ', cost_to_go)

    # construct a graph of nodes
    graph = ConstraintGraph()

    nodes = []
    for i in range(len(world.visible_obj_ids)):
        node = ConstraintGraphNode({'obj_id': world.visible_obj_ids[i]})
        node = graph.add_node(node)
        nodes.append(node)
    
    # add edges for visibility and accessibility constraint
    for i in range(len(world.visible_obj_ids)):
        for j in range(i+1, len(world.visible_obj_ids)):
            edged = False
            if not world.accessibility_check(world.visible_obj_ids[i], world.visible_obj_ids[j]):
                edged = True
            if (world.occlusion_obj_masks[world.visible_obj_ids[i]] & world.occlusion_obj_masks[world.visible_obj_ids[j]]).sum() > 0:
                edged = True
            if edged:
                graph.add_edge(nodes[i], nodes[j])
    
    # planning for the sequence
    total_arrangement = arrangement_plan(world, graph)
    print('arrangement: ')
    print(total_arrangement)

    # compute the cost for other arrangemnet
    arrangement1 = [1,0,2,3]
    arrangement2 = [1,2,0,3]
    from itertools import permutations

    perms = set(permutations(world.visible_obj_ids))
    perms = list(perms)
    for i in range(len(perms)):
        print('permutation: ', perms[i])
        print('cost %d: %f' % (i, world.cost_to_come(list(perms[i]), world.visible_obj_ids)[0]))


    # --- main game loop ---
    running = True
    while running:
        # Check the event queue
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                # The user closed the window or pressed escape
                running = False
        vis.display_setup(world)
        vis.display_occlusion(world, world.objects, world.occlusion)
        # res, _  = world.check_object_occlusion(world.objects, world.occlusion, cone_mask)
        vis.display_update()

    pygame.quit()