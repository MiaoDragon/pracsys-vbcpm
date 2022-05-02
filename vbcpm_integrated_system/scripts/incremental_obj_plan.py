"""
using incremental A* to plan under partial observation of the scene,
so we can reduce the number of unnecessary rearrangement actions.
action to plan:
- object to pick
- placement to sample
like D* and past work in incremental A*, we use optimisitc model
(ref: https://arxiv.org/pdf/1804.05804.pdf,
perception model U is optimistic model)
and assume that the revealed or visible objects are the only objects
in the scene. Hence there may be hidden objects but we don't consider them
and only leave them during replanning.
For A*, we use heuristics as the minimum number of actions required to
reconstruct all objects visible, which can be formulated as:
   # constraining objects + # revealed objects + # visible but not revealed objects
For constructing the tree node, we use a pick node to pick objects for extraction,
and a placement node which samples poses for placing the object. Each tree node
has a state representing the current scene, and g and h value to be prioritized.
"""
import pipeline_utils

class Node():
    """
    base class representing the tree node in the A* tree.
    *state:
    - object configuration: object pose
    - reconstructed objects
    - revealed objects
    - visbile but not revealed objects
    *=> implied state:
    constraints:
    - visibility constraint
    - blocking constraint (reachability)

    cost-to-come
    heuristics (cost-to-go)
    """
    def __init__(self, objs, reconstructed_list, revealed_list, visible_list,
                 modify_list, ignore_list, cost,
                 robot, workspace, occlusion, occlusion_label,
                 v_constrs=None, r_constrs=None):
        """
        given the state, construct the tree node
        can reuse already constructed constraints (for example, after picking an
        object, we might retain the constraints for other objects, and only edit
        the constraints which involve the picked object)
        if constraints are None, then we need to construct them raw from the state
        """
        # when we don't do replanning, we don't need to store the object models
        # and transforms. Just need to store the corresponding list and constraints
        self.reconstructed_list = reconstructed_list
        self.revealed_lsit = revealed_list
        self.visible_list = visible_list
        self.ignore_list = ignore_list
        if v_constrs is None:
            v_constr, r_constr = self.generate_constraints(objs, ignore_list,
                                                            robot, workspace, 
                                                            occlusion, occlusion_label)
            self.v_constr = v_constr
            self.r_constr = r_constr
        else:
            modify_objs = [objs[obj_id] for obj_id in objs if obj_id in modify_list]
            v_constr, r_constr = self.correct_constraints(modify_objs, modify_list, 
                                                        ignore_list, v_constrs, r_constrs)
            self.v_constr = v_constr
            self.r_constr = r_constr
        h = self.compute_heuristics()
        self.h = h
        self.g = cost
    def generate_constraints(self, objs, ignore_list, robot, workspace, occlusion, occlusion_label):
        """
        visibility constraints:
        obj1 -> [objs]
        if objs are in the visibility cone that occludes obj1
        reachability constraints:
        obj1 -> {joint_angle -> [objs]}
        for each object, sample several joint angles of the robot, and each joint angle
        compute the objects that block the target object obj1
        """

        pass
    def correct_constraints(self, modify_objs, modify_list, ignore_list, v_constrs, r_constrs):
        """
        correct the given constraints by 
        - modifying objects in the modify_objs
        - removing objects in the ignore_list
        """
        pass
    def compute_heuristics(self):
        """
        compute heuristics based on current state
        """
        pass