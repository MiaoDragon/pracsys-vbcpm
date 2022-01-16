def move_and_sense(obj):
    """
    move the valid object out of the workspace, sense it and the environment, and place back
    """
    pre_move(obj)

    suction_poses_in_obj = generate_poses(obj)  # suction pose in object frame
    suction_poses_in_obj = filter_poses(suction_poses_in_obj, obj)  # filter the suction poses by IK and collision
    intermediate_poses, suction_poses_in_obj = generate_intermediate_pose(obj, suction_poses_in_obj)
    # generate intermediate pose for the obj with valid suction pose
    for i in range(len(intermediate_poses)):
        intermediate_pose = intermediate_poses[i]
        suction_pose_in_obj = suction_poses_in_obj[i]
        pick_plan = plan_to_suction_pose(obj, suction_pose_in_obj)  # internally, plan_to_pre_pose, pre_to_suction, lift up
        lift_up_plan = straight_line_plan(obj, suction_pose_in_obj, pick_plan[-1])
        if len(pick_plan) == 0:
            continue
        retreat_plan = plan_to_intermediate_pose(obj, suction_pose_in_obj, pick_plan[-1])
        if len(retreat_plan) == 0:
            continue
        execute(pick_plan)
        execute_with_obj(lift_up_plan)
        execute_with_obj(retreat_plan)

        sense()  # sense the environment
        
        for k in range(10):
            plan = obj_sense_plan()
            execute_with_obj(plan)
            sense_object()
            sense()

        placement_pose = generate_placement_pose(obj, suction_pose_in_obj)  # if no valid pose, use previous pose
        place_plan = plan_to_placement_pose(obj, suction_pose_in_obj, placement_pose)
        execute_with_obj(place_plan)

        reset_robot()
        return True
    return False


def pre_move(obj):
    """
    before moving the object, check reachability constraints. Rearrange the blocking objects
    """
    pass

def rearrange(objs, moveable_objs):
    """
    rearrange the blocking objects
    """
    pass