"""
rearrangement functions to be used in the pipeline to move objects
so that they remain valid.
"""
from cv2 import transform
import numpy as np
import matplotlib.pyplot as plt
import heapq
import transformations as tf


def projection_rot_matrix(obj_poses):
    obj_2d_poses = []
    for i in range(len(obj_poses)):
        obj_pose = obj_poses[i]
        # projection from 3D rotation to 2D
        vec = np.array([1.0, 0, 0])
        rot_vec = obj_pose[:3,:3].dot(vec)
        rot_vec[2] = 0
        rot_vec = rot_vec / np.linalg.norm(rot_vec)

        print('obj pose: ')
        print(obj_pose)
        print('rot_vec: ')
        print(rot_vec)

        angle = np.arctan2(rot_vec[1], rot_vec[0])
        obj_2d_pose = np.zeros((3,3))
        obj_2d_pose[2,2] = 1
        obj_2d_pose[0,0] = np.cos(angle)
        obj_2d_pose[0,1] = -np.sin(angle)
        obj_2d_pose[1,0] = np.sin(angle)
        obj_2d_pose[1,1] = np.cos(angle)
        obj_2d_pose[:2,2] = obj_pose[:2,3]
        obj_2d_poses.append(obj_2d_pose)
    return obj_2d_poses

def projection_2d_state(obj_poses):
    obj_2d_states = []
    for i in range(len(obj_poses)):
        obj_pose = obj_poses[i]
        # projection from 3D rotation to 2D
        vec = np.array([1.0, 0, 0])
        rot_vec = obj_pose[:3,:3].dot(vec)
        rot_vec[2] = 0
        rot_vec = rot_vec / np.linalg.norm(rot_vec)
        angle = np.arctan2(rot_vec[1], rot_vec[0])
        obj_2d_states.append([obj_pose[0,3], obj_pose[1,3], angle])
    return obj_2d_states


def states_from_2d_pose(obj_poses):
    # TODO: debug

    obj_2d_states = []
    for i in range(len(obj_poses)):
        obj_pose = obj_poses[i]
        x = obj_pose[0,2]
        y = obj_pose[1,2]
        theta = np.arctan2(obj_pose[1,0], obj_pose[0,0])
        obj_2d_states.append([x,y,theta])
    return obj_2d_states

def poses_from_2d_state(obj_states):
    # TODO: debug

    obj_2d_poses = []
    for i in range(len(obj_states)):
        obj_2d_poses.append(pose_from_2d_state(obj_states[i]))
    return obj_2d_poses

def state_from_2d_pose(obj_pose):
    x = obj_pose[0,2]
    y = obj_pose[1,2]
    theta = np.arctan2(obj_pose[1,0], obj_pose[0,0])
    return [x,y,theta]

def pose_from_2d_state(obj_state):
    obj_2d_pose = np.zeros((3,3))
    obj_2d_pose[2,2] = 1
    obj_2d_pose[0,0] = np.cos(obj_state[2])
    obj_2d_pose[0,1] = -np.sin(obj_state[2])
    obj_2d_pose[1,0] = np.sin(obj_state[2])
    obj_2d_pose[1,1] = np.cos(obj_state[2])
    obj_2d_pose[:2,2] = obj_state[:2]
    return obj_2d_pose

def pose_from_2d_pose(obj_pose_2d, z):
    # from 2D pose to 3D pose. given the original z value 
    # NOTE: z is in voxel transform
    obj_pose = np.zeros((4,4))
    obj_pose[3,3] = 1
    theta = np.arctan2(obj_pose_2d[1,0], obj_pose_2d[0,0])

    R = tf.rotation_matrix(theta, [0,0,1])
    obj_pose[:3,:3] = R[:3,:3]
    obj_pose[:2,3] = obj_pose_2d[:2,2]
    obj_pose[2,3] = z
    return obj_pose


def obtain_pose_in_voxel(obj_poses, voxel_transform):
    # TODO: debug

    # trasnform the 3D object pose in world to 3D pose in voxel
    transformed_obj_poses = []
    for i in range(len(obj_poses)):
        transformed_obj_pose = obj_poses[i].dot(np.linalg.inv(voxel_transform))
        transformed_obj_poses.append(transformed_obj_pose)
    return transformed_obj_poses

def pose_to_pose_in_world(obj_poses, voxel_transform):
    # TODO: debug

    # transform the 3D object pose in voxel to 3D pose in world
    transformed_obj_poses = []
    for i in range(len(obj_poses)):
        transformed_obj_pose = voxel_transform.dot(obj_poses[i])
        transformed_obj_poses.append(transformed_obj_pose)
    return transformed_obj_poses

def state_to_pose_in_world(obj_states, voxel_transform, zs):
    # TODO: debug

    # transform the 3D object pose in voxel to 3D pose in world
    transformed_obj_poses = []
    for i in range(len(obj_states)):
        obj_2d_pose = pose_from_2d_state(obj_states[i])
        obj_pose = pose_from_2d_pose(obj_2d_pose, zs[i])
        transformed_obj_pose = voxel_transform.dot(obj_pose)
        transformed_obj_poses.append(transformed_obj_pose)
    return transformed_obj_poses



def rearrangement_plan(objs, obj_pcds, obj_start_poses, moveable_objs, moveable_obj_start_poses, 
                        collision_voxel, robot_collision_voxel, voxel_transform, voxel_resol, n_iter=15):
    # 3d voxel to 2d grid
    # TODO: debug
    collision_grid = collision_voxel.sum(axis=2)>0
    # in the z-axis, if there is at least one voxel occupied, then collision grid
    grid_resol = voxel_resol[:2]
    robot_collision_grid = robot_collision_voxel.sum(axis=2)>0
    
    # collision_grid, grid_resol,
    map_x, map_y = np.indices(collision_grid.shape).astype(int)

    # we use the collision voxel with robot since we don't want the object to collide with the robot at grasp pose
    obj_poses = sample_goal_locations(obj_pcds, moveable_objs, robot_collision_voxel, voxel_resol, n_iter)
    # obj_poses: for all objects (obj_pcds and moveable_objs)

    # sampled pose: relative to voxel frame. 2D


    obj_states = []
    for i in range(len(obj_poses)):
        obj_states.append(state_from_2d_pose(obj_poses[i]))
    if obj_poses is None:
        return None
    # convert the start poses to 2D     # TODO: debug
    obj_start_poses_in_voxel = obtain_pose_in_voxel(obj_start_poses, voxel_transform)
    obj_2d_start_poses = projection_rot_matrix(obj_start_poses_in_voxel)
    moveable_obj_start_poses_in_voxel = obtain_pose_in_voxel(moveable_obj_start_poses, voxel_transform)
    moveable_obj_2d_start_poses = projection_rot_matrix(moveable_obj_start_poses_in_voxel)

    obj_2d_start_states = states_from_2d_pose(obj_2d_start_poses)
    moveable_obj_2d_start_states = states_from_2d_pose(moveable_obj_2d_start_poses)

    obj_pcd_2ds = obj_pcd_2d_projection(obj_pcds)
    moveable_obj_pcd_2ds = obj_pcd_2d_projection(moveable_objs)

    # concatenate with moveable     # TODO: debug
    obj_pcd_2ds = obj_pcd_2ds + moveable_obj_pcd_2ds
    obj_2d_start_poses = obj_2d_start_poses + moveable_obj_2d_start_poses
    obj_2d_start_states = obj_2d_start_states + moveable_obj_2d_start_states
    obj_start_poses = obj_start_poses + moveable_obj_start_poses
    obj_start_poses_in_voxel = obj_start_poses_in_voxel + moveable_obj_start_poses_in_voxel


    # * given the target poses, rearrange the objects to those locations
    # DFS for searching
    # TODO: make sure that the trajectory is executable by the robot
    # preprocessed_data = preprocess(obj_pcds, objs)  # preprocess to generate Minkowski sum
    searched_objs = []  # record what object id at each step
    searched_objs_set = set()
    searched_trajs = []

    search_start = 0  # record where to start search for the current depth
    valid = False
    while len(searched_objs) < len(objs):
        for i in range(search_start, len(objs)):
            if i in searched_objs_set:
                continue
            traj = find_trajectory(i, obj_pcd_2ds[i], obj_2d_start_poses[i], obj_2d_start_states[i], 
                                        obj_poses[i], obj_states[i], obj_pcd_2ds, 
                                        obj_2d_start_poses, obj_2d_start_states, 
                                        obj_poses, obj_states, searched_objs,
                                        collision_grid, grid_resol)
            # NOTE: when finding trajectory, we ignore the robot grasp pose at the target object 
            valid_i = len(traj) > 0
            if valid_i:
                valid = True
                break
        if valid:
            # add the new object to rearrangement list
            searched_objs.append(i)
            searched_objs_set.add(i)
            searched_trajs.append(traj)
            search_start = 0  # reset
        else:
            # all possible choices fail at this depth. back up and change search_start
            if len(searched_objs) == 0:
                # if empty, then failed
                return None   
            idx = searched_objs.pop()
            searched_objs_set.remove(idx)
            searched_trajs.pop()
            search_start = idx + 1  # start from the next one

    # * now the order of arrangement is indicated by the list search_objs

    # transform the trajs which is a list of states to the trajectory of poses
    # TODO: debug
    input('before trasnforming search trajectory...')
    transformed_searched_trajs = []
    for i in range(len(searched_trajs)):
        zs = np.zeros(len(searched_trajs[i])) + obj_start_poses_in_voxel[searched_objs[i]][2,3]
        traj = state_to_pose_in_world(searched_trajs[i], voxel_transform, zs)
        transformed_searched_trajs.append(traj)

    input('after transforming trajectory...')

    return searched_objs, transformed_searched_trajs

def minkowski_diff(obj_indices_i, obj_indices_j):
    # obj_indices_i = np.array(obj_grid_i.nonzero()).astype(int).T
    # obj_indices_j = np.array(obj_grid_j.nonzero()).astype(int).T
    minkowski_diff = obj_indices_i.reshape((-1,1,2)) - obj_indices_j.reshape((1,-1,2))
    minkowski_diff = minkowski_diff.reshape((-1,2))
    
    return minkowski_diff

def preprocess(obj_pcds, objs):
    # Minkowski diff between pairs of objects
    obj_grids = [objs[i].get_conservative_model().sum(axis=2).astype(bool) for i in range(len(objs))]
    obj_indices = [obj_grids[i].nonzero().astype(int).T for i in range(len(objs))]  # N X 2
    
    minkowski_diffs = [[None for i in range(len(objs))] for j in range(len(objs))]
    for i in range(len(objs)):
        for j in range(i+1, len(objs)):
            # i - j
            minkowski_diff = obj_indices[i].reshape((-1,1,2)) - obj_indices[j].reshape((1,-1,2))
            minkowski_diffs[i][j] = minkowski_diff.reshape((-1,2))
    return minkowski_diffs

def find_trajectory(obj_i, obj, obj_start_pose, obj_start_state, obj_pose, obj_state, 
                    objs, obj_start_poses, obj_start_states, obj_poses, obj_states, searched_objs, 
                    collision_grid, grid_resol):
    # for objs in the saerched_objs, they use the final obj_poses
    # for other objects, they use the obj_start_poses
    """
    try using DFS resolution-complete search
    """
    # put object collision areas into the grid
    total_objs = set(list(range(len(objs))))
    serached_obj_set = set(searched_objs)
    unsearched_obj_set = total_objs - serached_obj_set
    unsearched_objs = list(unsearched_obj_set)

    print('searched objects: ', searched_objs)
    print('obj_i: ', obj_i)
    print('obj pcd: ')
    print(objs)
    mp_map = np.array(collision_grid)

    plt.clf()
    x_map, y_map = np.indices(mp_map.shape).astype(int)
    plt.pcolor(x_map * grid_resol[0], y_map * grid_resol[1], mp_map)        

    plt.scatter(objs[0][:,0], objs[0][:,1])
    input('object 0')

    plt.clf()


    for obj_idx in searched_objs:
        print('searched object idx: ', obj_idx)
        plt.scatter(objs[obj_idx][:,0], objs[obj_idx][:,1])
        input('object see')
        plt.clf()
        transformed_pcd = obj_poses[obj_idx][:2,:2].dot(objs[obj_idx].T).T + obj_poses[obj_idx][:2,2]
        transformed_pcd = transformed_pcd / grid_resol
        plt.scatter(objs[obj_idx][:,0], objs[obj_idx][:,1])
        input('object see')
        plt.clf()

        transformed_pcd = np.floor(transformed_pcd).astype(int)
        mp_map[transformed_pcd[:,0],transformed_pcd[:,1]] = 1
        plt.scatter(transformed_pcd[:,0], transformed_pcd[:,1])
        input('plotted searched_obj')
    for obj_idx in unsearched_objs:
        if obj_idx == obj_i:
            continue
        transformed_pcd = obj_start_poses[obj_idx][:2,:2].dot(objs[obj_idx].T).T + obj_start_poses[obj_idx][:2,2]
        transformed_pcd = transformed_pcd / grid_resol
        transformed_pcd = np.floor(transformed_pcd).astype(int)
        mp_map[transformed_pcd[:,0],transformed_pcd[:,1]] = 1        
        plt.scatter(transformed_pcd[:,0], transformed_pcd[:,1])
        input("plotted unsearched_obj")

    # A* search to find the path
    dlinear = grid_resol[0]
    dtheta = 0.01

    plt.clf()
    x_map, y_map = np.indices(mp_map.shape).astype(int)
    plt.pcolor(x_map * grid_resol[0], y_map * grid_resol[1], mp_map)        
    input("before a_star")


    # check collision for start and goal. If they're in collision, then automatically return failure
    transformed_pcd = obj_start_pose[:2,:2].dot(obj.T).T + obj_start_pose[:2,2]
    # plot to see if the start is in collision
    plt.clf()
    plt.pcolor(x_map * grid_resol[0], y_map * grid_resol[1], mp_map)        
    plt.scatter(transformed_pcd[:,0], transformed_pcd[:,1])
    input('after visualizing start pose of object')

    transformed_pcd = transformed_pcd / grid_resol
    transformed_pcd = np.floor(transformed_pcd).astype(int)
    valid_filter = (transformed_pcd[:,0] >= 0) & (transformed_pcd[:,0] < mp_map.shape[0]) & \
                    (transformed_pcd[:,1] >= 0) & (transformed_pcd[:,1] < mp_map.shape[1])
    transformed_pcd = transformed_pcd[valid_filter]
    if mp_map[transformed_pcd[:,0], transformed_pcd[:,1]].sum() > 0:
        return []


    transformed_pcd = obj_pose[:2,:2].dot(obj.T).T + obj_pose[:2,2]
    # plot to see if the start is in collision
    plt.clf()
    plt.pcolor(x_map * grid_resol[0], y_map * grid_resol[1], mp_map)        
    plt.scatter(transformed_pcd[:,0], transformed_pcd[:,1])
    input('after visualizing goal pose of object')


    transformed_pcd = transformed_pcd / grid_resol
    transformed_pcd = np.floor(transformed_pcd).astype(int)
    valid_filter = (transformed_pcd[:,0] >= 0) & (transformed_pcd[:,0] < mp_map.shape[0]) & \
                    (transformed_pcd[:,1] >= 0) & (transformed_pcd[:,1] < mp_map.shape[1])
    transformed_pcd = transformed_pcd[valid_filter]
    if mp_map[transformed_pcd[:,0], transformed_pcd[:,1]].sum() > 0:
        return []
    

    traj = a_star_se2(obj, obj_start_state, obj_start_pose, obj_state, obj_pose, mp_map, grid_resol, dlinear, dtheta)    

    print('found trajectory: ', traj)
    input("after found trajectory")
    # plot the trajectory by putting the object at each of the state
    x_grid, y_grid = np.indices(collision_grid.shape).astype(int)
    return traj


def se2_distance(state1, state2):
    dist = (state1[0] - state2[0])**2 + (state1[1] - state2[1])**2
    dist = np.sqrt(dist)
    ang_dist = state1[2] - state2[2]
    ang_dist = ang_dist % (np.pi*2)
    if ang_dist > np.pi:
        ang_dist = ang_dist - np.pi * 2
    # dist += ang_dist ** 2
    # dist = np.sqrt(dist)
    ang_dist = np.abs(ang_dist)
    return dist, ang_dist  # we return a tuple


def a_star_se2(obj_pcd, start_state, start_pose, goal_state, goal_pose, collision_map, map_resol, dlinear, dtheta):
    # expansion number: we use L1-distance equal to 1
    plt.clf()
    
    # expansion: move 6 directions
    corner = np.array([1.0, 1.0, 0.0])
    corner[:2] = corner[:2] * map_resol
    # corner = corner / np.linalg.norm(corner)
    expansions = [[dlinear, 0, 0], [-dlinear, 0, 0], [0, dlinear, 0], [0,-dlinear, 0]]
    # expansions += [[corner[0],corner[1],0],[corner[0],-corner[1],0],[-corner[0],corner[1],0],[-corner[0],-corner[1],0]]
    # expansions +=[[0,0,dtheta], [0,0,-dtheta]]
    # make sure we don't repeat states, we round the floating point number to .00
    # and also wrap the angle to [0, 2pi)
    heap = []  # for a star
    # initialization
    # item in the heap: (g_linear+h_linear, g_angle+h_angle, g_linear, g_angle, parent, item)
    start_state = np.array(start_state)
    start_state[2] = start_state[2] % (np.pi*2)
    goal_state = np.array(goal_state)
    goal_state[2] = goal_state[2] % (np.pi*2)

    rounded_start = np.round(start_state, 2)
    rounded_goal = np.round(goal_state, 2)
    rounded_start_pose = pose_from_2d_state(rounded_start)
    rounded_goal_pose = pose_from_2d_state(rounded_goal)

    rounded_start_pcd = rounded_start_pose[:2,:2].dot(obj_pcd.T).T + rounded_start_pose[:2,2]
    rounded_goal_pcd = rounded_goal_pose[:2,:2].dot(obj_pcd.T).T + rounded_goal_pose[:2,2]

    
    dist, ang_dist = se2_distance(rounded_start, rounded_goal)
    heapq.heappush(heap, (dist+ang_dist, 0, -1, None, rounded_start))
    explored_states = np.zeros(collision_map.shape).astype(bool)
    
    parent_of_states = dict()
    parent_of_states_x = np.zeros(collision_map.shape)
    parent_of_states_y = np.zeros(collision_map.shape)

    start_indices = rounded_start[:2] / map_resol
    start_indices = np.floor(start_indices).astype(int)

    explored_states[start_indices[0],start_indices[1]] = 1


    done = False
    iter_i = 0

    map_x, map_y = np.indices(collision_map.shape).astype(int)

    linear_threshold = 1e-3
    ang_threshold = 1e-3

    while len(heap)>0:
        # print('iteration: ', iter_i)
        iter_i += 1
        # expansion
        data = heapq.heappop(heap)
        g_linear = data[1]
        # g_angle = data[2]
        # g = data[1]
        state = data[4]

        state_indices = np.round(state[:2]/map_resol).astype(int)
        explored_states[state_indices[0],state_indices[1]] = 1

        plt.clf()
        plt.pcolor(map_x*map_resol[0], map_y*map_resol[1], collision_map)


        plt.scatter(rounded_start_pcd[:,0], rounded_start_pcd[:,1], c='g')
        plt.scatter(rounded_goal_pcd[:,0], rounded_goal_pcd[:,1], c='r')

        pose = pose_from_2d_state(state)
        pcd = pose[:2,:2].dot(obj_pcd.T).T + pose[:2,2]
        plt.scatter(pcd[:,0], pcd[:,1], c='b')
        plt.pause(0.1)

        # if the state is close to goal, then stop
        d_linear, d_angle = se2_distance(state, rounded_goal)

        if d_linear <= linear_threshold:# and d_angle <= ang_threshold:
            # when the position is right, we directly change the orientation of the object
            done = True
            print('done.')
            break

        for i in range(len(expansions)):
            # when the object is at the target location, not using linear moves anymore
            ds = expansions[i]
            if d_linear <= linear_threshold:
                if ds[2] == 0:
                    continue
                

            new_state = np.array([state[0]+ds[0], state[1]+ds[1], state[2]+ds[2]])
            new_state = np.round(new_state, 2)
            new_state[2] = new_state[2] % (np.pi*2)

            # check whether the new state has been explored before
            new_state_indices = np.round(new_state[:2] / map_resol).astype(int)
            print('new_state_indices: ', new_state_indices)
            if explored_states[new_state_indices[0],new_state_indices[1]]:
                print('explored...')
                continue
            # check whether the new state is in collision
            new_pose = pose_from_2d_state(new_state)
            transformed_pcd = new_pose[:2,:2].dot(obj_pcd.T).T + new_pose[:2,2]
            transformed_pcd = transformed_pcd / map_resol
            transformed_pcd = np.floor(transformed_pcd).astype(int)
            valid_filter = (transformed_pcd[:,0]>=0) & (transformed_pcd[:,0]<collision_map.shape[0]) & \
                            (transformed_pcd[:,1]>=0) & (transformed_pcd[:,1]<collision_map.shape[1])
            if valid_filter.sum() != len(transformed_pcd):
                print('outiside of workspace...')
                continue
            transformed_pcd = transformed_pcd[valid_filter]
            if collision_map[transformed_pcd[:,0], transformed_pcd[:,1]].sum() > 0:
                # collision
                print('collision with environment...')
                continue
            # add the new states
            # TODO: now hard to converge
            h_linear, h_angle = se2_distance(new_state, rounded_goal)
            
            parent_of_states_x[new_state_indices[0],new_state_indices[1]] = state[0]
            parent_of_states_y[new_state_indices[0],new_state_indices[1]] = state[1]

            c_linear = np.linalg.norm(new_state[:2]-state[:2])
            c_theta = np.abs(new_state[2]-state[2])

            heapq.heappush(heap, (g_linear+c_linear+5*h_linear, 
                                  g_linear+c_linear, iter_i*len(expansions)+i, state, new_state))
            explored_states[new_state_indices[0],new_state_indices[1]] = 1


        # TODO: correction: update the route to previous nodes if there is a shorter one
        # since we explore every time the shortest distance possible, the correction is not necessary

    if done:
        # retreive the path from start to goal
        trajs = [tuple(rounded_goal.tolist())]
        state = tuple(state.tolist())
        while state != tuple(rounded_start.tolist()):
            trajs.append(state)
            state_indices = np.array(state)[:2] / map_resol
            state_indices = np.round(state_indices).astype(int)
            state_x = parent_of_states_x[state_indices[0],state_indices[1]]
            state_y = parent_of_states_y[state_indices[0],state_indices[1]]
            state = tuple([state_x,state_y,state[2]])

        return trajs[::-1]
    return []



def sample_goal_locations(objs, moveable_objs, collision_voxel, voxel_resol, n_iter=15):
    # TODO: debug
    collision_grid = collision_voxel.sum(axis=2)>0
    # in the z-axis, if there is at least one voxel occupied, then collision grid
    grid_resol = voxel_resol[:2]
    # collision_grid, grid_resol,
    map_x, map_y = np.indices(collision_grid.shape).astype(int)

    plt.clf()
    plt.pcolor(map_x*grid_resol[0], map_y*grid_resol[1], collision_grid)


    obj_pcd_2ds = obj_pcd_2d_projection(objs)
    moveable_obj_pcd_2ds = obj_pcd_2d_projection(moveable_objs)

    # TODO get 2d moveable object poses
    moveable_obj_poses = []
    # moveable objs: use its initial pose first. If in collision then move them

    # NOTE: now we put all as obj
    obj_pcd_2ds = obj_pcd_2ds + moveable_obj_pcd_2ds

    for i in range(n_iter):
        obj_poses = initialize_poses(obj_pcd_2ds, collision_grid, grid_resol)
        # obj pose is 2D
        for trial_i in range(25):
            valid = True
            forces = np.zeros((len(objs),2))
            for i in range(len(objs)):
                for j in range(i+1, len(objs)):
                    if obj_collision(obj_pcd_2ds[i], obj_poses[i], obj_pcd_2ds[j], obj_poses[j], collision_grid, grid_resol):
                        forces[i] += obj_obj_force(obj_pcd_2ds[i], obj_poses[i], obj_pcd_2ds[j], obj_poses[j])
                        forces[j] += (-forces[i])
                        valid = False
            for i in range(len(objs)):
                if in_collision(obj_pcd_2ds[i], obj_poses[i], collision_grid, grid_resol):
                    forces[i] += obj_collision_force(obj_pcd_2ds[i], obj_poses[i], collision_grid, grid_resol)
                    valid = False

                if outside_boundary(obj_pcd_2ds[i], obj_poses[i], collision_grid, grid_resol):
                    forces[i] += obj_boundary_force(obj_pcd_2ds[i], obj_poses[i], collision_grid, grid_resol)
                    valid = False
            print('forces: ')
            print(forces)
            # update
            for i in range(len(objs)):
                obj_poses[i] = update(obj_poses[i], forces[i])
            print('obj_poses: ')
            print(obj_poses)
            # after update, plot
            plt.clf()
            plt.pcolor(map_x*grid_resol[0], map_y*grid_resol[1], collision_grid)
            for i in range(len(objs)):
                pcd = obj_poses[i][:2,:2].dot(obj_pcd_2ds[i].T).T + obj_poses[i][:2,2]
                plt.scatter(pcd[:,0], pcd[:,1])
            input('next...')

            # validate obj poses  
            if valid:
                return obj_poses
            if np.abs(forces).sum() == 0:
                # converged and not valid. try next time
                print('try another initialization...')
                break
    return None  # no solution


def update(obj_pose, force):
    res = np.array(obj_pose)
    res[:2,2] += force
    return res


def obj_pcd_2d_projection(objs):
    obj_pcds = []
    for i in range(len(objs)):
        pcd = objs[i]
        obj_pcds.append(pcd[:,:2])
    return obj_pcds

def filter_indices_map_size(indices, map):
    valid_i = (indices[:,0] >= 0) & (indices[:,0]<map.shape[0]) & \
                (indices[:,1] >= 0) & (indices[:,1]<map.shape[1])
    return indices[valid_i]


def initialize_poses(objs, collision_grid, grid_resol):
    # initialize 2d pose: x, y, theta
    # x, y are at the valid grids
    valid_indices = np.nonzero(~collision_grid)
    valid_indices = np.array(valid_indices).T
    valid_pos = valid_indices * grid_resol
    
    pos_indices = np.random.choice(len(valid_pos), size=len(objs))
    sampled_pos = valid_pos[pos_indices]
    thetas = np.random.uniform(low=-np.pi, high=np.pi, size=len(valid_pos))

    obj_poses = []
    for i in range(len(objs)):
        obj_pose_i = np.eye(3)
        obj_pose_i[0,0] = np.cos(thetas[i])
        obj_pose_i[0,1] = -np.sin(thetas[i])
        obj_pose_i[1,0] = np.sin(thetas[i])
        obj_pose_i[1,1] = np.cos(thetas[i])
        obj_pose_i[0,2] = sampled_pos[i,0]
        obj_pose_i[1,2] = sampled_pos[i,1]
        obj_poses.append(obj_pose_i)

    return obj_poses


def obj_collision(obj_i, obj_pose_i, obj_j, obj_pose_j, collision_grid, grid_resol):
    grid_i = np.zeros(collision_grid.shape).astype(bool)
    transformed_pcd_i = obj_pose_i[:2,:2].dot(obj_i.T).T + obj_pose_i[:2,2]
    transformed_pcd_i = transformed_pcd_i / grid_resol
    transformed_pcd_i = np.floor(transformed_pcd_i).astype(int)
    valid_i = (transformed_pcd_i[:,0] >= 0) & (transformed_pcd_i[:,0]<collision_grid.shape[0]) & \
                (transformed_pcd_i[:,1] >= 0) & (transformed_pcd_i[:,1]<collision_grid.shape[1])
    transformed_pcd_i = transformed_pcd_i[valid_i]
    grid_i[transformed_pcd_i[:,0],transformed_pcd_i[:,1]] = 1


    grid_j = np.zeros(collision_grid.shape).astype(bool)
    transformed_pcd_j = obj_pose_j[:2,:2].dot(obj_j.T).T + obj_pose_j[:2,2]
    transformed_pcd_j = transformed_pcd_j / grid_resol
    transformed_pcd_j = np.floor(transformed_pcd_j).astype(int)
    valid_j = (transformed_pcd_j[:,0] >= 0) & (transformed_pcd_j[:,0]<collision_grid.shape[0]) & \
                (transformed_pcd_j[:,1] >= 0) & (transformed_pcd_j[:,1]<collision_grid.shape[1])
    transformed_pcd_j = transformed_pcd_j[valid_j]
    grid_j[transformed_pcd_j[:,0],transformed_pcd_j[:,1]] = 1

    return (grid_i & grid_j).sum() > 0


def obj_obj_force(obj_i, pose_i, obj_j, pose_j):
    obj_i_transformed = pose_i[:2,:2].dot(obj_i.T).T + pose_i[:2,2]
    center_i = obj_i_transformed.mean(axis=0)    

    obj_j_transformed = pose_j[:2,:2].dot(obj_j.T).T + pose_j[:2,2]
    center_j = obj_j_transformed.mean(axis=0)    

    direction = center_i - center_j
    distance = np.linalg.norm(direction)
    direction = direction / np.linalg.norm(direction)

    # assuming bounding circle, obtain the radius
    r1 = obj_i - center_i.reshape(-1,2)
    r1 = np.linalg.norm(r1, axis=1)
    r1 = np.max(r1)

    r2 = obj_j - center_j.reshape(-1,2)
    r2 = np.linalg.norm(r2, axis=1)
    r2 = np.max(r2)

    dis = (r1 + r2 - distance) / 2
    force_factor = 0.01
    force = dis * direction * force_factor
    return force

def in_collision(obj_i, pose_i, collision_grid, grid_resol):
    obj_i_transformed = pose_i[:2,:2].dot(obj_i.T).T + pose_i[:2,2]
    obj_i_transformed = obj_i_transformed / grid_resol
    obj_i_transformed = np.floor(obj_i_transformed).astype(int)
    obj_i_transformed = filter_indices_map_size(obj_i_transformed, collision_grid)

    occupied_map_i = np.zeros(collision_grid.shape).astype(bool)
    occupied_map_i[obj_i_transformed[:,0], obj_i_transformed[:,1]] = 1

    intersection = occupied_map_i & collision_grid
    return intersection.sum() > 0

def obj_collision_force(obj_i, pose_i, collision_grid, grid_resol):
    # get occupied space of obj_i
    obj_i_transformed = pose_i[:2,:2].dot(obj_i.T).T + pose_i[:2,2]
    obj_i_transformed = obj_i_transformed / grid_resol
    obj_i_transformed = np.floor(obj_i_transformed).astype(int)
    obj_i_transformed = filter_indices_map_size(obj_i_transformed, collision_grid)

    occupied_map_i = np.zeros(collision_grid.shape).astype(bool)
    occupied_map_i[obj_i_transformed[:,0], obj_i_transformed[:,1]] = 1

    intersection = occupied_map_i & collision_grid
    inter_pts = np.nonzero(intersection)
    inter_pts = np.array(inter_pts).T
    inter_x = inter_pts[:,0] + 0.5
    inter_y = inter_pts[:,1] + 0.5

    # get the force from each intersection grid
    inter_x = inter_x * grid_resol[0]
    inter_y = inter_y * grid_resol[1]

    center = pose_i[:2,:2].dot(obj_i.T).T + pose_i[:2,2]
    center = center.mean(axis=0)
    print("center mean: ", center)

    direction = center.reshape(-1,2) - np.array([inter_x, inter_y]).T
    direction = direction / np.linalg.norm(direction, axis=1).reshape(-1,1)

    # direction = direction / np.linalg.norm(direction)
    # force_factor = np.linalg.norm(grid_resol)
    # forces = direction * force_factor
    force = direction.mean(axis=0)
    force = force / np.linalg.norm(force)
    force = force * grid_resol
    # force = force * force_factor

    # TODO: maybe not a good idea to sum all the forces
    return force


def outside_boundary(obj_i, pose_i, collision_grid, grid_resol):
    # see if the current object is outside of the map
    obj_i_transformed = pose_i[:2,:2].dot(obj_i.T).T + pose_i[:2,2]
    obj_i_transformed = obj_i_transformed / grid_resol
    obj_i_transformed = np.floor(obj_i_transformed).astype(int)
    outside_map_filter = (obj_i_transformed[:,0] < 0) | (obj_i_transformed[:,0] >= collision_grid.shape[0]) | \
                         (obj_i_transformed[:,1] < 0) | (obj_i_transformed[:,1] >= collision_grid.shape[1])
    return outside_map_filter.sum()>0

def obj_boundary_force(obj_i, pose_i, collision_grid, grid_resol):
    # see if the current object is outside of the map
    obj_i_transformed = pose_i[:2,:2].dot(obj_i.T).T + pose_i[:2,2]
    obj_i_transformed = obj_i_transformed / grid_resol
    obj_i_transformed = np.floor(obj_i_transformed).astype(int)
    outside_map_filter = (obj_i_transformed[:,0] < 0) | (obj_i_transformed[:,0] >= collision_grid.shape[0]) | \
                         (obj_i_transformed[:,1] < 0) | (obj_i_transformed[:,1] >= collision_grid.shape[1])
    print("outside map filter: ", outside_map_filter.sum() == 0)
    if outside_map_filter.sum() == 0:
        return np.zeros(2)
        
    x_force = np.zeros(2)
    y_force = np.zeros(2)
    if (obj_i_transformed[:,0] < 0).sum() > 0:
        x_force = np.array([-obj_i_transformed[:,0].min(),0.0])
    if (obj_i_transformed[:,0] >= collision_grid.shape[0]).sum() > 0:
        x_force = np.array([collision_grid.shape[0]-1-obj_i_transformed[:,0].max(), 0.0])
    if (obj_i_transformed[:,1] < 0).sum() > 0:
        y_force = np.array([0.0,-obj_i_transformed[:,1].min()])
    if (obj_i_transformed[:,1] >= collision_grid.shape[1]).sum() > 0:
        y_force = np.array([0.0, collision_grid.shape[1]-1-obj_i_transformed[:,1].max()])
        
    force = x_force + y_force
    force = force * grid_resol
    # force = force / np.linalg.norm(force)
    # force_factor = 0.1
    # force = force * force_factor
    print('move back to ws force: ', force)
    return force




def sample_circle(center, radius, n_samples=100):
    # sample a circle with center and radius
    # pcd_cylinder_r = np.random.uniform(low=0, high=radius, size=n_samples)
    circle_r = np.random.triangular(left=0., mode=radius, right=radius, size=n_samples)
    circle_xy = np.random.normal(loc=[0.,0.], scale=[1.,1.], size=(n_samples,2))
    circle_xy = circle_xy / np.linalg.norm(circle_xy, axis=1).reshape(-1,1)
    circle_xy = circle_xy * circle_r.reshape(-1,1)
    circle_xy = circle_xy + center.reshape(1,2)
    return circle_xy

def sample_triangle(p1, p2, p3, n_samples=100):
    # sample alpha, beta
    alpha = np.random.uniform(low=0, high=1, size=n_samples).reshape((-1,1))
    beta = np.random.uniform(low=0, high=1, size=n_samples).reshape((-1,1))
    # u = min(alpha, beta)
    u = np.minimum(alpha, beta)
    v = np.maximum(alpha, beta) - u
    w = 1 - u - v
    p1 = p1.reshape((1,2))
    p2 = p2.reshape((1,2))
    p3 = p3.reshape((1,2))
    return p1 * u + p2 * v + p3 * w
    


def sample_rectangle(x_size, y_size, n_samples=100):
    pcd = np.random.uniform(low=[-0.5,-0.5],high=[0.5,0.5], size=(n_samples,2))
    pcd = pcd * np.array([x_size, y_size])
    return pcd

    

def main():
    # construct the map
    plt.ion()
    plt.figure(figsize=(10,10))
    
    col_map = np.zeros((50, 50, 10)).astype(bool)
    voxel_resol = np.array([0.01, 0.01, 0.01])
    map_resol = np.array([0.01, 0.01])

    # col_map[:10,:,:] = 1
    # col_map[10:15,:,2] = 1

    # construct objects: 1 circle, 1 rectangular
    obj_1 = sample_circle(center=np.array([0.,0.]) * map_resol, radius=3*map_resol[0], n_samples=2000)
    # obj_1 = sample_triangle(np.array([-0.02,-0.02]), np.array([0, 0.03]), np.array([0.02, -0.02]), 2000)
    obj_2 = sample_rectangle(6*map_resol[0],6*map_resol[1], n_samples=2000)

    obj_3 = sample_circle(center=np.array([0.,0.])*map_resol, radius=3*map_resol[0], n_samples=2000)
    obj_4 = sample_rectangle(6*map_resol[0],6*map_resol[1], n_samples=2000)
    obj_5 = sample_rectangle(4*map_resol[0],8*map_resol[1], n_samples=2000)

    obj_1_min = np.min(obj_1, axis=0)
    obj_1_max = np.max(obj_1, axis=0)
    resol = 0.001
    size = obj_1_max - obj_1_min
    size = size / resol
    size = np.ceil(size).astype(int)
    # obj_1_grid = (obj_1 - obj_1_min) / resol
    obj_1_grid = np.floor(obj_1/resol).astype(int)
    obj_grid_1 = np.zeros((size[0], size[1])).astype(bool)


    obj_1_grid_ = (obj_1 - obj_1_min) / resol
    obj_1_grid_ = np.floor(obj_1_grid_).astype(int)

    obj_grid_1[obj_1_grid_[:,0], obj_1_grid_[:,1]] = 1

    obj_2_min = np.min(obj_2, axis=0)
    obj_2_max = np.max(obj_2, axis=0)
    resol = 0.001
    size = obj_2_max - obj_2_min
    size = size / resol
    size = np.ceil(size).astype(int)
    # obj_2_grid = (obj_2 - obj_2_min) / resol
    obj_2_grid = np.floor(obj_2/resol).astype(int)
    obj_grid_2 = np.zeros((size[0], size[1])).astype(bool)

    obj_2_grid_ = (obj_2 - obj_2_min) / resol
    obj_2_grid_ = np.floor(obj_2_grid_).astype(int)

    obj_grid_2[obj_2_grid_[:,0], obj_2_grid_[:,1]] = 1


    diff = minkowski_diff(obj_1_grid_, obj_2_grid_)
    x_1, y_1 = np.indices(obj_grid_1.shape).astype(int)
    x_2, y_2 = np.indices(obj_grid_2.shape).astype(int)

    mins = np.min(diff, axis=0)
    maxs = np.max(diff, axis=0)
    diff_plot = np.zeros(maxs-mins+1).astype(bool)
    diff_plot[(diff-mins)[:,0],(diff-mins)[:,1]] = 1
    x_diff, y_diff = np.indices(diff_plot.shape).astype(int)

    # rearrangement_plan([obj_1, obj_2], [], col_map, map_resol, n_iter=15)

    obj_1_start_pose = np.eye(4)
    obj_1_start_pose[:3,3] = np.array([4.0*map_resol[0],4.0*map_resol[1],0.0])
    obj_2_start_pose = np.eye(4)
    obj_2_start_pose[:3,3] = np.array([4.0*map_resol[0],16.0*map_resol[1],0.0])
    obj_3_start_pose = np.eye(4)
    obj_3_start_pose[:3,3] = np.array([16.0*map_resol[0],16.0*map_resol[1],0.0])
    obj_4_start_pose = np.eye(4)
    obj_4_start_pose[:3,3] = np.array([25.0*map_resol[0],5.0*map_resol[1],0.0])
    obj_5_start_pose = np.eye(4)
    obj_5_start_pose[:3,3] = np.array([40.0*map_resol[0],30.0*map_resol[1],0.0])

    obj_start_poses = [obj_1_start_pose, obj_2_start_pose, obj_3_start_pose, obj_4_start_pose, obj_5_start_pose]
    rearrangement_plan([obj_1, obj_2, obj_3, obj_4, obj_5], [obj_1, obj_2, obj_3, obj_4, obj_5], obj_start_poses, [], [], 
                        col_map, col_map, np.eye(4), voxel_resol, n_iter=15)
    # initialize object shapes & poses

if __name__ == "__main__":
    main()