"""
implement the rearragenment algorithm, and do a simple test
Map: 2d grid world. Some regions are occupied by objects or occluded
objects: with shape
moveable objects
"""
import numpy as np
import matplotlib.pyplot as plt

def filter_obj_pcd_map_size(map, pcd):
    shape = map.shape
    valid_filter = (pcd[:,0] >= 0) & (pcd[:,0] < shape[0]) & (pcd[:,1] >= 0) & (pcd[:,1] < shape[1])
    return pcd[valid_filter]

def collision(obj_i, pose_i, obj_j, pose_j, map, map_resol, map_x, map_y):
    # get occupied space of obj_i
    obj_i_transformed = pose_i[:2,:2].dot(obj_i.T).T + pose_i[:2,2]
    obj_i_transformed = obj_i_transformed / map_resol
    obj_i_transformed = np.floor(obj_i_transformed).astype(int)
    obj_i_transformed = filter_obj_pcd_map_size(map, obj_i_transformed)

    occupied_map_i = np.zeros(map.shape).astype(bool)
    occupied_map_i[obj_i_transformed[:,0], obj_i_transformed[:,1]] = 1

    # get occupied space of obj_j
    obj_j_transformed = pose_j[:2,:2].dot(obj_j.T).T + pose_j[:2,2]
    obj_j_transformed = obj_j_transformed / map_resol
    obj_j_transformed = np.floor(obj_j_transformed).astype(int)
    obj_j_transformed = filter_obj_pcd_map_size(map, obj_j_transformed)
    occupied_map_j = np.zeros(map.shape).astype(bool)
    occupied_map_j[obj_j_transformed[:,0], obj_j_transformed[:,1]] = 1

    # find if there is an intersection
    intersection = occupied_map_i & occupied_map_j
    return intersection.sum()>0

def compute_force(obj_i, pose_i, obj_j, pose_j):
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
    force_factor = 1.0
    force = dis * direction * force_factor
    return force

def in_collision(obj_i, pose_i, map, map_resol, map_x, map_y):
    # get occupied space of obj_i
    obj_i_transformed = pose_i[:2,:2].dot(obj_i.T).T + pose_i[:2,2]
    obj_i_transformed = obj_i_transformed / map_resol
    obj_i_transformed = np.floor(obj_i_transformed).astype(int)
    obj_i_transformed = filter_obj_pcd_map_size(map, obj_i_transformed)

    occupied_map_i = np.zeros(map.shape).astype(bool)
    occupied_map_i[obj_i_transformed[:,0], obj_i_transformed[:,1]] = 1

    intersection = occupied_map_i & map
    print('in collision: ')
    print(intersection.sum()>0)
    return intersection.sum() > 0

def compute_workspace_force(obj_i, pose_i, map, map_resol, map_x, map_y):
    # get occupied space of obj_i
    obj_i_transformed = pose_i[:2,:2].dot(obj_i.T).T + pose_i[:2,2]
    obj_i_transformed = obj_i_transformed / map_resol
    obj_i_transformed = np.floor(obj_i_transformed).astype(int)
    obj_i_transformed = filter_obj_pcd_map_size(map, obj_i_transformed)

    occupied_map_i = np.zeros(map.shape).astype(bool)
    occupied_map_i[obj_i_transformed[:,0], obj_i_transformed[:,1]] = 1

    intersection = occupied_map_i & map

    # get the force from each intersection grid
    inter_x = map_x[intersection] + 0.5
    inter_y = map_y[intersection] + 0.5
    inter_x = inter_x * map_resol[0]
    inter_y = inter_y * map_resol[1]

    center = pose_i[:2,:2].dot(obj_i.T).T + pose_i[:2,2]
    center = center.mean(axis=0)
    print("center mean: ", center)

    direction = center.reshape(-1,2) - np.array([inter_x, inter_y]).T
    direction = direction / np.linalg.norm(direction, axis=1).reshape(-1,1)

    # direction = direction / np.linalg.norm(direction)
    force_factor = 0.1
    # forces = direction * force_factor
    force = direction.mean(axis=0)
    force = force / np.linalg.norm(force)
    force = force * force_factor

    # TODO: maybe not a good idea to sum all the forces
    return force

def move_back_to_ws(obj_i, pose_i, map, map_resol, map_x, map_y):
    # see if the current object is outside of the map
    obj_i_transformed = pose_i[:2,:2].dot(obj_i.T).T + pose_i[:2,2]
    obj_i_transformed = obj_i_transformed / map_resol
    obj_i_transformed = np.floor(obj_i_transformed).astype(int)
    outside_map_filter = (obj_i_transformed[:,0] < 0) | (obj_i_transformed[:,0] >= map.shape[0]) | \
                         (obj_i_transformed[:,1] < 0) | (obj_i_transformed[:,1] >= map.shape[1])
    print("outside map filter: ", outside_map_filter.sum() == 0)
    if outside_map_filter.sum() == 0:
        return np.zeros(2)
        
    x_force = np.zeros(2)
    y_force = np.zeros(2)
    if (obj_i_transformed[:,0] < 0).sum() > 0:
        x_force = np.array([-obj_i_transformed[:,0].min(),0.0])
    if (obj_i_transformed[:,0] >= map.shape[0]).sum() > 0:
        x_force = np.array([map.shape[0]-1-obj_i_transformed[:,0].max(), 0.0])
    if (obj_i_transformed[:,1] < 0).sum() > 0:
        y_force = np.array([0.0,-obj_i_transformed[:,1].min()])
    if (obj_i_transformed[:,1] >= map.shape[1]).sum() > 0:
        y_force = np.array([0.0, map.shape[1]-1-obj_i_transformed[:,1].max()])
        
    force = x_force + y_force
    # force = force / np.linalg.norm(force)
    # force_factor = 0.1
    # force = force * force_factor
    print('move back to ws force: ', force)
    return force



def rearrangement(objs, moveable_objs, map, map_resol, map_x, map_y):
    # * sample poses for objects in the free areas of the map
    # randomly pick one from map

    # initialize pose
    obj_poses = []
    forces = np.zeros((len(objs),2))
    valid_x = map_x[map==0]
    valid_y = map_y[map==0]
    transformed_objs = []
    for i in range(len(objs)):
        pos_idx = np.random.choice(len(valid_x))
        x = valid_x[pos_idx] * map_resol[0]
        y = valid_y[pos_idx] * map_resol[1]

        # sample orientation
        theta = np.random.uniform(low=0.0, high=np.pi*2)
        obj_pose_i = np.eye(3)
        obj_pose_i[0,0] = np.cos(theta)
        obj_pose_i[0,1] = -np.sin(theta)
        obj_pose_i[1,0] = np.sin(theta)
        obj_pose_i[1,1] = np.cos(theta)
        obj_pose_i[0,2] = x
        obj_pose_i[1,2] = y
        obj_poses.append(obj_pose_i)
        transformed_obj = obj_pose_i[:2,:2].dot(objs[i].T).T + obj_pose_i[:2,2]

        transformed_objs.append(transformed_obj)
        plt.scatter(transformed_obj[:,1], transformed_obj[:,0])
    input('before we start...')
    obj_poses = np.array(obj_poses)


    for iter_i in range(100):
        print('iteration ', iter_i, '...')
        forces = np.zeros((len(objs),2))

        valid_x = map_x[map==0]
        valid_y = map_y[map==0]

        # * TODO for some probability, resample moveable objects that are in collision

        

        # * compute force: object-object, object-edge
        # object-object force
        for i in range(len(objs)):
            for j in range(i+1, len(objs)):
                if collision(objs[i], obj_poses[i], objs[j], obj_poses[j], map, map_resol, map_x, map_y):
                    forces[i] = compute_force(objs[i], obj_poses[i], objs[j], obj_poses[j])
                    forces[j] = compute_force(objs[j], obj_poses[j], objs[i], obj_poses[i])

        # object-edge
        for i in range(len(objs)):            
            forces[i] += move_back_to_ws(objs[i], obj_poses[i], map, map_resol, map_x, map_y)

            if in_collision(objs[i], obj_poses[i], map, map_resol, map_x, map_y):
                forces[i] += compute_workspace_force(objs[i], obj_poses[i], map, map_resol, map_x, map_y)

        # validate if not in collision
        free = True
        for i in range(len(objs)):
            if np.linalg.norm(forces[i]) > 0:
                free = False
                break
        if free:
            print('Free Free')
            break


        force_factor = 0.01
        plt.clf()
        plot_map(map, map_x, map_y, map_resol)

        for i in range(len(objs)):
            obj_poses[i][:2,2] = obj_poses[i][:2,2] + force_factor * forces[i]
            # plot the object after transform
            obj_pose_i = obj_poses[i]
            transformed_obj = obj_pose_i[:2,:2].dot(objs[i].T).T + obj_pose_i[:2,2]
            plt.scatter(transformed_obj[:,1], transformed_obj[:,0])
        input('next...')
    

# TODO: need to handle objects that are outside of the boundary

def sample_circle(center, radius, n_samples=100):
    # sample a circle with center and radius
    # pcd_cylinder_r = np.random.uniform(low=0, high=radius, size=n_samples)
    circle_r = np.random.triangular(left=0., mode=radius, right=radius, size=n_samples)
    circle_xy = np.random.normal(loc=[0.,0.], scale=[1.,1.], size=(n_samples,2))
    circle_xy = circle_xy / np.linalg.norm(circle_xy, axis=1).reshape(-1,1)
    circle_xy = circle_xy * circle_r.reshape(-1,1)
    circle_xy = circle_xy + center.reshape(1,2)
    return circle_xy

def sample_rectangle(x_size, y_size, n_samples=100):
    pcd = np.random.uniform(low=[-0.5,-0.5],high=[0.5,0.5], size=(n_samples,2))
    pcd = pcd * np.array([x_size, y_size])
    return pcd
    
def plot_map(map, map_x, map_y, map_resol):
    plt.pcolor(map_y*map_resol[0], map_x*map_resol[1], map)

def main():
    # construct the map
    col_map = np.zeros((50, 50)).astype(bool)
    map_resol = np.array([0.01, 0.01])
    map_x, map_y = np.indices(col_map.shape).astype(int)

    col_map[:10,:] = 1
    print(np.concatenate([map_x, map_y], axis=0).shape)
    map_x_ = np.array([map_x])
    map_y_ = np.array([map_y])    
    map_pos = np.concatenate([map_x_, map_y_], axis=0).transpose(1,2,0) * map_resol.reshape(1,1,2)
    plt.ion()

    plt.pcolor(map_y*map_resol[0], map_x*map_resol[1], col_map)

    # construct objects: 1 circle, 1 rectangular
    obj_1 = sample_circle(center=np.array([0.,0.]), radius=0.05)
    obj_2 = sample_rectangle(0.02,0.03)

    rearrangement([obj_1, obj_2], [], col_map, map_resol, map_x, map_y)


    # initialize object shapes & poses


main()