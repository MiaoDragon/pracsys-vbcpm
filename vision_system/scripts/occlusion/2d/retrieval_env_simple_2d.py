"""
Simple 2D retrieval problem, where we don't consider dynamics, and the only
actions avaiable is pick-and-place. We assume we can directly manipulate objects.

Some of the code can be used for setting up the planning scene.
"""
import numpy as np
from numpy.core.arrayprint import array_repr
import pygame
from pygame import surface
from pygame.locals import (QUIT, KEYDOWN, K_ESCAPE)

class RetrievalPickPlace2D():
    def __init__(self):
        self.world_w = 1.0  # in meter
        self.world_h = 1.0  # in meter
        self.objects = []
        obj = {'loc': np.array([0.5, 0.5]), 'type': 'circle', 'radius': 0.1}
        self.objects.append(obj)
        self.camera_loc = [self.world_w/2, -0.01]  # x axis is pointing right, y axis is pointing up

    def setup(self):
        self.world_w = 1.0  # in meter
        self.world_h = 1.0  # in meter
        self.objects = []
        obj = {'loc': np.array([0.3, 0.7]), 'type': 'circle', 'radius': 0.1}
        self.objects.append(obj)
        obj = {'loc': np.array([0.3, 0.2]), 'type': 'circle', 'radius': 0.1}
        self.objects.append(obj)
        obj = {'loc': np.array([0.6, 0.2]), 'type': 'circle', 'radius': 0.1}
        self.objects.append(obj)
        obj = {'loc': np.array([0.55, 0.7]), 'type': 'circle', 'radius': 0.1}
        self.objects.append(obj)

        self.camera_loc = np.array([self.world_w/2, -0.01])  # x axis is pointing right, y axis is pointing up
        
        self.resol = np.array([0.01, 0.01])
        self.occupied = np.zeros((int(self.world_w / self.resol[0]), int(self.world_h / self.resol[1]))).astype(bool)
        self.occlusion = np.zeros((int(self.world_w / self.resol[0]), int(self.world_h / self.resol[1]))).astype(bool)
        self.voxel_x, self.voxel_y = np.indices(self.occlusion.shape).astype(float)

        self.visible_grid = np.array(self.occlusion)

        self.vis_objs = np.zeros(len(self.objects)).astype(bool)

        self.move_times = 10. + np.zeros(len(self.objects))

    def get_occlusion(self, objects):
        """
        obtain the occlusion, which is represented by a cone (two lines centered at the camera) and visibility grid
        """
        # @Edit on Sep 6: add mapping to object indices so that we know which obj it corresponds to

        # obtain the intersection of camera vector with the object

        # if the object is a circle, we can compute the tangent vector
        rays = []
        angles = []
        for obj in objects:
            if obj['type'] == 'circle':
                cam_to_center = obj['loc'] - self.camera_loc
                angle = np.arcsin(obj['radius']/np.linalg.norm(cam_to_center))  # the angle between cam_to_center and ray
                cam_to_center_vec = cam_to_center / np.linalg.norm(cam_to_center)
                rot_mat1 = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
                ray1 = rot_mat1.dot(cam_to_center_vec)
                rot_mat2 = np.array([[np.cos(-angle), -np.sin(-angle)], [np.sin(-angle), np.cos(-angle)]])
                ray2 = rot_mat2.dot(cam_to_center_vec)

                angle1 = np.arctan2(ray1[1], ray1[0])
                angle2 = np.arctan2(ray2[1], ray2[0])
                rays.append((ray2,ray1))
                angles.append((angle2,angle1))

            elif obj['type'] == 'polygon':
                pass

        rays = np.array(rays)
        occlusion = np.zeros(self.occlusion.shape).astype(bool)
        occlusion, cone_mask, occlusion_obj_mask = self.occlusion_cone_map(rays, angles, objects)
        return occlusion, cone_mask, occlusion_obj_mask


    def occlusion_cone_map(self, rays, angles, objects):
        # given the cone (represented by two angles), set the occlusion map
        # 1. identify if the grid is within the cone
        # 2. identify if the grid is behind the first surface seen (trace the ray to see if there is anything before it
        #    that belongs to the object)
        # @Edit on Sep 3: add return value of cone map. This helps us know the boundary of the cone, and helps with 
        # identifying which object is in the visibility region. 
        # @Edit on Sep 6: add mapping to object indices so that we know which obj it corresponds to

        occlusion = np.zeros(self.occlusion.shape).astype(bool)
        occlusion_obj_masks = []
        xs = self.voxel_x+0.5 - self.camera_loc[0]/self.resol[0]
        ys = self.voxel_y+0.5 - self.camera_loc[1]/self.resol[1]
        # norm_vec = np.sqrt(xs ** 2 + ys ** 2)
        # norm_x = xs / norm_vec
        # norm_y = ys / norm_vec
        grid_angles = np.arctan2(ys, xs)  # the angle of the ray from cam to each cell
        total_cone_mask = np.zeros(self.occlusion.shape).astype(bool)
        for i in range(len(objects)):
            angle1 = angles[i][0]
            angle2 = angles[i][1]
            # filter the grid to obtain where the grid lies in angles
            # NOTE: we use center point for approximation
            cone_mask = (grid_angles >= angle1) & (grid_angles <= angle2)
            total_cone_mask = total_cone_mask | cone_mask

            occlusion_obj_i = np.zeros(self.occlusion.shape).astype(bool)
            # extract the ones that have intersections with the object
            if objects[i]['type'] == 'circle':
                # solve the quadratic equation
                A = xs **2 + ys**2
                B = 2*( xs * (self.camera_loc[0] - objects[i]['loc'][0])/self.resol[0] + \
                        ys * (self.camera_loc[1] - objects[i]['loc'][1])/self.resol[1])
                C = ((self.camera_loc[0]-objects[i]['loc'][0])/self.resol[0])**2 + \
                    ((self.camera_loc[1]-objects[i]['loc'][1])/self.resol[1])**2-(objects[i]['radius']/self.resol[0])**2

                sol_mask = (B**2 - 4*A*C >= 0)
                sol1 = (-B-np.sqrt(B**2-4*A*C))/2/A
                sol2 = (-B+np.sqrt(B**2-4*A*C))/2/A
                # intersect_mask = sol_mask

                intersect_mask = sol_mask & (((sol1 > 0) & (sol1 <= 1)) | ((sol2 > 0) & (sol2 <= 1)))
            occlusion = occlusion | (cone_mask & intersect_mask)
            occlusion_obj_i = cone_mask & intersect_mask

            occlusion_obj_masks.append(occlusion_obj_i)
        occlusion_obj_masks = np.array(occlusion_obj_masks)
        return occlusion, total_cone_mask, occlusion_obj_masks

    def intersect(self, ray):
        # given rays: 2, obtain the intersection maps for the occlusion grids
        # a ray is represented by: cam_pos + (dx, dy) * alpha
        # each ray vector represents the (dx, dy)

        # use resolution
        ray = ray / self.resol

        intersect = np.zeros(self.occlusion.shape).astype(bool)
        # for line: y = y0, x in (x0, x0+1)
        # not working for parallel lines
        # failed_mask = (self.voxel_y == ray[1])
        alphas = (self.voxel_y - self.camera_loc[1]/self.resol[1]) / ray[1]
        x = self.camera_loc[0]/self.resol[0] + alphas * ray[0]
        success_mask = (x > self.voxel_x) & (x < self.voxel_x+1)
        
        # for line: y = y0+1, x in (x0, x0+1)
        alphas = (self.voxel_y+1 - self.camera_loc[1]/self.resol[1]) / ray[1]
        x = self.camera_loc[0]/self.resol[0] + alphas * ray[0]
        success_mask = success_mask | ((x > self.voxel_x) & (x < self.voxel_x+1))

        # for line: x = x0, y in (y0, y0+1)
        alphas = (self.voxel_x - self.camera_loc[0]/self.resol[0]) / ray[0]
        y = self.camera_loc[1]/self.resol[1] + alphas * ray[1]
        success_mask = success_mask | ((y > self.voxel_y) & (y < self.voxel_y+1))

        # for line: x = x0+1, y in (y0, y0+1)
        alphas = (self.voxel_x+1 - self.camera_loc[0]/self.resol[0]) / ray[0]
        y = self.camera_loc[1]/self.resol[1] + alphas * ray[1]
        success_mask = success_mask | ((y > self.voxel_y) & (y < self.voxel_y+1))
        return success_mask

    def extract_surface(self, mask):
        # extract the surface which is the boundary between 0 and 1
        indices_i, indices_j = np.indices((len(mask), len(mask)))
        indices_i[0,:] = 1
        indices_i[-1,:] = len(mask)-2
        indices_j[:,0] = 1
        indices_j[:,-1] = len(mask[0])-2
        diff_mask = (mask[indices_i-1, indices_j-1] != mask[indices_i, indices_j])
        diff_mask = diff_mask | (mask[indices_i-1, indices_j] != mask[indices_i, indices_j])
        diff_mask = diff_mask | (mask[indices_i-1, indices_j+1] != mask[indices_i, indices_j])
        diff_mask = diff_mask | (mask[indices_i, indices_j-1] != mask[indices_i, indices_j])
        diff_mask = diff_mask | (mask[indices_i, indices_j+1] != mask[indices_i, indices_j])
        diff_mask = diff_mask | (mask[indices_i+1, indices_j-1] != mask[indices_i, indices_j])
        diff_mask = diff_mask | (mask[indices_i+1, indices_j] != mask[indices_i, indices_j])
        diff_mask = diff_mask | (mask[indices_i+1, indices_j+1] != mask[indices_i, indices_j])
        return diff_mask

    def check_object_occlusion(self, objects, occlusion, cone_mask):
        """
        given a list of object locations, and the occlusion region, check whether the objects complete remain in the occlusion
        area.
        We define occlusion to be a volume, and the visible region should include the surface which is the boundary between
        occlusion and visible regions. It should also exclude the cone, since the behavior is invalid in depth sensor.
        """
        # checking if the visible region has at least one grid that is inside the object
        res = []  # 1 for hidden, 0 for seen
        cone_mask = self.extract_surface(cone_mask)
        surface_mask = self.extract_surface(occlusion)
        vis_surface_mask = surface_mask & (~cone_mask)
        visibility_map = (~(occlusion & (~vis_surface_mask)))
        for obj in objects:
            # obtain the surface: the boundary between visible and occlusion, excluding the ray
            if obj['type'] == 'circle':
                center = obj['loc'] / self.resol
                radius = obj['radius'] / self.resol[0]
                # we use center of grid
                inside_mask = np.sqrt((self.voxel_x + 0.5 - center[0]) ** 2 + (self.voxel_y + 0.5 - center[1]) ** 2) <= radius
                visible_mask = inside_mask & visibility_map
            if visible_mask.sum() > 0:
                # at least one of the grids includes the circle, then the circle is seen
                res.append(0)
            else:
                res.append(1)
        print('occlusion result: ', res)
        return res, visibility_map

    def obtain_occupancy(self, obj):
        # obtain the occupancy area for the visible object
        if obj['type'] == 'circle':
            center = obj['loc'] / self.resol
            radius = obj['radius'] / self.resol[0]
            # we use center of grid
            inside_mask = np.sqrt((self.voxel_x + 0.5 - center[0]) ** 2 + (self.voxel_y + 0.5 - center[1]) ** 2) <= radius
        return inside_mask

    def compute_occlusion_volume(self, object_i, occlusion_obj_masks):
        return occlusion_obj_masks[object_i].sum()
    def compute_occlusion_volumes(self, object_is, occlusion_obj_masks):
        return occlusion_obj_masks[object_is].sum()

    def sense(self):
        """
        sense in the given world, give which objects are seen, and the location of the object
        NOTE: this will update the current state of the world, which is only going to be used for
        replanning.
        """
        # update the occlusion map
        occlusion, total_cone_mask, occlusion_obj_masks = self.get_occlusion(self.objects)
        self.occlusion_obj_masks = occlusion_obj_masks
        self.occlusion = occlusion
        self.total_cone_mask = total_cone_mask

        # check which object is visible
        occluded_objs, visible_grid = self.check_object_occlusion(self.objects, self.occlusion, self.total_cone_mask)
        occluded_objs = np.array(occluded_objs).astype(bool)
        # for visible objects, add the object shape to visible region as well
        vis_objs = ~occluded_objs
        self.vis_objs = self.vis_objs | vis_objs  # once an object becomes visible, we assume it's visible always
        # for visible objects, we can have access to their location
        obj_ids = []
        for obj_id in range(len(self.vis_objs)):
            if self.vis_objs[obj_id]:
                obj_ids.append(obj_id)
        self.visible_obj_ids = obj_ids
        # extract the occupied space for the visible objects, so that we have a larger "visible" area
        occupied = np.zeros(self.occlusion.shape).astype(bool)
        for i in range(len(self.objects)):
            if self.vis_objs[i]:
                occupied_i = self.obtain_occupancy(self.objects[i])
                occupied = occupied | occupied_i
        # update visibility history: union of past visible region and new visible region
        self.visible_grid = self.visible_grid | occupied
        self.visible_grid = self.visible_grid | visible_grid

        # update occlusion using the visibility map: if a place is visible in the past, we assume
        # we know it in the future too
        self.occlusion = self.occlusion & (~self.visible_grid)

    def obtain_sweep(self, obj):
        # obtain the sweeping area of the object
        inside_mask = self.obtain_occupancy(obj)
        min_x = int(self.voxel_x[inside_mask].min())
        max_x = int(self.voxel_x[inside_mask].max())
        x_range = np.arange(int(min_x), int(max_x)+1)

        xs = self.voxel_x[inside_mask].astype(int)
        ys = self.voxel_y[inside_mask].astype(int)
        y_range_vec = np.zeros(len(x_range)) + len(self.occlusion[0])
        for i in range(len(xs)):
            if ys[i] < y_range_vec[xs[i]-min_x]:
                y_range_vec[xs[i]-min_x] = ys[i]

        # max_ys = 
        y_range = np.zeros(self.occlusion.shape)
        print('min_x: ', min_x)
        print('max_x: ', max_x)
        y_range[min_x:max_x+1,:] = y_range_vec.reshape(-1,1)

        mask_x = (self.voxel_x >= min_x) & (self.voxel_x <= max_x)
        mask_y = (self.voxel_y <= y_range)
        sweep = np.zeros(self.occlusion.shape).astype(bool)
        sweep[mask_x & mask_y] = 1
        # max_ys_i = self.voxel_y[inside_mask].argmax(axis=1)
        # max_xs = self.voxel_x[inside_mask][max_ys_i]
        return sweep
    def remove_obj(self, obj_id):
        # if the object is visible and not colliding others, remove
        if self.vis_objs[obj_id] == False:
            return False
        if not self.accessibility_check(obj_id):
            return False
        print('object %d is removed.' % (obj_id))
        self.objects.pop(obj_id)
        return True
    def accessibility_check_all(self, obj_id, obj_ids):
        # check whether object is accessible
        sweep = self.obtain_sweep(self.objects[obj_id])
        for obj_i in obj_ids:
            if obj_i == obj_id:
                continue
            obj = self.objects[obj_i]
            inside_mask = self.obtain_occupancy(obj)
            if (inside_mask & sweep).sum() > 0:
                return False
        return True

    def accessibility_check(self, obj_id1, obj_id2):
        # check whether obj1 and obj2 are all accessible
        obj1 = self.objects[obj_id1]
        obj2 = self.objects[obj_id2]
        sweep1 = self.obtain_sweep(obj1)
        sweep2 = self.obtain_sweep(obj2)
        inside_mask1 = self.obtain_occupancy(obj1)
        inside_mask2 = self.obtain_occupancy(obj2)
        if (inside_mask1 & sweep2).sum() > 0 or (inside_mask2 & sweep1).sum() > 0:
            return False
        else:
            return True
    def cost_to_come(self, arrangement, visible_obj_ids):
        """
        notice that since we don't know if there are objects in the occluded area, we can't
        account for a "correct" calculation of the change of occlusion. The best we can do
        is to use the already visible objects, and assume they are the only objects in the scene.
        """
        # arrangement is a list of object id that we are removing
        # start_obj_ids is a set of all object ids
        obj_ids = [set(visible_obj_ids)]
        for i in range(len(arrangement)):
            obj_id = set(obj_ids[i])
            obj_id.remove(arrangement[i])
            obj_ids.append(obj_id)
        # obj_ids = obj_ids[1:]
        # compute the occlusion area
        occlusion_areas = []
        for i in range(len(obj_ids)):
            obj_id_list = list(obj_ids[i])
            objs = [self.objects[obj_id] for obj_id in obj_id_list]
            occlusion, _, _ = self.get_occlusion(objs)
            occlusion = occlusion.astype(float)
            occlusion_areas.append(occlusion.sum())
        # print('total occlusion: ', occlusion_areas[0])
        occlusion_diff = []
        for i in range(1, len(obj_ids)):
            occlusion_diff.append(occlusion_areas[i-1] - occlusion_areas[i])
        occlusion_areas = occlusion_diff
        occlusion_areas = np.array(occlusion_areas)

        move_times = self.move_times[arrangement]
        move_times = np.cumsum(move_times)
        return (occlusion_areas * move_times).sum(), move_times[-1]
    def cost_to_go(self, arrangement, visible_obj_ids):
        """
        notice that since we don't know if there are objects in the occluded area, we can't
        account for a "correct" calculation of the change of occlusion. The best we can do
        is to use the already visible objects, and assume they are the only objects in the scene.
        """
        # compute the total change of visiblity volume
        obj_ids = set(visible_obj_ids)
        for i in range(len(arrangement)):
            obj_ids.remove(arrangement[i])
        obj_ids = list(obj_ids)
        objs = [self.objects[obj_id] for obj_id in obj_ids]
        occlusion, _, _ = self.get_occlusion(objs)
        remaining_occlusion_area = occlusion.astype(float).sum()
        if len(obj_ids) == 0:
            # remaining occlusion area is 0
            return 0
        return remaining_occlusion_area * (self.move_times[arrangement].sum() + self.move_times[obj_ids].min())
    def cost_to_come_acc(self, obj_id, visible_obj_ids, prev_cost, prev_total_time):
        # visible_obj_ids will be the visible ones at the start
        obj_ids = list(visible_obj_ids)
        objs = [self.objects[obj_id] for obj_id in obj_ids]
        occlusion, _, _ = self.get_occlusion(objs)
        occlusion_area = occlusion.astype(float).sum()

        # after removing the object
        new_visible_obj_ids = set(visible_obj_ids)
        new_visible_obj_ids.remove(obj_id)
        obj_ids = list(new_visible_obj_ids)
        objs = [self.objects[obj_id] for obj_id in obj_ids]
        occlusion, _, _ = self.get_occlusion(objs)
        new_occlusion_area = occlusion.astype(float).sum()
        
        occlusion_diff = occlusion_area - new_occlusion_area

        # times the total time
        cost_i = occlusion_diff * (prev_total_time + self.move_times[obj_id])
        return prev_cost + cost_i, prev_total_time + self.move_times[obj_id]
    def cost_to_go_acc(self, obj_id, visible_obj_ids, prev_total_time):
        # obtain the occlusion after removing obj_id
        obj_ids = set(visible_obj_ids)
        obj_ids.remove(obj_id)
        obj_ids = list(obj_ids)
        objs = [self.objects[obj_id] for obj_id in obj_ids]
        occlusion, _, _ = self.get_occlusion(objs)
        remaining_occlusion_area = occlusion.astype(float).sum()
        if len(obj_ids) == 0:
            return 0
        return remaining_occlusion_area * (prev_total_time + self.move_times[obj_id] + self.move_times[obj_ids].min())

    def collective_utility(self, arrangement, visible_obj_ids):

        obj_ids = list(visible_obj_ids)
        objs = [self.objects[obj_id] for obj_id in obj_ids]
        occlusion, _, _ = self.get_occlusion(objs)
        occlusion_area = occlusion.astype(float).sum()

        # after removing the object
        new_visible_obj_ids = set(visible_obj_ids) - set(arrangement)
        obj_ids = list(new_visible_obj_ids)
        objs = [self.objects[obj_id] for obj_id in obj_ids]
        occlusion, _, _ = self.get_occlusion(objs)
        new_occlusion_area = occlusion.astype(float).sum()
        
        occlusion_diff = occlusion_area - new_occlusion_area

        total_time = self.move_times[arrangement].sum()

        return occlusion_diff / total_time
class RetrievalPickPlace2DVisual():
    """
    a visualizer for PickPlace2D Simple
    """
    def __init__(self):
        self.TARGET_FPS = 60
        self.TIME_STEP = 1.0 / self.TARGET_FPS
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 960, 1280
        self.PPM = self.SCREEN_WIDTH
        # --- pygame setup ---
        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), 0, 32)
        pygame.display.set_caption('Simple pygame example')
        self.clock = pygame.time.Clock()

    def display_setup(self, world):
        # setup the view
        self.screen.fill((255, 255, 255, 255))
        # fill the workspace with orange color
        left = 0
        top = self.SCREEN_HEIGHT-self.SCREEN_HEIGHT
        width = self.SCREEN_WIDTH
        height = self.PPM * world.world_h
        pygame.draw.rect(self.screen, (240,230,140), (left, top, width, height))

        # workspace boundary
        pygame.draw.line(self.screen, (150, 75, 0, 255), (0, world.world_h*self.PPM),\
                     (self.SCREEN_WIDTH, world.world_h*self.PPM))
        # draw camera
        pygame.draw.circle(surface=self.screen, \
                center=[world.camera_loc[0]*self.PPM, height-world.camera_loc[1]*self.PPM], \
                color=(255,0,0), radius=5)

    def display_state(self, world):
        # display the current state of the world
        height = self.PPM * world.world_h

        for obj in world.objects:
            if obj['type'] == 'circle':
                pygame.draw.circle(surface=self.screen, 
                    center=[obj['loc'][0]*self.PPM, height-obj['loc'][1]*self.PPM], 
                    color=(0,0,255), radius=obj['radius']*self.PPM)
            elif obj['type'] == 'polygon':
                # pygame.draw.polygon(screen, colors[body.type], vertices)
                pass
        
        # draw occlusion
        for i in range(len(world.occlusion)):
            for j in range(len(world.occlusion[0])):
                if world.occlusion[i,j]:
                    pygame.draw.rect(self.screen, (255,255,255), \
                        (i*world.resol[0]*self.PPM, height-(j+1)*world.resol[1]*self.PPM,self.PPM,self.PPM))
                
    
    def display_occlusion(self, world, objects, occlusion):
        # display the occlusion due to objects
        height = self.PPM * world.world_h

        for obj in objects:
            if obj['type'] == 'circle':
                pygame.draw.circle(surface=self.screen, 
                    center=[obj['loc'][0]*self.PPM, height-obj['loc'][1]*self.PPM], 
                    color=(0,0,255), radius=obj['radius']*self.PPM)            
            elif obj['type'] == 'polygon':
                # pygame.draw.polygon(screen, colors[body.type], vertices)
                pass
        # draw occlusion
        for i in range(len(occlusion)):
            for j in range(len(occlusion[0])):
                if occlusion[i,j]:
                    pygame.draw.rect(self.screen, (255,255,255), \
                        (i*world.resol[0]*self.PPM,  height-(j+1)*world.resol[1]*self.PPM,
                        world.resol[0]*self.PPM,world.resol[1]*self.PPM))
                
    
    def display_update(self):
        pygame.display.flip()
        self.clock.tick(self.TARGET_FPS)



if __name__ == '__main__':

    world = RetrievalPickPlace2D()
    world.setup()
    occlusion, cone_mask, _ = world.get_occlusion(world.objects)
    print('occlusion: ')
    print(occlusion)
    sweep = world.obtain_sweep(world.objects[0])

    vis = RetrievalPickPlace2DVisual()
    world.sense()
    world.move_times = np.array([0., 1.])
    cost = world.cost_to_come([1,0], [0,1])
    print('world cost: ', cost)
    cost_to_go = world.cost_to_go([1], [0,1])
    print('cost to go: ', cost_to_go)
    # --- main game loop ---
    running = True
    while running:
        # Check the event queue
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                # The user closed the window or pressed escape
                running = False
            if event.type == KEYDOWN:
                world.objects.pop(0)
                world.sense()
        vis.display_setup(world)
        vis.display_occlusion(world, world.objects, world.occlusion)
        # res, _  = world.check_object_occlusion(world.objects, world.occlusion, cone_mask)
        vis.display_update()

    pygame.quit()