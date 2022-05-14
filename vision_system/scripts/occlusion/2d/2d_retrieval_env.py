import yaml
import pygame
from pygame.locals import (QUIT, KEYDOWN, K_ESCAPE)

import Box2D  # The main library
# Box2D.b2 maps Box2D.b2Vec2 to vec2 (and so on)
from Box2D.b2 import (world, polygonShape, circleShape, edgeShape, staticBody, dynamicBody)
import Box2D.b2 as box2Db2
import numpy as np
class Retrieval2D():
    def __init__(self):
        f = open('2d_default_configs.yaml', 'r')
        world_configs = yaml.load(f)
        print(world_configs)
        self.world_configs = world_configs
        self.world_configs['TIME_STEP'] = 1.0 / self.world_configs['TARGET_FPS']

        # set up the world
        self.screen = pygame.display.set_mode((self.world_configs['world_width'], self.world_configs['world_height']), 0, 32)
        pygame.display.set_caption('Simple pygame example')
        self.clock = pygame.time.Clock()

        # --- pybox2d world setup ---
        # Create the world
        world = box2Db2.world(gravity=(0, 0), doSleep=True)  # 0 gravity since we are top-down view

        # horizontal line between workspace and camera
        self.workspace_lower = self.world_configs['world_height'] * 0.1


        # static body for the camera
        self.camera_position = [self.world_configs['world_width']/self.world_configs['PPM']/2, \
                            0.08 * self.world_configs['world_height']/self.world_configs['PPM']]

        self.colors = {
            'camera': (255, 0, 0),
            'objects': (127, 127, 127, 255)
        }

        # robot
        link_length = np.sqrt((self.world_configs['world_width']/2/self.world_configs['PPM'])**2 + \
                    (self.world_configs['world_height']/self.world_configs['PPM'])**2)
        link_length = link_length / 2
        link_width = 0.01 * self.world_configs['world_height'] / self.world_configs['PPM']
        link1 = world.CreateDynamicBody(position=\
                    (self.world_configs['world_width']/2/self.world_configs['PPM']-link_length/2,
                    0.04*self.world_configs['world_height']/self.world_configs['PPM']))
        link1.CreatePolygonFixture(box=(link_length/2, link_width/2), density=0.0001, friction=1.)
        link2 = world.CreateDynamicBody(position=\
                (self.world_configs['world_width']/2/self.world_configs['PPM']-link_length-link_length/2,
                0.04*self.world_configs['world_height']/self.world_configs['PPM']))
        link2.CreatePolygonFixture(box=(link_length/2, link_width/2), density=0.0001, friction=1.)

        self.link_length = link_length
        self.link_width = link_width
        self.robot_links = [link1, link2]

        robot_fix = world.CreateStaticBody(shapes=circleShape(radius=1),
                                        position=(self.world_configs['world_width']/self.world_configs['PPM']/2, 
                                                0.04*self.world_configs['world_height']/self.world_configs['PPM']))
        joint1 = world.CreateRevoluteJoint(bodyA=robot_fix, bodyB=link1, anchor=robot_fix.worldCenter, 
                                            maxMotorTorque = 100.0,
                                            # motorSpeed = 0.0,
                                            # enableMotor = True,
                                            collideConnected=False)
        joint1.motorEnabled = True
        joint2 = world.CreateRevoluteJoint(bodyA=link1, bodyB=link2, 
                                            anchor=(self.world_configs['world_width']/self.world_configs['PPM']/2-link_length, 
                                                0.04*self.world_configs['world_height']/self.world_configs['PPM']),
                                            maxMotorTorque = 100.0,
                                            # motorSpeed = 0.0,
                                            # enableMotor = True,
                                            collideConnected=False)
        joint2.motorEnabled = True

        self.robot_joints = [joint1, joint2]

        self.objects = []

        self.world = world

    def load_object_config(self):
        # load the object configuration file
        pass

    def reset(self):
        pass
    def step(self):
        # step simulation

        self.screen.fill((255, 255, 255, 255))
        # fill the workspace with orange color
        left = 0
        top = self.world_configs['world_height'] - self.world_configs['world_height']
        width = self.world_configs['world_width']
        height = 0.9 * self.world_configs['world_height']
        pygame.draw.rect(self.screen, (240,230,140), (left, top, width, height))
        # Draw the world
        for body in self.objects:  # or: world.bodies
            # The body gives us the position and angle of its shapes
            for fixture in body.fixtures:
                # The fixture holds information like density and friction,
                # and also the shape.
                shape = fixture.shape

                # Naively assume that this is a polygon shape. (not good normally!)
                # We take the body's transform and multiply it with each
                # vertex, and then convert from meters to pixels with the scale
                # factor.
                vertices = [(body.transform * v) * self.world_configs['PPM'] for v in shape.vertices]

                # But wait! It's upside-down! Pygame and Box2D orient their
                # axes in different ways. Box2D is just like how you learned
                # in high school, with positive x and y directions going
                # right and up. Pygame, on the other hand, increases in the
                # right and downward directions. This means we must flip
                # the y components.
                vertices = [(v[0], self.world_configs['world_height'] - v[1]) for v in vertices]

                pygame.draw.polygon(self.screen, self.colors['objects'], vertices)

        # camera
        pos = self.camera_position
        # pos = body.transform * shape.pos * PPM
        pos = [pos[0], self.world_configs['world_height'] - pos[1]]
        
        pygame.draw.circle(surface=self.screen, center=pos, color=self.colors['camera'], radius=5)

        # draw robot
        for body in self.robot_links:
            # The body gives us the position and angle of its shapes
            for fixture in body.fixtures:
                # The fixture holds information like density and friction,
                # and also the shape.
                shape = fixture.shape

                # Naively assume that this is a polygon shape. (not good normally!)
                # We take the body's transform and multiply it with each
                # vertex, and then convert from meters to pixels with the scale
                # factor.
                # print(shape.vertices)
                vertices = [(body.transform * v) * self.world_configs['PPM'] for v in shape.vertices]

                # But wait! It's upside-down! Pygame and Box2D orient their
                # axes in different ways. Box2D is just like how you learned
                # in high school, with positive x and y directions going
                # right and up. Pygame, on the other hand, increases in the
                # right and downward directions. This means we must flip
                # the y components.
                vertices = [(v[0], self.world_configs['world_height'] - v[1]) for v in vertices]
                # pygame.draw.line(screen, [0, 0, 0], vertices[0], vertices[1])
                pygame.draw.polygon(self.screen, [0,0,0], vertices)

        for body in self.robot_joints:
            # The body gives us the position and angle of its shapes
            # print(body.anchorB)
            anchorB = body.anchorB * self.world_configs['PPM']

            # But wait! It's upside-down! Pygame and Box2D orient their
            # axes in different ways. Box2D is just like how you learned
            # in high school, with positive x and y directions going
            # right and up. Pygame, on the other hand, increases in the
            # right and downward directions. This means we must flip
            # the y components.
            anchorB = [anchorB[0], self.world_configs['world_height'] - anchorB[1]]
            # vertices = [(v[0], world_height - v[1]) for v in vertices]
            # pygame.draw.line(screen, [0, 0, 0], vertices[0], vertices[1])
            pygame.draw.circle(surface=self.screen, center=anchorB, color=[255,0,0], radius=5)


        # workspace boundary
        pygame.draw.line(self.screen, (150, 75, 0, 255), (0, self.world_configs['world_height']-self.workspace_lower), 
                        (self.world_configs['world_width'], self.world_configs['world_height']-self.workspace_lower))


        # control robot
        joint1, joint2 = self.robot_joints
        joint1_target_angle = 0#joint1.angle
        angleError = joint1.angle - joint1_target_angle
        print('angle1 error: ', angleError)
        gain = 1.
        joint1.motorSpeed = -gain * angleError

        print('joint2 angle: ')
        print(joint2.angle)
        joint2_target_angle = np.pi/2
        angleError = joint2.angle - joint2_target_angle
        print('angle2 error: ', angleError)
        gain = 1
        joint2.motorSpeed = -gain * angleError


        # Make Box2D simulate the physics of our world for one step.
        # Instruct the world to perform a single step of simulation. It is
        # generally best to keep the time step and iterations fixed.
        # See the manual (Section "Simulating the World") for further discussion
        # on these parameters and their implications.
        self.world.Step(self.world_configs['TIME_STEP'], 10, 10)

        # Flip the screen and try to keep at the target FPS
        pygame.display.flip()
        self.clock.tick(self.world_configs['TARGET_FPS'])
    def sense(self):
        pass
    def execute(self, traj):
        # execute a trajectory
        pass





retrieval2d = Retrieval2D()

# --- main game loop ---
running = True
while running:
    # Check the event queue
    for event in pygame.event.get():
        if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
            # The user closed the window or pressed escape
            running = False
    retrieval2d.step()
