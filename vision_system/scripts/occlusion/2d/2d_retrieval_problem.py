#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
An attempt at some simple, self-contained pygame-based examples.

Example 01

In short:
One static body: a big polygon to represent the ground
One dynamic body: a rotated big polygon
And some drawing code to get you going.

kne
"""
import pygame
from pygame.locals import (QUIT, KEYDOWN, K_ESCAPE)

import Box2D  # The main library
# Box2D.b2 maps Box2D.b2Vec2 to vec2 (and so on)
from Box2D.b2 import (world, polygonShape, circleShape, edgeShape, staticBody, dynamicBody)
import numpy as np
# --- constants ---
# Box2D deals with meters, but we want to display pixels,
# so define a conversion factor:
PPM = 20.0  # pixels per meter
TARGET_FPS = 60
TIME_STEP = 1.0 / TARGET_FPS
SCREEN_WIDTH, SCREEN_HEIGHT = 1280, 960

# --- pygame setup ---
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)
pygame.display.set_caption('Simple pygame example')
clock = pygame.time.Clock()

# --- pybox2d world setup ---
# Create the world
world = world(gravity=(0, 0), doSleep=True)  # 0 gravity since we are top-down view

# horizontal line between workspace and camera
workspace_lower = SCREEN_HEIGHT * 0.1


# static body for the camera
camera_body = world.CreateStaticBody(
    position=(SCREEN_WIDTH/PPM/2, 0.08 * SCREEN_HEIGHT/PPM),
    shapes=circleShape(radius=1),)

# dynamic body for robot link
link_length = np.sqrt((SCREEN_WIDTH/2/PPM)**2 + (SCREEN_HEIGHT/PPM)**2)
link_length = 10#link_length / 2
link_width = 0.01 * SCREEN_HEIGHT / PPM
link1 = world.CreateDynamicBody(position=(SCREEN_WIDTH/2/PPM-link_length/2,0.04*SCREEN_HEIGHT/PPM))
link1.CreatePolygonFixture(box=(link_length/2, link_width/2), density=0.01, friction=10.)
link2 = world.CreateDynamicBody(position=(SCREEN_WIDTH/2/PPM-link_length-link_length/2,0.04*SCREEN_HEIGHT/PPM))
link2.CreatePolygonFixture(box=(link_length/2, link_width/2), density=0.01, friction=10.)


# link1 = world.CreateDynamicBody(shapes=edgeShape(vertices=[(SCREEN_WIDTH/PPM/2, 0.04*SCREEN_HEIGHT/PPM), \
#                                                             (SCREEN_WIDTH/PPM/2-link_length, 0.04*SCREEN_HEIGHT/PPM)]),
#                                 position=(0,0))
# link2 = world.CreateDynamicBody(shapes=edgeShape(vertices=[(SCREEN_WIDTH/PPM/2-link_length, 0.04*SCREEN_HEIGHT/PPM), \
#                                                             (SCREEN_WIDTH/PPM/2-2*link_length, 0.04*SCREEN_HEIGHT/PPM)]),
#                                 position=(0,0))
robot_fix = world.CreateStaticBody(shapes=circleShape(radius=1),
                                   position=(SCREEN_WIDTH/PPM/2, 0.04*SCREEN_HEIGHT/PPM))
joint1 = world.CreateRevoluteJoint(bodyA=robot_fix, bodyB=link1, anchor=robot_fix.worldCenter, 
                                    maxMotorTorque = 100.0,
                                    # motorSpeed = 0.0,
                                    # enableMotor = True,
                                    collideConnected=False)
joint1.motorEnabled = True
joint2 = world.CreateRevoluteJoint(bodyA=link1, bodyB=link2, anchor=(SCREEN_WIDTH/PPM/2-link_length, 0.04*SCREEN_HEIGHT/PPM),
                                    maxMotorTorque = 100.0,
                                    # motorSpeed = 0.0,
                                    # enableMotor = True,
                                    collideConnected=False)
joint2.motorEnabled = True

# Create a dynamic body
dynamic_body = world.CreateDynamicBody(position=(10, 15), angle=0)

# And add a box fixture onto it (with a nonzero density, so it will move)
box = dynamic_body.CreatePolygonFixture(box=(2, 1), density=1, friction=0.3)

colors = {
    staticBody: (255, 0, 0, 255),
    dynamicBody: (127, 127, 127, 255)
}

# --- main game loop ---
running = True
while running:
    # Check the event queue
    for event in pygame.event.get():
        if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
            # The user closed the window or pressed escape
            running = False

    screen.fill((255, 255, 255, 255))
    # fill the workspace with orange color
    left = 0
    top = SCREEN_HEIGHT - SCREEN_HEIGHT
    width = SCREEN_WIDTH
    height = 0.9 * SCREEN_HEIGHT
    pygame.draw.rect(screen, (240,230,140), (left, top, width, height))
    # Draw the world
    for body in [dynamic_body]:  # or: world.bodies
        # The body gives us the position and angle of its shapes
        for fixture in body.fixtures:
            # The fixture holds information like density and friction,
            # and also the shape.
            shape = fixture.shape

            # Naively assume that this is a polygon shape. (not good normally!)
            # We take the body's transform and multiply it with each
            # vertex, and then convert from meters to pixels with the scale
            # factor.
            vertices = [(body.transform * v) * PPM for v in shape.vertices]

            # But wait! It's upside-down! Pygame and Box2D orient their
            # axes in different ways. Box2D is just like how you learned
            # in high school, with positive x and y directions going
            # right and up. Pygame, on the other hand, increases in the
            # right and downward directions. This means we must flip
            # the y components.
            vertices = [(v[0], SCREEN_HEIGHT - v[1]) for v in vertices]

            pygame.draw.polygon(screen, colors[body.type], vertices)
    for body in [camera_body]:
        for fixture in body.fixtures:
            shape = fixture.shape
            pos = body.transform * shape.pos * PPM
            pos = [pos[0], SCREEN_HEIGHT - pos[1]]
            
            pygame.draw.circle(surface=screen, center=pos, color=colors[body.type], radius=5)

    # draw robot
    for body in [link1, link2]:
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
            vertices = [(body.transform * v) * PPM for v in shape.vertices]

            # But wait! It's upside-down! Pygame and Box2D orient their
            # axes in different ways. Box2D is just like how you learned
            # in high school, with positive x and y directions going
            # right and up. Pygame, on the other hand, increases in the
            # right and downward directions. This means we must flip
            # the y components.
            vertices = [(v[0], SCREEN_HEIGHT - v[1]) for v in vertices]
            # pygame.draw.line(screen, [0, 0, 0], vertices[0], vertices[1])
            pygame.draw.polygon(screen, [0,0,0], vertices)

    for body in [joint1, joint2]:
        # The body gives us the position and angle of its shapes
        # print(body.anchorB)
        anchorB = body.anchorB * PPM
        # vertices = [(body.transform * v) * PPM for v in shape.vertices]

        # But wait! It's upside-down! Pygame and Box2D orient their
        # axes in different ways. Box2D is just like how you learned
        # in high school, with positive x and y directions going
        # right and up. Pygame, on the other hand, increases in the
        # right and downward directions. This means we must flip
        # the y components.
        anchorB = [anchorB[0], SCREEN_HEIGHT - anchorB[1]]
        # vertices = [(v[0], SCREEN_HEIGHT - v[1]) for v in vertices]
        # pygame.draw.line(screen, [0, 0, 0], vertices[0], vertices[1])
        pygame.draw.circle(surface=screen, center=anchorB, color=[255,0,0], radius=5)


    # workspace boundary
    pygame.draw.line(screen, (150, 75, 0, 255), (0, SCREEN_HEIGHT-workspace_lower), (SCREEN_WIDTH, SCREEN_HEIGHT-workspace_lower))


    # control robot
    joint1_target_angle = joint1.angle
    angleError = joint1_target_angle - joint1_target_angle
    print('angle1 error: ', angleError)
    gain = 1.
    joint1.motorSpeed = -gain * angleError

    print('joint2 angle: ')
    print(joint2.angle)
    joint2_target_angle = np.pi/2
    angleError = joint2.angle - joint2_target_angle
    print('angle2 error: ', angleError)
    gain = 1.
    joint2.motorSpeed = -gain * angleError


    # Make Box2D simulate the physics of our world for one step.
    # Instruct the world to perform a single step of simulation. It is
    # generally best to keep the time step and iterations fixed.
    # See the manual (Section "Simulating the World") for further discussion
    # on these parameters and their implications.
    world.Step(TIME_STEP, 10, 10)

    # Flip the screen and try to keep at the target FPS
    pygame.display.flip()
    clock.tick(TARGET_FPS)

pygame.quit()
print('Done!')