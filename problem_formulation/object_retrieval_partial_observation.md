# This defines a object retrieval problem given partial observation

```
elements: robot, [object_i], target_object, shelf, sensor
problem:
    given:
        object_i at pose_i for all i
        target_object at target_pose
        robot at pose (relative to sensor)
        sensor input
    find:
        robot sequences tau (including actions grasp and push)
        sensor sequence z
    such that:
        target_object(z) has a known pose
        target_object.propagate(tau).endpoint is in robot hand
        object[i].propagate(tau) is stable and within shelf, for all i
        sweep(robot, tau) does not collide with the shelf
```

A robot sequence $$\tau$$ is defined to be several levels, each lower level takes care of
details that the levels above lack. Going down the levels, the plan transits from task planning
to pure motion and control planning. The upper levels introduce domain knowledge and heuristics,
but helps solve the problem more efficiently.
In this problem, we define different levels of planning as follows:

- cosntraint removal: remove visibility constraint, accessibility constraint, or both
- object-level inference: which object(s) is going to move where? (rough locations vs. accurate location)
- action: grasping, pushing, sensing
- trajectory of choice

Ths hierarhical formulation of the problem follows the TAMP literature, and LGP formulation, where rough
snapshot/skeleton is first planned at higher level, and lower-level trajectory is later planned.
The top-down flow prunes the search space of the trajectory to make planning more efficient, while the bottom-up
flow gives information that reality requires.


=================================================================================================================
Below describes the constraint-level representation:

The given problem instance defines constraints among objects, which forms a constraint graph. Some objects in the scene
might block/occlude the target object, while others might make it unaccessible for grasping.
Once we have the constraint graph, we can then determine which constraint to remove from the scene.

-------------------------------------------------------------------------------------------------------------
object-level inference:

Once we know which constraint and which object to move, we can make an inference on where to move the objects.
An object that occludes the target object needs to move to somewhere that maximially removes the occlusion area.
Meanwhile, an object that makes the target unaccessible needs to move so that the target is reachable.

This module should output the object moving sequence in the format:
([objects], [target_poses])
OR
([objects], [target_regions])

Notice that the second format relaxes the constraint of the motion, since objects sometimes don't need to go to an 
accurate pose. Instead, some rough estimate, or a set definition is good enough.

-------------------------------------------------------------------------------------------------------------
action inference:

Once the object pose sequence is obtained, we need to consider the right actions to achieve that. Notice that 
grasping and pushing achieve totally different effects.
Grasping acheives accurate position of the grasped object, and can also push the other objects away. It requires
the grasped object to be accessible when grasping, hence a separate consideration of the accessibiltiy constraint
is necessary.
Pushing achieves non-accurate pose transform of a set of objects. This has the benefit of using less time and less
trajectory length. However, there is a requirement on the pose of pushed objects, since pushing relies on contact
between the arm and the objects.

-------------------------------------------------------------------------------------------------------------
Below describes the trajectory-level representation:

```
#############################################################################
- Basic Version:
robot sequence tau is a list of trajectories with an action tag:
[action_i: trajectory_i]

If action_i=grasp, then at the end of the trajectory, the end effector will be closed
If action_i=ungrasp, then at the end of the trajectory, the end effector will be open


trajectory is defined to be a mapping from time to joint angles:
t -> [joint_angle_i]

An instance of such a trajectory can be a piecewise linear trajectory:
piecewise linear trajectory: [(joint_angle_i, time_i)]

#############################################################################
- Optimization Version:
robot sequence tau is a list of trajectories defined by:
    - trajectory constraint (e.g. tracking a certain trajectory, or making sure an object moves along a trajectory)
    - startpoint constraint, endpoint constraint (e.g. reach the grasp pose)
    - endpoint action

#############################################################################

```
=================================================================================================================

Task and Motion Plan Formulation:
=================================================================================================================
Task Plan:

which object needs to be moved to where? Using what actions?

```
grasp(o, x1, x2, o', x1', x2'):
grasp object o from pose x1 to pose x2, while pushing objects o' from poses x1' to poses x2'.

Pre-Condition:
object o is graspable/accessible. Pushed objects o' satisfy geometric constraints for being pushed.

Effect:
object o is grasped from x1 to x2. Objects o' are pushed from x1' to x2'
```

```
push(o, x1, x2):
push objects o from poses x1 to poses x2

Pre-Condition:
Pushed objects o satisfy geometric constraints for being pushed.

Effect:
objects o are pushed from x1 to x2.
```

```
sense():
sense the scene to update the occlusion information

Effect:
unknown regions are revealed.
```

=================================================================================================================


Below describes the propagation method for single object:
```
object.propagate(tau):

given:
    initial object pose s
    object geometry O
output:
    a trajectory of the object:
        t -> pose
implementation:
    from tau, find the first contact point at time t such that:
        robot(tau(t)) is in contact with object, at contact point p
    if the object is not grasped by the manipulator: 
        object will move with the arm by following Newton's Law
            # Note:
            1. can approximate this process by assuming table has high friction,
            and thus the object directly stop once the contact is over.

            2. need to consider the rotational torque around the contact point. The
            motion can be decoupled into the torque around the contact point, and
            the translation.
            Assuming zero friction bewteen the robot and the object, then sliding
            doesn't change the orientation of the object.
            The the contact point relative to the object will change only due to
            the rotation of the arm relative to the contact point.
            The contact point will move in the same distance relative to the robot link and the object.
    if the object is grasped by the manipulator:
        object will stay stationary with the gripper

implementations of motion:
    alternatives:
    1. can be implemented by a simulator (which is still a rough estimate, and might be different from real world)
    2. can be implemented by a simplified analytical model (rough estimate)
    3. with the rough estimate, we can assume the motion is noisy with some noise model
```
Below describes the propagation method for multiple objects:

```
objects.propagate(tau):

given:
    initial poses [si]
    object geometry [oi]
output:
    trajectories for the objects
implementation:
    find first contact object: -> t, object_i at pi
    moving_object_graph.append(object_i)  # link between two nodes has the condition: they are in contact with each other
    when moving_object_graph hits another object, add that to the graph

```

