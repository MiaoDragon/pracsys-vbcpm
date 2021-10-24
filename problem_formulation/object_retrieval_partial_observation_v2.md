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
        task plan pi
        robot sequence tau from pi
    such that:
        target object has a known pose, and is retrieved
        objects remain in the workspace
        robot doesn't collide with the workspace
    objective:
        minimize the total time of execution, the number of objects that are moved, and trajectory length
```

Task Plan definition:
=================================================================================================================
A task plan is defined as (a, o, x1, x2)
which contain an action a, a set of objects o, start pose x1, and goal pose x2
In essense, it transforms objects o from poses x1 to poses x2

---------------------------------------------------------------------------------------------------
Another definition can be:
(a, predicates)
where predicates are conditions that need to be True after the action takes place.
One example is to push an object so that the target object becomes visible.
Another example is to grasp an objec so that the target object brcomes accessible.

This has the advantage of giving more freedom for the positions to place to objects, since most of the time
we don't care where the objects are actually moved, as long as the conditions are valid.
=================================================================================================================


Action Defintion:
=================================================================================================================
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



Visibility:
=================================================================================================================
Visibility is defined given a threshold r.
An object becomes visible when more than r of the object becomes visible. (i.e., less than 1-r is in occlusion)

Another definition is when the uncertainty of the perception module of the object is less than a threshold.
=================================================================================================================
