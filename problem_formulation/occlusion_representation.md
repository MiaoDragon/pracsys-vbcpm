There are several ways to represent the occlusion. To represent the occlusion, we need the following functionalities:
- check whther object A occludes object B, and how much it occludes
- obtain the region where object A is occluding, and the associated volumes. This is useful for inference on how much to move the object

The following are some of the proposals for different representations:

## Method 1
The occlusion can be written as:
$$sO_{cc}=\{\frac{x-c}{||x-c||_2}\alpha, ||x-c||_2\leq\alpha\leq\alpha_{max}, (\frac{x-c}{||x-c||_2}\alpha)[2]\geq0\}$$
Here c represents the camera center position, and each x is a point in the object.
This basically means we shoot a ray from the camera to the object until the ray intersects with the ground, or reaches
some max limit.
Exact representation of the object as a set of point {x} is often not possible, unless the object is known to have some
geometry shapes such as cubes or spheres (which form equations). To represent the object approximately, here are some of
the possible ways:
- voxelize the object. Then each object is decomposed into a number of cubes.
  With this approximation, we can easily check if a ray intersects with the object by checking the intersection with the
  voxels.
  However, the downside of this is that we have to loop over each of the ray from camera to the target object to check
  intersections, which can be time consuming.
  Another downside is that it's hard to approximate the volume of the occlusion region, since the voxel shape can be arbitrary. But there might be some **computer graphics** methods to achieve that.
  Another potential benefit is that this method may allow the computations to be **differentiable**, which can ease the optimization of trajectories.
- use point cloud. 
- This is more accurate to represent the object shape, but can be computationally challenging.

## Method 2
Voxelize the workspace, and directly obtain the occlusion region represented in the voxels. After obtaining the occlusion
region, the occluded-relation can be checked by seeing if the object voxel intersects with the occlusion voxel.
In this method, object can be represented as PCD, and then converted to voxels in the workspace voxel.
This method is easy when we have a depth image and want to generate the occlusion region using that. However, it is not easy to know which occlusion region corresponds to which object in this case. The following are some of the ways:
- for each object given the known pose, generate a depth image of it. Then use the depth image to check occlusion region.
The downside of this method is that it is not differentiable, since we need to first generate a depth image, and then
obtain the occlusion area.
However, the advantage is that we can easily compute the volume of the occlusion directly from the voxels.
- for each object pcd, convert that to voxel in workspace. Then for each pcd of target object, check the ray from camera
  to that object if it intersects with the object voxel. If so, then there is occlusion.

Using Method 2, we can obtain the occlusion associated with each object, and can label the occluded region by them. Hence
from this information, we have the region-object relationship. This tells us to reveal a certain region, which object(s) we should move to achieve it.


## Other Method
Another Method of choice is to use a decision tree to represent the occlusion. Each upper level in the decision tree tells us the location of the occlusion at a more coarse place, and then we can get the finer level location by going down the decision tree. This has the advantage of telling us where approximately we need to move the object, instead of giving us a accurate value.


## Keypoint and Mode Method
The occlusion region is sufficient to define based on the keypoint (which maps to the outmost occlusion points) and the plane. Hence If the keypoints can be identified, the boundary of the occlusion can be specified. Then any motion of hte keypoints can transfer to the occlusion analytically, based on the height of the keypoint.
This is valid until any mode change, which means a change of the keypoint.

However, notice that the occlusion region includes not only the occlusion intersection with the plane, but also the points in the middle between the object and the occlusion.


## Approximation by the Shadow Region

---------------------------------------------------------------------------------------------------------------------------
# What does the region-to-reveal tell us about which object to move? and to which location(s)?

We can use the closest boundary point to tell us the relative position we should move. Since to reveal the target region, we
basically need to move the object so that the target region becomes at the boundary of the occlusion region. Hence we need to move the boundary to the target region.
We assume the boundary point is the intersection points with the plane. Then the boundary point motion gives us a 2D vector,
which also can transfer to the keypoint on the object (which sheds light at the boundary point).
Hence by applying the 2D motion, as long as the mode of the occlusion doesn't change, in most cases the target region can be revealed.

What this means is we can define a policy:
reveal(point) -> object translation vector


Another way is to find the keypoints that map to the outmost occlusion. The motion of these keypoints then corresponds to the motion of the occlusion.
Then 2D translation becomes the motion of all the occlusion points with some ratio of the translated distance (the ratio is proportional to the object height)
The 2D rotation can also be mapped to the motion of the occlusion points by a rotation.
This is valid until any mode change.


---------------------------------------------------------------------------------------------------------------------------
# visibility-constraint graph

## Method 1: Object-To-Object
We can set a constraint graph representing the visibility of objects. However we need to notice that in reality the visibility relations are unknown for the target object, since it is hidden by others.
However, we can set a region-based visibility constraint graph. In this graph, each object is hiding a region, and the regions might overlap.
A->B if object A is hiding object B

## Method 2: Pose
There is also a region-to-region visibility constraint graph, where each node represents the pose. 
Then two nodes have edge
A->B with tag o1,
if object o1 at pose A will hide region B

A, o1 -> B, o2
if objec o1 at pose A will hide object o2 at pose B

## Method 3: 3D Point to 3D Point
Another representation is 3D point to 3D point:
A->B with tag of camera pose,
if occupation of a point at A will hide point at B.
This means that the visibility constraint can be precomputed.

# Method 4: Object, Pose -> Region
[objects], [poses] -> region where the objects are occluding
