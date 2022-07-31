# pracsys-vbcpm: Vision-Based Constrained (Not Yet) Manipulation
TO run experiments, first open a terminal and go to folder motoman_moveit_config/launch/, and run
```
roslaunch move_group.launch
```
to launch the moveit motion planning server.
open another terminal and go to folder vbcpm_integrated_system/scripts/ and run
```
python execution_system.py [0/1] [problem name] [y/n]
```
where first 0/1 indicates whether to load previously saved problem instances (1) or not (0). Some saved problem instances have been provided in the same folder with ".pkl" extension. "y/n" indicates whether to use simple geometry (y) or to use YCB object (n).
YCB objects will be uploaded later.
Afterwards, open another terminal and go to folder vbcpm_integrated_system/scripts, and run
```
python task_planner.py [problem name] [trial num]
```
to run our algorithm. Or run
```
python task_planner_random.py [problem name] [trial num] [0/1] [timeout] [num_obj]
```
to select random algorithm (0 for [0/1]) or greedy algorithm (1 for [0/1]) as baseline to run.
