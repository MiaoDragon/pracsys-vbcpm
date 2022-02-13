all: ./baxter ./baxter_common ./baxter_tools ./baxter_examples ./baxter_interface ./baxter_simulator ./moveit_resources ./pysdf ./planit# ./moveit_grasps # ./timed_roslaunch ./sim_ros_interface ./moveit_robots

./baxter:
	git clone https://github.com/RethinkRobotics/baxter.git $@

./baxter_common:
	git clone https://github.com/RethinkRobotics/baxter_common.git $@

./baxter_tools:
	git clone https://github.com/DanManN/baxter_tools.git $@

./baxter_examples:
	git clone https://github.com/DanManN/baxter_examples.git $@

./baxter_interface:
	git clone https://github.com/DanManN/baxter_interface.git $@

./baxter_simulator:
	git clone https://github.com/DanManN/baxter_simulator.git $@

./moveit_resources:
	git clone https://github.com/ros-planning/moveit_resources.git $@

./pysdf:
	git clone https://github.com/DanManN/pysdf.git $@

./planit:
	git clone https://github.com/DanManN/planning_baxter.git
	mv ./planning_baxter/src/planit .
	mv ./planning_baxter/src/baxter_moveit .
	rm -rf ./planning_baxter

# ./moveit_grasps:
#         git clone -b melodic-devel https://github.com/ros-planning/moveit_grasps.git $@

# ./timed_roslaunch:
#         git clone https://github.com/MoriKen254/timed_roslaunch.git $@

# ./sim_ros_interface:
#         git clone --branch coppeliasim-v4.1.0 --recursive https://github.com/CoppeliaRobotics/simExtROSInterface.git $@

# ./moveit_robots:
#         https://github.com/ros-planning/moveit_robots.git $@

deps:
	./install_deps.sh

clean:
	rm -rf ./baxter ./baxter_common ./baxter_tools ./baxter_examples ./baxter_interface ./baxter_simulator ./moveit_resources ./pysdf ./planit ./baxter_planit ./baxter_moveit
