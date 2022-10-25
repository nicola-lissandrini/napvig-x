cmake_minimum_required (VERSION 3.0.2)

project (napvig)

add_compile_options(-std=c++17)
add_compile_options (-ffast-math)
add_compile_options(-Wno-pmf-conversions)

set(Torch_DIR "/opt/libtorch/share/cmake/Torch")

find_package (catkin REQUIRED COMPONENTS
	nlib
	roscpp
	rospy
	std_msgs
)

find_package (Torch REQUIRED)

catkin_package (
	INCLUDE_DIRS include ${TORCH_INCLUDE_DIRS}
	CATKIN_DEPENDS nlib
)

include_directories (
	include
	${TORCH_INCLUDE_DIRS}
	${catkin_INCLUDE_DIRS}
)

add_executable (napvig
	src/napvig_node.cpp
	src/napvig_modflow.cpp
	src/napvig_x.cpp
	src/napvig.cpp
	src/landscape.cpp
	src/ros_conversions.cpp
	src/policy/policy.cpp
	src/policy/legacy.cpp
	src/policy/predictive.cpp
	src/policy/exploitative.cpp
        src/policy/explorative.cpp
)
target_link_libraries (napvig dl ${catkin_LIBRARIES} ${TORCH_LIBRARIES} /usr/local/lib/liblietorch.so )
