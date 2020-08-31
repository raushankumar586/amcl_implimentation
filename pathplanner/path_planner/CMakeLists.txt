cmake_minimum_required(VERSION 3.1.3)

project(path_planner)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV 3 REQUIRED)

find_package(catkin REQUIRED COMPONENTS
        roscpp
        roslib
        nav_msgs
        geometry_msgs
        )

catkin_package()

add_executable(path_planner
        src/astar.cpp
        )

add_dependencies(path_planner ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

target_include_directories(path_planner SYSTEM PRIVATE ${OpenCV_INCLUDE_DIRS} ${catkin_INCLUDE_DIRS})
target_include_directories(path_planner PRIVATE include)

target_link_libraries(path_planner ${OpenCV_LIBS} ${catkin_LIBRARIES})
