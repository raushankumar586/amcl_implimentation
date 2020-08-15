cmake_minimum_required(VERSION 3.1.3)

project(particle_filter_localisation)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

find_package(catkin REQUIRED COMPONENTS
        roscpp
        sensor_msgs
        geometry_msgs
        nav_msgs
        tf2_ros
        )

catkin_package()

add_executable(particle_filter_localisation
        src/particle_filter_localisation.cpp
        )

target_include_directories(particle_filter_localisation SYSTEM PRIVATE ${OpenCV_INCLUDE_DIRS} ${catkin_INCLUDE_DIRS})
target_include_directories(particle_filter_localisation PRIVATE include)

target_link_libraries(particle_filter_localisation ${OpenCV_LIBS} ${catkin_LIBRARIES})

