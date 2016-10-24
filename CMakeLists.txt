project(Clust)
cmake_minimum_required(VERSION 3.0)

find_package(OpenCV REQUIRED)
include(mini-caffe/mini-caffe.cmake)

set(BOOST_DIR $ENV{BOOST_DIR})
SET(CMAKE_LIBRARY_PATH ${BOOST_DIR}/stage/lib)

file(GLOB HEADERS ARecog/*.h*)
file(GLOB SOURCES src/*.c* newClust.cpp)
set(ALL_FILES ${HEADERS} ${SOURCES})

add_executable(${PROJECT_NAME} ${ALL_FILES})

target_link_libraries(${PROJECT_NAME} libcaffe ${OpenCV_LIBS} ${Boost_LIBS})