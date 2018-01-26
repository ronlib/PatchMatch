cmake_minimum_required(VERSION 2.8)
project(PatchMatch)
find_package(OpenCV REQUIRED)
# set(OpenCV_INCLUDE_DIRS "/usr/include/opencv/")
# set(OpenCV_LIBS "/usr/lib")



include_directories(${INCLUDE_DIRECTORIES} "include/" ${OpenCV_INCLUDE_DIRS})
aux_source_directory("source/" SOURCES)
list(REMOVE_ITEM SOURCES "source//pywrapped_inpaint.c")

message("${SOURCES}")

SET(GCC_COVERAGE_COMPILE_FLAGS "-Og -g")
add_executable(patchmatch ${SOURCES})
# add_executable(cv_test source/pywrapped_inpaint.c)
add_library(inpaint STATIC source//pywrapped_inpaint.c)
target_link_libraries(patchmatch ${OpenCV_LIBS})
# target_link_libraries(cv_test ${OpenCV_LIBS})
