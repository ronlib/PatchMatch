cmake_minimum_required(VERSION 2.8)
project(PatchMatch)
find_package(OpenCV REQUIRED)
# set(OpenCV_INCLUDE_DIRS "/usr/include/opencv/")
# set(OpenCV_LIBS "/usr/lib")


# aux_source_directory("source/" patchmatch_SOURCES)
# list(REMOVE_ITEM patchmatch_SOURCES "source//pywrapped_inpaint.c" "source//pyinpaint.c" "source//lua_inpaint.c" "source//qualitymesures.cpp")
# add_executable(patchmatch ${patchmatch_SOURCES})
# # target_include_directories(patchmatch PRIVATE ${INCLUDE_DIRECTORIES} "include/" ${OpenCV_INCLUDE_DIRS})
# target_include_directories(patchmatch PRIVATE "include/")

# # SET(GCC_COVERAGE_COMPILE_FLAGS "-Og -g")
# target_link_libraries(patchmatch ${OpenCV_LIBS})

find_package(Lua51)
FIND_PACKAGE(Torch REQUIRED)

if (NOT LUA51_FOUND)
	message(FATAL_ERROR "Could not find lua")
endif (NOT LUA51_FOUND)

if (NOT TORCH_PATH)
	message(FATAL_ERROR "Must provide torch repository path")
endif (NOT TORCH_PATH)
file (TO_CMAKE_PATH "/home/ron/git/torch/install/include" TH_INCLUDE_PATH)

# add_library(luainpaint2 SHARED "main.c")
# target_link_libraries(luainpaint2 "TH")

add_library(luainpaint SHARED "source//inpaint.c" "source//nearestneighborfield.c" "source//maskedimage.c" "source//lua_inpaint.c" "source//wrapped_inpaint.c")
target_include_directories(luainpaint PRIVATE "include")
target_include_directories(luainpaint PRIVATE ${TH_INCLUDE_PATH})
target_include_directories(luainpaint PRIVATE ${LUA_INCLUDE_DIR}//lua5.1)
target_link_libraries(luainpaint ${OpenCV_LIBS} "/usr/lib/liblua.so.5.1")

# /home/ron/git/torch/pkg/torch/lib/TH/generic/.
