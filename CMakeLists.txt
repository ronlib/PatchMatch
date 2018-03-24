cmake_minimum_required(VERSION 2.8)
project(PatchMatch)


if (OPENCV_SEARCH_PATH)
	message("Searching OpenCV at " ${OPENCV_SEARCH_PATH})
	find_package(
		OpenCV REQUIRED
		HINTS OPENCV_SEARCH_PATH)
else()
	find_package(OpenCV REQUIRED)
endif()

if (LUA51_SEARCH_PATH)
	message("Searching Lua 5.1 at " ${LUA51_SEARCH_PATH})
	find_package(
		Lua51 REQUIRED
		HINTS LUA51_SEARCH_PATH)
else()
	find_package(Lua51 REQUIRED)
endif()

if (TORCH_SEARCH_PATH)
	message("Searching Torch at " ${TORCH_SEARCH_PATH})
	find_package(
		Torch REQUIRED
		HINTS TORCH_SEARCH_PATH)
else()
	find_package(Torch REQUIRED)
endif()

file (TO_CMAKE_PATH "${Torch_DIR}/../../../include" TH_INCLUDE_PATH)

add_library(luainpaint SHARED "source//inpaint.c" "source//nearestneighborfield.c" "source//maskedimage.c" "source//lua_inpaint.c" "source//wrapped_inpaint.c")
target_include_directories(luainpaint PRIVATE "include")
target_include_directories(luainpaint PRIVATE ${TH_INCLUDE_PATH})
target_include_directories(luainpaint PRIVATE ${LUA_INCLUDE_DIR}//lua5.1)
# target_link_libraries(luainpaint ${OpenCV_LIBS} "/usr/lib/liblua.so.5.1")
target_link_libraries(luainpaint ${OpenCV_LIBS})
target_compile_definitions(luainpaint PRIVATE DPNN_PATCH_DISTANCE)
