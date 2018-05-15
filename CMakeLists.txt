cmake_minimum_required(VERSION 2.8)
project(PatchMatch)

if (MSVC AND OPENCV_SEARCH_PATH)
	set(OpenCV_FOUND 1)
	set(OpenCV_FIND_QUIETLY ON)
	set(OpenCV_STATIC ON)
	set(MSVC_VERSION 1900) 	# It doesn't really matter which version,
							# as the interfaces are defined in C
	set(OPENCV_SEARCH_PATH ${OPENCV_SEARCH_PATH}/build)
	file (TO_CMAKE_PATH ${OPENCV_SEARCH_PATH} OPENCV_SEARCH_PATH)
	message("Searching OpenCV at " ${OPENCV_SEARCH_PATH})
	find_package(
		OpenCV REQUIRED
		PATHS ${OPENCV_SEARCH_PATH}
		)
else()
	find_package(OpenCV REQUIRED)
endif()

if (MSVC AND TORCH_SEARCH_PATH)
	file (TO_CMAKE_PATH ${TORCH_SEARCH_PATH}/share/cmake/torch TORCH_SEARCH_PATH)
	message("Searching Torch at " ${TORCH_SEARCH_PATH})
	LIST(APPEND CMAKE_MODULE_PATH "${TORCH_SEARCH_PATH}")
	find_package(
		Torch REQUIRED
		PATHS ${TORCH_SEARCH_PATH}
		)

	set(LUA_INCLUDE_DIR "${Torch_DIR}/../../../include/")
else()
	find_package(Torch REQUIRED)
endif()

add_library(luainpaint SHARED "source//inpaint.c" "source//nearestneighborfield.c" "source//maskedimage.c" "source//lua_inpaint.c" "source//wrapped_inpaint.c")
target_include_directories(luainpaint PRIVATE "include")
file (TO_CMAKE_PATH "${Torch_DIR}/../../../include" TH_INCLUDE_PATH)
target_include_directories(luainpaint PRIVATE ${TH_INCLUDE_PATH})
target_include_directories(luainpaint PRIVATE ${LUA_INCLUDE_DIR}//lua5.1)
target_include_directories(luainpaint PRIVATE ${OpenCV_INCLUDE_DIRS})
# target_link_libraries(luainpaint ${OpenCV_LIBS} "/usr/lib/liblua.so.5.1")
target_link_libraries(luainpaint ${OpenCV_LIBS})
if(MSVC)
	target_link_libraries(luainpaint
						"${TORCH_SEARCH_PATH}/../../../lib/libluajit.lib"
						"${TORCH_SEARCH_PATH}/../../../lib/luaT.lib"
						"${TORCH_SEARCH_PATH}/../../../lib/TH.lib")
endif()
target_compile_definitions(luainpaint PRIVATE DPNN_PATCH_DISTANCE)
add_custom_command(TARGET luainpaint POST_BUILD
    # COMMAND ${CMAKE_COMMAND} -E make_directory "${PatchMatch_SOURCE_DIR}/lib/"
	# COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:luainpaint>" "${PatchMatch_SOURCE_DIR}/lib/"
	# COMMAND ${CMAKE_COMMAND} -E copy "${PatchMatch_SOURCE_DIR}/patch2vec.lua" "${PatchMatch_SOURCE_DIR}/lib/"
	COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:luainpaint>" "${PatchMatch_SOURCE_DIR}/../"
	COMMAND ${CMAKE_COMMAND} -E copy "${PatchMatch_SOURCE_DIR}/patch2vec.lua" "${PatchMatch_SOURCE_DIR}/../"
	VERBATIM
)

add_library(patchmatch2 SHARED
	patchmatch2.1/lua_inpaint.cpp
	patchmatch2.1/allegro_emu.cpp
	patchmatch2.1/allegro_emu.h
	patchmatch2.1/knn.cpp
	patchmatch2.1/knn.h
	patchmatch2.1/nn.cpp
	patchmatch2.1/nn.h
	patchmatch2.1/patch.cpp
	patchmatch2.1/patch.h
	patchmatch2.1/simnn.cpp
	patchmatch2.1/pm_minimal.cpp
	patchmatch2.1/simnn.h
	patchmatch2.1/simpatch.h
	patchmatch2.1/vecnn.cpp
	patchmatch2.1/vecnn.h
	patchmatch2.1/vecpatch.h
	patchmatch2.1/inpaint.cpp
	patchmatch2.1/inpaint.h
	)

target_include_directories(patchmatch2 PRIVATE ${LUA_INCLUDE_DIR}//lua5.1)
