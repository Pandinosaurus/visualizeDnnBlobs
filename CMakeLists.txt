cmake_minimum_required(VERSION 2.8)
project( visualizeDnnBlobs )

find_package( OpenCV )
include_directories( ${OpenCV_INCLUDE_DIRS} )
link_directories(  ${OpenCV_INCLUDE_DIRS} )
SET(CMAKE_CXX_FLAGS "-std=c++11")

add_executable( visualizeDnnBlobs 
		src/main.cpp 
		src/extractImagesFromABlob.hpp )

target_link_libraries( visualizeDnnBlobs 
			${OpenCV_LIBS} )

#You may need to change the change the "${OpenCV_DIR}/install/x64/vc15/bin/" to your installation path
add_custom_command(TARGET visualizeDnnBlobs POST_BUILD
COMMAND ${CMAKE_COMMAND} -E copy_directory
${OpenCV_DIR}/install/x64/vc15/bin/
"$<TARGET_FILE_DIR:visualizeDnnBlobs>"
)