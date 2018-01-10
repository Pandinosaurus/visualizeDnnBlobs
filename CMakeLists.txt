cmake_minimum_required(VERSION 2.8)
project( main )

find_package( OpenCV )
include_directories( ${OpenCV_INCLUDE_DIRS} )
SET(CMAKE_CXX_FLAGS "-std=c++11")
add_executable( main src/main.cpp )
target_link_libraries( main ${OpenCV_LIBS} )

add_custom_command(TARGET main POST_BUILD
COMMAND ${CMAKE_COMMAND} -E copy_directory
"D:/lib/opencv/build/x64/vs2017/install/x64/vc15/bin/"
"$<TARGET_FILE_DIR:main>"
)