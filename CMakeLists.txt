
cmake_minimum_required (VERSION 2.6) 
project (SIFT_CUDA) 

find_package(OpenCV)
find_package(CUDA)


###############################################################################
# Find THRUST
# (from here: https://groups.google.com/forum/#!topic/thrust-users/UX7Gm4piBiU)
find_path( THRUST_INCLUDE_DIR
HINTS ${CUDA_INCLUDE_DIRS} /usr/include/cuda /usr/local/include
NAMES thrust/version.h
DOC "Thrust headers"
)
if( THRUST_INCLUDE_DIR )
list( REMOVE_DUPLICATES THRUST_INCLUDE_DIR )
include_directories( ${THRUST_INCLUDE_DIR} )
endif( THRUST_INCLUDE_DIR )


file( STRINGS ${THRUST_INCLUDE_DIR}/thrust/version.h
version
REGEX "#define THRUST_VERSION[ \t]+([0-9x]+)"
)
string( REGEX REPLACE "#define THRUST_VERSION[ \t]+" "" version $
{version} )

string( REGEX MATCH "^[0-9]" major ${version} )
string( REGEX REPLACE "^${major}00" "" version ${version} )
string( REGEX MATCH "^[0-9]" minor ${version} )
string( REGEX REPLACE "^${minor}0" "" version ${version} )
set( THRUST_VERSION "${major}.${minor}.${version}")

# Check for required components
set( THRUST_FOUND TRUE )

include( FindPackageHandleStandardArgs )
find_package_handle_standard_args( Thrust
REQUIRED_VARS
THRUST_INCLUDE_DIR
VERSION_VAR
THRUST_VERSION
)
######################################################################################


include_directories( ${OpenCV_INCLUDE_DIRS})
include_directories( ${CMAKE_CURRENT_SOURCE_DIR})

link_directories(${OpenCV_LIBRARY_DIRS})
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS})

cuda_compile(source_cu PyramidDoG.cu PyramidDoG.h )
#cuda_add_executable(pruCuda ${source_cu} )

add_executable(SIFT_features SiftFeatures.cpp ${source_cu} )


target_link_libraries(SIFT_features ${OpenCV_LIBS} )

