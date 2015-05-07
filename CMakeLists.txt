
cmake_minimum_required (VERSION 2.6) 
project (SIFT_CUDA) 

find_package(OpenCV)
find_package(CUDA)



include_directories( ${OpenCV_INCLUDE_DIRS})
include_directories( ${CMAKE_CURRENT_SOURCE_DIR})

link_directories(${OpenCV_LIBRARY_DIRS})
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS})

cuda_compile(source_cu SiftFeatures.cu SiftFeatures.h )
#cuda_add_executable(pruCuda ${source_cu} )

add_executable(SIFT_Test main.cpp ${source_cu} )


target_link_libraries(SIFT_Test ${OpenCV_LIBS} )

