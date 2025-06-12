cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(example-app)

#CUDA
find_package(CUDA REQUIRED)
set(CUDA_ARCH "30 35 50 52 60 61 70 86" CACHE STRING "CUDA architectures to compile for")
string(REPLACE " " ";" CUDA_ARCH_LIST ${CUDA_ARCH})
foreach(a ${CUDA_ARCH_LIST})
    set(CUDA_ARCH_OPTIONS "${CUDA_ARCH_OPTIONS} -gencode arch=compute_${a},code=compute_${a}")
endforeach()
include_directories(${CUDA_INCLUDE_DIRS})

include_directories("/usr/local/cuda/include")
link_directories("/usr/local/cuda/lib64")

set(CMAKE_PREFIX_PATH "/home/weleslie/libtorch/libtorch")
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

find_package(GLEW REQUIRED)
include_directories(${GLEW_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FILES} ${TORCH_CXX_FLAGS}")

cuda_add_executable(example-app example-app.cpp BackWarp.cu backward_code.cu)
target_link_libraries(example-app ${TORCH_LIBRARIES} ${OpenCV_LIBS} ${GLEW_LIBRARIES} -lcuda -lcudart -lcublas -lcusolver -lglfw3 -lX11
        -lXxf86vm -lXrandr -lpthread -lXi -lXcursor -lXinerama -lGL -lGLU -ldl)
set_property(TARGET example-app PROPERTY CXX_STANDARD 14)