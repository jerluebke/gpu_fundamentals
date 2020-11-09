cmake_minimum_required( VERSION 3.18 )

set( CMAKE_EXPORT_COMPILE_COMMANDS ON )

project( hello_cuda LANGUAGES CUDA CXX )
include( CheckLanguage )
check_language( CUDA )

add_executable(hello_cuda main.cu)

set_target_properties(hello_cuda PROPERTIES
    CMAKE_CXX_STANDARD 17
    CMAKE_CUDA_STANDARD 11
    CUDA_ARCHITECTURES 50
    CUDA_SEPERABLE_COMPILATION ON
)
