cmake_minimum_required( VERSION 3.18 )

set( CMAKE_EXPORT_COMPILE_COMMANDS ON )

project( vec_add LANGUAGES CUDA CXX )
include( CheckLanguage )
check_language( CUDA )

add_executable(vec_add main.cu)

set_target_properties(vec_add PROPERTIES
    CMAKE_CXX_STANDARD 17
    CMAKE_CUDA_STANDARD 11
    CUDA_ARCHITECTURES 50
    CUDA_SEPERABLE_COMPILATION ON
)
