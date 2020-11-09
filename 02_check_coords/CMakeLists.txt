cmake_minimum_required( VERSION 3.18 )

set( CMAKE_EXPORT_COMPILE_COMMANDS ON )

project( check_coords LANGUAGES CUDA CXX )
include( CheckLanguage )
check_language( CUDA )

add_executable(check_coords main.cu)

set_target_properties(check_coords PROPERTIES
    CMAKE_CXX_STANDARD 17
    CMAKE_CUDA_STANDARD 11
    CUDA_ARCHITECTURES 37
    CUDA_SEPERABLE_COMPILATION ON
)
