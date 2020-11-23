cmake_minimum_required( VERSION 3.18 )

set( CMAKE_EXPORT_COMPILE_COMMANDS ON )

project( ilp LANGUAGES CUDA C )
include( CheckLanguage )
check_language( CUDA )

add_executable(ilp main.cu)

target_include_directories(ilp
    PRIVATE /usr/local/cuda-10.0/samples/common/inc/
)

target_compile_options(ilp PRIVATE -g)

set_target_properties(ilp PROPERTIES
    CMAKE_C_STANDARD 99
    CMAKE_CUDA_STANDARD 11
    CUDA_ARCHITECTURES 50
    CUDA_SEPERABLE_COMPILATION ON
)
