cmake_minimum_required( VERSION 3.18 )

set( CMAKE_EXPORT_COMPILE_COMMANDS ON )

project( scan LANGUAGES CUDA C )
include( CheckLanguage )
check_language( CUDA )

add_executable(scan main.cu)

target_include_directories(scan
    PRIVATE /usr/local/cuda-10.0/samples/common/inc/
)

target_compile_options(scan PRIVATE -G -g)

set_target_properties(scan PROPERTIES
    CMAKE_C_STANDARD 99
    CMAKE_CUDA_STANDARD 11
    CUDA_ARCHITECTURES 37
    CUDA_SEPERABLE_COMPILATION ON
)
