cmake_minimum_required( VERSION 3.18 )

set( CMAKE_EXPORT_COMPILE_COMMANDS ON )

project( transpose LANGUAGES CUDA C )
include( CheckLanguage )
check_language( CUDA )

add_executable(transpose main.cu)

target_include_directories(transpose
    PRIVATE /usr/local/cuda-10.0/samples/common/inc/
)

# target_compile_options(transpose PRIVATE -G -g)

set_target_properties(transpose PROPERTIES
    CMAKE_C_STANDARD 99
    CMAKE_CUDA_STANDARD 11
    CUDA_ARCHITECTURES 37
    CUDA_SEPERABLE_COMPILATION ON
)
