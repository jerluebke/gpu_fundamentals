cmake_minimum_required( VERSION 3.18 )

set( CMAKE_EXPORT_COMPILE_COMMANDS ON )

project( tiled_mat_mul LANGUAGES CUDA C )
include( CheckLanguage )
check_language( CUDA )

add_executable(tiled_mat_mul main.cu)

target_include_directories(tiled_mat_mul
    PRIVATE C:/ProgramData/NVIDIA\ Corporation/CUDA\ Samples/v11.1/common/inc
)

# target_compile_options(tiled_mat_mul PRIVATE -G -g)

set_target_properties(tiled_mat_mul PROPERTIES
    CMAKE_C_STANDARD 99
    CMAKE_CUDA_STANDARD 11
    CUDA_ARCHITECTURES 50
    CUDA_SEPERABLE_COMPILATION ON
)
