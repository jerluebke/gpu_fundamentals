cmake_minimum_required( VERSION 3.18 )

set( CMAKE_EXPORT_COMPILE_COMMANDS ON )

project( mat_add LANGUAGES CUDA C )
include( CheckLanguage )
check_language( CUDA )

add_executable(mat_add main.cu)

target_include_directories(mat_add
    PRIVATE C:/ProgramData/NVIDIA\ Corporation/CUDA\ Samples/v11.1/common/inc
)

set_target_properties(mat_add PROPERTIES
    CMAKE_C_STANDARD 99
    CMAKE_CUDA_STANDARD 11
    CUDA_ARCHITECTURES 50
    CUDA_SEPERABLE_COMPILATION ON
)
