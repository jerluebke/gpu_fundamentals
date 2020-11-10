cmake_minimum_required( VERSION 3.18 )

set( CMAKE_EXPORT_COMPILE_COMMANDS ON )

project( mat_mul LANGUAGES CUDA C )
include( CheckLanguage )
check_language( CUDA )

add_executable(mat_mul main.cu)

# target_compile_options(mat_mul PRIVATE -Wall -pedantic)
target_compile_options(mat_mul PRIVATE -g)

set_target_properties(mat_mul PROPERTIES
    # CMAKE_CXX_STANDARD 17
    CMAKE_C_STANDARD 99
    CMAKE_CUDA_STANDARD 11
    CUDA_ARCHITECTURES 37
    CUDA_SEPERABLE_COMPILATION ON
)
