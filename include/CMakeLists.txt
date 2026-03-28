add_library(robust_gpu_indexing_src INTERFACE)

file(GLOB_RECURSE   robust_gpu_indexing_src
                    "*.h"
                    "*.hpp"
                    "*.cuh"
                    "*.cu")
set(SOURCE_LIST ${robust_gpu_indexing_src})
target_sources(robust_gpu_indexing INTERFACE ${robust_gpu_indexing_src})
target_include_directories(robust_gpu_indexing INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}")