project(evaluation)

set_property(DIRECTORY PROPERTY EXCLUDE_FROM_ALL YES)

set(EVALUATION_INCLUDE_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

add_executable(universal_bench universal_bench.cu)
set_target_properties(universal_bench PROPERTIES FOLDER "evaluation")
target_include_directories(universal_bench PRIVATE ${EVALUATION_INCLUDE_DIRS})
target_compile_definitions(universal_bench PRIVATE UNIVERSAL_BENCH_WITH_ROBUST_INDEX)
target_link_libraries(universal_bench PRIVATE robust_gpu_indexing)

add_executable(universal_bench_with_gpu_baseline universal_bench.cu)
set_target_properties(universal_bench_with_gpu_baseline PROPERTIES FOLDER "evaluation")
target_include_directories(universal_bench_with_gpu_baseline PRIVATE
  ${EVALUATION_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/../baselines/MVGpuBTree/include
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)
target_sources(universal_bench_with_gpu_baseline PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/gpu_dycuckoo_dynamic.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/gpu_dycuckoo_dynamic_lock.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/../baselines/DyCuckoo/dynamicHash/thirdParty/cnmem.cpp
)
target_compile_definitions(universal_bench_with_gpu_baseline PRIVATE UNIVERSAL_BENCH_WITH_GPU_BASELINE)
target_link_libraries(universal_bench_with_gpu_baseline PRIVATE robust_gpu_indexing)
