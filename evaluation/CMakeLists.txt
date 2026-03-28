project(evaluation)

add_executable(universal_bench universal_bench.cu)
set_target_properties(universal_bench PROPERTIES FOLDER "evaluation")
target_include_directories(universal_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(universal_bench PRIVATE robust_gpu_indexing)

