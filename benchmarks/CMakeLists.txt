project(benchmarks)

set(BENCHMARKS
    masstree_bench
    fixedhashtable_bench
    linearhashtable_bench
    masstree_mix_bench
    fixedhashtable_mix_bench
    linearhashtable_mix_bench
)

foreach(BENCHMARK ${BENCHMARKS})
    add_executable(${BENCHMARK} ${BENCHMARK}.cu)
    set_target_properties(${BENCHMARK} PROPERTIES FOLDER "benchmarks")
    target_link_libraries(${BENCHMARK} PRIVATE robust_gpu_indexing)
endforeach()
