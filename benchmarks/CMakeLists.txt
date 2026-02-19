project(benchmarks)

set(BENCHMARKS
    masstree_insert_find_bench
    masstree_erase_bench
    masstree_scan_bench
    hashtable_insert_find_bench
    hashtable_erase_bench
    linearhashtable_insert_find_bench
    linearhashtable_erase_bench
)

foreach(BENCHMARK ${BENCHMARKS})
    add_executable(${BENCHMARK} ${BENCHMARK}.cu)
    set_target_properties(${BENCHMARK} PROPERTIES FOLDER "benchmarks")
    target_link_libraries(${BENCHMARK} PRIVATE gpu_btrees)
endforeach()
