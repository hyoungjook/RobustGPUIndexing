/*
 *   Copyright 2026 Hyoungjoo Kim, Carnegie Mellon University
 *
 *   Licensed under the Apache License, Version 2.0 (the "License");
 *   you may not use this file except in compliance with the License.
 *   You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 *   Unless required by applicable law or agreed to in writing, software
 *   distributed under the License is distributed on an "AS IS" BASIS,
 *   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *   See the License for the specific language governing permissions and
 *   limitations under the License.
 */

#include <stdlib.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <thread>
#include <vector>
#include <cmd.hpp>
#include <generate_workload.hpp>
#include <cpu_libcuckoo_adapter.hpp>
#include <cpu_masstree_adapter.hpp>
#include <cpu_art_adapter.hpp>

namespace universal {

template <class F, class ThreadEnter, class ThreadExit>
void helper_multithread(F&& f, std::size_t num_tasks, ThreadEnter&& thread_enter, ThreadExit&& thread_exit) {
  const unsigned num_workers = std::max(1u, std::thread::hardware_concurrency());
  std::vector<std::thread> workers;
  for (unsigned tid = 0; tid < num_workers; tid++) {
    workers.emplace_back([&](unsigned thread_id) {
      thread_enter();
      for (std::size_t task_idx = thread_id; task_idx < num_tasks; task_idx += num_workers) {
        std::forward<F>(f)(task_idx);
      }
      thread_exit();
    }, tid);
  }
  for (auto& w: workers) { w.join(); }
}

template <typename adapter_type>
void run_bench(adapter_type& adapter,
               uint32_t keylen_max,
               std::vector<key_slice_type>& keys,
               std::vector<size_type>& key_lengths,
               std::vector<value_type>& values,
               std::size_t num_keys,
               std::size_t num_deletes,
               std::vector<key_slice_type>& lookup_keys,
               std::vector<size_type>& lookup_key_lengths,
               std::size_t num_lookups,
               uint32_t scan_count,
               std::size_t num_scans,
               std::vector<value_type>& results,
               std::size_t repeats_insert,
               std::size_t repeats_delete,
               std::size_t repeats_lookup,
               std::size_t repeats_scan) {
  auto print_rate_Mops = [&](std::string name, std::size_t size, float seconds, std::size_t repeats) {
    auto rate = static_cast<float>(size) / 1e6 / (seconds / repeats);
    std::cout << name << ": " << rate << " Mop/s" << std::endl;
  };
  std::chrono::time_point<std::chrono::high_resolution_clock> timer_start, timer_end;
  // measure insert & delete
  float insert_seconds = 0, delete_seconds = 0;
  for (std::size_t r = 0; r < repeats_insert; r++) {
    adapter.initialize();
    timer_start = std::chrono::high_resolution_clock::now();
    helper_multithread([&](std::size_t task_idx) {
      adapter.insert(&keys[task_idx * keylen_max], key_lengths[task_idx], values[task_idx]);
    }, num_keys,
    [&]() { adapter.thread_enter(); },
    [&]() { adapter.thread_exit(); });
    timer_end = std::chrono::high_resolution_clock::now();
    insert_seconds += std::chrono::duration_cast<std::chrono::duration<float>>(timer_end - timer_start).count();

    if (r < repeats_delete) {
      timer_start = std::chrono::high_resolution_clock::now();
      helper_multithread([&](std::size_t task_idx) {
        adapter.erase(&keys[task_idx * keylen_max], key_lengths[task_idx]);
      }, num_deletes,
      [&]() { adapter.thread_enter(); },
      [&]() { adapter.thread_exit(); });
      timer_end = std::chrono::high_resolution_clock::now();
      delete_seconds += std::chrono::duration_cast<std::chrono::duration<float>>(timer_end - timer_start).count();
    }
    adapter.destroy();
  }
  if (repeats_insert > 0) {
    print_rate_Mops("insert", num_keys, insert_seconds, repeats_insert);
  }
  if (repeats_delete > 0) {
    print_rate_Mops("delete", num_deletes, delete_seconds, repeats_delete);
  }

  // measure lookup & scan
  float lookup_seconds = 0, scan_seconds = 0;
  adapter.initialize();
  helper_multithread([&](std::size_t task_idx) {
    adapter.insert(&keys[task_idx * keylen_max], key_lengths[task_idx], values[task_idx]);
  }, num_keys,
  [&]() { adapter.thread_enter(); },
  [&]() { adapter.thread_exit(); });
  for (std::size_t r = 0; r < repeats_lookup; r++) {
    timer_start = std::chrono::high_resolution_clock::now();
    helper_multithread([&](std::size_t task_idx) {
      results[task_idx] = adapter.find(&lookup_keys[task_idx * keylen_max], lookup_key_lengths[task_idx]);
    }, num_lookups,
    [&]() { adapter.thread_enter(); },
    [&]() { adapter.thread_exit(); });
    timer_end = std::chrono::high_resolution_clock::now();
    lookup_seconds += std::chrono::duration_cast<std::chrono::duration<float>>(timer_end - timer_start).count();
  }

  if constexpr (adapter_type::is_ordered) {
    for (std::size_t r = 0; r < repeats_scan; r++) {
      timer_start = std::chrono::high_resolution_clock::now();
      helper_multithread([&](std::size_t task_idx) {
        adapter.scan(&lookup_keys[task_idx * keylen_max], lookup_key_lengths[task_idx], scan_count, &results[task_idx * scan_count]);
      }, num_scans,
      [&]() { adapter.thread_enter(); },
      [&]() { adapter.thread_exit(); });
      timer_end = std::chrono::high_resolution_clock::now();
      scan_seconds += std::chrono::duration_cast<std::chrono::duration<float>>(timer_end - timer_start).count();
    }
  }
  adapter.destroy();
  if (repeats_lookup > 0) {
    print_rate_Mops("lookup", num_lookups, lookup_seconds, repeats_lookup);
  }
  if (adapter_type::is_ordered && repeats_scan > 0) {
    print_rate_Mops("scan", num_scans, scan_seconds, repeats_scan);
  }
}

} // namespace universal

int main(int argc, char** argv) {
  auto arguments = std::vector<std::string>(argv, argv + argc);
  bool verbose = get_arg_value<bool>(arguments, "verbose").value_or(true);
  #define PARSE_ARGUMENTS(arg, type, default_value) \
  auto arg = get_arg_value<type>(arguments, #arg).value_or(default_value);
  FORALL_ARGUMENTS(PARSE_ARGUMENTS)
  #undef PARSE_ARGUMENTS
  check_argument(num_keys < std::numeric_limits<size_type>::max() &&
                 num_lookups < std::numeric_limits<size_type>::max() &&
                 num_scans < std::numeric_limits<size_type>::max());

  #define FORALL_INDEXES(x) \
  x(cpu_libcuckoo) x(cpu_masstree) x(cpu_art)

  #define INDEX_NAME_CHECK(index) (index_type == #index) ||
  check_argument(FORALL_INDEXES(INDEX_NAME_CHECK) false);
  #undef INDEX_NAME_CHECK
  #define ADAPTER_DECLARE(index) index##_adapter index##_adapter_;
  FORALL_INDEXES(ADAPTER_DECLARE)
  #undef ADAPTER_DECLARE
  #define ADAPTER_PARSE_ARGS(index) \
  if (index_type == #index) { index##_adapter_.parse(arguments); }
  FORALL_INDEXES(ADAPTER_PARSE_ARGS)
  #undef ADAPTER_PARSE_ARGS

  // print arguments
  repeats_insert = std::max(repeats_insert, repeats_delete);
  check_argument(0.0 < delete_ratio && delete_ratio <= 1.0);
  std::size_t num_deletes = static_cast<std::size_t>(delete_ratio * num_keys);
  if (verbose) {
    std::cout << "arguments:" << std::endl;
    #define PRINT_ARGUMENT(arg, type, default_value) \
    std::cout << "  " #arg "=" << arg << std::endl;
    FORALL_ARGUMENTS(PRINT_ARGUMENT)
    #undef PRINT_ARGUMENT
    #define ADAPTER_PRINT_ARGS(index) \
    if (index_type == #index) { index##_adapter_.print_args(); }
    FORALL_INDEXES(ADAPTER_PRINT_ARGS)
    #undef ADAPTER_PRINT_ARGS
  }

  // generate keys and queries
  std::vector<key_slice_type> h_keys;
  std::vector<size_type> h_key_lengths;
  std::vector<value_type> h_values;
  std::vector<key_slice_type> h_lookup_keys;
  std::vector<size_type> h_lookup_key_lengths;
  universal::generate_key_values(h_keys, h_key_lengths, h_values,
                                 num_keys, keylen_prefix, keylen_min, keylen_max, keylen_theta, true);
  std::size_t num_lookups_keys = std::max(
    (repeats_lookup > 0) ? num_lookups : 0,
    (repeats_scan > 0) ? num_scans : 0
  );
  if (repeats_lookup > 0 || repeats_scan > 0) {
    universal::generate_lookup_keys(h_lookup_keys, h_lookup_key_lengths, h_keys, h_key_lengths,
                                    num_keys, keylen_prefix, keylen_min, keylen_max, keylen_theta,
                                    num_lookups_keys, lookup_theta, lookup_exist_ratio, true);
  }
  std::size_t results_buffer_size = std::max(
    (repeats_lookup > 0) ? num_lookups : 0,
    (repeats_scan > 0) ? (index_type != "cpu_art" ? num_scans * scan_count : num_scans * scan_count * 2) : 0
  );
  auto h_results = std::vector<value_type>(results_buffer_size);
  

  // run benchmark
  #define ADAPTER_RUN_BENCH(index) \
  if (index_type == #index) { \
    index##_adapter_.register_dataset(h_keys.data(), h_key_lengths.data(), h_values.data()); \
    universal::run_bench(index##_adapter_, \
      keylen_max, h_keys, h_key_lengths, h_values, num_keys, num_deletes, \
      h_lookup_keys, h_lookup_key_lengths, num_lookups, \
      scan_count, num_scans, h_results, \
      repeats_insert, repeats_delete, repeats_lookup, repeats_scan); \
  }
  FORALL_INDEXES(ADAPTER_RUN_BENCH)
  #undef ADAPTER_RUN_BENCH

}
