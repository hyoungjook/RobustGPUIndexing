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
  // key distribution
  std::size_t num_keys = get_arg_value<std::size_t>(arguments, "num-keys").value_or(1000000);
  uint32_t keylen_prefix = get_arg_value<uint32_t>(arguments, "keylen-prefix").value_or(0);
  uint32_t keylen_min = get_arg_value<uint32_t>(arguments, "keylen-min").value_or(1);
  uint32_t keylen_max = get_arg_value<uint32_t>(arguments, "keylen-max").value_or(1);
  double keylen_theta = get_arg_value<double>(arguments, "keylen-theta").value_or(0.0);
  // delete config
  double delete_ratio = get_arg_value<double>(arguments, "delete-ratio").value_or(0.1);
  // lookup distribution
  std::size_t num_lookups = get_arg_value<std::size_t>(arguments, "num-lookups").value_or(1000000);
  double lookup_theta = get_arg_value<double>(arguments, "lookup-theta").value_or(0.0);
  double lookup_exist_ratio = get_arg_value<double>(arguments, "lookup-exist-ratio").value_or(1.0);
  // scan config
  std::size_t num_scans = get_arg_value<std::size_t>(arguments, "num-scans").value_or(1000000);
  uint32_t scan_count = get_arg_value<uint32_t>(arguments, "scan-count").value_or(1);
  // repeats
  std::size_t repeats_insert = get_arg_value<std::size_t>(arguments, "repeats-insert").value_or(10);
  std::size_t repeats_delete = get_arg_value<std::size_t>(arguments, "repeats-delete").value_or(10);
  std::size_t repeats_lookup = get_arg_value<std::size_t>(arguments, "repeats-lookup").value_or(10);
  std::size_t repeats_scan = get_arg_value<std::size_t>(arguments, "repeats-scan").value_or(0);
  // index config
  std::string index_type = get_arg_value<std::string>(arguments, "index-type").value_or("cpu_libcuckoo");
  check_argument(num_keys < std::numeric_limits<size_type>::max() &&
                 num_lookups < std::numeric_limits<size_type>::max() &&
                 num_scans < std::numeric_limits<size_type>::max());

  #define FORALL_INDEXES(x) \
  x(cpu_libcuckoo) x(cpu_masstree)

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
    std::cout << "arguments: " << std::endl
              << "  num-keys: " << num_keys << std::endl
              << "  keylen-prefix: " << keylen_prefix << std::endl
              << "  keylen-min: " << keylen_min << std::endl
              << "  keylen-max: " << keylen_max << std::endl
              << "  keylen-theta: " << keylen_theta << std::endl
              << "  delete-ratio: " << delete_ratio << std::endl
              << "  num-lookups: " << num_lookups << std::endl
              << "  lookup-theta: " << lookup_theta << std::endl
              << "  lookup-exist-ratio: " << lookup_exist_ratio << std::endl
              << "  num-scans: " << num_scans << std::endl
              << "  scan-count: " << scan_count << std::endl
              << "  repeats-insert: " << repeats_insert << std::endl
              << "  repeats-delete: " << repeats_delete << std::endl
              << "  repeats-lookup: " << repeats_lookup << std::endl
              << "  repeats-scan: " << repeats_scan << std::endl
              << "  index-type: " << index_type << std::endl
              ;
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
    (repeats_scan > 0) ? num_scans * scan_count : 0
  );
  auto h_results = std::vector<value_type>(results_buffer_size);

  // run benchmark
  #define ADAPTER_RUN_BENCH(index) \
  if (index_type == #index) { \
    universal::run_bench(index##_adapter_, \
      keylen_max, h_keys, h_key_lengths, h_values, num_keys, num_deletes, \
      h_lookup_keys, h_lookup_key_lengths, num_lookups, \
      scan_count, num_scans, h_results, \
      repeats_insert, repeats_delete, repeats_lookup, repeats_scan); \
  }
  FORALL_INDEXES(ADAPTER_RUN_BENCH)
  #undef ADAPTER_RUN_BENCH

}
