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

#include <cuda_profiler_api.h>
#include <stdlib.h>
#include <thrust/sequence.h>
#include <algorithm>
#include <cstdint>
#include <numeric>
#include <vector>
#include <cmd.hpp>
#include <macros.hpp>
#include <gpu_timer.hpp>
#include <generate_workload.hpp>
#if defined(UNIVERSAL_BENCH_WITH_ROBUST_INDEX)
#include <gpu_masstree_adapter.hpp>
#include <gpu_chainhashtable_adapter.hpp>
#include <gpu_cuckoohashtable_adapter.hpp>
#include <gpu_linearhashtable_adapter.hpp>
#elif defined(UNIVERSAL_BENCH_WITH_GPU_BASELINE)
#include <gpu_blink_tree_adapter.hpp>
#include <gpu_dycuckoo_adapter.hpp>
#endif

namespace universal {

template <typename adapter_type>
void run_bench(adapter_type& adapter,
               uint32_t keylen_max,
               thrust::device_vector<key_slice_type>& keys,
               thrust::device_vector<size_type>& key_lengths,
               thrust::device_vector<value_type>& values,
               std::size_t num_keys,
               std::size_t num_deletes,
               thrust::device_vector<key_slice_type>& lookup_keys,
               thrust::device_vector<size_type>& lookup_key_lengths,
               std::size_t num_lookups,
               thrust::device_vector<key_slice_type>& scan_upper_keys_if_btree,
               uint32_t scan_count,
               std::size_t num_scans,
               thrust::device_vector<value_type>& results,
               std::size_t repeats_insert,
               std::size_t repeats_delete,
               std::size_t repeats_lookup,
               std::size_t repeats_scan) {
  auto print_rate_Mops = [&](std::string name, std::size_t size, float seconds, std::size_t repeats) {
    auto rate = static_cast<float>(size) / 1e6 / (seconds / repeats);
    std::cout << name << ": " << rate << " Mop/s" << std::endl;
  };
  // measure insert & delete
  float insert_seconds = 0, delete_seconds = 0;
  for (std::size_t r = 0; r < repeats_insert; r++) {
    adapter.initialize();
    gpu_timer insert_timer;
    insert_timer.start_timer();
    adapter.insert(keys.data().get(), keylen_max, key_lengths.data().get(), values.data().get(), num_keys);
    insert_timer.stop_timer();
    cuda_try(cudaDeviceSynchronize());
    insert_seconds += insert_timer.get_elapsed_s();

    if (r < repeats_delete) {
      gpu_timer delete_timer;
      delete_timer.start_timer();
      adapter.erase(keys.data().get(), keylen_max, key_lengths.data().get(), num_deletes);
      delete_timer.stop_timer();
      cuda_try(cudaDeviceSynchronize());
      delete_seconds += delete_timer.get_elapsed_s();
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
  adapter.insert(keys.data().get(), keylen_max, key_lengths.data().get(), values.data().get(), num_keys);
  for (std::size_t r = 0; r < repeats_lookup; r++) {
    gpu_timer lookup_timer;
    lookup_timer.start_timer();
    adapter.find(lookup_keys.data().get(), keylen_max, lookup_key_lengths.data().get(), results.data().get(), num_lookups);
    lookup_timer.stop_timer();
    cuda_try(cudaDeviceSynchronize());
    lookup_seconds += lookup_timer.get_elapsed_s();
  }

  if constexpr (adapter_type::is_ordered) {
    for (std::size_t r = 0; r < repeats_scan; r++) {
      gpu_timer scan_timer;
      scan_timer.start_timer();
      adapter.scan(lookup_keys.data().get(), keylen_max, lookup_key_lengths.data().get(), scan_count, results.data().get(), num_scans, scan_upper_keys_if_btree.data().get());
      scan_timer.stop_timer();
      cuda_try(cudaDeviceSynchronize());
      scan_seconds += scan_timer.get_elapsed_s();
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

} // namespace universal_bench

int main(int argc, char** argv) {
  auto arguments = std::vector<std::string>(argv, argv + argc);
  int device_id = get_arg_value<int>(arguments, "device").value_or(0);
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
  std::string index_type = get_arg_value<std::string>(arguments, "index-type").value_or("gpu_masstree");
  check_argument(num_keys < std::numeric_limits<size_type>::max() &&
                 num_lookups < std::numeric_limits<size_type>::max() &&
                 num_scans < std::numeric_limits<size_type>::max());

  #if defined(UNIVERSAL_BENCH_WITH_ROBUST_INDEX)
  #define FORALL_INDEXES(x) \
  x(gpu_masstree) x(gpu_chainhashtable) x(gpu_cuckoohashtable) x(gpu_linearhashtable)
  #elif defined(UNIVERSAL_BENCH_WITH_GPU_BASELINE)
  #define FORALL_INDEXES(x) \
  x(gpu_blink_tree) x(gpu_dycuckoo)
  #endif

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
  std::vector<key_slice_type> h_scan_upper_keys_if_btree;
  universal::generate_key_values(h_keys, h_key_lengths, h_values,
                           num_keys, keylen_prefix, keylen_min, keylen_max, keylen_theta, false);
  std::size_t num_lookups_keys = std::max(
    (repeats_lookup > 0) ? num_lookups : 0,
    (repeats_scan > 0) ? num_scans : 0
  );
  if (repeats_lookup > 0 || repeats_scan > 0) {
    universal::generate_lookup_keys(h_lookup_keys, h_lookup_key_lengths, h_keys, h_key_lengths,
                                    num_keys, keylen_prefix, keylen_min, keylen_max, keylen_theta,
                                    num_lookups_keys, lookup_theta, lookup_exist_ratio, false);
  }
  if (index_type == "gpu_blink_tree" && repeats_scan > 0) {
    check_argument(keylen_max == 1);
    h_scan_upper_keys_if_btree = std::vector<key_slice_type>(num_scans);
    for (std::size_t i = 0; i < num_scans; i++) {
      h_scan_upper_keys_if_btree[i] = h_lookup_keys[i] + scan_count - 1;
    }
  }
  std::size_t results_buffer_size = std::max(
    (repeats_lookup > 0) ? num_lookups : 0,
    (repeats_scan > 0) ? (index_type != "gpu_blink_tree" ? num_scans * scan_count : num_scans * scan_count * 2) : 0
  );

  // copy vectors to device
  auto d_keys = thrust::device_vector<key_slice_type>(h_keys.size());
  auto d_key_lengths = thrust::device_vector<size_type>(h_key_lengths.size());
  auto d_values = thrust::device_vector<value_type>(h_values.size());
  auto d_lookup_keys = thrust::device_vector<key_slice_type>(h_lookup_keys.size());
  auto d_lookup_key_lengths = thrust::device_vector<key_slice_type>(h_lookup_key_lengths.size());
  auto d_results = thrust::device_vector<value_type>(results_buffer_size);
  auto d_scan_upper_keys_if_btree = thrust::device_vector<key_slice_type>(h_scan_upper_keys_if_btree.size());
  d_keys = h_keys;
  d_key_lengths = h_key_lengths;
  d_values = h_values;
  d_lookup_keys = h_lookup_keys;
  d_lookup_key_lengths = h_lookup_key_lengths;
  if (h_scan_upper_keys_if_btree.size() > 0) {
    d_scan_upper_keys_if_btree = h_scan_upper_keys_if_btree;
  }

  // run benchmark
  #define ADAPTER_RUN_BENCH(index) \
  if (index_type == #index) { \
    universal::run_bench(index##_adapter_, \
      keylen_max, d_keys, d_key_lengths, d_values, num_keys, num_deletes, \
      d_lookup_keys, d_lookup_key_lengths, num_lookups, \
      d_scan_upper_keys_if_btree, scan_count, num_scans, d_results, \
      repeats_insert, repeats_delete, repeats_lookup, repeats_scan); \
  }
  FORALL_INDEXES(ADAPTER_RUN_BENCH)
  #undef ADAPTER_RUN_BENCH
}
