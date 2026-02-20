/*
 *   Copyright 2022 The Regents of the University of California, Davis
 *   Copyright 2025 Hyoungjoo Kim, Carnegie Mellon University
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
#include <gpu_index.h>
#include <stdlib.h>
#include <thrust/sequence.h>
#include <thrust/logical.h>
#include <thrust/count.h>
#include <algorithm>
#include <cmd.hpp>
#include <cstdint>
#include <gpu_timer.hpp>
#include <numeric>
#include <random>
#include <rkg.hpp>
#include <string>
#include <unordered_set>
#include <vector>
#include <thread>

using key_slice_type = uint32_t;
using value_type = uint32_t;
using size_type = uint32_t;

struct bench_rates {
  float insertion_rate;
  float find_rate;
};
template <typename linearhashtable_type, bool concurrent_find, bool use_hash_tag, bool tag_use_same_hash>
bench_rates bench_linearhashtable_insertion_find(thrust::device_vector<key_slice_type>& d_keys,
                                                 thrust::device_vector<size_type>& d_lengths,
                                                 thrust::device_vector<value_type>& d_values,
                                                 thrust::device_vector<key_slice_type>& d_query_keys,
                                                 thrust::device_vector<size_type>& d_query_lengths,
                                                 thrust::device_vector<value_type>& d_query_results,
                                                 uint32_t num_keys,
                                                 uint32_t max_key_length,
                                                 std::size_t num_experiments,
                                                 uint32_t initial_directory_size,
                                                 float resize_policy,
                                                 bool validate_result = false,
                                                 bool validate_index = false) {
  cudaStream_t insertion_stream{0};
  cudaStream_t find_stream{0};
  float average_insertion_seconds(0.0f);
  float average_find_seconds(0.0f);
  std::size_t valid_count = 0;

  for (std::size_t exp = 0; exp < num_experiments; exp++) {
    typename linearhashtable_type::host_allocator_type host_alloc;
    typename linearhashtable_type::host_reclaimer_type host_reclaim;
    linearhashtable_type table(host_alloc, host_reclaim, initial_directory_size, resize_policy);
    auto memory_usage = utils::compute_device_memory_usage();
    std::cout << "Using: " << double(memory_usage.used_bytes) / double(1 << 30) << " GiBs"
              << std::endl;
    gpu_timer insert_timer(insertion_stream);
    insert_timer.start_timer();
    table.insert(d_keys.data().get(), max_key_length, d_lengths.data().get(), d_values.data().get(), num_keys, insertion_stream, false, use_hash_tag, tag_use_same_hash);
    insert_timer.stop_timer();
    cuda_try(cudaDeviceSynchronize());
    auto insertion_elapsed = insert_timer.get_elapsed_s();
    average_insertion_seconds += insertion_elapsed;

    gpu_timer find_timer(find_stream);
    find_timer.start_timer();
    table.find(d_query_keys.data().get(), max_key_length, d_query_lengths.data().get(), d_query_results.data().get(), num_keys, find_stream, concurrent_find, use_hash_tag, tag_use_same_hash);
    find_timer.stop_timer();
    cuda_try(cudaDeviceSynchronize());
    auto find_elapsed = find_timer.get_elapsed_s();
    average_find_seconds += find_elapsed;
    std::cout << exp << std::setw(6) << '\t';
    std::cout << insertion_elapsed << std::setw(6) << '\t';
    std::cout << find_elapsed << std::setw(6) << '\n';

    if (validate_result) {
      thrust::device_vector<bool> cmp_result(num_keys);
      thrust::transform(d_values.begin(), d_values.end(), d_query_results.begin(), cmp_result.begin(), thrust::equal_to<value_type>());
      uint32_t matching_count = (uint32_t)thrust::count(cmp_result.begin(), cmp_result.end(), true);
      if (matching_count == num_keys) {
        valid_count++;
      }
      else {
        std::cout << "validation failed: " << matching_count << "/" << num_keys << " matches" << std::endl;
      }
    }
    if (validate_index) {
      table.validate();
    }
  }

  average_insertion_seconds /= float(num_experiments);
  average_find_seconds /= float(num_experiments);
  float insertion_rate = float(d_lengths.size()) / 1e6 / average_insertion_seconds;
  float find_rate      = float(d_query_lengths.size()) / 1e6 / average_find_seconds;
  std::cout << "insertion_rate: " << insertion_rate << '\n';
  std::cout << "find_rate: " << find_rate << std::endl;
  if (validate_result) {
    if (valid_count == num_experiments) {
      std::cout << "all results valid" << std::endl;
    }
    else {
      std::cout << "validation: " << valid_count << "/" << num_experiments << " valid" << std::endl;
    }
  }
  return {insertion_rate, find_rate};
}

int main(int argc, char** argv) {
  auto arguments    = std::vector<std::string>(argv, argv + argc);
  uint32_t num_keys = get_arg_value<uint32_t>(arguments, "num-keys").value_or(1'000'000);
  int device_id     = get_arg_value<int>(arguments, "device").value_or(0);
  uint32_t initial_directory_size = get_arg_value<uint32_t>(arguments, "initial-directory-size").value_or(1024u);
  float resize_policy = get_arg_value<float>(arguments, "resize-policy").value_or(2.0f);
  uint32_t min_key_length = get_arg_value<uint32_t>(arguments, "min-key-length").value_or(1u);
  uint32_t max_key_length = get_arg_value<uint32_t>(arguments, "max-key-length").value_or(1u);
  float common_prefix_ratio = get_arg_value<float>(arguments, "common-prefix-ratio").value_or(0.1f);
  bool validate_result   = get_arg_value<bool>(arguments, "validate-result").value_or(false);
  bool validate_index   = get_arg_value<bool>(arguments, "validate-index").value_or(false);
  std::string dataset_file = get_arg_value<std::string>(arguments, "dataset-file").value_or("");
  std::size_t num_experiments =
      get_arg_value<std::size_t>(arguments, "num-experiments").value_or(1llu);
  if (min_key_length > max_key_length) {
    std::cerr << "min_key_lenght is larger than max_key_length" << std::endl;
    exit(1);
  }

  int device_count;
  cudaGetDeviceCount(&device_count);
  cudaDeviceProp devProp;
  if (device_id < device_count) {
    cudaSetDevice(device_id);
    cudaGetDeviceProperties(&devProp, device_id);
    std::cout << "Device[" << device_id << "]: " << devProp.name << std::endl;
  } else {
    std::cout << "No capable CUDA device found." << std::endl;
    std::terminate();
  }
  std::string device_name(devProp.name);
  std::replace(device_name.begin(), device_name.end(), ' ', '-');

  std::cout << "Generating input...\n";
  static constexpr value_type invalid_value = std::numeric_limits<value_type>::max();

  std::vector<std::string> dataset;
  if (dataset_file != "") {
    // parse dataset
    uint32_t min_length, max_length;
    dataset = rkg::parse_dataset_file<key_slice_type>(dataset_file, min_length, max_length);
    std::cout << "Parsed dataset " << dataset_file << ", " <<
        "found " << dataset.size() << " keys " <<
        "with key length in [" << min_length << ", " << max_length << "]" << std::endl;
    if (num_keys > dataset.size()) {
      std::cout << "Dataset is smaller than the givne num_keys. Using the full dataset..." << std::endl;
      num_keys = dataset.size();
    }
    if (min_key_length > min_length) {
      std::cout << "Dataset has keys smaller than the given min_key_length. Ignoring the parameter..." << std::endl;
    }
    if (max_key_length > max_length) {
      std::cout << "All keys in the dataset is smaller than the givne max_key_length. Not trimming keys..." << std::endl;
      max_key_length = max_length;
    }
  }

  // device vectors
  auto d_keys      = thrust::device_vector<key_slice_type>(num_keys * max_key_length, 0);
  auto d_lengths      = thrust::device_vector<size_type>(num_keys, 0);
  auto d_values    = thrust::device_vector<value_type>(num_keys, invalid_value);
  auto d_find_keys = thrust::device_vector<key_slice_type>(num_keys * max_key_length, 0);
  auto d_find_lengths = thrust::device_vector<size_type>(num_keys, 0);
  auto d_results   = thrust::device_vector<value_type>(num_keys, invalid_value);

  // host vectors
  std::vector<key_slice_type> h_keys;
  std::vector<size_type> h_lengths;
  if (dataset_file != "") {
    rkg::generate_varlen_keys_from_dataset(dataset, h_keys, h_lengths, num_keys, max_key_length);
  }
  else {
    std::random_device rd;
    //std::mt19937 rng(0);
    std::mt19937 rng(rd());
    rkg::generate_varlen_keys<key_slice_type, size_type>(
      h_keys, h_lengths, num_keys, min_key_length, max_key_length, rng, rkg::distribution_type::unique_random,
      common_prefix_ratio);
  }

  // copy to device
  d_keys = h_keys;
  d_lengths = h_lengths;
  d_find_keys = h_keys;
  d_find_lengths = h_lengths;

  // assign values
  auto to_value = [](key_slice_type* key, size_type length) {
    // fnv1a_32 hash
    const unsigned char* byte_data = (const unsigned char*)key;
    uint32_t hash = 2166136261UL;
    for (size_t i = 0; i < sizeof(key_slice_type) * length; ++i) {
        hash ^= byte_data[i];
        hash *= 16777619UL;
    }
    return hash;
  };
  std::vector<value_type> h_values(num_keys, invalid_value);
  std::vector<std::thread> value_threads;
  const uint32_t num_threads = std::thread::hardware_concurrency();
  for (uint32_t tid = 0; tid < num_threads; tid++) {
    value_threads.emplace_back([&](uint32_t tid) {
      for (uint32_t i = tid; i < num_keys; i += num_threads) {
        h_values[i] = to_value(&h_keys[i * max_key_length], h_lengths[i]);
      }
    }, tid);
  }
  for (uint32_t tid = 0; tid < num_threads; tid++) {
    value_threads[tid].join();
  }
  d_values = h_values;

  std::cout << "Benchmarking...\n";
  std::cout << "num_keys = " << num_keys << ", ";
  std::cout << "min_key_length = " << min_key_length << ", ";
  std::cout << "max_key_length = " << max_key_length << ", ";
  std::cout << "common_prefix_ratio = " << common_prefix_ratio << std::endl;
  using simple_bump_alloc_type = simple_bump_allocator<128>;
  using simple_slab_alloc_type = simple_slab_allocator<128>;
  using simple_dummy_reclaim_type = simple_dummy_reclaimer;
  using simple_debra_reclaim_type = simple_debra_reclaimer<>;
  using linearhashtable_slab_type = GpuLinearHashtable::gpu_linearhashtable<simple_slab_alloc_type, simple_dummy_reclaim_type>;
  using linearhashtable_slab_reclaim_type = GpuLinearHashtable::gpu_linearhashtable<simple_slab_alloc_type, simple_debra_reclaim_type>;

  std::cout << "Benchmarking linearhashtable_slab_reclaim_type readonlyfind prefix4longkey" << std::endl;
  bench_linearhashtable_insertion_find<linearhashtable_slab_reclaim_type, false, false, false>(
    d_keys, d_lengths, d_values, d_find_keys, d_find_lengths, d_results,
    num_keys, max_key_length, num_experiments, initial_directory_size, resize_policy, validate_result, validate_index
  );
  std::cout << "Benchmarking linearhashtable_slab_reclaim_type concurrentfind prefix4longkey" << std::endl;
  bench_linearhashtable_insertion_find<linearhashtable_slab_reclaim_type, true, false, false>(
    d_keys, d_lengths, d_values, d_find_keys, d_find_lengths, d_results,
    num_keys, max_key_length, num_experiments, initial_directory_size, resize_policy, validate_result, validate_index
  );
  std::cout << "Benchmarking linearhashtable_slab_reclaim_type readonlyfind hash4longkey" << std::endl;
  bench_linearhashtable_insertion_find<linearhashtable_slab_reclaim_type, false, true, false>(
    d_keys, d_lengths, d_values, d_find_keys, d_find_lengths, d_results,
    num_keys, max_key_length, num_experiments, initial_directory_size, resize_policy, validate_result, validate_index
  );
  std::cout << "Benchmarking linearhashtable_slab_reclaim_type concurrentfind hash4longkey" << std::endl;
  bench_linearhashtable_insertion_find<linearhashtable_slab_reclaim_type, true, true, false>(
    d_keys, d_lengths, d_values, d_find_keys, d_find_lengths, d_results,
    num_keys, max_key_length, num_experiments, initial_directory_size, resize_policy, validate_result, validate_index
  );
  std::cout << "Benchmarking linearhashtable_slab_reclaim_type readonlyfind samehash4longkey" << std::endl;
  bench_linearhashtable_insertion_find<linearhashtable_slab_reclaim_type, false, true, true>(
    d_keys, d_lengths, d_values, d_find_keys, d_find_lengths, d_results,
    num_keys, max_key_length, num_experiments, initial_directory_size, resize_policy, validate_result, validate_index
  );
  std::cout << "Benchmarking linearhashtable_slab_reclaim_type concurrentfind samehash4longkey" << std::endl;
  bench_linearhashtable_insertion_find<linearhashtable_slab_reclaim_type, true, true, true>(
    d_keys, d_lengths, d_values, d_find_keys, d_find_lengths, d_results,
    num_keys, max_key_length, num_experiments, initial_directory_size, resize_policy, validate_result, validate_index
  );
  
}
