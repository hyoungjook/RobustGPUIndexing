/*
 *   Copyright 2022 The Regents of the University of California, Davis
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
#include <gpu_btree.h>
#include <stdlib.h>
#include <thrust/sequence.h>
#include <algorithm>
#include <cmd.hpp>
#include <cstdint>
#include <gpu_timer.hpp>
#include <numeric>
#include <random>
#include <rkg.hpp>
#include <string>
#include <unordered_set>
#include <validation.hpp>
#include <vector>
#include <thread>

#include <device_bump_allocator.hpp>
#include <slab_alloc.hpp>

using key_slice_type = uint32_t;
using value_type = uint32_t;
using size_type = uint32_t;

struct bench_rates {
  float insertion_rate;
  float find_rate;
};
template <typename BTree, bool use_masstree, bool fixlen_key>
bench_rates bench_masstree_insertion_find(thrust::device_vector<key_slice_type>& d_keys,
                                          thrust::device_vector<size_type>& d_lengths,
                                          thrust::device_vector<value_type>& d_values,
                                          thrust::device_vector<key_slice_type>& d_query_keys,
                                          thrust::device_vector<size_type>& d_query_lengths,
                                          thrust::device_vector<value_type>& d_query_results,
                                          uint32_t max_key_length,
                                          std::size_t num_experiments) {
  cudaStream_t insertion_stream{0};
  cudaStream_t find_stream{0};
  float average_insertion_seconds(0.0f);
  float average_find_seconds(0.0f);

  for (std::size_t exp = 0; exp < num_experiments; exp++) {
    BTree tree;
    auto memory_usage = utils::compute_device_memory_usage();
    std::cout << "Using: " << double(memory_usage.used_bytes) / double(1 << 30) << " GiBs"
              << std::endl;
    gpu_timer insert_timer(insertion_stream);
    insert_timer.start_timer();
    if constexpr (use_masstree) {
      if constexpr (fixlen_key) {
        tree.insert_fixlen(d_keys.data().get(), max_key_length, d_values.data().get(), d_lengths.size(), insertion_stream);
      }
      else {
        tree.insert_varlen(d_keys.data().get(), max_key_length, d_lengths.data().get(), d_values.data().get(), d_lengths.size(), insertion_stream);
      }
    }
    else {
      tree.insert(d_keys.data().get(), d_values.data().get(), d_keys.size(), insertion_stream);
    }
    insert_timer.stop_timer();
    cuda_try(cudaDeviceSynchronize());
    auto insertion_elapsed = insert_timer.get_elapsed_s();
    average_insertion_seconds += insertion_elapsed;

    gpu_timer find_timer(find_stream);
    find_timer.start_timer();
    if constexpr (use_masstree) {
      if constexpr (fixlen_key) {
        tree.find_fixlen(d_query_keys.data().get(), max_key_length, d_query_results.data().get(), d_query_lengths.size(), find_stream);
      }
      else {
        tree.find_varlen(d_query_keys.data().get(), max_key_length, d_query_lengths.data().get(), d_query_results.data().get(), d_query_lengths.size(), find_stream);
      }
    }
    else {
      tree.find(d_query_keys.data().get(), d_query_results.data().get(), d_query_keys.size(), find_stream);
    }
    find_timer.stop_timer();
    cuda_try(cudaDeviceSynchronize());
    auto find_elapsed = find_timer.get_elapsed_s();
    average_find_seconds += find_elapsed;
    std::cout << exp << std::setw(6) << '\t';
    std::cout << insertion_elapsed << std::setw(6) << '\t';
    std::cout << find_elapsed << std::setw(6) << '\n';
  }

  average_insertion_seconds /= float(num_experiments);
  average_find_seconds /= float(num_experiments);
  float insertion_rate = float(d_lengths.size()) / 1e6 / average_insertion_seconds;
  float find_rate      = float(d_query_lengths.size()) / 1e6 / average_find_seconds;
  std::cout << "insertion_rate: " << insertion_rate << '\n';
  std::cout << "find_rate: " << find_rate << std::endl;
  return {insertion_rate, find_rate};
}

int main(int argc, char** argv) {
  auto arguments    = std::vector<std::string>(argv, argv + argc);
  uint32_t num_keys = get_arg_value<uint32_t>(arguments, "num-keys").value_or(1'000'000);
  int device_id     = get_arg_value<int>(arguments, "device").value_or(0);
  uint32_t min_key_length = get_arg_value<uint32_t>(arguments, "min-key-length").value_or(1u);
  uint32_t max_key_length = get_arg_value<uint32_t>(arguments, "max-key-length").value_or(1u);
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

  unsigned seed = 0;
  std::random_device rd;
  std::mt19937 rng(seed);

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
  rkg::generate_varlen_keys<key_slice_type, size_type>(h_keys, h_lengths, num_keys, min_key_length, max_key_length, rng, rkg::distribution_type::unique_random);

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
    for (size_t i = 0; i < length; ++i) {
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
  std::cout << "max_key_length = " << max_key_length << std::endl;
  static constexpr int branching_factor = 16;
  using node_type           = GpuBTree::node_type<key_slice_type, value_type, branching_factor>;
  using slab_allocator_type = device_allocator::SlabAllocLight<node_type, 4, 1024 * 8, 32, 128>;
  using masstree_slab_type =
      GpuBTree::gpu_masstree<slab_allocator_type>;

  using slab_allocator_type_blink = device_allocator::SlabAllocLight<node_type, 4, 1024 * 8, 16, 128>;
  using blink_tree_slab_type =
      GpuBTree::gpu_blink_tree<key_slice_type, value_type, branching_factor, slab_allocator_type_blink>;

  {
    std::cout << "Benchmarking masstree_slab_type" << std::endl;
    bench_masstree_insertion_find<masstree_slab_type, true, false>(
      d_keys, d_lengths, d_values, d_find_keys, d_find_lengths, d_results,
      max_key_length, num_experiments
    );
  }
  if (min_key_length == max_key_length) {
    std::cout << "Benchmarking masstree_slab_type with fixlen keys" << std::endl;
    bench_masstree_insertion_find<masstree_slab_type, true, true>(
      d_keys, d_lengths, d_values, d_find_keys, d_find_lengths, d_results,
      max_key_length, num_experiments
    );
  }
  if (max_key_length == 1) {
    std::cout << "Benchmarking blink_tree_slab_type" << std::endl;
    bench_masstree_insertion_find<blink_tree_slab_type, false, false>(
      d_keys, d_lengths, d_values, d_find_keys, d_find_lengths, d_results,
      max_key_length, num_experiments
    );
  }
}
