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

template <typename masstree_type,
          bool enable_suffix = true,
          bool erase_remove_empty_root = true,
          bool erase_merge = true,
          bool reuse_root = true>
void mix_bench_masstree(thrust::device_vector<key_slice_type>& d_insert_keys,
                        thrust::device_vector<size_type>& d_insert_lengths,
                        thrust::device_vector<value_type>& d_insert_values,
                        uint32_t insert_num_keys,
                        thrust::device_vector<kernels::request_type>& d_mix_types,
                        thrust::device_vector<key_slice_type>& d_mix_keys,
                        thrust::device_vector<size_type>& d_mix_lengths,
                        thrust::device_vector<value_type>& d_mix_values,
                        uint32_t mix_num_requests,
                        uint32_t max_key_length,
                        std::size_t num_experiments,
                        bool verbose = false) {
  float average_insert_seconds = 0.f;
  float average_mix_seconds = 0.f;

  for (std::size_t exp = 0; exp < num_experiments; exp++) {
    typename masstree_type::host_allocator_type host_alloc;
    typename masstree_type::host_reclaimer_type host_reclaim;
    masstree_type tree(host_alloc, host_reclaim);
    if (verbose) {
      auto memory_usage = utils::compute_device_memory_usage();
      std::cout << "Using: " << double(memory_usage.used_bytes) / double(1 << 30) << " GiBs"
                << std::endl;
    }
    gpu_timer insert_timer;
    insert_timer.start_timer();
    tree.template insert<enable_suffix, reuse_root>(
      d_insert_keys.data().get(), max_key_length, d_insert_lengths.data().get(), d_insert_values.data().get(), insert_num_keys);
    insert_timer.stop_timer();
    cuda_try(cudaDeviceSynchronize());
    float insert_elapsed = insert_timer.get_elapsed_s();
    average_insert_seconds += insert_elapsed;

    gpu_timer mix_timer;
    mix_timer.start_timer();
    tree.template mixed_batch<enable_suffix,
                              erase_remove_empty_root,
                              erase_merge || erase_remove_empty_root,
                              reuse_root>(
      d_mix_types.data().get(), d_mix_keys.data().get(), max_key_length, d_mix_lengths.data().get(), d_mix_values.data().get(), nullptr, mix_num_requests);
    mix_timer.stop_timer();
    cuda_try(cudaDeviceSynchronize());
    float mix_elapsed = mix_timer.get_elapsed_s();
    average_mix_seconds += mix_elapsed;

    if (verbose) {
      std::cout << exp << " "
                << insert_elapsed << " "
                << mix_elapsed << std::endl;
    }
  }

  average_insert_seconds /= float(num_experiments);
  average_mix_seconds /= float(num_experiments);
  float insertion_rate = float(insert_num_keys) / 1e6 / average_insert_seconds;
  float mix_rate = float(mix_num_requests) / 1e6 / average_mix_seconds;
  std::cout << "insert: " << insertion_rate << " Mop/s" << std::endl;
  std::cout << "mixed: " << mix_rate << " Mop/s" << std::endl;
}

int main(int argc, char** argv) {
  auto arguments    = std::vector<std::string>(argv, argv + argc);
  uint32_t num_keys = get_arg_value<uint32_t>(arguments, "num-keys").value_or(1'000'000);
  int device_id     = get_arg_value<int>(arguments, "device").value_or(0);
  uint32_t min_key_length = get_arg_value<uint32_t>(arguments, "min-key-length").value_or(1u);
  uint32_t max_key_length = get_arg_value<uint32_t>(arguments, "max-key-length").value_or(1u);
  float common_prefix_ratio = get_arg_value<float>(arguments, "common-prefix-ratio").value_or(0.1f);
  float insert_ratio = get_arg_value<float>(arguments, "insert-ratio").value_or(0.33f);
  float erase_ratio = get_arg_value<float>(arguments, "erase-ratio").value_or(0.33f);
  float find_ratio = 1.0f - insert_ratio - erase_ratio;
  if (find_ratio < 0.f) {
    std::cerr << "insert-ratio " << insert_ratio << " + erase_ratio " << erase_ratio << " > 1" << std::endl;
  }
  bool verbose   = get_arg_value<bool>(arguments, "verbose").value_or(false);
  std::string dataset_file = get_arg_value<std::string>(arguments, "dataset-file").value_or("");
  std::size_t num_experiments = get_arg_value<std::size_t>(arguments, "num-experiments").value_or(1llu);
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

  uint32_t half_num_keys = num_keys / 2;
  std::random_device rd;
  //std::mt19937 rng(0);
  std::mt19937 rng(rd());

  // device vectors
  auto d_keys      = thrust::device_vector<key_slice_type>(num_keys * max_key_length, 0);
  auto d_lengths      = thrust::device_vector<size_type>(num_keys, 0);
  auto d_values    = thrust::device_vector<value_type>(num_keys, invalid_value);

  // host vectors
  std::vector<key_slice_type> h_keys;
  std::vector<size_type> h_lengths;
  if (dataset_file != "") {
    rkg::generate_varlen_keys_from_dataset(dataset, h_keys, h_lengths, num_keys, max_key_length);
  }
  else {
    rkg::generate_varlen_keys<key_slice_type, size_type>(
      h_keys, h_lengths, num_keys, min_key_length, max_key_length, rng, rkg::distribution_type::unique_random,
      common_prefix_ratio);
  }

  // copy to device
  d_keys = h_keys;
  d_lengths = h_lengths;

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

  // generate mix
  auto d_mix_types = thrust::device_vector<kernels::request_type>(half_num_keys);
  auto d_mix_keys      = thrust::device_vector<key_slice_type>(half_num_keys * max_key_length, 0);
  auto d_mix_lengths      = thrust::device_vector<size_type>(half_num_keys, 0);
  auto d_mix_values    = thrust::device_vector<value_type>(half_num_keys, invalid_value);
  std::vector<uint32_t> mix_order(half_num_keys);
  std::vector<uint32_t> insert_mix_order(half_num_keys);
  std::vector<uint32_t> erase_mix_order(half_num_keys);
  std::vector<uint32_t> find_mix_order(2 * half_num_keys);
  for (uint32_t i = 0; i < half_num_keys; i++) {
    mix_order[i] = i;
    insert_mix_order[i] = i;
    erase_mix_order[i] = i;
  }
  for (uint32_t i = 0; i < 2 * half_num_keys; i++) {
    find_mix_order[i] = i;
  }
  std::shuffle(mix_order.begin(), mix_order.end(), rng);
  std::shuffle(insert_mix_order.begin(), insert_mix_order.end(), rng);
  std::shuffle(erase_mix_order.begin(), erase_mix_order.end(), rng);
  std::shuffle(find_mix_order.begin(), find_mix_order.end(), rng);
  uint32_t mix_num_inserts = (uint32_t)(((float)half_num_keys) * insert_ratio);
  uint32_t mix_num_erases = (uint32_t)(((float)half_num_keys) * erase_ratio);
  std::vector<kernels::request_type> h_mix_types(half_num_keys);
  std::vector<key_slice_type> h_mix_keys(half_num_keys * max_key_length);
  std::vector<size_type> h_mix_lengths(half_num_keys);
  std::vector<value_type> h_mix_values(half_num_keys);
  for (uint32_t src_i = 0; src_i < half_num_keys; src_i++) {
    uint32_t dst_i = mix_order[src_i];
    if (src_i < mix_num_inserts) {
      h_mix_types[dst_i] = kernels::request_type_insert;
      uint32_t src_key_i = half_num_keys + insert_mix_order[src_i];
      for (uint32_t s = 0; s < max_key_length; s++) {
        h_mix_keys[dst_i * max_key_length + s] = h_keys[src_key_i * max_key_length + s];
      }
      h_mix_lengths[dst_i] = h_lengths[src_key_i];
      h_mix_values[dst_i] = h_values[src_key_i];
    }
    else if (src_i < mix_num_inserts + mix_num_erases) {
      h_mix_types[dst_i] = kernels::request_type_erase;
      uint32_t src_key_i = erase_mix_order[src_i - mix_num_inserts];
      for (uint32_t s = 0; s < max_key_length; s++) {
        h_mix_keys[dst_i * max_key_length + s] = h_keys[src_key_i * max_key_length + s];
      }
      h_mix_lengths[dst_i] = h_lengths[src_key_i];
      h_mix_values[dst_i] = invalid_value;
    }
    else {
      h_mix_types[dst_i] = kernels::request_type_find;
      uint32_t src_key_i = find_mix_order[src_i - (mix_num_inserts + mix_num_erases)];
      for (uint32_t s = 0; s < max_key_length; s++) {
        h_mix_keys[dst_i * max_key_length + s] = h_keys[src_key_i * max_key_length + s];
      }
      h_mix_lengths[dst_i] = h_lengths[src_key_i];
      h_mix_values[dst_i] = invalid_value;
    }
  }
  d_mix_types = h_mix_types;
  d_mix_keys = h_mix_keys;
  d_mix_lengths = h_mix_lengths;
  d_mix_values = h_mix_values;

  std::cout << "Benchmarking...\n";
  std::cout << "num_keys = " << num_keys << ", ";
  std::cout << "min_key_length = " << min_key_length << ", ";
  std::cout << "max_key_length = " << max_key_length << ", ";
  std::cout << "common_prefix_ratio = " << common_prefix_ratio << " ";
  std::cout << "with workload mix insert:erase:find = "
    << insert_ratio << ":"
    << erase_ratio << ":"
    << find_ratio << std::endl;
  using simple_slab_alloc_type = simple_slab_allocator<128>;
  using simple_debra_reclaim_type = simple_debra_reclaimer<>;
  using masstree_type = GpuMasstree::gpu_masstree<simple_slab_alloc_type, simple_debra_reclaim_type>;

  std::cout << "Benchmarking masstree_type" << std::endl;
  mix_bench_masstree<masstree_type>(
    d_keys, d_lengths, d_values, half_num_keys,
    d_mix_types, d_mix_keys, d_mix_lengths, d_mix_values, half_num_keys,
    max_key_length, num_experiments, verbose
  );
  
}
