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
template <typename hashtable_type, bool do_merge, bool use_hash_for_longkey>
bench_rates bench_hashtable_insertion_erase(thrust::device_vector<key_slice_type>& d_keys,
                                            thrust::device_vector<size_type>& d_lengths,
                                            thrust::device_vector<value_type>& d_values,
                                            thrust::device_vector<key_slice_type>& d_query_keys,
                                            thrust::device_vector<size_type>& d_query_lengths,
                                            uint32_t num_keys,
                                            float fill_factor,
                                            uint32_t max_key_length,
                                            float erase_ratio,
                                            std::size_t num_experiments) {
  cudaStream_t insertion_stream{0};
  cudaStream_t erase_stream{0};
  float average_insertion_seconds(0.0f);
  float average_erase_seconds(0.0f);

  for (std::size_t exp = 0; exp < num_experiments; exp++) {
    typename hashtable_type::host_allocator_type host_alloc;
    typename hashtable_type::host_reclaimer_type host_reclaim;
    hashtable_type table(host_alloc, host_reclaim, num_keys, fill_factor);
    auto memory_usage = utils::compute_device_memory_usage();
    std::cout << "Using: " << double(memory_usage.used_bytes) / double(1 << 30) << " GiBs"
              << std::endl;
    gpu_timer insert_timer(insertion_stream);
    insert_timer.start_timer();
    table.insert(d_keys.data().get(), max_key_length, d_lengths.data().get(), d_values.data().get(), num_keys, insertion_stream, false, use_hash_for_longkey);
    insert_timer.stop_timer();
    cuda_try(cudaDeviceSynchronize());
    auto insertion_elapsed = insert_timer.get_elapsed_s();
    average_insertion_seconds += insertion_elapsed;

    gpu_timer erase_timer(erase_stream);
    uint32_t num_erase = (uint32_t)(((float)num_keys) * erase_ratio);
    erase_timer.start_timer();
    table.erase(d_query_keys.data().get(), max_key_length, d_query_lengths.data().get(), num_erase, erase_stream, do_merge, use_hash_for_longkey);
    erase_timer.stop_timer();
    cuda_try(cudaDeviceSynchronize());
    auto erase_elapsed = erase_timer.get_elapsed_s();
    average_erase_seconds += erase_elapsed;
    std::cout << exp << std::setw(6) << '\t';
    std::cout << insertion_elapsed << std::setw(6) << '\t';
    std::cout << erase_elapsed << std::setw(6) << '\n';
  }

  average_insertion_seconds /= float(num_experiments);
  average_erase_seconds /= float(num_experiments);
  float insertion_rate = float(d_lengths.size()) / 1e6 / average_insertion_seconds;
  float erase_rate      = float(d_query_lengths.size()) * erase_ratio / 1e6 / average_erase_seconds;
  std::cout << "insertion_rate: " << insertion_rate << '\n';
  std::cout << "erase_rate: " << erase_rate << std::endl;
  return {insertion_rate, erase_rate};
}

int main(int argc, char** argv) {
  auto arguments    = std::vector<std::string>(argv, argv + argc);
  uint32_t num_keys = get_arg_value<uint32_t>(arguments, "num-keys").value_or(1'000'000);
  float fill_factor = get_arg_value<float>(arguments, "fill-factor").value_or(1.0f);
  int device_id     = get_arg_value<int>(arguments, "device").value_or(0);
  uint32_t min_key_length = get_arg_value<uint32_t>(arguments, "min-key-length").value_or(1u);
  uint32_t max_key_length = get_arg_value<uint32_t>(arguments, "max-key-length").value_or(1u);
  float common_prefix_ratio = get_arg_value<float>(arguments, "common-prefix-ratio").value_or(0.1f);
  float erase_ratio = get_arg_value<float>(arguments, "erase-ratio").value_or(0.1f);
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
  std::cout << "fill_factor = " << fill_factor << ", ";
  std::cout << "min_key_length = " << min_key_length << ", ";
  std::cout << "max_key_length = " << max_key_length << ", ";
  std::cout << "common_prefix_ratio = " << common_prefix_ratio << std::endl;
  using simple_bump_alloc_type = simple_bump_allocator<128>;
  using simple_slab_alloc_type = simple_slab_allocator<128>;
  using simple_dummy_reclaim_type = simple_dummy_reclaimer;
  using simple_debra_reclaim_type = simple_debra_reclaimer<131072>;
  using chainhashtable_slab_type = GpuHashtable::gpu_chainhashtable<simple_slab_alloc_type, simple_dummy_reclaim_type>;
  using chainhashtable_slab_reclaim_type = GpuHashtable::gpu_chainhashtable<simple_slab_alloc_type, simple_debra_reclaim_type>;
  using cuckoohashtable_slab_type = GpuHashtable::gpu_cuckoohashtable<simple_slab_alloc_type, simple_dummy_reclaim_type>;
  using cuckoohashtable_slab_reclaim_type = GpuHashtable::gpu_cuckoohashtable<simple_slab_alloc_type, simple_debra_reclaim_type>;

  std::cout << "Benchmarking chainhashtable_slab_reclaim_type no-merge prefix4longkey" << std::endl;
  bench_hashtable_insertion_erase<chainhashtable_slab_reclaim_type, false, false>(
    d_keys, d_lengths, d_values, d_find_keys, d_find_lengths,
    num_keys, fill_factor, max_key_length, erase_ratio, num_experiments
  );
  std::cout << "Benchmarking chainhashtable_slab_reclaim_type merge prefix4longkey" << std::endl;
  bench_hashtable_insertion_erase<chainhashtable_slab_reclaim_type, true, false>(
    d_keys, d_lengths, d_values, d_find_keys, d_find_lengths,
    num_keys, fill_factor, max_key_length, erase_ratio, num_experiments
  );
  std::cout << "Benchmarking chainhashtable_slab_reclaim_type no-merge hash4longkey" << std::endl;
  bench_hashtable_insertion_erase<chainhashtable_slab_reclaim_type, false, true>(
    d_keys, d_lengths, d_values, d_find_keys, d_find_lengths,
    num_keys, fill_factor, max_key_length, erase_ratio, num_experiments
  );
  std::cout << "Benchmarking chainhashtable_slab_reclaim_type merge hash4longkey" << std::endl;
  bench_hashtable_insertion_erase<chainhashtable_slab_reclaim_type, true, true>(
    d_keys, d_lengths, d_values, d_find_keys, d_find_lengths,
    num_keys, fill_factor, max_key_length, erase_ratio, num_experiments
  );
  std::cout << "Benchmarking cuckoohashtable_slab_reclaim_type prefix4longkey" << std::endl;
  bench_hashtable_insertion_erase<cuckoohashtable_slab_reclaim_type, false, false>(
    d_keys, d_lengths, d_values, d_find_keys, d_find_lengths,
    num_keys, fill_factor, max_key_length, erase_ratio, num_experiments
  );
  std::cout << "Benchmarking cuckoohashtable_slab_reclaim_type hash4longkey" << std::endl;
  bench_hashtable_insertion_erase<cuckoohashtable_slab_reclaim_type, false, true>(
    d_keys, d_lengths, d_values, d_find_keys, d_find_lengths,
    num_keys, fill_factor, max_key_length, erase_ratio, num_experiments
  );
}
