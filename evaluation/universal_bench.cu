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
#include <random>
#include <string>
#include <unordered_set>
#include <vector>
#include <thread>
#include <mutex>
#include <cmd.hpp>
#include <gpu_timer.hpp>
#include <gpu_masstree_adapter.hpp>

using key_slice_type = uint32_t;
using value_type = uint32_t;
using size_type = uint32_t;

namespace universal {

template <typename T>
struct zipfian_int_distribution {
  // theta = 0 equals to uniform distribution
 public:
  zipfian_int_distribution(T min_value, T max_value, double theta) 
      : min_value_(min_value)
      , max_value_(max_value)
      , theta_(theta) {
    check_argument(0.0 <= theta && theta < 1.0);
    check_argument(min_value <= max_value);
    uint64_t num_values = max_value_ - min_value_ + 1;
    zetan_ = zeta(num_values);
    alpha_ = 1.0 / (1.0 - theta_);
    eta_ = (1 - std::pow(2.0 / num_values, 1 - theta_)) / (1 - zeta(2) / zetan_);
  }
  T operator()(std::mt19937& rng) {
    std::uniform_real_distribution<double> double_dist(0.0, std::nextafter(1.0, 2.0));
    double u = double_dist(rng);
    double uz = u * zetan_;
    uint64_t base;
    if (uz < 1.0) { base = 0; }
    else if (uz < 1.0 + std::pow(0.5, theta_)) { base = 1; }
    else {
      uint64_t max_base = max_value_ - min_value_;
      base = static_cast<uint64_t>(max_base * std::pow(eta_ * u - eta_ + 1, alpha_));
    }
    return min_value_ + static_cast<T>(base);
  }
 private:
  double zeta(uint64_t n) {
    double sum = 0;
    for (uint64_t i = 0; i < n; i++) {
      sum += (1.0 / std::pow(i + 1, theta_));
    }
    return sum;
  }
  uint64_t min_value_, max_value_;
  double theta_, alpha_, eta_, zetan_;
};

std::size_t key_hasher(const key_slice_type* key, size_type length) {
  std::size_t hash = 0;
  for (size_type i = 0; i < length; i++) {
    hash ^= std::hash<uint32_t>{}(key[i]) + 0x9e3779b9 + (hash<<6) + (hash>>2);
  }
  return hash;
}

void generate_key_values(std::vector<key_slice_type>& keys,
                         std::vector<size_type>& key_lengths,
                         std::vector<value_type>& values,
                         std::size_t num_keys,
                         uint32_t keylen_prefix,
                         uint32_t keylen_min,
                         uint32_t keylen_max,
                         double keylen_theta) {
  // key: [common prefix (keylen_prefix)], [random slices], [unique_id]
  // normally unique_id is 1 slice, but if num_keys exceeds uint32, should be 2 slices
  // also spare uint32_max to represent non-existing key
  const uint32_t unique_slices = (num_keys < std::numeric_limits<key_slice_type>::max()) ? 1 : 2;
  check_argument(
    keylen_prefix + unique_slices <= keylen_min &&
    keylen_min <= keylen_max
  );
  // generate prefix
  std::mt19937 main_thd_rng(0);
  std::vector<key_slice_type> prefix(keylen_prefix);
  for (uint32_t slice = 0; slice < keylen_prefix; slice++) {
    std::uniform_int_distribution<key_slice_type> prefix_dist(0, std::numeric_limits<key_slice_type>::max());
    prefix[slice] = prefix_dist(main_thd_rng);
  }
  // generate unique_id
  std::vector<std::size_t> unique_id_mix(num_keys);
  for (std::size_t i = 0; i < num_keys; i++) unique_id_mix[i] = i;
  std::shuffle(unique_id_mix.begin(), unique_id_mix.end(), main_thd_rng);
  // generate keys
  keys = std::vector<key_slice_type>(num_keys * keylen_max);
  key_lengths = std::vector<size_type>(num_keys);
  values = std::vector<value_type>(num_keys);
  std::atomic<std::size_t> num_generated(0);
  zipfian_int_distribution<key_slice_type> length_dist(keylen_min, keylen_max, keylen_theta);
  std::uniform_int_distribution<key_slice_type> slice_dist(0, std::numeric_limits<key_slice_type>::max());
  std::vector<std::thread> workers;
  for (unsigned tid = 0; tid < std::thread::hardware_concurrency(); tid++) {
    workers.emplace_back([&](unsigned thread_id) {
      std::mt19937 per_thd_rng(1 + thread_id);
      while (true) {
        // check idx
        auto key_idx = num_generated.fetch_add(1);
        if (key_idx >= num_keys) { break; }
        auto* key = &keys[key_idx * keylen_max];
        // decide key length
        uint32_t length = length_dist(per_thd_rng);
        key_lengths[key_idx] = length;
        // fill slices
        for (uint32_t slice = 0; slice < length; slice++) {
          //  key[0:keylen_prefix) = prefix[]
          //  key[keylen_prefix:length-unique_slices) = random
          //  key[length-unique_slices:length-1] = key_idx
          if (slice < keylen_prefix) {
            key[slice] = prefix[slice];
          }
          else if (slice < length - unique_slices) {
            key[slice] = slice_dist(per_thd_rng);
          }
          else if (slice == length - 2) {
            key[slice] = static_cast<key_slice_type>(unique_id_mix[key_idx] >> (sizeof(key_slice_type) * 8));
          }
          else {
            key[slice] = static_cast<key_slice_type>(unique_id_mix[key_idx]);
          }
        }
        // compute value
        values[key_idx] = key_hasher(key, length);
      }
    }, tid);
  }
  for (auto& w: workers) {
    w.join();
  }
}

void generate_lookup_keys(std::vector<key_slice_type>& lookup_keys,
                          std::vector<size_type>& lookup_key_lengths,
                          std::vector<key_slice_type>& keys,
                          std::vector<size_type>& key_lengths,
                          std::size_t num_keys,
                          uint32_t keylen_prefix,
                          uint32_t keylen_min,
                          uint32_t keylen_max,
                          double keylen_theta,
                          std::size_t num_queries,
                          double lookup_theta,
                          double lookup_exist_ratio) {
  // randomly select lookup key from given keys
  // not-existing key is made with unique index num_keys
  check_argument(0.0 <= lookup_exist_ratio && lookup_exist_ratio <= 1.0);
  const uint32_t unique_slices = (num_keys < std::numeric_limits<key_slice_type>::max()) ? 1 : 2;
  // generate queries
  lookup_keys = std::vector<key_slice_type>(num_queries * keylen_max);
  lookup_key_lengths = std::vector<size_type>(num_queries);
  std::atomic<std::size_t> num_generated(0);
  std::uniform_real_distribution<double> exist_dist(0.0, 1.0);
  zipfian_int_distribution<std::size_t> key_choose_dist(0, num_keys - 1, lookup_theta);
  zipfian_int_distribution<uint32_t> length_dist(keylen_min, keylen_max, keylen_theta);
  std::uniform_int_distribution<key_slice_type> slice_dist(0, std::numeric_limits<key_slice_type>::max());
  std::vector<std::thread> workers;
  for (unsigned tid = 0; tid < std::thread::hardware_concurrency(); tid++) {
    workers.emplace_back([&](unsigned thread_id) {
      std::mt19937 per_thd_rng(1 + thread_id);
      while (true) {
        // check idx
        auto lookup_idx = num_generated.fetch_add(1);
        if (lookup_idx >= num_queries) { break; }
        auto* lookup_key = &lookup_keys[lookup_idx * keylen_max];
        // decide existance
        if (exist_dist(per_thd_rng) < lookup_exist_ratio) {
          // copy randomly from key
          std::size_t key_idx = key_choose_dist(per_thd_rng);
          uint32_t length = key_lengths[key_idx];
          lookup_key_lengths[lookup_idx] = length;
          memcpy(lookup_key, &keys[key_idx * keylen_max], sizeof(key_slice_type) * length);
        }
        else {
          // create non-exist key
          uint32_t length = length_dist(per_thd_rng);
          lookup_key_lengths[lookup_idx] = length;
          for (uint32_t slice = 0; slice < length; slice++) {
            // key[0:keylen_prefix) = prefix
            // key[keylen_prefix:length-unique_slices) = random
            // key[length-unique_slices:length-1] = num_keys
            if (slice < keylen_prefix) {
              lookup_key[slice] = keys[slice];
            }
            else if (slice < length - unique_slices) {
              lookup_key[slice] = slice_dist(per_thd_rng);
            }
            else if (slice == length - 2) {
              lookup_key[slice] = static_cast<key_slice_type>(num_keys >> (sizeof(key_slice_type) * 8));
            }
            else {
              lookup_key[slice] = static_cast<key_slice_type>(num_keys);
            }
          }
        }
      }
    }, tid);
  }
  for (auto& w: workers) {
    w.join();
  }
}

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
      adapter.scan(lookup_keys.data().get(), keylen_max, lookup_key_lengths.data().get(), scan_count, results.data().get(), num_scans);
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

  #define FORALL_INDEXES(x) \
  x(gpu_masstree)

  #define INDEX_NAME_CHECK(index) (index_type == #index) ||
  check_argument(FORALL_INDEXES(INDEX_NAME_CHECK) false);
  #undef INDEX_NAME_CHECK
  #define ADAPTER_DECLARE(index) index##_adapter index##_adapter_;
  FORALL_INDEXES(ADAPTER_DECLARE)
  #undef DECLARE_ADAPTERS
  #define ADAPTER_PARSE_ARGS(index) \
  if (index_type == #index) { index##_adapter_.parse(arguments); }
  FORALL_INDEXES(ADAPTER_PARSE_ARGS)
  #undef ADAPTER_PARSE_ARGS

  // print arguments
  repeats_insert = max(repeats_insert, repeats_delete);
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
                           num_keys, keylen_prefix, keylen_min, keylen_max, keylen_theta);
  if (repeats_lookup > 0) {
    universal::generate_lookup_keys(h_lookup_keys, h_lookup_key_lengths, h_keys, h_key_lengths,
                                    num_keys, keylen_prefix, keylen_min, keylen_max, keylen_theta,
                                    num_lookups, lookup_theta, lookup_exist_ratio);
  }
  std::size_t results_buffer_size = max(
    (repeats_lookup > 0) ? num_lookups : 0,
    (repeats_scan > 0) ? (num_scans * scan_count) : 0
  );

  // copy vectors to device
  auto d_keys = thrust::device_vector<key_slice_type>(h_keys.size());
  auto d_key_lengths = thrust::device_vector<size_type>(h_key_lengths.size());
  auto d_values = thrust::device_vector<value_type>(h_values.size());
  auto d_lookup_keys = thrust::device_vector<key_slice_type>(h_lookup_keys.size());
  auto d_lookup_key_lengths = thrust::device_vector<key_slice_type>(h_lookup_key_lengths.size());
  auto d_results = thrust::device_vector<value_type>(results_buffer_size);
  d_keys = h_keys;
  d_key_lengths = h_key_lengths;
  d_values = h_values;
  d_lookup_keys = h_lookup_keys;
  d_lookup_key_lengths = h_lookup_key_lengths;

  // run benchmark
  #define ADAPTER_RUN_BENCH(index) \
  if (index_type == #index) { \
    universal::run_bench(index##_adapter_, \
      keylen_max, d_keys, d_key_lengths, d_values, num_keys, num_deletes, \
      d_lookup_keys, d_lookup_key_lengths, num_lookups, \
      scan_count, num_scans, d_results, \
      repeats_insert, repeats_delete, repeats_lookup, repeats_scan); \
  }
  FORALL_INDEXES(ADAPTER_RUN_BENCH)
  #undef ADAPTER_RUN_BENCH
}
