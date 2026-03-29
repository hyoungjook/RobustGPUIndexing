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
#pragma once
#include <iostream>
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>
#include <thread>
#include <macros.hpp>

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
                         double keylen_theta,
                         bool big_endian) {
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
  const unsigned num_workers = std::max(1u, std::thread::hardware_concurrency());
  std::vector<std::thread> workers;
  for (unsigned tid = 0; tid < num_workers; tid++) {
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
        // big endian
        if (big_endian) {
          for (uint32_t slice = 0; slice < length; slice++) {
            uint32_t key_slice = key[slice];
            key[slice] = __builtin_bswap32(key_slice);
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
                          double lookup_exist_ratio,
                          bool big_endian) {
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
  const unsigned num_workers = std::max(1u, std::thread::hardware_concurrency());
  std::vector<std::thread> workers;
  for (unsigned tid = 0; tid < num_workers; tid++) {
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
          // big endian
          if (big_endian) {
            for (uint32_t slice = 0; slice < length; slice++) {
              uint32_t key_slice = lookup_key[slice];
              lookup_key[slice] = __builtin_bswap32(key_slice);
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

} // namespace universal
