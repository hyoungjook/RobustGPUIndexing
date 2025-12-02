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

#pragma once
#include <stdlib.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <filesystem>
#include <numeric>
#include <pair_type.hpp>
#include <set>
#include <string>
#include <string_view>
#include <typeinfo>
#include <unordered_set>

namespace rkg {

template <typename key_type, typename value_type, typename size_type>
inline void prep_experiment_find_with_exist_ratio(float exist_ratio,
                                                  size_type num_keys,
                                                  const std::vector<key_type>& keys,
                                                  std::vector<key_type>& queries) {
  if (exist_ratio < 1.0f) { assert(num_keys * 2 == keys.size()); }
  unsigned int end_index   = static_cast<unsigned int>(num_keys * (-exist_ratio + 2));
  unsigned int start_index = end_index - num_keys;

  // Need to copy our range [start_index, end_index) from keys into queries.
  std::copy(keys.begin() + start_index, keys.begin() + end_index, queries.begin());
}

template <typename key_type, typename rng_type>
inline void prep_experiment_range_query(const std::vector<key_type>& keys,
                                        const std::size_t num_keys,
                                        std::vector<key_type>& queries,
                                        const std::size_t num_rqs,
                                        rng_type& rng) {
  // copy all keys
  queries.resize(std::max(num_keys, num_rqs));
  std::copy(keys.begin(), keys.begin() + num_keys, queries.begin());

  // sample the rest
  if (num_rqs > num_keys) {
    auto current_rqs = num_keys;
    while (current_rqs < num_rqs) {
      auto sampled         = rng() % num_keys;
      queries[current_rqs] = keys[sampled];
      current_rqs++;
    }
  }
  // shuffle
  std::shuffle(queries.begin(), queries.end(), rng);

  // bring size back to required number of queries
  queries.resize(num_rqs);
}

enum class distribution_type { unique_ascending, unique_descending, unique_random, has_duplicates };

template <typename key_type, typename rng_type>
inline std::vector<key_type> generate_keys(const uint32_t num_keys,
                                           rng_type& rng,
                                           const distribution_type dist,
                                           const float duplicates_ratio = 1.0f) {
  std::vector<key_type> keys(num_keys, 1);
  std::iota(keys.begin(), keys.end(), 1);
  uint32_t num_unique = 0;
  switch (dist) {
    case distribution_type::unique_random: std::shuffle(keys.begin(), keys.end(), rng); break;
    case distribution_type::unique_ascending: break;
    case distribution_type::unique_descending: std::reverse(keys.begin(), keys.end()); break;
    case distribution_type::has_duplicates:
      num_unique = static_cast<uint32_t>(static_cast<double>(num_keys) * duplicates_ratio);
    default: break;
  }
  if (dist == distribution_type::has_duplicates) {
    for (size_t key_id = 0; key_id < num_keys; key_id++) { keys[key_id] = rng() % num_unique + 1; }
  }
  return keys;
}

template <typename key_slice_type, typename size_type, typename rng_type>
inline void generate_varlen_keys(std::vector<key_slice_type>& keys,
                                 std::vector<size_type>& key_lengths,
                                 const uint32_t num_keys,
                                 const uint32_t min_key_length,
                                 const uint32_t max_key_length,
                                 rng_type& rng,
                                 const distribution_type dist,
                                 const float common_prefix_ratio = 0.1f) {
  keys = std::vector<key_slice_type>(num_keys * max_key_length, 0);
  key_lengths = std::vector<size_type>(num_keys, 0);
  auto base_keys = generate_keys<key_slice_type>(num_keys, rng, dist);
  const uint32_t key_length_mod = max_key_length - min_key_length + 1;
  for (uint32_t i = 0; i < num_keys; i++) {
    const size_type length = min_key_length + (i % key_length_mod);
    key_lengths[i] = length;
    uint32_t prefix_denominator = 1;
    for (uint32_t s = 0; s < length; s++) {
      uint32_t keylen_idx = length - 1 - s;
      keys[i * max_key_length + keylen_idx] = base_keys[i] / prefix_denominator;
      prefix_denominator = (float)prefix_denominator / common_prefix_ratio;
      prefix_denominator = std::min<uint32_t>(prefix_denominator, num_keys);
    }
  }
}

}  // namespace rkg
