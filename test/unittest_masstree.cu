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

#include <gpu_index.h>
#include <gtest/gtest.h>
#include <cmd.hpp>
#include <cstdint>
#include <random>

std::size_t num_keys;

namespace {
using key_slice_type   = uint32_t;
using value_type = uint32_t;
using size_type = uint32_t;

const auto invalid_value = std::numeric_limits<value_type>::max();
template <typename Map>
struct MapData {
  using map = Map;
  using host_allocator = typename Map::host_allocator_type;
  using host_reclaimer = typename Map::host_reclaimer_type;
};

template <class MapData>
class MapTest : public testing::Test {
 protected:
  MapTest() {
    host_allocator_ = new typename map_data::host_allocator();
    host_reclaimer_ = new typename map_data::host_reclaimer();
    map_ = new typename map_data::map(*host_allocator_, *host_reclaimer_);
  }
  ~MapTest() override {
    //host_allocator_->print_stats();
    delete map_;
    delete host_allocator_;
    delete host_reclaimer_;
  }
  using map_data = MapData;
  typename map_data::map* map_;
  typename map_data::host_allocator* host_allocator_;
  typename map_data::host_reclaimer* host_reclaimer_;
};

template <typename T>
struct mapped_vector {
  mapped_vector(std::size_t capacity) : capacity_(capacity) { allocate(capacity); }
  T& operator[](std::size_t index) { return dh_buffer_[index]; }
  ~mapped_vector() {}
  void free() {
    cuda_try(cudaDeviceSynchronize());
    cuda_try(cudaFreeHost(dh_buffer_));
  }
  T* data() const { return dh_buffer_; }
  std::size_t size() const { return capacity_; }

  std::vector<T> to_std_vector() {
    std::vector<T> copy(capacity_);
    for (std::size_t i = 0; i < capacity_; i++) { copy[i] = dh_buffer_[i]; }
    return copy;
  }

 private:
  void allocate(std::size_t count) { cuda_try(cudaMallocHost(&dh_buffer_, sizeof(T) * count)); }
  std::size_t capacity_;
  T* dh_buffer_;
};

struct testing_input {
  testing_input(std::size_t input_num_keys, uint32_t input_min_key_length, uint32_t input_max_key_length, bool duplicate = false)
      : num_keys(input_num_keys)
      , min_key_length(input_min_key_length)
      , max_key_length(input_max_key_length)
      , keys(num_keys * max_key_length)
      , lengths(num_keys)
      , values(num_keys)
      , values2(num_keys)
      , keys_not_exist(input_num_keys * max_key_length)
  {
    assert(min_key_length <= max_key_length);
    make_input(duplicate);
  }
  void make_input(bool duplicate = false) {
    uint32_t key_length_modulo = max_key_length - min_key_length + 1;
    for (std::size_t i = 0; i < num_keys; i++) {
      uint32_t key_length = min_key_length + (i % key_length_modulo);
      value_type value = 0;
      for (uint32_t s = 0; s < key_length; s++) {
        uint32_t common_prefix_factor = 1u << (key_length - 1 - s);
        uint32_t effective_i = i / common_prefix_factor;
        key_slice_type key_slice = static_cast<key_slice_type>(effective_i + 1) * 2 + s;
        keys[i * max_key_length + s] = key_slice;
        keys_not_exist[i * max_key_length + s] = key_slice + 1;
        value += key_slice;
      }
      lengths[i] = key_length;
      values[i] = value;
      values2[i] = value + 7;
    }
    if (duplicate) {
      for (std::size_t i = (num_keys + 1) / 2; i < num_keys; i++) {
        std::size_t src_i = i - ((num_keys + 1) / 2);
        for (uint32_t s = 0; s < max_key_length; s++) {
          keys[i * max_key_length + s] = keys[src_i * max_key_length + s];
          keys_not_exist[i * max_key_length + s] = keys_not_exist[src_i * max_key_length + s];
        }
        lengths[i] = lengths[src_i];
        values[i] = values[src_i];
        values2[i] = values[i] + 7;
      }
    }
  }
  void free() {
    keys.free();
    lengths.free();
    values.free();
    values2.free();
    keys_not_exist.free();
  }
  void sort() {
    // decide order
    std::vector<std::size_t> order(num_keys);
    for (std::size_t i = 0; i < num_keys; i++) order[i] = i;
    std::sort(order.begin(), order.end(), [&](const std::size_t& a, const std::size_t& b) {
      // returns true if a < b
      const key_slice_type* a_key = &keys[a * max_key_length];
      const key_slice_type* b_key = &keys[b * max_key_length];
      const size_type a_length = lengths[a], b_length = lengths[b];
      return std::lexicographical_compare(a_key, a_key + a_length, b_key, b_key + b_length);
    });
    // rearrange
    mapped_vector<key_slice_type> sorted_keys(keys.size());
    mapped_vector<size_type> sorted_lengths(lengths.size());
    mapped_vector<value_type> sorted_values(values.size());
    for (std::size_t i = 0; i < num_keys; i++) {
      std::size_t old_i = order[i];
      for (uint32_t s = 0; s < max_key_length; s++) {
        sorted_keys[i * max_key_length + s] = keys[old_i * max_key_length + s];
      }
      sorted_lengths[i] = lengths[old_i];
      sorted_values[i] = values[old_i];
    }
    keys.free();
    lengths.free();
    values.free();
    keys = sorted_keys;
    lengths = sorted_lengths;
    values = sorted_values;
  }
  void shuffle() {
    // decide order
    std::vector<std::size_t> order(num_keys);
    for (std::size_t i = 0; i < num_keys; i++) order[i] = i;
    std::mt19937 rng(0);
    std::shuffle(order.begin(), order.end(), rng);
    // rearrange
    mapped_vector<key_slice_type> sorted_keys(keys.size());
    mapped_vector<size_type> sorted_lengths(lengths.size());
    mapped_vector<value_type> sorted_values(values.size());
    for (std::size_t i = 0; i < num_keys; i++) {
      std::size_t old_i = order[i];
      for (uint32_t s = 0; s < max_key_length; s++) {
        sorted_keys[i * max_key_length + s] = keys[old_i * max_key_length + s];
      }
      sorted_lengths[i] = lengths[old_i];
      sorted_values[i] = values[old_i];
    }
    keys.free();
    lengths.free();
    values.free();
    keys = sorted_keys;
    lengths = sorted_lengths;
    values = sorted_values;
  }

  std::size_t num_keys;
  uint32_t min_key_length;
  uint32_t max_key_length;
  mapped_vector<key_slice_type> keys;
  mapped_vector<size_type> lengths;
  mapped_vector<value_type> values;
  mapped_vector<value_type> values2;
  mapped_vector<key_slice_type> keys_not_exist;
};

struct testing_range_input {
  testing_range_input(testing_input& input_, uint32_t input_num_queries, uint32_t input_max_count_per_query)
      : input(input_)
      , max_key_length(input.max_key_length)
      , num_queries(input_num_queries)
      , max_count_per_query(input_max_count_per_query)
      , lower_keys(num_queries * max_key_length)
      , upper_keys(num_queries * max_key_length)
      , lower_lengths(num_queries)
      , upper_lengths(num_queries)
      , counts(num_queries)
      , values(num_queries * max_count_per_query)
      , out_keys(num_queries * max_count_per_query * max_key_length)
      , out_key_lengths(num_queries * max_count_per_query)
  {
    input.sort();
    make_input();
  }
  void make_input() {
    for (uint32_t i = 0; i < num_queries; i++) {
      uint32_t begin = i % num_keys;
      size_type count = 1 + (i % (2* max_count_per_query));
      uint32_t end = begin + count - 1;
      if (end >= num_keys) { end = num_keys - 1; }
      count = end - begin + 1;
      for (uint32_t s = 0; s < max_key_length; s++) {
        lower_keys[i * max_key_length + s] = input.keys[begin * max_key_length + s];
      }
      lower_lengths[i] = input.lengths[begin];
      for (uint32_t s = 0; s < max_key_length; s++) {
        upper_keys[i * max_key_length + s] = input.keys[end * max_key_length + s];
      }
      upper_lengths[i] = input.lengths[end];
      counts[i] = min(count, max_count_per_query);
      for (uint32_t v = 0; v < counts[i]; v++) {
        values[i * max_count_per_query + v] = input.values[begin + v];
        for (uint32_t ss = 0; ss < max_key_length; ss++) {
          out_keys[i * max_count_per_query * max_key_length + v * max_key_length + ss] =
              input.keys[(begin + v) * max_key_length + ss];
        }
        out_key_lengths[i * max_count_per_query + v] = input.lengths[begin + v];
      }
    }
  }
  void free() {
    lower_keys.free();
    upper_keys.free();
    lower_lengths.free();
    upper_lengths.free();
    counts.free();
    values.free();
    out_keys.free();
    out_key_lengths.free();
  }

  testing_input& input;
  uint32_t max_key_length;
  uint32_t num_queries;
  uint32_t max_count_per_query;
  mapped_vector<key_slice_type> lower_keys, upper_keys;
  mapped_vector<size_type> lower_lengths, upper_lengths;
  mapped_vector<size_type> counts;
  mapped_vector<size_type> values;
  mapped_vector<key_slice_type> out_keys;
  mapped_vector<size_type> out_key_lengths;
};

using simple_bump_alloc_type = simple_bump_allocator<128>;
using simple_slab_alloc_type = simple_slab_allocator<128>;
using simple_dummy_reclaim_type = simple_dummy_reclaimer;
using simple_debra_reclaim_type = simple_debra_reclaimer<>;

typedef testing::Types<
    //MapData<GpuMasstree::gpu_masstree<simple_bump_alloc_type, simple_dummy_reclaim_type>>,
    //MapData<GpuMasstree::gpu_masstree<simple_slab_alloc_type, simple_dummy_reclaim_type>>,
    //MapData<GpuMasstree::gpu_masstree<simple_slab_alloc_type, simple_debra_reclaim_type>>,
    MapData<GpuMasstree::gpu_masstree_subwarp<simple_slab_alloc_type, simple_debra_reclaim_type>>>
    Implementations;

TYPED_TEST_SUITE(MapTest, Implementations);

template <typename map_type>
void validate(map_type* map, uint32_t min_key_length_bytes, uint32_t max_key_length_bytes) {
  const size_type min_key_length = min_key_length_bytes / sizeof(key_slice_type);
  const size_type max_key_length = max_key_length_bytes / sizeof(key_slice_type);
  testing_input input(num_keys, min_key_length, max_key_length);
  map->insert(input.keys.data(), max_key_length, input.lengths.data(), input.values.data(), num_keys);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  map->validate();
  input.free();
}

template <typename map_type>
void test_exist(map_type* map, uint32_t min_key_length_bytes, uint32_t max_key_length_bytes) {
  const size_type min_key_length = min_key_length_bytes / sizeof(key_slice_type);
  const size_type max_key_length = max_key_length_bytes / sizeof(key_slice_type);
  mapped_vector<value_type> find_results(num_keys);
  testing_input input(num_keys, min_key_length, max_key_length);
  map->insert(input.keys.data(), max_key_length, input.lengths.data(), input.values.data(), num_keys);
  cuda_try(cudaDeviceSynchronize());
  map->find(input.keys.data(), max_key_length, input.lengths.data(), find_results.data(), num_keys);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  for (std::size_t i = 0; i < num_keys; i++) {
    auto expected_value = input.values[i];
    auto found_value    = find_results[i];
    ASSERT_EQ(found_value, expected_value);
  }
  find_results.free();
  input.free();
}

template <typename map_type>
void test_notexist(map_type* map, uint32_t min_key_length_bytes, uint32_t max_key_length_bytes) {
  const size_type min_key_length = min_key_length_bytes / sizeof(key_slice_type);
  const size_type max_key_length = max_key_length_bytes / sizeof(key_slice_type);
  mapped_vector<value_type> find_results(num_keys);
  testing_input input(num_keys, min_key_length, max_key_length);
  map->insert(input.keys.data(), max_key_length, input.lengths.data(), input.values.data(), num_keys);
  cuda_try(cudaDeviceSynchronize());
  map->find(input.keys_not_exist.data(), max_key_length, input.lengths.data(), find_results.data(), num_keys);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  for (std::size_t i = 0; i < num_keys; i++) {
    auto expected_value = invalid_value;
    auto found_value    = find_results[i];
    ASSERT_EQ(found_value, expected_value);
  }
  find_results.free();
  input.free();
}

template <typename map_type>
void test_update(map_type* map, uint32_t min_key_length_bytes, uint32_t max_key_length_bytes) {
  const size_type min_key_length = min_key_length_bytes / sizeof(key_slice_type);
  const size_type max_key_length = max_key_length_bytes / sizeof(key_slice_type);
  mapped_vector<value_type> find_results(num_keys);
  testing_input input(num_keys, min_key_length, max_key_length);
  map->insert(input.keys.data(), max_key_length, input.lengths.data(), input.values.data(), num_keys);
  cuda_try(cudaDeviceSynchronize());
  map->insert(input.keys.data(), max_key_length, input.lengths.data(), input.values2.data(), num_keys, 0, true);
  cuda_try(cudaDeviceSynchronize());
  map->find(input.keys.data(), max_key_length, input.lengths.data(), find_results.data(), num_keys);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  for (std::size_t i = 0; i < num_keys; i++) {
    auto expected_value = input.values2[i];
    auto found_value    = find_results[i];
    ASSERT_EQ(found_value, expected_value);
  }
  find_results.free();
  input.free();
}

template <typename map_type>
void test_eraseall(map_type* map, uint32_t min_key_length_bytes, uint32_t max_key_length_bytes) {
  const size_type min_key_length = min_key_length_bytes / sizeof(key_slice_type);
  const size_type max_key_length = max_key_length_bytes / sizeof(key_slice_type);
  mapped_vector<value_type> find_results(num_keys);
  testing_input input(num_keys, min_key_length, max_key_length);
  map->insert(input.keys.data(), max_key_length, input.lengths.data(), input.values.data(), num_keys);
  cuda_try(cudaDeviceSynchronize());
  map->erase(input.keys.data(), max_key_length, input.lengths.data(), num_keys);
  cuda_try(cudaDeviceSynchronize());
  map->find(input.keys.data(), max_key_length, input.lengths.data(), find_results.data(), num_keys);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  for (std::size_t i = 0; i < num_keys; i++) {
    auto expected_value = invalid_value;
    auto found_value    = find_results[i];
    ASSERT_EQ(found_value, expected_value);
  }
  map->validate();
  find_results.free();
  input.free();
}

template <typename map_type>
void test_erasenone(map_type* map, uint32_t min_key_length_bytes, uint32_t max_key_length_bytes) {
  const size_type min_key_length = min_key_length_bytes / sizeof(key_slice_type);
  const size_type max_key_length = max_key_length_bytes / sizeof(key_slice_type);
  mapped_vector<value_type> find_results(num_keys);
  testing_input input(num_keys, min_key_length, max_key_length);
  map->insert(input.keys.data(), max_key_length, input.lengths.data(), input.values.data(), num_keys);
  cuda_try(cudaDeviceSynchronize());
  map->erase(input.keys_not_exist.data(), max_key_length, input.lengths.data(), num_keys);
  cuda_try(cudaDeviceSynchronize());
  map->find(input.keys.data(), max_key_length, input.lengths.data(), find_results.data(), num_keys);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  for (std::size_t i = 0; i < num_keys; i++) {
    auto expected_value = input.values[i];
    auto found_value    = find_results[i];
    ASSERT_EQ(found_value, expected_value);
  }
  find_results.free();
  input.free();
}

template <typename map_type>
void test_inserttwiceeraseall(map_type* map, uint32_t min_key_length_bytes, uint32_t max_key_length_bytes) {
  const size_type min_key_length = min_key_length_bytes / sizeof(key_slice_type);
  const size_type max_key_length = max_key_length_bytes / sizeof(key_slice_type);
  mapped_vector<value_type> find_results(num_keys);
  testing_input input(num_keys, min_key_length, max_key_length, true);
  map->insert(input.keys.data(), max_key_length, input.lengths.data(), input.values.data(), num_keys);
  cuda_try(cudaDeviceSynchronize());
  map->erase(input.keys.data(), max_key_length, input.lengths.data(), num_keys);
  cuda_try(cudaDeviceSynchronize());
  map->find(input.keys.data(), max_key_length, input.lengths.data(), find_results.data(), num_keys);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  for (std::size_t i = 0; i < num_keys; i++) {
    auto expected_value = invalid_value;
    auto found_value    = find_results[i];
    ASSERT_EQ(found_value, expected_value);
  }
  find_results.free();
  input.free();
}

template <typename map_type>
void test_eraseallinsertall(map_type* map, uint32_t min_key_length_bytes, uint32_t max_key_length_bytes) {
  const size_type min_key_length = min_key_length_bytes / sizeof(key_slice_type);
  const size_type max_key_length = max_key_length_bytes / sizeof(key_slice_type);
  mapped_vector<value_type> find_results(num_keys);
  testing_input input(num_keys, min_key_length, max_key_length);
  map->insert(input.keys.data(), max_key_length, input.lengths.data(), input.values.data(), num_keys);
  cuda_try(cudaDeviceSynchronize());
  map->erase(input.keys.data(), max_key_length, input.lengths.data(), num_keys);
  cuda_try(cudaDeviceSynchronize());
  map->find(input.keys.data(), max_key_length, input.lengths.data(), find_results.data(), num_keys);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  for (std::size_t i = 0; i < num_keys; i++) {
    auto expected_value = invalid_value;
    auto found_value    = find_results[i];
    ASSERT_EQ(found_value, expected_value);
  }
  map->insert(input.keys.data(), max_key_length, input.lengths.data(), input.values.data(), num_keys);
  cuda_try(cudaDeviceSynchronize());
  map->find(input.keys.data(), max_key_length, input.lengths.data(), find_results.data(), num_keys);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  for (std::size_t i = 0; i < num_keys; i++) {
    auto expected_value = input.values[i];
    auto found_value    = find_results[i];
    ASSERT_EQ(found_value, expected_value);
  }
  find_results.free();
  input.free();
}

template <typename map_type>
void test_erasealltwice(map_type* map, uint32_t min_key_length_bytes, uint32_t max_key_length_bytes) {
  const size_type min_key_length = min_key_length_bytes / sizeof(key_slice_type);
  const size_type max_key_length = max_key_length_bytes / sizeof(key_slice_type);
  mapped_vector<value_type> find_results(num_keys);
  testing_input input(num_keys, min_key_length, max_key_length, true);
  auto half_num_keys = (num_keys + 1) / 2;
  map->insert(input.keys.data(), max_key_length, input.lengths.data(), input.values.data(), half_num_keys);
  cuda_try(cudaDeviceSynchronize());
  map->erase(input.keys.data(), max_key_length, input.lengths.data(), num_keys);
  cuda_try(cudaDeviceSynchronize());
  map->find(input.keys.data(), max_key_length, input.lengths.data(), find_results.data(), half_num_keys);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  for (std::size_t i = 0; i < half_num_keys; i++) {
    auto expected_value = invalid_value;
    auto found_value    = find_results[i];
    ASSERT_EQ(found_value, expected_value);
  }
  find_results.free();
  input.free();
}

template <typename map_type>
void test_scan(map_type* map, uint32_t min_key_length_bytes, uint32_t max_key_length_bytes) {
  const size_type min_key_length = min_key_length_bytes / sizeof(key_slice_type);
  const size_type max_key_length = max_key_length_bytes / sizeof(key_slice_type);
  uint32_t num_queries = num_keys / 2;
  uint32_t max_count_per_query = 10;
  mapped_vector<size_type> result_counts(num_queries);
  mapped_vector<value_type> result_values(num_queries * max_count_per_query);
  mapped_vector<key_slice_type> result_keys(num_queries * max_count_per_query * max_key_length);
  mapped_vector<size_type> result_key_lengths(num_queries * max_count_per_query);
  testing_input input(num_keys, min_key_length, max_key_length);
  testing_range_input rinput(input, num_queries, max_count_per_query);
  map->insert(input.keys.data(), max_key_length, input.lengths.data(), input.values.data(), num_keys);
  cuda_try(cudaDeviceSynchronize());
  map->scan(rinput.lower_keys.data(), rinput.lower_lengths.data(),
             max_key_length, max_count_per_query, num_queries,
             rinput.upper_keys.data(), rinput.upper_lengths.data(),
             result_counts.data(), result_values.data(),
             result_keys.data(), result_key_lengths.data());
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  for (std::size_t i = 0; i < num_queries; i++) {
    auto expected_count = rinput.counts[i];
    auto found_count    = result_counts[i];
    ASSERT_EQ(found_count, expected_count);
    for (uint32_t v = 0; v < expected_count; v++) {
      auto expected_value = rinput.values[i * max_count_per_query + v];
      auto found_value = result_values[i * max_count_per_query + v];
      ASSERT_EQ(found_value, expected_value);
      auto expected_length = rinput.out_key_lengths[i * max_count_per_query + v];
      auto found_length = result_key_lengths[i * max_count_per_query + v];
      ASSERT_EQ(found_length, expected_length);
      for (uint32_t s = 0; s < expected_length; s++) {
        auto expected_key_slice = rinput.out_keys[i * max_count_per_query * max_key_length + v * max_key_length + s];
        auto found_key_slice = result_keys[i * max_count_per_query * max_key_length + v * max_key_length + s];
        ASSERT_EQ(found_key_slice, expected_key_slice);
      }
    }
  }
  result_counts.free();
  result_values.free();
  rinput.free();
  input.free();
}

template <typename map_type>
void test_concurrentmix(map_type* map, uint32_t min_key_length_bytes, uint32_t max_key_length_bytes) {
  const size_type min_key_length = min_key_length_bytes / sizeof(key_slice_type);
  const size_type max_key_length = max_key_length_bytes / sizeof(key_slice_type);
  mapped_vector<value_type> find_results(num_keys);
  testing_input input(num_keys, min_key_length, max_key_length);
  input.shuffle();
  // keys: [A: num_keys/4][B: num_keys/4][C: num_keys/4][D: num_keys/4]
  std::size_t num_keyset = num_keys / 4;
  std::size_t offset_keysetB = num_keyset;
  std::size_t offset_keysetC = 2 * num_keyset;
  std::size_t offset_keysetD = 3 * num_keyset;
  // at step 2, we will execute concurrent mix of:
  //    - insert A, C
  //    - find A, B, C, D
  //    - erase B, D
  std::size_t mix_num_requests = 8 * num_keyset;
  mapped_vector<kernels::request_type> mix_types(mix_num_requests);
  mapped_vector<key_slice_type> mix_keys(mix_num_requests * max_key_length);
  mapped_vector<size_type> mix_lengths(mix_num_requests);
  mapped_vector<value_type> mix_values(mix_num_requests);
  mapped_vector<bool> mix_results(mix_num_requests);
  std::vector<std::size_t> shuffle_order(mix_num_requests);
  std::vector<std::size_t> shuffle_order_inverse(mix_num_requests);
  for (std::size_t i = 0; i < mix_num_requests; i++) { shuffle_order[i] = i; }
  std::mt19937 rng(0);
  std::shuffle(shuffle_order.begin(), shuffle_order.end(), rng);
  auto fill_requests = [&](std::size_t dst_begin, std::size_t src_begin,
                           kernels::request_type type) {
    for (std::size_t i = 0; i < num_keyset; i++) {
      auto src_i = src_begin + i;
      auto dst_i = dst_begin + i;
      auto shuffled_dst_i = shuffle_order[dst_i];
      mix_types[shuffled_dst_i] = type;
      for (uint32_t s = 0; s < max_key_length; s++) {
        mix_keys[shuffled_dst_i * max_key_length + s] = input.keys[src_i * max_key_length + s];
      }
      mix_lengths[shuffled_dst_i] = input.lengths[src_i];
      if (type == kernels::request_type_insert) { mix_values[shuffled_dst_i] = input.values[src_i]; }
      else { mix_values[shuffled_dst_i] = invalid_value; }
      mix_results[shuffled_dst_i] = false;
      shuffle_order_inverse[shuffled_dst_i] = dst_i;
    }
  };
  fill_requests(0, 0, kernels::request_type_insert);
  fill_requests(num_keyset, offset_keysetC, kernels::request_type_insert);
  fill_requests(2 * num_keyset, 0, kernels::request_type_find);
  fill_requests(3 * num_keyset, offset_keysetB, kernels::request_type_find);
  fill_requests(4 * num_keyset, offset_keysetC, kernels::request_type_find);
  fill_requests(5 * num_keyset, offset_keysetD, kernels::request_type_find);
  fill_requests(6 * num_keyset, offset_keysetB, kernels::request_type_erase);
  fill_requests(7 * num_keyset, offset_keysetD, kernels::request_type_erase);
  // 1. insert A, B
  map->insert(input.keys.data(), max_key_length, input.lengths.data(), input.values.data(), 2 * num_keyset);
  cuda_try(cudaDeviceSynchronize());
  // 2. concurrent mix
  map->mixed_batch(mix_types.data(), mix_keys.data(), max_key_length, mix_lengths.data(), mix_values.data(), mix_results.data(), mix_num_requests);
  cuda_try(cudaDeviceSynchronize());
  // on order before shuffle, result should be:
  //    [insert A: fail][insert C: success]
  //    [find A: exist][find B: ??][find C: ??][find D: not exist]
  //    [erase B: success][erase D: fail]
  for (std::size_t i = 0; i < mix_num_requests; i++) {
    std::size_t dst_i = shuffle_order_inverse[i];
    if (dst_i < 2 * num_keyset) {
      ASSERT_EQ(mix_types[i], kernels::request_type_insert);
      auto expected_result = (dst_i < num_keyset) ? false : true;
      auto found_result = mix_results[i];
      ASSERT_EQ(expected_result, found_result);
    }
    else if (dst_i < 6 * num_keyset) {
      ASSERT_EQ(mix_types[i], kernels::request_type_find);
      if (dst_i < 3 * num_keyset) {
        auto expected_result = input.values[dst_i - 2 * num_keyset];
        auto found_result = mix_values[i];
        ASSERT_EQ(expected_result, found_result);
      }
      else if (5 * num_keyset <= dst_i) {
        auto expected_result = invalid_value;
        auto found_result = mix_values[i];
        ASSERT_EQ(expected_result, found_result);
      }
    }
    else {
      ASSERT_EQ(mix_types[i], kernels::request_type_erase);
      auto expected_result = (dst_i < 7 * num_keyset) ? true : false;
      auto found_result = mix_results[i];
      ASSERT_EQ(expected_result, found_result);
    }
  }
  // 3. find all
  map->find(input.keys.data(), max_key_length, input.lengths.data(), find_results.data(), 4 * num_keyset);
  cuda_try(cudaDeviceSynchronize());
  // A, C should exist; B, D should not
  for (std::size_t i = 0; i < 4 * num_keyset; i++) {
    auto expected_value = (i < num_keyset || (2 * num_keyset <= i && i < 3 * num_keyset)) ? input.values[i] : invalid_value;
    auto found_value    = find_results[i];
    ASSERT_EQ(found_value, expected_value);
  }
  find_results.free();
  input.free();
  mix_types.free();
  mix_keys.free();
  mix_lengths.free();
  mix_values.free();
  mix_results.free();
}

#define DECLARE_TESTS_FOR_KEY_LENGTHS(min_length, max_length) \
TYPED_TEST(MapTest, Validate##min_length##_##max_length) { \
  validate(this->map_, min_length, max_length); \
} \
TYPED_TEST(MapTest, FindExist##min_length##_##max_length) { \
  test_exist(this->map_, min_length, max_length); \
} \
TYPED_TEST(MapTest, FindNotExist##min_length##_##max_length) { \
  test_notexist(this->map_, min_length, max_length); \
} \
TYPED_TEST(MapTest, Update##min_length##_##max_length) { \
  test_update(this->map_, min_length, max_length); \
} \
TYPED_TEST(MapTest, EraseAll##min_length##_##max_length) { \
  test_eraseall(this->map_, min_length, max_length); \
} \
TYPED_TEST(MapTest, EraseNone##min_length##_##max_length) { \
  test_erasenone(this->map_, min_length, max_length); \
} \
TYPED_TEST(MapTest, InsertTwiceEraseAll##min_length##_##max_length) { \
  test_inserttwiceeraseall(this->map_, min_length, max_length); \
} \
TYPED_TEST(MapTest, EraseAllInsertAll##min_length##_##max_length) { \
  test_eraseallinsertall(this->map_, min_length, max_length); \
} \
TYPED_TEST(MapTest, EraseAllTwice##min_length##_##max_length) { \
  test_erasealltwice(this->map_, min_length, max_length); \
} \
TYPED_TEST(MapTest, ConcurrentMix##min_length##_##max_length) { \
  test_concurrentmix(this->map_, min_length, max_length); \
} \
TYPED_TEST(MapTest, RangeQuery##min_length##_##max_length) { \
  test_scan(this->map_, min_length, max_length); \
}

DECLARE_TESTS_FOR_KEY_LENGTHS(4, 4)
DECLARE_TESTS_FOR_KEY_LENGTHS(64, 64)
DECLARE_TESTS_FOR_KEY_LENGTHS(4, 64)
DECLARE_TESTS_FOR_KEY_LENGTHS(200, 200)
DECLARE_TESTS_FOR_KEY_LENGTHS(100, 200)
DECLARE_TESTS_FOR_KEY_LENGTHS(100, 800)

}  // namespace

int main(int argc, char** argv) {
  auto arguments = std::vector<std::string>(argv, argv + argc);
  num_keys       = get_arg_value<uint32_t>(arguments, "num-keys").value_or(1024);
  std::cout << "Testing using " << num_keys << " keys\n";
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
