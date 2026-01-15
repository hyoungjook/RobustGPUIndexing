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

#include <gpu_btree.h>
#include <gtest/gtest.h>
#include <cmd.hpp>
#include <cstdint>

std::size_t num_keys;

namespace {
using key_slice_type   = uint32_t;
using value_type = uint32_t;
using size_type = uint32_t;

const auto invalid_value = std::numeric_limits<value_type>::max();
template <typename BTreeMap>
struct BTreeMapData {
  using btree_map = BTreeMap;
  using host_allocator = typename BTreeMap::host_allocator_type;
};

template <class MapData>
class BTreeMapTest : public testing::Test {
 protected:
  BTreeMapTest() {
    host_allocator_ = new typename map_data::host_allocator();
    btree_map_ = new typename map_data::btree_map(*host_allocator_);
  }
  ~BTreeMapTest() override {
    //host_allocator_->print_stats();
    delete btree_map_;
    delete host_allocator_;
  }
  using map_data = MapData;
  typename map_data::btree_map* btree_map_;
  typename map_data::host_allocator* host_allocator_;
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
      for (uint32_t s = 0; s < key_length; s++) {
        uint32_t common_prefix_factor = 1u << (key_length - 1 - s);
        uint32_t effective_i = i / common_prefix_factor;
        keys[i * max_key_length + s] = static_cast<key_slice_type>(effective_i + 1) * 2 + s;
        keys_not_exist[i * max_key_length + s] = static_cast<key_slice_type>(effective_i + 1) * 2 + 1 + s;
      }
      lengths[i] = key_length;
      values[i] = static_cast<value_type>(keys[(i + 1) * key_length - 1] + 1);
      values2[i] = values[i] + 7;
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

struct TreeParam {
  static constexpr int BranchingFactor = 16;
};
struct SlabAllocParam {
  static constexpr uint32_t NumSuperBlocks  = 4;
  static constexpr uint32_t NumMemoryBlocks = 1024 * 8;
  static constexpr uint32_t TileSize        = 2 * TreeParam::BranchingFactor;
  static constexpr uint32_t SlabSize        = 128;
};
using node_type           = GpuBTree::node_type<key_slice_type, value_type, TreeParam::BranchingFactor>;
using bump_allocator_type = device_bump_allocator<node_type>;
using slab_allocator_type = device_allocator::SlabAllocLight<node_type,
                                                             SlabAllocParam::NumSuperBlocks,
                                                             SlabAllocParam::NumMemoryBlocks,
                                                             SlabAllocParam::TileSize,
                                                             SlabAllocParam::SlabSize>;

using simple_bump_alloc_type = simple_bump_allocator<128>;
using simple_slab_alloc_type = simple_slab_allocator<128>;

typedef testing::Types<
    //BTreeMapData<
    //    GpuBTree::
    //        gpu_masstree<bump_allocator_type>>,
    //BTreeMapData<
    //    GpuBTree::
    //        gpu_masstree<slab_allocator_type>>>
    //BTreeMapData<GpuBTree::gpu_masstree<simple_bump_alloc_type>>,
    BTreeMapData<GpuBTree::gpu_masstree<simple_slab_alloc_type>>>
    Implementations;

TYPED_TEST_SUITE(BTreeMapTest, Implementations);

template <typename btree>
void validate(btree* tree, uint32_t min_key_length_bytes, uint32_t max_key_length_bytes) {
  const size_type min_key_length = min_key_length_bytes / sizeof(key_slice_type);
  const size_type max_key_length = max_key_length_bytes / sizeof(key_slice_type);
  testing_input input(num_keys, min_key_length, max_key_length);
  tree->insert(input.keys.data(), max_key_length, input.lengths.data(), input.values.data(), num_keys);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  tree->validate_tree();
  input.free();
}

template <typename btree>
void test_exist(btree* tree, uint32_t min_key_length_bytes, uint32_t max_key_length_bytes) {
  const size_type min_key_length = min_key_length_bytes / sizeof(key_slice_type);
  const size_type max_key_length = max_key_length_bytes / sizeof(key_slice_type);
  mapped_vector<value_type> find_results(num_keys);
  testing_input input(num_keys, min_key_length, max_key_length);
  tree->insert(input.keys.data(), max_key_length, input.lengths.data(), input.values.data(), num_keys);
  cuda_try(cudaDeviceSynchronize());
  tree->find(input.keys.data(), max_key_length, input.lengths.data(), find_results.data(), num_keys);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  for (std::size_t i = 0; i < num_keys; i++) {
    auto expected_value = input.values[i];
    auto found_value    = find_results[i];
    ASSERT_EQ(found_value, expected_value);
  }
  find_results.free();
  input.free();
}

template <typename btree>
void test_notexist(btree* tree, uint32_t min_key_length_bytes, uint32_t max_key_length_bytes) {
  const size_type min_key_length = min_key_length_bytes / sizeof(key_slice_type);
  const size_type max_key_length = max_key_length_bytes / sizeof(key_slice_type);
  mapped_vector<value_type> find_results(num_keys);
  testing_input input(num_keys, min_key_length, max_key_length);
  tree->insert(input.keys.data(), max_key_length, input.lengths.data(), input.values.data(), num_keys);
  cuda_try(cudaDeviceSynchronize());
  tree->find(input.keys_not_exist.data(), max_key_length, input.lengths.data(), find_results.data(), num_keys);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  for (std::size_t i = 0; i < num_keys; i++) {
    auto expected_value = invalid_value;
    auto found_value    = find_results[i];
    ASSERT_EQ(found_value, expected_value);
  }
  find_results.free();
  input.free();
}

template <typename btree>
void test_update(btree* tree, uint32_t min_key_length_bytes, uint32_t max_key_length_bytes) {
  const size_type min_key_length = min_key_length_bytes / sizeof(key_slice_type);
  const size_type max_key_length = max_key_length_bytes / sizeof(key_slice_type);
  mapped_vector<value_type> find_results(num_keys);
  testing_input input(num_keys, min_key_length, max_key_length);
  tree->insert(input.keys.data(), max_key_length, input.lengths.data(), input.values.data(), num_keys);
  cuda_try(cudaDeviceSynchronize());
  tree->insert(input.keys.data(), max_key_length, input.lengths.data(), input.values2.data(), num_keys, 0, true);
  cuda_try(cudaDeviceSynchronize());
  tree->find(input.keys.data(), max_key_length, input.lengths.data(), find_results.data(), num_keys);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  for (std::size_t i = 0; i < num_keys; i++) {
    auto expected_value = input.values2[i];
    auto found_value    = find_results[i];
    ASSERT_EQ(found_value, expected_value);
  }
  find_results.free();
  input.free();
}

template <typename btree>
void test_eraseall(btree* tree, uint32_t min_key_length_bytes, uint32_t max_key_length_bytes) {
  const size_type min_key_length = min_key_length_bytes / sizeof(key_slice_type);
  const size_type max_key_length = max_key_length_bytes / sizeof(key_slice_type);
  mapped_vector<value_type> find_results(num_keys);
  testing_input input(num_keys, min_key_length, max_key_length);
  tree->insert(input.keys.data(), max_key_length, input.lengths.data(), input.values.data(), num_keys);
  cuda_try(cudaDeviceSynchronize());
  tree->erase(input.keys.data(), max_key_length, input.lengths.data(), num_keys);
  cuda_try(cudaDeviceSynchronize());
  tree->find(input.keys.data(), max_key_length, input.lengths.data(), find_results.data(), num_keys);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  for (std::size_t i = 0; i < num_keys; i++) {
    auto expected_value = invalid_value;
    auto found_value    = find_results[i];
    ASSERT_EQ(found_value, expected_value);
  }
  tree->validate_tree();
  find_results.free();
  input.free();
}

template <typename btree>
void test_erasenone(btree* tree, uint32_t min_key_length_bytes, uint32_t max_key_length_bytes) {
  const size_type min_key_length = min_key_length_bytes / sizeof(key_slice_type);
  const size_type max_key_length = max_key_length_bytes / sizeof(key_slice_type);
  mapped_vector<value_type> find_results(num_keys);
  testing_input input(num_keys, min_key_length, max_key_length);
  tree->insert(input.keys.data(), max_key_length, input.lengths.data(), input.values.data(), num_keys);
  cuda_try(cudaDeviceSynchronize());
  tree->erase(input.keys_not_exist.data(), max_key_length, input.lengths.data(), num_keys);
  cuda_try(cudaDeviceSynchronize());
  tree->find(input.keys.data(), max_key_length, input.lengths.data(), find_results.data(), num_keys);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  for (std::size_t i = 0; i < num_keys; i++) {
    auto expected_value = input.values[i];
    auto found_value    = find_results[i];
    ASSERT_EQ(found_value, expected_value);
  }
  find_results.free();
  input.free();
}

template <typename btree>
void test_inserttwiceeraseall(btree* tree, uint32_t min_key_length_bytes, uint32_t max_key_length_bytes) {
  const size_type min_key_length = min_key_length_bytes / sizeof(key_slice_type);
  const size_type max_key_length = max_key_length_bytes / sizeof(key_slice_type);
  mapped_vector<value_type> find_results(num_keys);
  testing_input input(num_keys, min_key_length, max_key_length);
  tree->insert(input.keys.data(), max_key_length, input.lengths.data(), input.values.data(), num_keys);
  cuda_try(cudaDeviceSynchronize());
  tree->insert(input.keys.data(), max_key_length, input.lengths.data(), input.values.data(), num_keys);
  cuda_try(cudaDeviceSynchronize());
  tree->erase(input.keys.data(), max_key_length, input.lengths.data(), num_keys);
  cuda_try(cudaDeviceSynchronize());
  tree->find(input.keys.data(), max_key_length, input.lengths.data(), find_results.data(), num_keys);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  for (std::size_t i = 0; i < num_keys; i++) {
    auto expected_value = invalid_value;
    auto found_value    = find_results[i];
    ASSERT_EQ(found_value, expected_value);
  }
  find_results.free();
  input.free();
}

template <typename btree>
void test_eraseallinsertall(btree* tree, uint32_t min_key_length_bytes, uint32_t max_key_length_bytes) {
  const size_type min_key_length = min_key_length_bytes / sizeof(key_slice_type);
  const size_type max_key_length = max_key_length_bytes / sizeof(key_slice_type);
  mapped_vector<value_type> find_results(num_keys);
  testing_input input(num_keys, min_key_length, max_key_length);
  tree->insert(input.keys.data(), max_key_length, input.lengths.data(), input.values.data(), num_keys);
  cuda_try(cudaDeviceSynchronize());
  tree->erase(input.keys.data(), max_key_length, input.lengths.data(), num_keys);
  cuda_try(cudaDeviceSynchronize());
  tree->find(input.keys.data(), max_key_length, input.lengths.data(), find_results.data(), num_keys);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  for (std::size_t i = 0; i < num_keys; i++) {
    auto expected_value = invalid_value;
    auto found_value    = find_results[i];
    ASSERT_EQ(found_value, expected_value);
  }
  tree->insert(input.keys.data(), max_key_length, input.lengths.data(), input.values.data(), num_keys);
  cuda_try(cudaDeviceSynchronize());
  tree->find(input.keys.data(), max_key_length, input.lengths.data(), find_results.data(), num_keys);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  for (std::size_t i = 0; i < num_keys; i++) {
    auto expected_value = input.values[i];
    auto found_value    = find_results[i];
    ASSERT_EQ(found_value, expected_value);
  }
  find_results.free();
  input.free();
}

template <typename btree>
void test_erasealltwice(btree* tree, uint32_t min_key_length_bytes, uint32_t max_key_length_bytes) {
  const size_type min_key_length = min_key_length_bytes / sizeof(key_slice_type);
  const size_type max_key_length = max_key_length_bytes / sizeof(key_slice_type);
  mapped_vector<value_type> find_results(num_keys);
  testing_input input(num_keys, min_key_length, max_key_length, true);
  auto half_num_keys = (num_keys + 1) / 2;
  tree->insert(input.keys.data(), max_key_length, input.lengths.data(), input.values.data(), half_num_keys);
  cuda_try(cudaDeviceSynchronize());
  tree->erase(input.keys.data(), max_key_length, input.lengths.data(), num_keys);
  cuda_try(cudaDeviceSynchronize());
  tree->find(input.keys.data(), max_key_length, input.lengths.data(), find_results.data(), half_num_keys);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  for (std::size_t i = 0; i < half_num_keys; i++) {
    auto expected_value = invalid_value;
    auto found_value    = find_results[i];
    ASSERT_EQ(found_value, expected_value);
  }
  find_results.free();
  input.free();
}

template <typename btree>
void test_concurrentinserterase(btree* tree, uint32_t min_key_length_bytes, uint32_t max_key_length_bytes) {
  const size_type min_key_length = min_key_length_bytes / sizeof(key_slice_type);
  const size_type max_key_length = max_key_length_bytes / sizeof(key_slice_type);
  mapped_vector<value_type> find_results(num_keys);
  testing_input input(num_keys, min_key_length, max_key_length);
  // keys: [A: num_keys/3][B: num_keys/3][C: the rest]
  std::size_t num_keysetA = num_keys / 3, num_keysetB = num_keys / 3;
  std::size_t offset_keysetB = num_keysetA, offset_keysetC = num_keysetA + num_keysetB;
  std::size_t num_keysetC = num_keys - offset_keysetC;
  // 1. insert A, B
  tree->insert(input.keys.data(), max_key_length, input.lengths.data(), input.values.data(), num_keysetA + num_keysetB);
  cuda_try(cudaDeviceSynchronize());
  // 2. concurrently insert C & erase B
  tree->test_concurrent_insert_erase(
      input.keys.data() + (max_key_length * offset_keysetC), input.lengths.data() + offset_keysetC, input.values.data() + offset_keysetC, num_keysetC,
      input.keys.data() + (max_key_length * offset_keysetB), input.lengths.data() + offset_keysetB, num_keysetB,
      max_key_length);
  cuda_try(cudaDeviceSynchronize());
  // 3. check validity: A, C should exist, B should not
  tree->find(input.keys.data(), max_key_length, input.lengths.data(), find_results.data(), num_keys);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  for (std::size_t i = 0; i < num_keys; i++) {
    auto expected_value = (offset_keysetB <= i && i < offset_keysetC) ? invalid_value : input.values[i];
    auto found_value    = find_results[i];
    ASSERT_EQ(found_value, expected_value);
  }
  find_results.free();
  input.free();
}

template <typename btree>
void test_range(btree* tree, uint32_t min_key_length_bytes, uint32_t max_key_length_bytes) {
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
  tree->insert(input.keys.data(), max_key_length, input.lengths.data(), input.values.data(), num_keys);
  cuda_try(cudaDeviceSynchronize());
  tree->range(rinput.lower_keys.data(), rinput.lower_lengths.data(),
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
      //for (uint32_t s = 0; s < expected_length; s++) {
      //  auto expected_key_slice = rinput.out_keys[i * max_count_per_query * max_key_length + v * max_key_length + s];
      //  auto found_key_slice = result_keys[i * max_count_per_query * max_key_length + v * max_key_length + s];
      //  ASSERT_EQ(found_key_slice, expected_key_slice);
      //}
    }
  }
  result_counts.free();
  result_values.free();
  rinput.free();
  input.free();
}

#define DECLARE_TESTS_FOR_KEY_LENGTHS(min_length, max_length) \
TYPED_TEST(BTreeMapTest, Validate##min_length##_##max_length) { \
  validate(this->btree_map_, min_length, max_length); \
} \
TYPED_TEST(BTreeMapTest, FindExist##min_length##_##max_length) { \
  test_exist(this->btree_map_, min_length, max_length); \
} \
TYPED_TEST(BTreeMapTest, FindNotExist##min_length##_##max_length) { \
  test_notexist(this->btree_map_, min_length, max_length); \
} \
TYPED_TEST(BTreeMapTest, Update##min_length##_##max_length) { \
  test_update(this->btree_map_, min_length, max_length); \
} \
TYPED_TEST(BTreeMapTest, EraseAll##min_length##_##max_length) { \
  test_eraseall(this->btree_map_, min_length, max_length); \
} \
TYPED_TEST(BTreeMapTest, EraseNone##min_length##_##max_length) { \
  test_erasenone(this->btree_map_, min_length, max_length); \
} \
TYPED_TEST(BTreeMapTest, InsertTwiceEraseAll##min_length##_##max_length) { \
  test_inserttwiceeraseall(this->btree_map_, min_length, max_length); \
} \
TYPED_TEST(BTreeMapTest, EraseAllInsertAll##min_length##_##max_length) { \
  test_eraseallinsertall(this->btree_map_, min_length, max_length); \
} \
TYPED_TEST(BTreeMapTest, EraseAllTwice##min_length##_##max_length) { \
  test_erasealltwice(this->btree_map_, min_length, max_length); \
} \
TYPED_TEST(BTreeMapTest, ConcurrentInsertErase##min_length##_##max_length) { \
  test_concurrentinserterase(this->btree_map_, min_length, max_length); \
} \
TYPED_TEST(BTreeMapTest, RangeQuery##min_length##_##max_length) { \
  test_range(this->btree_map_, min_length, max_length); \
}

DECLARE_TESTS_FOR_KEY_LENGTHS(4, 4)
DECLARE_TESTS_FOR_KEY_LENGTHS(64, 64)
DECLARE_TESTS_FOR_KEY_LENGTHS(4, 64)

}  // namespace

int main(int argc, char** argv) {
  auto arguments = std::vector<std::string>(argv, argv + argc);
  num_keys       = get_arg_value<uint32_t>(arguments, "num-keys").value_or(1024);
  std::cout << "Testing using " << num_keys << " keys\n";
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}