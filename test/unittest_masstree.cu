
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
};

template <class MapData>
class BTreeMapTest : public testing::Test {
 protected:
  BTreeMapTest() { btree_map_ = new typename map_data::btree_map(); }
  ~BTreeMapTest() override { delete btree_map_; }
  using map_data = MapData;
  typename map_data::btree_map* btree_map_;
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
  testing_input(std::size_t input_num_keys, uint32_t input_min_key_length, uint32_t input_max_key_length)
      : num_keys(input_num_keys)
      , min_key_length(input_min_key_length)
      , max_key_length(input_max_key_length)
      , keys(num_keys * max_key_length)
      , lengths(num_keys)
      , values(num_keys)
      , keys_not_exist(input_num_keys * max_key_length)
  {
    assert(min_key_length <= max_key_length);
    make_input();
  }
  void make_input() {
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
    }
  }
  void free() {
    keys.free();
    lengths.free();
    values.free();
  }

  std::size_t num_keys;
  uint32_t min_key_length;
  uint32_t max_key_length;
  mapped_vector<key_slice_type> keys;
  mapped_vector<size_type> lengths;
  mapped_vector<value_type> values;
  mapped_vector<key_slice_type> keys_not_exist;
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

typedef testing::Types<
    BTreeMapData<
        GpuBTree::
            gpu_masstree<bump_allocator_type>>,
    BTreeMapData<
        GpuBTree::
            gpu_masstree<slab_allocator_type>>>
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

/*template <typename btree>
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
void test_eraseallinsertall(btree* tree, uint32_t min_key_length_bytes, uint32_t max_key_length_bytes) {
  const size_type min_key_length = min_key_length_bytes / sizeof(key_slice_type);
  const size_type max_key_length = max_key_length_bytes / sizeof(key_slice_type);
  mapped_vector<value_type> find_results(num_keys);
  testing_input input(num_keys, min_key_length, max_key_length);
  tree->insert(input.keys.data(), max_key_length, input.lengths.data(), input.values.data(), num_keys);
  cuda_try(cudaDeviceSynchronize());
  tree->validate_tree();
  tree->erase(input.keys.data(), max_key_length, input.lengths.data(), num_keys);
  cuda_try(cudaDeviceSynchronize());
  tree->validate_tree();
  tree->find(input.keys.data(), max_key_length, input.lengths.data(), find_results.data(), num_keys);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  for (std::size_t i = 0; i < num_keys; i++) {
    auto expected_value = invalid_value;
    auto found_value    = find_results[i];
    ASSERT_EQ(found_value, expected_value);
  }
  tree->insert(input.keys.data(), max_key_length, input.lengths.data(), input.values.data(), num_keys);
  cuda_try(cudaDeviceSynchronize());
  tree->validate_tree();
  tree->find(input.keys.data(), max_key_length, input.lengths.data(), find_results.data(), num_keys);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  for (std::size_t i = 0; i < num_keys; i++) {
    auto expected_value = input.values[i];
    auto found_value    = find_results[i];
    ASSERT_EQ(found_value, expected_value);
  }
  find_results.free();
  input.free();
}*/

#define DECLARE_TESTS_FOR_KEY_LENGTHS(min_length, max_length) \
TYPED_TEST(BTreeMapTest, ValidateKey##min_length##_##max_length) { \
  validate(this->btree_map_, min_length, max_length); \
} \
TYPED_TEST(BTreeMapTest, FindExistKey##min_length##_##max_length) { \
  test_exist(this->btree_map_, min_length, max_length); \
} \
TYPED_TEST(BTreeMapTest, FindNotExistKey##min_length##_##max_length) { \
  test_notexist(this->btree_map_, min_length, max_length); \
}

DECLARE_TESTS_FOR_KEY_LENGTHS(4, 4)
DECLARE_TESTS_FOR_KEY_LENGTHS(8, 8)
DECLARE_TESTS_FOR_KEY_LENGTHS(4, 8)
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