
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

struct testing_input_fixlen {
  testing_input_fixlen(std::size_t input_num_keys, uint32_t input_key_length)
      : num_keys(input_num_keys)
      , key_length(input_key_length)
      , keys(input_num_keys * input_key_length)
      , values(input_num_keys)
      , keys_not_exist(input_num_keys * input_key_length)
  {
    make_input();
  }
  void make_input() {
    for (std::size_t i = 0; i < num_keys; i++) {
      for (uint32_t s = 0; s < key_length; s++) {
        uint32_t common_prefix_factor = 1u << (key_length - 1 - s);
        uint32_t effective_i = i / common_prefix_factor;
        keys[i * key_length + s] = static_cast<key_slice_type>(effective_i + 1) * 2 + s;
        keys_not_exist[i * key_length + s] = static_cast<key_slice_type>(effective_i + 1) * 2 + 1 + s;
      }
      values[i] = static_cast<value_type>(keys[(i + 1) * key_length - 1] + 1);
    }
  }
  void free() {
    keys.free();
    values.free();
    keys_not_exist.free();
  }

  std::size_t num_keys;
  uint32_t key_length;
  mapped_vector<key_slice_type> keys;
  mapped_vector<value_type> values;
  mapped_vector<key_slice_type> keys_not_exist;
};

struct testing_input_varlen {
  testing_input_varlen(std::size_t input_num_keys, uint32_t input_max_key_length)
      : num_keys(input_num_keys)
      , max_key_length(input_max_key_length)
      , keys(num_keys * max_key_length)
      , lengths(num_keys)
      , values(num_keys)
  {
    make_input();
  }
  void make_input() {
    for (std::size_t i = 0; i < num_keys; i++) {
      uint32_t key_length = 1 + (i % max_key_length);
      for (uint32_t s = 0; s < key_length; s++) {
        uint32_t common_prefix_factor = 1u << (key_length - 1 - s);
        uint32_t effective_i = i / common_prefix_factor;
        keys[i * max_key_length + s] = static_cast<key_slice_type>(effective_i + 1) * 2 + s;
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
  uint32_t max_key_length;
  mapped_vector<key_slice_type> keys;
  mapped_vector<size_type> lengths;
  mapped_vector<value_type> values;
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

/*TYPED_TEST(BTreeMapTest, Debug) {
  uint32_t key_length = 8 / sizeof(key_slice_type);
  testing_input_fixlen input (num_keys, key_length);
  mapped_vector<value_type> find_results(num_keys);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  for (size_t i = 0; i < num_keys; i++) {
    printf("insert ");
    for (size_t s = 0; s < key_length; s++) printf("%u ", input.keys[i * key_length + s]);
    printf("\n");
    this->btree_map_->insert_fixlen(input.keys.data() + (key_length * i), key_length, input.values.data() + i, 1);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    printf("===\n");
    this->btree_map_->print_tree_nodes();
    printf("===\n");
  }
  for (size_t i = 0; i < num_keys; i++) {
    printf("find ");
    for (size_t s = 0; s < key_length; s++) printf("%u ", input.keys[i * key_length + s]);
    printf("\n");
    this->btree_map_->find_fixlen(input.keys.data() + (key_length * i), key_length, find_results.data() + i, 1);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    printf("result: %u\n", find_results[i]);
  }
  input.free();
  exit(1);
}*/

/*TYPED_TEST(BTreeMapTest, Debug) {
  uint32_t max_key_length = 8 / sizeof(key_slice_type);
  testing_input_varlen input (num_keys, max_key_length);
  mapped_vector<value_type> find_results(num_keys);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  for (size_t i = 0; i < num_keys; i++) {
    printf("insert ");
    for (size_t s = 0; s < input.lengths[i]; s++) printf("%u ", input.keys[i * max_key_length + s]);
    printf("\n");
    this->btree_map_->insert_varlen(input.keys.data() + (max_key_length * i), max_key_length, input.lengths.data() + i, input.values.data() + i, 1);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    printf("===\n");
    this->btree_map_->print_tree_nodes();
    printf("===\n");
  }
  for (size_t i = 0; i < num_keys; i++) {
    printf("find ");
    for (size_t s = 0; s < input.lengths[i]; s++) printf("%u ", input.keys[i * max_key_length + s]);
    printf("\n");
    this->btree_map_->find_varlen(input.keys.data() + (max_key_length * i), max_key_length, input.lengths.data() + i, find_results.data() + i, 1);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    printf("result: %u %s\n", find_results[i],
      (find_results[i] == input.values[i]) ? "" : "(WRONG)");
  }
  input.free();
  exit(1);
}*/

template <typename btree>
void validate_key_fixlen(btree* tree, uint32_t key_length_bytes) {
  const size_type key_length = key_length_bytes / sizeof(key_slice_type);
  testing_input_fixlen input(num_keys, key_length);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  tree->insert_fixlen(input.keys.data(), key_length, input.values.data(), num_keys);
  cuda_try(cudaDeviceSynchronize());
  tree->validate_tree();
  input.free();
}

template <typename btree>
void test_exist_key_fixlen(btree* tree, uint32_t key_length_bytes) {
  const size_type key_length = key_length_bytes / sizeof(key_slice_type);
  mapped_vector<value_type> find_results(num_keys);
  testing_input_fixlen input(num_keys, key_length);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  tree->insert_fixlen(input.keys.data(), key_length, input.values.data(), num_keys);
  cuda_try(cudaDeviceSynchronize());
  tree->find_fixlen(input.keys.data(), key_length, find_results.data(), num_keys);
  cuda_try(cudaDeviceSynchronize());
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
void test_notexist_key_fixlen(btree* tree, uint32_t key_length_bytes) {
  const size_type key_length = key_length_bytes / sizeof(key_slice_type);
  mapped_vector<value_type> find_results(num_keys);
  testing_input_fixlen input(num_keys, key_length);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  tree->insert_fixlen(input.keys.data(), key_length, input.values.data(), num_keys);
  cuda_try(cudaDeviceSynchronize());
  tree->find_fixlen(input.keys_not_exist.data(), key_length, find_results.data(), num_keys);
  cuda_try(cudaDeviceSynchronize());
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  for (std::size_t i = 0; i < num_keys; i++) {
    auto expected_value = invalid_value;
    auto found_value    = find_results[i];
    ASSERT_EQ(found_value, expected_value);
  }
  find_results.free();
  input.free();
}

TYPED_TEST(BTreeMapTest, ValidateKey4) {
  validate_key_fixlen(this->btree_map_, 4);
}
TYPED_TEST(BTreeMapTest, FindExistKey4) {
  test_exist_key_fixlen(this->btree_map_, 4);
}
TYPED_TEST(BTreeMapTest, FindNotExist4) {
  test_notexist_key_fixlen(this->btree_map_, 4);
}
TYPED_TEST(BTreeMapTest, ValidateKey8) {
  validate_key_fixlen(this->btree_map_, 8);
}
TYPED_TEST(BTreeMapTest, FindExistKey8) {
  test_exist_key_fixlen(this->btree_map_, 8);
}
TYPED_TEST(BTreeMapTest, FindNotExist8) {
  test_notexist_key_fixlen(this->btree_map_, 8);
}
TYPED_TEST(BTreeMapTest, ValidateKey64) {
  validate_key_fixlen(this->btree_map_, 64);
}
TYPED_TEST(BTreeMapTest, FindExistKey64) {
  test_exist_key_fixlen(this->btree_map_, 64);
}
TYPED_TEST(BTreeMapTest, FindNotExist64) {
  test_notexist_key_fixlen(this->btree_map_, 64);
}

template <typename btree>
void validate_key_varlen(btree* tree, uint32_t max_key_length_bytes) {
  const size_type max_key_length = max_key_length_bytes / sizeof(key_slice_type);
  testing_input_varlen input(num_keys, max_key_length);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  tree->insert_varlen(input.keys.data(), max_key_length, input.lengths.data(), input.values.data(), num_keys);
  cuda_try(cudaDeviceSynchronize());
  tree->validate_tree();
  input.free();
}

template <typename btree>
void test_exist_key_varlen(btree* tree, uint32_t max_key_length_bytes) {
  const size_type max_key_length = max_key_length_bytes / sizeof(key_slice_type);
  mapped_vector<value_type> find_results(num_keys);
  testing_input_varlen input(num_keys, max_key_length);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  tree->insert_varlen(input.keys.data(), max_key_length, input.lengths.data(), input.values.data(), num_keys);
  cuda_try(cudaDeviceSynchronize());
  tree->find_varlen(input.keys.data(), max_key_length, input.lengths.data(), find_results.data(), num_keys);
  cuda_try(cudaDeviceSynchronize());
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  for (std::size_t i = 0; i < num_keys; i++) {
    auto expected_value = input.values[i];
    auto found_value    = find_results[i];
    ASSERT_EQ(found_value, expected_value);
  }
  find_results.free();
  input.free();
}

TYPED_TEST(BTreeMapTest, ValidateKeyVarLen8) {
  validate_key_varlen(this->btree_map_, 8);
}
TYPED_TEST(BTreeMapTest, FindExistKeyVarLen8) {
  test_exist_key_varlen(this->btree_map_, 8);
}
TYPED_TEST(BTreeMapTest, ValidateKeyVarLen64) {
  validate_key_varlen(this->btree_map_, 64);
}
TYPED_TEST(BTreeMapTest, FindExistKeyVarLen64) {
  test_exist_key_varlen(this->btree_map_, 64);
}

/*TYPED_TEST(BTreeMapTest, EraseAllTest) {
  mapped_vector<value_type> find_results(num_keys);
  testing_input input(num_keys);
  this->btree_map_->insert(input.keys.data(), input.values.data(), num_keys);
  cuda_try(cudaDeviceSynchronize());
  this->btree_map_->erase(input.keys.data(), num_keys);
  cuda_try(cudaDeviceSynchronize());
  this->btree_map_->find(input.keys_exist.data(), find_results.data(), num_keys);
  cuda_try(cudaDeviceSynchronize());
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  for (std::size_t i = 0; i < num_keys; i++) {
    auto expected_value = sentinel_value;
    auto found_value    = find_results[i];
    ASSERT_EQ(found_value, expected_value);
  }
  find_results.free();
  input.free();
}

TYPED_TEST(BTreeMapTest, EraseNoneTest) {
  mapped_vector<value_type> find_results(num_keys);
  testing_input input(num_keys);
  this->btree_map_->insert(input.keys.data(), input.values.data(), num_keys);
  cuda_try(cudaDeviceSynchronize());
  this->btree_map_->erase(input.keys_not_exist.data(), num_keys);
  cuda_try(cudaDeviceSynchronize());
  this->btree_map_->find(input.keys_exist.data(), find_results.data(), num_keys);
  cuda_try(cudaDeviceSynchronize());
  // EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  for (std::size_t i = 0; i < num_keys; i++) {
    auto expected_value = input.values[i];
    auto found_value    = find_results[i];
    ASSERT_EQ(found_value, expected_value);
  }
  find_results.free();
  input.free();
}

TYPED_TEST(BTreeMapTest, EraseAllInsertAllTest) {
  mapped_vector<value_type> find_results(num_keys);
  testing_input input(num_keys);
  this->btree_map_->insert(input.keys.data(), input.values.data(), num_keys);
  cuda_try(cudaDeviceSynchronize());
  this->btree_map_->erase(input.keys.data(), num_keys);
  cuda_try(cudaDeviceSynchronize());
  this->btree_map_->find(input.keys_exist.data(), find_results.data(), num_keys);
  cuda_try(cudaDeviceSynchronize());
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  for (std::size_t i = 0; i < num_keys; i++) {
    auto expected_value = sentinel_value;
    auto found_value    = find_results[i];
    ASSERT_EQ(found_value, expected_value);
  }
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  this->btree_map_->insert(input.keys.data(), input.values.data(), num_keys);
  cuda_try(cudaDeviceSynchronize());
  this->btree_map_->find(input.keys_exist.data(), find_results.data(), num_keys);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  cuda_try(cudaDeviceSynchronize());
  for (std::size_t i = 0; i < num_keys; i++) {
    auto expected_value = input.values[i];
    auto found_value    = find_results[i];
    ASSERT_EQ(found_value, expected_value);
  }
  find_results.free();
  input.free();
}*/

}  // namespace

int main(int argc, char** argv) {
  auto arguments = std::vector<std::string>(argv, argv + argc);
  num_keys       = get_arg_value<uint32_t>(arguments, "num-keys").value_or(1024);
  std::cout << "Testing using " << num_keys << " keys\n";
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}