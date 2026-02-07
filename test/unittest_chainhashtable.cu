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

std::size_t num_keys;
float fill_factor;

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
    std::size_t num_buckets = std::max(static_cast<std::size_t>(static_cast<double>(num_keys) / 15.0f / fill_factor), 1UL);
    map_ = new typename map_data::map(*host_allocator_, *host_reclaimer_, num_buckets);
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

  std::size_t num_keys;
  uint32_t min_key_length;
  uint32_t max_key_length;
  mapped_vector<key_slice_type> keys;
  mapped_vector<size_type> lengths;
  mapped_vector<value_type> values;
  mapped_vector<value_type> values2;
  mapped_vector<key_slice_type> keys_not_exist;
};

using simple_bump_alloc_type = simple_bump_allocator<128>;
using simple_slab_alloc_type = simple_slab_allocator<128>;
using simple_dummy_reclaim_type = simple_dummy_reclaimer;
using simple_debra_reclaim_type = simple_debra_reclaimer<>;

typedef testing::Types<
    //MapData<GpuChainHashtable::gpu_chainhashtable<simple_bump_alloc_type, simple_dummy_reclaim_type>>,
    //MapData<GpuChainHashtable::gpu_chainhashtable<simple_slab_alloc_type, simple_dummy_reclaim_type>>,
    MapData<GpuChainHashtable::gpu_chainhashtable<simple_slab_alloc_type, simple_debra_reclaim_type>>>
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
  testing_input input(num_keys, min_key_length, max_key_length);
  map->insert(input.keys.data(), max_key_length, input.lengths.data(), input.values.data(), num_keys);
  cuda_try(cudaDeviceSynchronize());
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
void test_concurrentinserterase(map_type* map, uint32_t min_key_length_bytes, uint32_t max_key_length_bytes) {
  const size_type min_key_length = min_key_length_bytes / sizeof(key_slice_type);
  const size_type max_key_length = max_key_length_bytes / sizeof(key_slice_type);
  mapped_vector<value_type> find_results(num_keys);
  testing_input input(num_keys, min_key_length, max_key_length);
  // keys: [A: num_keys/3][B: num_keys/3][C: the rest]
  std::size_t num_keysetA = num_keys / 3, num_keysetB = num_keys / 3;
  std::size_t offset_keysetB = num_keysetA, offset_keysetC = num_keysetA + num_keysetB;
  std::size_t num_keysetC = num_keys - offset_keysetC;
  // 1. insert A, B
  map->insert(input.keys.data(), max_key_length, input.lengths.data(), input.values.data(), num_keysetA + num_keysetB);
  cuda_try(cudaDeviceSynchronize());
  // 2. concurrently insert C & erase B
  map->test_concurrent_insert_erase(
      input.keys.data() + (max_key_length * offset_keysetC), input.lengths.data() + offset_keysetC, input.values.data() + offset_keysetC, num_keysetC,
      input.keys.data() + (max_key_length * offset_keysetB), input.lengths.data() + offset_keysetB, num_keysetB,
      max_key_length);
  cuda_try(cudaDeviceSynchronize());
  // 3. check validity: A, C should exist, B should not
  map->find(input.keys.data(), max_key_length, input.lengths.data(), find_results.data(), num_keys);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  for (std::size_t i = 0; i < num_keys; i++) {
    auto expected_value = (offset_keysetB <= i && i < offset_keysetC) ? invalid_value : input.values[i];
    auto found_value    = find_results[i];
    ASSERT_EQ(found_value, expected_value);
  }
  find_results.free();
  input.free();
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
TYPED_TEST(MapTest, ConcurrentInsertErase##min_length##_##max_length) { \
  test_concurrentinserterase(this->map_, min_length, max_length); \
} \

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
  fill_factor    = get_arg_value<float>(arguments, "fill-factor").value_or(1.0f);
  std::cout << "Testing using " << num_keys << " keys, fill factor " << fill_factor << "\n";
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}