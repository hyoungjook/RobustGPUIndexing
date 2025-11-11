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
#include <cstdint>
#include <macros.hpp>
#include <memory_utils.hpp>
#include <utils.hpp>

template <typename elem_type, typename tile_type, int node_width = 16>
struct masstree_node {
  
  using key_type = elem_type;
  using value_type = elem_type;
  using size_type = uint32_t;
  using unsigned_type =
      typename std::conditional<sizeof(elem_type) == sizeof(uint32_t), uint32_t, uint64_t>::type;
  static constexpr value_type invalid_value = std::numeric_limits<unsigned_type>::max();
  DEVICE_QUALIFIER masstree_node(elem_type* ptr, const tile_type& tile)
      : node_ptr_(ptr), tile_(tile), is_locked_(false) {
    assert(tile_.size() == 2 * node_width);
  }
  DEVICE_QUALIFIER masstree_node(elem_type* ptr,
                                 const tile_type& tile,
                                 const elem_type elem,
                                 bool is_locked,
                                 bool is_intermediate,
                                 bool has_sibling,
                                 size_type num_keys)
      : node_ptr_(ptr)
      , lane_elem_(elem)
      , tile_(tile)
      , is_locked_(is_locked)
      , is_intermediate_(is_intermediate)
      , has_sibling_(has_sibling)
      , num_keys_(num_keys) {}

  DEVICE_QUALIFIER void initialize_root() {
    lane_elem_ = 0;
    is_locked_ = false;
    is_intermediate_ = false;
    has_sibling_ = false;
    num_keys_ = 1; // zero as sentinel?
    write_metadata_to_registers();
  }

  DEVICE_QUALIFIER void load(cuda_memory_order order = cuda_memory_order::memory_order_weak) {
    lane_elem_ = cuda_memory<elem_type>::load(node_ptr_ + tile_.thread_rank(), order);
    is_intermediate_ = !get_leaf_bit();
    is_locked_ = get_lock_bit();
    num_keys_ = get_num_keys();
    has_sibling_ = get_sibling_bit();
  }
  DEVICE_QUALIFIER void store(cuda_memory_order order = cuda_memory_order::memory_order_weak) {
    cuda_memory<elem_type>::store(node_ptr_ + tile_.thread_rank(), lane_elem_, order);
  }

  DEVICE_QUALIFIER elem_type set_metadata_bit(const elem_type& data) const {
    return data | metadata_bit_mask_;
  }
  DEVICE_QUALIFIER elem_type unset_metadata_bit(const elem_type& data) const {
    return data & ~metadata_bit_mask_;
  }
  DEVICE_QUALIFIER elem_type set_metadata_bit_as(const elem_type& data, const bool bit) const {
    return bit ? set_metadata_bit(data) : unset_metadata_bit(data);
  }
  DEVICE_QUALIFIER bool is_metadata_bit_set(const elem_type& data) const {
    return (data & metadata_bit_mask_) != 0;
  }
  DEVICE_QUALIFIER elem_type set_except_metadata_bit(const elem_type& data, const elem_type& value) const {
    return (data & metadata_bit_mask_) | (value & ~metadata_bit_mask_);
  }

  DEVICE_QUALIFIER int get_key_lane_from_location(const int location) const {
    assert(0 <= location && location < node_width);
    return location;
  }
  DEVICE_QUALIFIER int get_value_lane_from_location(const int location) const {
    assert(0 <= location && location < node_width);
    return node_width + location;
  }

  DEVICE_QUALIFIER key_type get_key_from_location(const int location) const {
    return tile_.shfl(lane_elem_, get_key_lane_from_location(location));
  }
  DEVICE_QUALIFIER value_type get_value_from_location(const int location) const {
    value_type value = tile_.shfl(lane_elem_, get_value_lane_from_location(location));
    return unset_metadata_bit(value);
  }
  DEVICE_QUALIFIER bool is_valid_key_lane() const {
    return tile_.thread_rank() < num_keys_;
  }
  DEVICE_QUALIFIER bool is_valid_value_lane() const {
    return node_width <= tile_.thread_rank() && tile_.thread_rank() < node_width + num_keys_;
  }

  DEVICE_QUALIFIER bool get_lock_bit() const {
    return tile_.shfl(is_metadata_bit_set(lane_elem_), lock_bit_lane_);
  }
  DEVICE_QUALIFIER bool get_leaf_bit() const {
    return tile_.shfl(is_metadata_bit_set(lane_elem_), leaf_bit_lane_);
  }
  DEVICE_QUALIFIER bool get_sibling_bit() const {
    return tile_.shfl(is_metadata_bit_set(lane_elem_), sibling_bit_lane_);
  }
  DEVICE_QUALIFIER size_type get_num_keys() const {
    uint32_t metadata_bitmap = tile_.ballot(is_metadata_bit_set(lane_elem_));
    size_type num_keys = static_cast<size_type>(metadata_bitmap >> num_keys_lane_offset_);
    assert(num_keys <= max_num_keys_);
    return num_keys;
  }

  DEVICE_QUALIFIER int find_next_location(const key_type& key) const {
    const bool key_greater_equal = is_valid_key_lane() && (key >= lane_elem_);
    uint32_t key_greater_equal_bitmap = tile_.ballot(key_greater_equal);
    auto next_location = utils::bits::bfind(key_greater_equal_bitmap);
    assert(0 <= next_location && next_location < num_keys_);
    return next_location;
  }
  DEVICE_QUALIFIER value_type find_next(const key_type& key) const {
    auto next_location = find_next_location(key);
    auto value = tile_.shfl(lane_elem_, get_value_lane_from_location(next_location));
    return unset_metadata_bit(value);
  }
  DEVICE_QUALIFIER bool key_is_in_upperhalf(const key_type& key) const {
    auto next_location = find_next_location(key);
    return next_location >= half_node_width_;
  }

  DEVICE_QUALIFIER int find_key_location_in_node(const key_type& key) const {
    auto key_exist = tile_.ballot(is_valid_key_lane() && lane_elem_ == key);
    return __ffs(key_exist) - 1;
  }
  DEVICE_QUALIFIER bool key_is_in_node(const key_type& key) const {
    auto key_exist = tile_.ballot(is_valid_key_lane() && lane_elem_ == key);
    return (key_exist != 0);
  }
  DEVICE_QUALIFIER int find_ptr_location_in_node(const value_type& ptr) const {
    auto ptr_exist = tile_.ballot(is_valid_value_lane() &&
                                  unset_metadata_bit(lane_elem_) == ptr);
    return __ffs(ptr_exist >> node_width) - 1;
  }
  DEVICE_QUALIFIER bool ptr_is_in_node(const value_type& ptr) const {
    auto ptr_exist = tile_.ballot(is_valid_value_lane() &&
                                  unset_metadata_bit(lane_elem_) == ptr);
    return (ptr_exist != 0);
  }
  DEVICE_QUALIFIER value_type get_key_value_from_node(const key_type& key) const {
    auto key_location = find_key_location_in_node(key);
    return key_location == -1 ? invalid_value : get_value_from_location(key_location);
  }

  DEVICE_QUALIFIER bool is_leaf() const { return !is_intermediate_; }
  DEVICE_QUALIFIER bool is_intermediate() const { return is_intermediate_; }
  DEVICE_QUALIFIER bool is_full() const {
    assert(num_keys_ <= max_num_keys_);
    return (num_keys_ == max_num_keys_);
  }
  DEVICE_QUALIFIER bool has_sibling() const { return has_sibling_; }
  DEVICE_QUALIFIER key_type get_high_key() const {
    return get_key_from_location(sibling_location_);
  }
  DEVICE_QUALIFIER key_type get_low_key() const {
    return get_key_from_location(0);
  }
  DEVICE_QUALIFIER value_type get_sibling_index() const {
    return get_value_from_location(sibling_location_);
  }

  DEVICE_QUALIFIER void set_lock_in_registers() {
    if (tile_.thread_rank() == lock_bit_lane_) {
      lane_elem_ = set_metadata_bit(lane_elem_);
    }
    is_locked_ = true;
  }
  DEVICE_QUALIFIER void unset_lock_in_registers() {
    if (tile_.thread_rank() == lock_bit_lane_) {
      lane_elem_ = unset_metadata_bit(lane_elem_);
    }
    is_locked_ = false;
  }
  DEVICE_QUALIFIER void set_num_keys_in_registers(const size_type& num_keys) {
    int bits_offset = static_cast<int>(tile_.thread_rank()) - static_cast<int>(num_keys_lane_offset_);
    if (bits_offset >= 0) {
      bool bit = ((num_keys >> bits_offset) & 1u) != 0;
      lane_elem_ = set_metadata_bit_as(lane_elem_, bit);
    }
    num_keys_ = num_keys;
  }
  DEVICE_QUALIFIER void write_metadata_to_registers() {
    // is_locked_
    if (tile_.thread_rank() == lock_bit_lane_) {
      lane_elem_ = set_metadata_bit_as(lane_elem_, is_locked_);
    }
    // is_intermediate_
    else if (tile_.thread_rank() == leaf_bit_lane_) {
      lane_elem_ = set_metadata_bit_as(lane_elem_, !is_intermediate_);
    }
    // has_sibling_
    else if (tile_.thread_rank() == sibling_bit_lane_) {
      lane_elem_ = set_metadata_bit_as(lane_elem_, has_sibling_);
    }
    // num_keys_
    else {
      set_num_keys_in_registers(num_keys_);
    }
  }

  DEVICE_QUALIFIER bool is_locked() const { return is_locked_; }
  DEVICE_QUALIFIER bool try_lock() {
    unsigned_type old;
    if (tile_.thread_rank() == lock_bit_lane_) {
      old = atomicOr(reinterpret_cast<unsigned_type*>(&node_ptr_[lock_bit_lane_]),
                     static_cast<unsigned_type>(metadata_bit_mask_));
    }
    old = tile_.shfl(old, lock_bit_lane_);
    is_locked_ = !is_metadata_bit_set(old);
    if (is_locked_) {
      set_lock_in_registers();
      __threadfence();
    }
    else {
      unset_lock_in_registers();
    }
    return is_locked_;
  }
  DEVICE_QUALIFIER void lock() {
    while (auto failed = !try_lock()) {}
    is_locked_ = true;
  }
  DEVICE_QUALIFIER void unlock() {
    __threadfence();
    unsigned_type old;
    if (tile_.thread_rank() == lock_bit_lane_) {
      old = atomicAnd(reinterpret_cast<unsigned_type*>(&node_ptr_[lock_bit_lane_]),
                      static_cast<unsigned_type>(~metadata_bit_mask_));
    }
    unset_lock_in_registers();
  }

  DEVICE_QUALIFIER masstree_node do_split(const value_type right_sibling_index,
                                          elem_type* right_sibling_ptr,
                                          const bool make_sibling_locked = false) {
    // prepare the upper half in right sibling
    auto right_sibling_elem = tile_.shfl_down(lane_elem_, half_node_width_);
    auto right_sibling_minimum = get_key_from_location(half_node_width_);
    size_type right_num_keys = num_keys_ - half_node_width_;
    num_keys_ = half_node_width_;

    // reconnect sibling pointers
    auto this_node_previous_sibling_key = get_high_key();
    auto this_node_previous_sibling_index = get_sibling_index();
    if (tile_.thread_rank() == get_key_lane_from_location(sibling_location_)) {
      // left node (this node)'s sibling key = minimum of right node
      // right node (new node)'s sibling key = this node's previous sibling key
      right_sibling_elem = this_node_previous_sibling_key;
      lane_elem_ = right_sibling_minimum;
    }
    else if (tile_.thread_rank() == get_value_lane_from_location(sibling_location_)) {
      // left node (this node)'s sibling index = right_sibling_index
      // right node (new node)'s sibling index = this node's previous sibling index
      right_sibling_elem = this_node_previous_sibling_index;
      lane_elem_ = right_sibling_index;
    }

    // create right sibling node
    masstree_node right_sibling_node =
        masstree_node(right_sibling_ptr, tile_, right_sibling_elem, make_sibling_locked, is_intermediate_, has_sibling_, right_num_keys);
    has_sibling_ = true; // left node now has a sibling.

    // flush metadata
    write_metadata_to_registers();
    right_sibling_node.write_metadata_to_registers();

    return right_sibling_node;
  }

  struct split_intermediate_result {
    masstree_node parent;
    masstree_node sibling;
  };
  // Note parent must be locked before this gets called
  DEVICE_QUALIFIER split_intermediate_result split(const value_type right_sibling_index,
                                                   const value_type parent_index,
                                                   elem_type* right_sibling_ptr,
                                                   elem_type* parent_ptr,
                                                   const bool make_sibling_locked = false) {
    // We assume here that the parent is locked
    auto split_result = do_split(right_sibling_index, right_sibling_ptr, make_sibling_locked);

    // Update parent
    auto parent_node = masstree_node(parent_ptr, tile_);
    parent_node.load(cuda_memory_order::memory_order_relaxed);

    // update the parent
    auto pivot_key = split_result.get_key_from_location(0);
    parent_node.insert(pivot_key, right_sibling_index);
    return {parent_node, split_result};
  }

  struct two_nodes_result {
    masstree_node left;
    masstree_node right;
  };
  DEVICE_QUALIFIER two_nodes_result split_as_root(const value_type left_sibling_index,
                                                  const value_type right_sibling_index,
                                                  elem_type* left_sibling_ptr,
                                                  elem_type* right_sibling_ptr,
                                                  const bool make_children_locked = false) {
    // Create a new root
    auto right_node_minimum = get_key_from_location(half_node_width_);

    // Copy the current node into a child
    auto left_child =
        masstree_node(left_sibling_ptr, tile_, lane_elem_, make_children_locked, is_intermediate_, has_sibling_, num_keys_);
    // if the root was a leaf, now it should be intermediate
    if (!is_intermediate_) { is_intermediate_ = true; }
    // Make new root
    num_keys_ = 2;
    if (tile_.thread_rank() == get_value_lane_from_location(0)) {
      lane_elem_ = left_sibling_index;
    }
    else if (tile_.thread_rank() == get_key_lane_from_location(1)) {
      lane_elem_ = right_node_minimum;
    }
    else if (tile_.thread_rank() == get_value_lane_from_location(1)) {
      lane_elem_ = right_sibling_index;
    }
    write_metadata_to_registers();

    // now split the left child
    auto right_child =
        left_child.do_split(right_sibling_index, right_sibling_ptr, make_children_locked);
    return {left_child, right_child};
  }

  DEVICE_QUALIFIER bool insert(const key_type& key, const value_type& value) {
    auto key_location = find_key_location_in_node(key);
    // if the key exists, we update the value
    if (key_location >= 0) {
      if (tile_.thread_rank() == get_value_lane_from_location(key_location)) {
        lane_elem_ = set_except_metadata_bit(lane_elem_, value);
      }
      return false;
    }
    else {
      // else we shuffle the keys and do the insertion
      assert(!is_full());
      const bool key_is_larger = is_valid_key_lane() && (key > lane_elem_);
      uint32_t key_is_larger_bitmap = tile_.ballot(key_is_larger);
      auto key_location = utils::bits::bfind(key_is_larger_bitmap) + 1;
      
      ++num_keys_;
      const int key_lane = get_key_lane_from_location(key_location);
      const int value_lane = get_value_lane_from_location(key_location);
      auto up_elem = tile_.shfl_up(lane_elem_, 1);
      if (is_valid_key_lane()) {
        if (tile_.thread_rank() == key_lane) { lane_elem_ = key; }
        else if (tile_.thread_rank() > key_lane) { lane_elem_ = up_elem; }
      }
      else if (is_valid_value_lane()) {
        if (tile_.thread_rank() == value_lane) { lane_elem_ = value; }
        else if (tile_.thread_rank() > value_lane) {lane_elem_ = up_elem; }
      }
      write_metadata_to_registers();
      return true;
    }
  }

  DEVICE_QUALIFIER bool erase(const key_type& key) {
    // check if key exists
    auto key_location = find_key_location_in_node(key);
    if (key_location >= 0) {
      assert(num_keys_ > 0);
      --num_keys_;
      const int key_lane = get_key_lane_from_location(key_location);
      const int value_lane = get_value_lane_from_location(key_location);
      auto down_elem = tile_.shfl_down(lane_elem_, 1);
      if (is_valid_key_lane()) {
        if (tile_.thread_rank() >= key_lane) { lane_elem_ = down_elem; }
      }
      else if (is_valid_value_lane()) {
        if (tile_.thread_rank() >= value_lane) { lane_elem_ = down_elem; }
      }
      write_metadata_to_registers();
      return true;
    }
    else {
      return false;
    }
  }

  DEVICE_QUALIFIER masstree_node<elem_type, tile_type, node_width>& operator=(
      const masstree_node<elem_type, tile_type, node_width>& other) {
    node_ptr_ = other.node_ptr_;
    lane_elem_ = other.lane_elem_;
    is_locked_ = other.is_locked_;
    is_intermediate_ = other.is_intermediate_;
    has_sibling_ = other.has_sibling_;
    num_keys_ = other.num_keys_;
    return *this;
  }

  DEVICE_QUALIFIER void print(uint32_t node_index = 0) const {
    bool lead_lane = (tile_.thread_rank() == 0);
    if (lead_lane) printf("node[%u](%p): {", node_index, node_ptr_);
    if (num_keys_ > max_num_keys_) {
      if (lead_lane) printf("num_keys too large: skip}\n");
      return;
    } 
    for (size_type i = 0; i < num_keys_; ++i) {
      elem_type key = tile_.shfl(lane_elem_, get_key_lane_from_location(i));
      elem_type value = tile_.shfl(lane_elem_, get_value_lane_from_location(i));
      if (lead_lane) printf("(%u %u) ", key, unset_metadata_bit(value));
    }
    if (lead_lane) printf("%s %s ", is_locked_ ? "lckd" : "free", is_intermediate_ ? "intr" : "leaf");
    elem_type sibling_key = get_high_key();
    elem_type sibling_index = get_sibling_index();
    if (has_sibling_ && lead_lane) printf("sibl(%u %u)", sibling_key, sibling_index);
    if (lead_lane) printf("}\n");
  }

 private:
  elem_type* node_ptr_;
  elem_type lane_elem_;
  const tile_type tile_;
  
  // node consists of 2*node_width elements, each mapped to a lane in the tile.
  // [key0][key1]...[key14][key15][value0][value1]...[value14][value15]
  // the 15'th pair is [key15 = highKey = sibling's min key], [value15 = sibling ptr].
  // Each value's MSB forms a 16-bit metadata.
  //    value0's MSB: lock bit
  //    value1's MSB: leaf bit
  //    value2's MSB: sibling bit (whether key15, value15 is null pointer or not)
  //    value3's MSB and on : num keys
  static constexpr uint32_t bits_per_byte_ = 8;
  static constexpr uint32_t metadata_bit_offset_ = sizeof(elem_type) * bits_per_byte_ - 1;
  static constexpr elem_type metadata_bit_mask_ = elem_type(1u) << metadata_bit_offset_;
  // value0's MSB is lock bit (which is node_width'th elem)
  static constexpr uint32_t lock_bit_lane_ = node_width;
  // value1's MSB is leaf bit
  static constexpr uint32_t leaf_bit_lane_ = node_width + 1;
  // value2's MSB is sibling bit
  static constexpr uint32_t sibling_bit_lane_ = node_width + 2;
  // value3's MSB and on is num keys
  static constexpr uint32_t num_keys_lane_offset_ = node_width + 3;

  static constexpr size_type max_num_keys_ = node_width - 1; // last pair reserved for sibling
  static constexpr uint32_t sibling_location_ = node_width - 1; // last location is sibling
  static constexpr int half_node_width_ = node_width >> 1;

  bool is_locked_;
  bool is_intermediate_;
  bool has_sibling_;
  size_type num_keys_;
};

template <typename btree>
__global__ void print_tree_nodes_kernel(btree tree) {
  auto block = cooperative_groups::this_thread_block();
  auto tile  = cooperative_groups::tiled_partition<btree::cg_tile_size>(block);
  tree.print_tree_nodes_device_func(tile);
}
