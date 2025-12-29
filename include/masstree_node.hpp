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

#pragma once
#include <cstdint>
#include <macros.hpp>
#include <memory_utils.hpp>
#include <utils.hpp>

#ifndef NDEBUG
#define NODE_DEBUG
#endif
template <typename tile_type>
struct masstree_node {
  using elem_type = uint32_t;
  using key_type = elem_type;
  using value_type = elem_type;
  using size_type = uint32_t;
  static constexpr int node_width = 16;
  static constexpr key_type max_key = std::numeric_limits<key_type>::max();
  DEVICE_QUALIFIER masstree_node(elem_type* ptr, const size_type index, const tile_type& tile)
      : node_ptr_(ptr), tile_(tile)
      #ifdef NODE_DEBUG
      , node_index_(index)
      #endif
  {
    assert(tile_.size() == 2 * node_width);
  }
  DEVICE_QUALIFIER masstree_node(elem_type* ptr,
                                 const size_type index,
                                 const tile_type& tile,
                                 const elem_type elem,
                                 uint32_t metadata,
                                 bool key_end_bit)
      : node_ptr_(ptr)
      #ifdef NODE_DEBUG
      , node_index_(index)
      #endif
      , lane_elem_(elem)
      , tile_(tile)
      , metadata_(metadata)
      , key_end_bit_(key_end_bit) {}

  DEVICE_QUALIFIER void initialize_root() {
    lane_elem_ = 0;
    metadata_ = (
      (0u << num_keys_offset_) |  // num_keys = 0;
      (0u & lock_bit_mask_) |     // is_locked = false;
      (border_bit_mask_) |        // is_border = true;
      (0u & sibling_bit_mask_) |  // has_sibling = false;
      (0u & garbage_bit_mask_)    // is_garbage = false;
    );
    key_end_bit_ = 0;
    write_metadata_to_registers();
  }

  DEVICE_QUALIFIER void load(cuda_memory_order order = cuda_memory_order::memory_order_weak) {
    lane_elem_ = cuda_memory<elem_type>::load(node_ptr_ + tile_.thread_rank(), order);
    read_metadata_from_registers();
  }
  DEVICE_QUALIFIER void store(cuda_memory_order order = cuda_memory_order::memory_order_weak) {
    cuda_memory<elem_type>::store(node_ptr_ + tile_.thread_rank(), lane_elem_, order);
  }

  DEVICE_QUALIFIER void read_metadata_from_registers() {
    metadata_ = tile_.shfl(lane_elem_, metadata_lane_);
    if (is_border()) {
      uint32_t key_end_bits = tile_.shfl(lane_elem_, key_end_bits_lane_);
      key_end_bit_ = (key_end_bits & (1u << tile_.thread_rank())) != 0;
    }
  }
  DEVICE_QUALIFIER void write_metadata_to_registers() {
    if (tile_.thread_rank() == metadata_lane_) {
      lane_elem_ = metadata_;
    }
    if (is_border()) {
      uint32_t key_end_bits = tile_.ballot(key_end_bit_);
      if (tile_.thread_rank() == key_end_bits_lane_) {
        lane_elem_ = key_end_bits;
      }
    }
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
    return tile_.shfl(lane_elem_, get_value_lane_from_location(location));
  }
  DEVICE_QUALIFIER bool get_key_end_bit_from_location(const int location) const {
    return tile_.shfl(key_end_bit_, get_key_lane_from_location(location));
  }
  DEVICE_QUALIFIER bool is_valid_key_lane() const {
    return tile_.thread_rank() < num_keys();
  }
  DEVICE_QUALIFIER bool is_valid_value_lane() const {
    // intermediate node has children one more than keys
    return node_width <= tile_.thread_rank() && tile_.thread_rank() < node_width + num_keys();
  }

  DEVICE_QUALIFIER uint32_t num_keys() const {
    return (metadata_ & num_keys_mask_) >> num_keys_offset_;
  }
  DEVICE_QUALIFIER void set_num_keys(const uint32_t& value) {
    assert(value <= (num_keys_mask_ >> num_keys_offset_));
    metadata_ &= ~num_keys_mask_;
    metadata_ |= (value << num_keys_offset_);
  }
  DEVICE_QUALIFIER bool is_border() const {
    return static_cast<bool>(metadata_ & border_bit_mask_);
  }
  DEVICE_QUALIFIER bool is_full() const {
    assert(is_border() ? (num_keys() <= border_max_num_keys_) : (num_keys() <= interior_max_num_keys_));
    return (is_border() ? (num_keys() == border_max_num_keys_) : (num_keys() == interior_max_num_keys_));
  }
  DEVICE_QUALIFIER bool is_underflow() const {
    return (num_keys() <= underflow_num_keys_);
  }
  DEVICE_QUALIFIER bool is_mergeable(const masstree_node& sibling_node) const {
    return (num_keys() + sibling_node.num_keys()) <= (is_border() ? border_max_num_keys_ : interior_max_num_keys_);
  }
  DEVICE_QUALIFIER bool is_garbage() const {
    return static_cast<bool>(metadata_ & garbage_bit_mask_);
  }
  DEVICE_QUALIFIER key_type get_high_key() const {
    assert(is_border() || num_keys() > 0); // never called
    return tile_.shfl(lane_elem_, is_border() ? (border_high_key_lane_) : (num_keys() - 1));
  }
  DEVICE_QUALIFIER value_type get_sibling_index() const {
    return tile_.shfl(lane_elem_, sibling_ptr_lane_);
  }

  DEVICE_QUALIFIER bool is_locked() const {
    return static_cast<bool>(metadata_ & lock_bit_mask_);
  }
  DEVICE_QUALIFIER bool try_lock() {
    elem_type old;
    if (tile_.thread_rank() == metadata_lane_) {
      old = atomicOr(reinterpret_cast<elem_type*>(&node_ptr_[metadata_lane_]),
                     static_cast<elem_type>(lock_bit_mask_));
    }
    old = tile_.shfl(old, metadata_lane_);
    bool is_locked = (old & lock_bit_mask_) == 0; // if previously not locked, now it's locked
    if (is_locked) { __threadfence(); }
    return is_locked;
    // do not need to update registers; if locked, the code will load() again.
    // if lock failed, this node object will be disposed.
  }
  DEVICE_QUALIFIER void lock() {
    while (!try_lock());
    // the code will load() again
  }
  DEVICE_QUALIFIER void unlock() {
    assert(is_locked());
    __threadfence();
    if (tile_.thread_rank() == metadata_lane_) {
      atomicAnd(reinterpret_cast<elem_type*>(&node_ptr_[metadata_lane_]),
                static_cast<elem_type>(~lock_bit_mask_));
    }
    // the node object can be used after this, so update regsiters
    metadata_ &= ~lock_bit_mask_;
    if (tile_.thread_rank() == metadata_lane_) {
      lane_elem_ &= ~lock_bit_mask_;
    }
  }

  DEVICE_QUALIFIER bool has_sibling() const {
    return static_cast<bool>(metadata_ & sibling_bit_mask_);
  }
  DEVICE_QUALIFIER bool traverse_required(const key_type& key) const {
    return has_sibling() && (is_garbage() || (get_high_key() < key));
  }
  DEVICE_QUALIFIER int find_next_location(const key_type& key) const {
    assert(!is_border());
    //const bool key_less_equal = is_valid_key_lane() && (key <= lane_elem_);
    // __ffs() will filter out the values from non-valid key lanes
    const bool key_less_equal = (key <= lane_elem_);
    uint32_t key_less_equal_bitmap = tile_.ballot(key_less_equal);
    auto next_location = __ffs(key_less_equal_bitmap) - 1;
    assert(0 <= next_location && next_location < num_keys() + 1);
    return next_location;
  }
  DEVICE_QUALIFIER value_type find_next(const key_type& key) const {
    auto next_location = find_next_location(key);
    return tile_.shfl(lane_elem_, get_value_lane_from_location(next_location));
  }
  DEVICE_QUALIFIER bool key_is_in_upperhalf(const key_type& pivot_key, const key_type& key) const {
    return (pivot_key < key);
  }
  DEVICE_QUALIFIER value_type find_next_and_sibling(const key_type& key, value_type& sibling_index, bool& sibling_at_left) const {
    auto next_location = find_next_location(key);
    // same decision with get_merge_plan()
    sibling_at_left = (next_location >= num_keys() - 1);
    sibling_index = tile_.shfl(lane_elem_, get_value_lane_from_location(
        next_location + (sibling_at_left ? -1 : 1)));
    return tile_.shfl(lane_elem_, get_value_lane_from_location(next_location));
  }

  DEVICE_QUALIFIER uint32_t match_key_in_node(const key_type& key, bool key_end) const {
    assert(is_border());
    return tile_.ballot(is_valid_key_lane() &&
                        lane_elem_ == key &&
                        key_end_bit_ == key_end);
  }
  DEVICE_QUALIFIER bool key_is_in_node(const key_type& key, bool key_end) const {
    assert(is_border());
    auto key_exist = match_key_in_node(key, key_end);
    return (key_exist != 0);
  }
  DEVICE_QUALIFIER uint32_t match_ptr_in_node(const value_type& ptr) const {
    assert(!is_border());
    return tile_.ballot(is_valid_value_lane() && lane_elem_ == ptr);
  }
  DEVICE_QUALIFIER bool ptr_is_in_node(const value_type& ptr) const {
    auto ptr_exist = match_ptr_in_node(ptr);
    return (ptr_exist != 0);
  }
  DEVICE_QUALIFIER bool get_key_value_from_node(const key_type& key, value_type& value, bool key_end) const {
    assert(is_border());
    auto key_exists = match_key_in_node(key, key_end);
    if (key_exists == 0) return false;
    value = get_value_from_location(__ffs(key_exists) - 1);
    return true;
  }

  DEVICE_QUALIFIER int get_split_left_width() const {
    // normally, location is left_half_width_
    // but we should put same-key non-last-slice and first last-slice in the same node
    // to avoid comparing keys, we just shift if last elem of the left half is non-last-slice
    bool shift_required = false;
    if (is_border()) {
      bool key_end_of_default_pivot = get_key_end_bit_from_location(half_node_width_ - 1);
      shift_required = !key_end_of_default_pivot; // shift if it's non-last-slice
    }
    return shift_required ? (half_node_width_ - 1) : half_node_width_;
  }

  DEVICE_QUALIFIER masstree_node do_split(const value_type right_sibling_index,
                                          elem_type* right_sibling_ptr,
                                          const int left_width) {
    // same-key non-last-slice entry and first last-slice entry (if both exist) go to the same node
    //assert(!(
    //  is_border_ &&
    //  (get_key_from_location(left_width - 1) == get_key_from_location(left_width)) &&
    //  (get_key_end_bit_from_location(left_width - 1) == false)
    //));
    assert(is_full());
    // prepare the upper half in right sibling
    elem_type right_sibling_elem = tile_.shfl_down(lane_elem_, left_width);
    bool right_sibling_key_end_bit = is_border() ? tile_.shfl_down(key_end_bit_, left_width) : false;
    // reconnect right sibling pointers
    if (tile_.thread_rank() == sibling_ptr_lane_) {
      // right's right sibling = this node's previous right sibling
      right_sibling_elem = lane_elem_;
      // left's right sibling = right
      lane_elem_ = right_sibling_index;
    }
    // reassign high keys
    if (is_border()) {
      auto pivot_key = get_key_from_location(left_width - 1);
      if (tile_.thread_rank() == border_high_key_lane_) {
        // right's high key = this node's previous high key
        right_sibling_elem = lane_elem_;
        // left's high key = pivot key
        lane_elem_ = pivot_key;
      }
    }
    // update metadata
    uint32_t right_metadata = (metadata_ & ~(num_keys_mask_ | sibling_bit_mask_));
    right_metadata |= ((num_keys() - left_width) << num_keys_offset_); // right.num_keys = num_keys - left_width;
    if (has_sibling()) right_metadata |= sibling_bit_mask_;   // right.has_sibling = has_sibling;
    set_num_keys(left_width);
    metadata_ |= sibling_bit_mask_; // has_sibling = true;
    // create right sibling node
    masstree_node right_sibling_node =
        masstree_node(right_sibling_ptr, right_sibling_index, tile_, right_sibling_elem,
                      right_metadata, right_sibling_key_end_bit);
    // flush metadata
    write_metadata_to_registers();
    right_sibling_node.write_metadata_to_registers();

    return right_sibling_node;
  }

  struct split_intermediate_result {
    masstree_node sibling;
    key_type pivot_key;
  };
  // Note parent must be locked before this gets called
  DEVICE_QUALIFIER split_intermediate_result split(const value_type right_sibling_index,
                                                   const value_type parent_index,
                                                   elem_type* right_sibling_ptr,
                                                   masstree_node& parent_node) {
    // We assume here that the parent is locked
    auto left_width = get_split_left_width();
    auto split_result = do_split(right_sibling_index, right_sibling_ptr, left_width);

    // update the parent
    auto pivot_key = get_key_from_location(left_width - 1);
    parent_node.insert(pivot_key, right_sibling_index, false);
    return {split_result, pivot_key};
  }

  struct two_nodes_result {
    masstree_node left;
    masstree_node right;
  };
  DEVICE_QUALIFIER two_nodes_result split_as_root(const value_type left_sibling_index,
                                                  const value_type right_sibling_index,
                                                  elem_type* left_sibling_ptr,
                                                  elem_type* right_sibling_ptr) {
    // Copy the current node into a child
    auto left_child =
        masstree_node(left_sibling_ptr, left_sibling_index, tile_, lane_elem_, metadata_, key_end_bit_);
    // if the root was a border node, now it should be interior
    metadata_ &= ~border_bit_mask_; // is_border = false;
    // Make new root
    set_num_keys(2);
    auto left_width = left_child.get_split_left_width();
    auto pivot_key = left_child.get_key_from_location(left_width - 1);
    if (tile_.thread_rank() == get_key_lane_from_location(0)) {
      lane_elem_ = pivot_key;
    }
    else if (tile_.thread_rank() == get_value_lane_from_location(0)) {
      lane_elem_ = left_sibling_index;
    }
    else if (tile_.thread_rank() == get_key_lane_from_location(1)) {
      lane_elem_ = max_key;
    }
    else if (tile_.thread_rank() == get_value_lane_from_location(1)) {
      lane_elem_ = right_sibling_index;
    }
    assert(!has_sibling()); // root has no sibling
    write_metadata_to_registers();

    // now split the left child
    auto right_child = left_child.do_split(right_sibling_index, right_sibling_ptr, left_width);
    return {left_child, right_child};
  }

  DEVICE_QUALIFIER void do_insert(const key_type& key, const value_type& value, bool key_end, const size_type& key_location) {
    // shuffle the keys and do the insertion
    assert(!is_full());
    metadata_++;  // equiv. to num_keys++;
    const int key_lane = get_key_lane_from_location(key_location);
    const int value_lane = get_value_lane_from_location(key_location + (is_border() ? 0 : 1));
    auto up_elem = tile_.shfl_up(lane_elem_, 1);
    bool up_key_end_bit = tile_.shfl_up(key_end_bit_, 1);
    if (is_valid_key_lane()) {
      if (tile_.thread_rank() == key_lane) {
        lane_elem_ = key;
        key_end_bit_ = key_end;
      }
      else if (tile_.thread_rank() > key_lane) {
        lane_elem_ = up_elem;
        key_end_bit_ = up_key_end_bit;
      }
    }
    else if (is_valid_value_lane()) {
      if (tile_.thread_rank() == value_lane) {
        lane_elem_ = value;
      }
      else if (tile_.thread_rank() > value_lane) {
        lane_elem_ = up_elem;
      }
    }
    write_metadata_to_registers();
  }

  DEVICE_QUALIFIER void insert(const key_type& key,
                               const value_type& value,
                               bool last_slice_and_leaf) {
    assert(!is_full());
    const bool key_is_larger = is_valid_key_lane() && (key > lane_elem_);
    uint32_t key_is_larger_bitmap = tile_.ballot(key_is_larger);
    auto key_location = utils::bits::bfind(key_is_larger_bitmap) + 1;
    // if duplicates, this is the location of the first duplicate.
    if (last_slice_and_leaf &&
        (key_location < num_keys()) &&
        (!get_key_end_bit_from_location(key_location)) &&
        (get_key_from_location(key_location) == key)) {
      // if inserting a last-slice entry but there's a same-key non-last slice,
      // we should insert after the non-last slice.
      key_location++;
    }
    do_insert(key, value, last_slice_and_leaf, key_location);
  }

  struct merge_plan {
    value_type sibling_index;
    int left_location;
    bool sibling_at_left;
  };
  DEVICE_QUALIFIER merge_plan get_merge_plan(const value_type& child_index) const {
    static_assert(underflow_num_keys_ >= 2);
    assert(num_keys() >= 2);
    auto ptr_exist = match_ptr_in_node(child_index);
    if (ptr_exist == 0) return {0, -1, false};
    merge_plan plan;
    plan.left_location = __ffs(ptr_exist) - (1 + node_width);
    plan.sibling_at_left = false;
    if (plan.left_location < num_keys() - 1) {
      // use right sibling
      plan.sibling_index = get_value_from_location(plan.left_location + 1);
    }
    else {
      // use left sibling
      plan.left_location--;
      plan.sibling_index = get_value_from_location(plan.left_location);
      plan.sibling_at_left = true;
    }
    return plan;
  }

  DEVICE_QUALIFIER void make_garbage_node(bool has_sibling, value_type sibling_index = 0) {
    // num_keys = 0, has_sibling = has_sibling, is_garbage = true;
    metadata_ &= ~(num_keys_mask_ | sibling_bit_mask_);
    metadata_ |= garbage_bit_mask_;
    if (has_sibling) metadata_ |= sibling_bit_mask_;
    if (has_sibling && tile_.thread_rank() == sibling_ptr_lane_) {
      lane_elem_ = sibling_index;
    }
    write_metadata_to_registers();
  }

  DEVICE_QUALIFIER void merge(const value_type& left_sibling_index,
                              masstree_node& right_sibling_node,
                              masstree_node& parent_node,
                              int left_location) { 
    // this node is the left sibling
    // copy elements from right sibling node
    elem_type shifted_elem = tile_.shfl_up(right_sibling_node.lane_elem_, num_keys());
    bool shifted_key_end_bit = is_border() ? tile_.shfl_up(right_sibling_node.key_end_bit_, num_keys()) : false;
    auto new_num_keys = num_keys() + right_sibling_node.num_keys();
    if ((num_keys() <= tile_.thread_rank() && tile_.thread_rank() < new_num_keys) ||
        (node_width + num_keys() <= tile_.thread_rank() && tile_.thread_rank() < node_width + new_num_keys)) {
      lane_elem_ = shifted_elem;
      key_end_bit_ = shifted_key_end_bit;
    }
    set_num_keys(new_num_keys);
    // copy right child's sibling info and high key
    metadata_ = (metadata_ & ~sibling_bit_mask_) ^ (right_sibling_node.metadata_ & sibling_bit_mask_);  // has_sibling = right.has_sibling;
    if ((tile_.thread_rank() == sibling_ptr_lane_) ||
        (is_border() && tile_.thread_rank() == border_high_key_lane_)) {
      lane_elem_ = right_sibling_node.lane_elem_;
    }
    write_metadata_to_registers();
    // remove key from parent
    assert(!parent_node.is_border());
    assert(left_location < parent_node.num_keys() - 1);
    parent_node.do_erase(get_key_lane_from_location(left_location),
                         get_value_lane_from_location(left_location + 1));
    // set right sibling as empty node and connect it to this node
    right_sibling_node.make_garbage_node(true, left_sibling_index);
  }

  DEVICE_QUALIFIER void merge_to_root(const value_type& parent_index,
                                      masstree_node& left_child_node,
                                      masstree_node& right_child_node) {
    // this node is parent
    assert(num_keys() == 2);
    // copy the children into current node
    lane_elem_ = left_child_node.lane_elem_;
    key_end_bit_ = left_child_node.key_end_bit_;
    set_num_keys(left_child_node.num_keys() + right_child_node.num_keys());
    metadata_ = (metadata_ & ~border_bit_mask_) ^ (left_child_node.metadata_ & border_bit_mask_); // is_border = left.is_border;
    auto right_elem = tile_.shfl_up(right_child_node.lane_elem_, left_child_node.num_keys());
    bool right_key_end_bit = is_border() ? tile_.shfl_up(right_child_node.key_end_bit_, left_child_node.num_keys()) : false;
    if ((left_child_node.num_keys() <= tile_.thread_rank() && tile_.thread_rank() < num_keys()) ||
        (node_width + left_child_node.num_keys() <= tile_.thread_rank() && tile_.thread_rank() < node_width + num_keys())) {
      lane_elem_ = right_elem;
      key_end_bit_ = right_key_end_bit;
    }
    // copy right child's sibling info and high key
    metadata_ = (metadata_ & ~sibling_bit_mask_) ^ (right_child_node.metadata_ & sibling_bit_mask_);  // has_sibling = right.has_sibling;
    if ((tile_.thread_rank() == sibling_ptr_lane_) ||
        (is_border() && tile_.thread_rank() == border_high_key_lane_)) {
      lane_elem_ = right_child_node.lane_elem_;
    }
    write_metadata_to_registers();
    // make both children empty
    left_child_node.make_garbage_node(true, parent_index);
    right_child_node.make_garbage_node(true, parent_index);
  }

  DEVICE_QUALIFIER void borrow_left(masstree_node& sibling_node,
                                    masstree_node& parent_node,
                                    int left_location) {
    // compute num shift; adjust similar to get_split_left_width()
    uint32_t num_shift = (sibling_node.num_keys() - num_keys()) / 2;
    if (is_border()) {
      if (!sibling_node.get_key_end_bit_from_location(sibling_node.num_keys() - num_shift - 1)) {
        num_shift++;
      }
    }
    // copy last num_shift entries of the sibling into current
    elem_type shifted_elem = tile_.shfl_up(lane_elem_, num_shift);
    bool shifted_key_end_bit = is_border() ? tile_.shfl_up(key_end_bit_, num_shift) : false;
    metadata_ += num_shift; // equiv. to num_keys += num_shift;
    if ((tile_.thread_rank() < num_keys()) ||
        (node_width <= tile_.thread_rank() && tile_.thread_rank() < node_width + num_keys())) {
      lane_elem_ = shifted_elem;
      key_end_bit_ = shifted_key_end_bit;
    }
    sibling_node.metadata_ -= num_shift;  // equiv. to num_keys -= num_shift;
    shifted_elem = tile_.shfl_down(sibling_node.lane_elem_, sibling_node.num_keys());
    shifted_key_end_bit = is_border() ? tile_.shfl_down(sibling_node.key_end_bit_, sibling_node.num_keys()) : false;
    if ((tile_.thread_rank() < num_shift) ||
        (node_width <= tile_.thread_rank() && tile_.thread_rank() < node_width + num_shift)) {
      lane_elem_ = shifted_elem;
      key_end_bit_ = shifted_key_end_bit;
    }
    write_metadata_to_registers();
    // remove last num_shift entries from the sibling
    key_type pivot_key = sibling_node.get_key_from_location(sibling_node.num_keys() - 1);
    if (is_border()) {
      if (tile_.thread_rank() == border_high_key_lane_) {
        sibling_node.lane_elem_ = pivot_key;
      }
    }
    sibling_node.write_metadata_to_registers();
    // update parent
    if (tile_.thread_rank() == get_key_lane_from_location(left_location)) {
      parent_node.lane_elem_ = pivot_key;
    }
    //assert(!(
    //  is_border_ &&
    //  (sibling_node.get_key_from_location(sibling_node.num_keys_ - 1) == get_key_from_location(0)) &&
    //  (sibling_node.get_key_end_bit_from_location(sibling_node.num_keys_ - 1) == false)
    //));
  }

  DEVICE_QUALIFIER void borrow_right(masstree_node& sibling_node,
                                     masstree_node& parent_node,
                                     int left_location,
                                     const value_type& current_node_index,
                                     const value_type& new_sibling_index,
                                     masstree_node& new_sibling_node) {
    // compute num shift; adjust similar to get_split_left_width()
    uint32_t num_shift = (sibling_node.num_keys() - num_keys()) / 2;
    if (is_border()) {
      if (!sibling_node.get_key_end_bit_from_location(num_shift - 1)) {
        num_shift--;
      }
    }
    // copy first num_shift entries of the sibling into current
    elem_type shifted_elem = tile_.shfl_up(sibling_node.lane_elem_, num_keys());
    bool shifted_key_end_bit = is_border() ? tile_.shfl_up(sibling_node.key_end_bit_, num_keys()) : false;
    if ((num_keys() <= tile_.thread_rank() && tile_.thread_rank() < num_keys() + num_shift) ||
        (node_width + num_keys() <= tile_.thread_rank() && tile_.thread_rank() < node_width + num_keys() + num_shift)) {
      lane_elem_ = shifted_elem;
      key_end_bit_ = shifted_key_end_bit;
    }
    metadata_ += num_shift; // equiv. to num_keys += num_shift;
    key_type pivot_key = get_key_from_location(num_keys() - 1);
    if (is_border()) {
      if (tile_.thread_rank() == border_high_key_lane_) {
        lane_elem_ = pivot_key;
      }
    }
    write_metadata_to_registers();
    // current points to the new sibling
    if (tile_.thread_rank() == sibling_ptr_lane_) {
      lane_elem_ = new_sibling_index;
    }
    // copy sibling to new_sibling
    new_sibling_node = masstree_node(new_sibling_node.node_ptr_,
                                     new_sibling_index,
                                     new_sibling_node.tile_,
                                     sibling_node.lane_elem_,
                                     sibling_node.metadata_,
                                     sibling_node.key_end_bit_);
    // remove first num_shift entries from the new_sibling
    shifted_elem = tile_.shfl_down(new_sibling_node.lane_elem_, num_shift);
    shifted_key_end_bit = is_border() ? tile_.shfl_down(new_sibling_node.key_end_bit_, num_shift) : false;
    new_sibling_node.metadata_ -= num_shift;  // equiv. to new_sibling_node.num_keys -= num_shift;
    if ((tile_.thread_rank() < new_sibling_node.num_keys()) ||
        (node_width <= tile_.thread_rank() && tile_.thread_rank() < node_width + new_sibling_node.num_keys())) {
      new_sibling_node.lane_elem_ = shifted_elem;
      new_sibling_node.key_end_bit_ = shifted_key_end_bit;
    }
    new_sibling_node.write_metadata_to_registers();
    // make old sibling empty
    sibling_node.make_garbage_node(true, current_node_index);
    // update parent
    if (tile_.thread_rank() == get_key_lane_from_location(left_location)) {
      parent_node.lane_elem_ = pivot_key;
    }
    else if (tile_.thread_rank() == get_value_lane_from_location(left_location + 1)) {
      parent_node.lane_elem_ = new_sibling_index;
    }
    //assert(!(
    //  is_border_ &&
    //  (get_key_from_location(num_keys_ - 1) == new_sibling_node.get_key_from_location(0)) &&
    //  (get_key_end_bit_from_location(num_keys_ - 1) == false)
    //));
  }

  DEVICE_QUALIFIER void do_erase(int key_lane, int value_lane) {
    assert(num_keys() > 0);
    metadata_--;  // equiv. to num_keys--;
    auto down_elem = tile_.shfl_down(lane_elem_, 1);
    bool down_key_end_bit = tile_.shfl_down(key_end_bit_, 1);
    if (is_valid_key_lane()) {
      if (tile_.thread_rank() >= key_lane) {
        lane_elem_ = down_elem;
        key_end_bit_ = down_key_end_bit;
      }
    }
    else if (is_valid_value_lane()) {
      if (tile_.thread_rank() >= value_lane) {
        lane_elem_ = down_elem;
      }
    }
    write_metadata_to_registers();
  }

  DEVICE_QUALIFIER bool erase(const key_type& key, bool key_end) {
    // erase entry (key, key_end=true)
    assert(is_border());
    uint32_t key_exists = match_key_in_node(key, key_end);
    if (key_exists == 0) return false;
    uint32_t key_location = __ffs(key_exists) - 1;
    do_erase(get_key_lane_from_location(key_location),
             get_value_lane_from_location(key_location));
    return true;
  }

  DEVICE_QUALIFIER masstree_node<tile_type>& operator=(
      const masstree_node<tile_type>& other) {
    node_ptr_ = other.node_ptr_;
    #ifdef NODE_DEBUG
    node_index_ = other.node_index_;
    #endif
    lane_elem_ = other.lane_elem_;
    metadata_ = other.metadata_;
    key_end_bit_ = other.key_end_bit_;
    return *this;
  }

  DEVICE_QUALIFIER void print() const {
    #ifdef NODE_DEBUG
    bool lead_lane = (tile_.thread_rank() == 0);
    if (lead_lane) printf("node[%u]: {", node_index_);
    if (num_keys() > interior_max_num_keys_) {
      if (lead_lane) printf("num_keys too large: skip}\n");
      return;
    } 
    if (is_garbage()) {
      if (lead_lane) printf("garbage}\n");
      return;
    }
    if (num_keys() == 0) {
      if (lead_lane) printf("empty}\n");
      return;
    }
    if (lead_lane) printf("%u ", num_keys());
    for (size_type i = 0; i < num_keys(); ++i) {
      elem_type key = tile_.shfl(lane_elem_, get_key_lane_from_location(i));
      elem_type value = tile_.shfl(lane_elem_, get_value_lane_from_location(i));
      bool key_end_bit = tile_.shfl(key_end_bit_, get_key_lane_from_location(i));
      if (lead_lane) printf("(%u %u %s) ", key, value, key_end_bit ? "$" : ":");
    }
    if (lead_lane) printf("%s %s ", is_locked() ? "locked" : "unlocked", is_border() ? "border" : "interior");
    elem_type sibling_key = get_high_key();
    elem_type sibling_index = get_sibling_index();
    if (has_sibling()) {
      if (lead_lane) printf("next(%u %u)", sibling_key, sibling_index);
    }
    else {
      if (lead_lane) printf("nullnext");
    }
    if (lead_lane) printf("}\n");
    #endif
  }

 private:
  elem_type* node_ptr_;
  #ifdef NODE_DEBUG
  size_type node_index_;
  #endif
  elem_type lane_elem_;
  const tile_type tile_;
  
  // node consists of 2*node_width elements, each mapped to a lane in the tile.
  //  [key0] [key1] ... [key13] [key14] | [metadata]
  //  [ptr0] [ptr1] ... [ptr13] [ptr14] | [ptr15 = sibling_ptr]

  // for interior nodes,
  //    ptr_i contains subtree with keys: key_(i-1) < key <= key_i.
  //    key[num_keys-1] is high key (upper bound of the subtree), used for B-link traversal.

  // for border nodes,
  //    key14 is high key, used for B-link traversal
  //    ptr14 is key_end_bits; bits13-0 corresponds to key13-0
  //      bit=0 (pointer of root node to next masstree layer), bit=1 (pointer of leaf value)

  // metadata is 32bits.
  //    (MSB)
  //    [empty:24]
  //    [is_garbage:1][has_sibling:1]
  //    [is_border:1][is_locked:1]
  //    [num_keys:4]
  //    (LSB)
  static_assert(sizeof(elem_type) == sizeof(uint32_t));
  static constexpr uint32_t metadata_lane_ = node_width - 1;
  static constexpr uint32_t sibling_ptr_lane_ = node_width * 2 - 1;
  static constexpr uint32_t border_high_key_lane_ = node_width - 2;
  static constexpr uint32_t key_end_bits_lane_ = node_width * 2 - 2;
  
  static constexpr uint32_t num_keys_offset_ = 0;
  static constexpr uint32_t num_keys_bits_ = 4;
  static constexpr uint32_t num_keys_mask_ = ((1u << num_keys_bits_) - 1) << num_keys_offset_;
  static constexpr uint32_t lock_bit_offset_ = 4;
  static constexpr uint32_t lock_bit_mask_ = 1u << lock_bit_offset_;
  static constexpr uint32_t border_bit_offset_ = 5;
  static constexpr uint32_t border_bit_mask_ = 1u << border_bit_offset_;
  static constexpr uint32_t sibling_bit_offset_ = 6;
  static constexpr uint32_t sibling_bit_mask_ = 1u << sibling_bit_offset_;
  static constexpr uint32_t garbage_bit_offset_ = 7;
  static constexpr uint32_t garbage_bit_mask_ = 1u << garbage_bit_offset_;

  static constexpr uint32_t interior_max_num_keys_ = node_width - 1;
  static constexpr uint32_t border_max_num_keys_ = node_width - 2;
  static constexpr uint32_t underflow_num_keys_ = node_width / 3; // TODO adjust
  static constexpr uint32_t half_node_width_ = node_width / 2;

  static_assert(num_keys_offset_ == 0); // this allows (metadata +/- N) equivalent to (num_keys +/- N) within range

  uint32_t metadata_;
  bool key_end_bit_; // per-lane value. only threads0-13's values are meaningful
};

template <int MAX_STACK_SIZE, typename Func, typename BTree>
__global__ void traverse_tree_nodes_kernel(BTree tree) {
  // called with single warp, BFS
  assert(gridDim.x == 1 && gridDim.y == 1 && gridDim.z == 1);
  assert(blockDim.x == 32 && blockDim.y == 1 && blockDim.z == 1);
  auto block = cooperative_groups::this_thread_block();
  auto tile  = cooperative_groups::tiled_partition<BTree::cg_tile_size>(block);
  __shared__ uint32_t stack[MAX_STACK_SIZE], metadata[MAX_STACK_SIZE];
  Func task;
  task.init(tile.thread_rank() == 0);
  tree.cooperative_traverse_tree_nodes<MAX_STACK_SIZE>(&stack[0], &metadata[0], tile, task);
  task.fini();
}

template <typename BTree>
__global__ void debug_find_varlen_print_kernel(BTree tree, uint32_t* key, uint32_t* len) {
  auto block = cooperative_groups::this_thread_block();
  auto tile  = cooperative_groups::tiled_partition<BTree::cg_tile_size>(block);
  tree.cooperative_debug_find_varlen_print(key, len[0], tile);
}
