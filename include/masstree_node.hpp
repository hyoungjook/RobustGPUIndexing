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
      : node_ptr_(ptr), tile_(tile), is_locked_(false)
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
                                 uint16_t num_keys,
                                 bool is_locked,
                                 bool is_leaf,
                                 bool has_sibling,
                                 bool key_meta_bit)
      : node_ptr_(ptr)
      #ifdef NODE_DEBUG
      , node_index_(index)
      #endif
      , lane_elem_(elem)
      , tile_(tile)
      , num_keys_(num_keys)
      , is_locked_(is_locked)
      , is_leaf_(is_leaf)
      , has_sibling_(has_sibling)
      , key_meta_bit_(key_meta_bit) {}

  DEVICE_QUALIFIER void initialize_root() {
    lane_elem_ = 0;
    num_keys_ = 0;
    is_locked_ = false;
    is_leaf_ = true;
    has_sibling_ = false;
    key_meta_bit_ = 0;
    write_metadata_to_registers();
  }

  DEVICE_QUALIFIER void load(cuda_memory_order order = cuda_memory_order::memory_order_weak) {
    lane_elem_ = cuda_memory<elem_type>::load(node_ptr_ + tile_.thread_rank(), order);
    read_metadata_from_registers();
  }
  DEVICE_QUALIFIER void store(cuda_memory_order order = cuda_memory_order::memory_order_weak) {
    cuda_memory<elem_type>::store(node_ptr_ + tile_.thread_rank(), lane_elem_, order);
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
  DEVICE_QUALIFIER bool get_key_meta_bit_from_location(const int location) const {
    return tile_.shfl(key_meta_bit_, get_key_lane_from_location(location));
  }
  DEVICE_QUALIFIER bool is_valid_key_lane() const {
    return tile_.thread_rank() < num_keys_;
  }
  DEVICE_QUALIFIER bool is_valid_value_lane() const {
    // intermediate node has children one more than keys
    return node_width <= tile_.thread_rank() && tile_.thread_rank() < node_width + num_keys_;
  }

  DEVICE_QUALIFIER uint16_t num_keys() const { return num_keys_; }
  DEVICE_QUALIFIER bool is_leaf() const { return is_leaf_; }
  DEVICE_QUALIFIER bool is_intermediate() const { return !is_leaf_; }
  DEVICE_QUALIFIER bool is_full() const {
    assert(num_keys_ <= max_num_keys_);
    return (num_keys_ == max_num_keys_);
  }
  DEVICE_QUALIFIER bool has_sibling() const { return has_sibling_; }
  DEVICE_QUALIFIER key_type get_high_key() const {
    assert(num_keys_ > 0); // never called
    return tile_.shfl(lane_elem_, num_keys_ - 1);
  }
  DEVICE_QUALIFIER value_type get_sibling_index() const {
    return tile_.shfl(lane_elem_, sibling_ptr_lane_);
  }

  DEVICE_QUALIFIER void read_metadata_from_registers() {
    uint32_t metadata = tile_.shfl(lane_elem_, metadata_lane_);
    key_meta_bit_ = (metadata & (1u << tile_.thread_rank())) != 0;
    num_keys_ = static_cast<uint16_t>((metadata & num_keys_mask_) >> num_keys_offset_);
    is_locked_ = (metadata & lock_bit_mask_) != 0;
    is_leaf_ = (metadata & leaf_bit_mask_) != 0;
    has_sibling_ = (metadata & sibling_bit_mask_) != 0;
  }
  DEVICE_QUALIFIER void write_metadata_to_registers() {
    // key_meta_bit_ @ threads 15-31 will be overwritten by other metadata bits.
    uint32_t metadata = tile_.ballot(key_meta_bit_) & key_meta_bits_mask_;
    metadata |= ((num_keys_ << num_keys_offset_) & num_keys_mask_);
    if (is_leaf_) metadata |= leaf_bit_mask_;
    if (is_locked_) metadata |= lock_bit_mask_;
    if (has_sibling_) metadata |= sibling_bit_mask_;
    if (tile_.thread_rank() == metadata_lane_) {
      lane_elem_ = metadata;
    }
  }
  template <uint32_t offset, bool bit>
  DEVICE_QUALIFIER void set_bit_in_metadata() {
    if (tile_.thread_rank() == metadata_lane_) {
      if (bit) { lane_elem_ |= (1u << offset); }
      else { lane_elem_ &= ~(1u << offset); }
    }
  }
  DEVICE_QUALIFIER bool is_locked() const { return is_locked_; }
  DEVICE_QUALIFIER bool try_lock() {
    elem_type old;
    if (tile_.thread_rank() == metadata_lane_) {
      old = atomicOr(reinterpret_cast<elem_type*>(&node_ptr_[metadata_lane_]),
                     static_cast<elem_type>(lock_bit_mask_));
    }
    old = tile_.shfl(old, metadata_lane_);
    is_locked_ = (old & lock_bit_mask_) == 0; // if previously not locked, now it's locked
    if (is_locked_) {
      set_bit_in_metadata<lock_bit_offset_, true>();
      __threadfence();
    }
    else {
      set_bit_in_metadata<lock_bit_offset_, false>();
    }
    return is_locked_;
  }
  DEVICE_QUALIFIER void lock() {
    while (auto failed = !try_lock()) {}
    is_locked_ = true;
  }
  DEVICE_QUALIFIER void unlock() {
    __threadfence();
    elem_type old;
    if (tile_.thread_rank() == metadata_lane_) {
      old = atomicAnd(reinterpret_cast<elem_type*>(&node_ptr_[metadata_lane_]),
                      static_cast<elem_type>(~lock_bit_mask_));
    }
    is_locked_ = false;
    set_bit_in_metadata<lock_bit_offset_, false>();
  }

  DEVICE_QUALIFIER bool traverse_required(const key_type& key) const {
    return has_sibling() && (get_high_key() < key);
  }
  DEVICE_QUALIFIER int find_next_location(const key_type& key) const {
    assert(!is_leaf_);
    const bool key_less_equal = is_valid_key_lane() && (key <= lane_elem_);
    uint32_t key_less_equal_bitmap = tile_.ballot(key_less_equal);
    auto next_location = __ffs(key_less_equal_bitmap) - 1;
    assert(0 <= next_location && next_location < num_keys_ + 1);
    return next_location;
  }
  DEVICE_QUALIFIER value_type find_next(const key_type& key) const {
    auto next_location = find_next_location(key);
    return tile_.shfl(lane_elem_, get_value_lane_from_location(next_location));
  }
  DEVICE_QUALIFIER bool key_is_in_upperhalf(const key_type& pivot_key, const key_type& key) const {
    return (pivot_key < key);
  }

  DEVICE_QUALIFIER int find_key_location_in_node(const key_type& key, bool last_slice) const {
    assert(is_leaf_);
    auto key_exist = tile_.ballot(
        is_valid_key_lane() && lane_elem_ == key && key_meta_bit_ == last_slice);
    return __ffs(key_exist) - 1;
  }
  DEVICE_QUALIFIER bool key_is_in_node(const key_type& key, bool last_slice) const {
    assert(is_leaf_);
    auto key_exist = tile_.ballot(
      is_valid_key_lane() && lane_elem_ == key && key_meta_bit_ == last_slice);
    return (key_exist != 0);
  }
  DEVICE_QUALIFIER bool ptr_is_in_node(const value_type& ptr) const {
    auto ptr_exist = tile_.ballot(is_valid_value_lane() && lane_elem_ == ptr);
    return (ptr_exist != 0);
  }
  DEVICE_QUALIFIER bool get_key_value_from_node(const key_type& key, value_type& value, bool last_slice) const {
    assert(is_leaf_);
    auto key_location = find_key_location_in_node(key, last_slice);
    if (key_location < 0) return false;
    value = get_value_from_location(key_location);
    return true;
  }

  DEVICE_QUALIFIER int get_split_left_width() const {
    // normally, location is left_half_width_
    // but we should put same-key non-last-slice and first last-slice in the same node
    // to avoid comparing keys, we just shift if last elem of the left half is non-last-slice
    bool shift_required = false;
    if (is_leaf_) {
      bool key_meta_of_default_pivot = get_key_meta_bit_from_location(left_half_width_ - 1);
      shift_required = !key_meta_of_default_pivot; // shift if it's non-last-slice
    }
    return shift_required ? (left_half_width_ - 1) : left_half_width_;
  }

  DEVICE_QUALIFIER masstree_node do_split(const value_type right_sibling_index,
                                          elem_type* right_sibling_ptr,
                                          const int left_width,
                                          const bool make_sibling_locked = false) {
    // same-key non-last-slice entry and first last-slice entry (if both exist) go to the same node
    //assert(!(
    //  is_leaf_ &&
    //  (get_key_from_location(left_width - 1) == get_key_from_location(left_width)) &&
    //  (get_key_meta_bit_from_location(left_width - 1) == false)
    //));
    // prepare the upper half in right sibling
    elem_type right_sibling_elem = tile_.shfl_down(lane_elem_, left_width);
    bool right_sibling_key_meta_bit = tile_.shfl_down(key_meta_bit_, left_width);
    
    // distribute num keys
    uint16_t right_num_keys = num_keys_ - left_width;
    num_keys_ = left_width;

    // reconnect right sibling pointers
    if (tile_.thread_rank() == sibling_ptr_lane_) {
      // right's right sibling = this node's previous right sibling
      right_sibling_elem = lane_elem_;
      // left's right sibling = right
      lane_elem_ = right_sibling_index;
    }
    bool right_has_sibling = has_sibling_;
    has_sibling_ = true;

    // create right sibling node
    masstree_node right_sibling_node =
        masstree_node(right_sibling_ptr, right_sibling_index, tile_, right_sibling_elem, 
                      right_num_keys, make_sibling_locked, is_leaf_,
                      right_has_sibling, right_sibling_key_meta_bit);

    // flush metadata
    write_metadata_to_registers();
    right_sibling_node.write_metadata_to_registers();

    return right_sibling_node;
  }

  struct split_intermediate_result {
    masstree_node parent;
    masstree_node sibling;
    key_type pivot_key;
  };
  // Note parent must be locked before this gets called
  DEVICE_QUALIFIER split_intermediate_result split(const value_type right_sibling_index,
                                                   const value_type parent_index,
                                                   elem_type* right_sibling_ptr,
                                                   elem_type* parent_ptr,
                                                   const bool make_sibling_locked = false) {
    // We assume here that the parent is locked
    auto left_width = get_split_left_width();
    auto split_result = do_split(right_sibling_index, right_sibling_ptr, left_width, make_sibling_locked);

    // Update parent
    auto parent_node = masstree_node(parent_ptr, parent_index, tile_);
    parent_node.load(cuda_memory_order::memory_order_relaxed);

    // update the parent
    auto pivot_key = get_key_from_location(left_width - 1);
    parent_node.insert(pivot_key, right_sibling_index, false);
    return {parent_node, split_result, pivot_key};
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
    // Copy the current node into a child
    auto left_child =
        masstree_node(left_sibling_ptr, left_sibling_index, tile_, lane_elem_, 
                      num_keys_, make_children_locked, is_leaf_, has_sibling_, key_meta_bit_);
    // if the root was a leaf, now it should be intermediate
    if (is_leaf_) { is_leaf_ = false; }
    // Make new root
    num_keys_ = 2;
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
    assert(!has_sibling_); // root has no sibling
    write_metadata_to_registers();

    // now split the left child
    auto right_child =
        left_child.do_split(right_sibling_index, right_sibling_ptr, left_width, make_children_locked);
    return {left_child, right_child};
  }

  DEVICE_QUALIFIER void do_insert(const key_type& key, const value_type& value, bool last_slice, const size_type& key_location) {
    // shuffle the keys and do the insertion
    ++num_keys_;
    const int key_lane = get_key_lane_from_location(key_location);
    const int value_lane = get_value_lane_from_location(key_location + (is_leaf_ ? 0 : 1));
    auto up_elem = tile_.shfl_up(lane_elem_, 1);
    bool up_key_meta_bit = tile_.shfl_up(key_meta_bit_, 1);
    if (is_valid_key_lane()) {
      if (tile_.thread_rank() == key_lane) {
        lane_elem_ = key;
        key_meta_bit_ = last_slice; // embed last_slice into key_meta_bit
      }
      else if (tile_.thread_rank() > key_lane) {
        lane_elem_ = up_elem;
        key_meta_bit_ = up_key_meta_bit;
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
        (!get_key_meta_bit_from_location(key_location)) &&
        (get_key_from_location(key_location) == key)) {
      // if inserting a last-slice entry but there's a same-key non-last slice,
      // we should insert after the non-last slice.
      key_location++;
    }
    do_insert(key, value, last_slice_and_leaf, key_location);
  }

  /*DEVICE_QUALIFIER bool erase(const key_type& key) {
    // check if key exists
    auto key_location = find_key_location_in_node(key);
    if (key_location) {
      assert(num_keys_ > 0);
      --num_keys_;
      const int key_lane = get_key_lane_from_location(key_location);
      const int value_lane = get_value_lane_from_location(key_location);
      auto down_elem = tile_.shfl_down(lane_elem_, 1);
      bool down_ptr_meta_bit = tile_.shfl_down(ptr_meta_bit_, 1);
      if (is_valid_key_lane()) {
        if (tile_.thread_rank() >= key_lane) { lane_elem_ = down_elem; }
      }
      else if (is_valid_value_lane()) {
        if (tile_.thread_rank() >= value_lane) {
          lane_elem_ = down_elem;
          ptr_meta_bit_ = down_ptr_meta_bit;
        }
      }
      write_metadata_to_registers();
      return true;
    }
    else {
      return false;
    }
  }*/

  DEVICE_QUALIFIER masstree_node<tile_type>& operator=(
      const masstree_node<tile_type>& other) {
    node_ptr_ = other.node_ptr_;
    #ifdef NODE_DEBUG
    node_index_ = other.node_index_;
    #endif
    lane_elem_ = other.lane_elem_;
    num_keys_ = other.num_keys_;
    is_locked_ = other.is_locked_;
    is_leaf_ = other.is_leaf_;
    has_sibling_ = other.has_sibling_;
    key_meta_bit_ = other.key_meta_bit_;
    return *this;
  }

  DEVICE_QUALIFIER void print() const {
    #ifdef NODE_DEBUG
    bool lead_lane = (tile_.thread_rank() == 0);
    if (lead_lane) printf("node[%u]: {", node_index_);
    if (num_keys_ > max_num_keys_) {
      if (lead_lane) printf("num_keys too large: skip}\n");
      return;
    } 
    if (num_keys_ == 0) {
      if (lead_lane) printf("empty}\n");
      return;
    }
    if (lead_lane) printf("%u ", num_keys_);
    for (size_type i = 0; i < num_keys_; ++i) {
      elem_type key = tile_.shfl(lane_elem_, get_key_lane_from_location(i));
      elem_type value = tile_.shfl(lane_elem_, get_value_lane_from_location(i));
      bool key_meta_bit = tile_.shfl(key_meta_bit_, get_key_lane_from_location(i));
      if (lead_lane) printf("(%u %u %s) ", key, value, key_meta_bit ? "$" : ":");
    }
    if (lead_lane) printf("%s %s ", is_locked_ ? "locked" : "unlocked", is_leaf_ ? "leaf" : "nonleaf");
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
  size_type node_index_; // TODO only use on debug mode
  #endif
  elem_type lane_elem_;
  const tile_type tile_;
  
  // node consists of 2*node_width elements, each mapped to a lane in the tile.
  //  [key0] [key1] ... [key13] [key14] | [metadata]
  //  [ptr0] [ptr1] ... [ptr13] [ptr14] | [ptr15 = sibling_ptr]

  // for intermediates nodes,
  //    ptr_i contains subtree with keys: key_(i-1) < key <= key_i.
  //    key[num_keys-1] is high key (upper bound of the subtree), used for B-link traversal.

  // metadata is 32bits.
  // MSB - [empty:9][has_sibling:1][is_leaf:1][is_locked:1][num_keys:4][empty:1][key_meta_bits:15] - LSB
  //    key_meta_bits: each corresponds to key14-key0 (MSB-LSB).
  //      for intermediate node's child pointers: meaningless
  //      for leaf (border) node's child pointers: bit=0 (root node pointer to next masstree layer), bit=1 (leaf pointer)
  static_assert(sizeof(elem_type) == sizeof(uint32_t));
  static constexpr uint32_t metadata_lane_ = node_width - 1;
  static constexpr uint32_t sibling_ptr_lane_ = node_width * 2 - 1;
  
  static constexpr uint32_t num_keys_offset_ = 16;
  static constexpr uint32_t num_keys_bits_ = 4;
  static constexpr uint32_t num_keys_mask_ = ((1u << num_keys_bits_) - 1) << num_keys_offset_;
  static constexpr uint32_t lock_bit_offset_ = 20;
  static constexpr uint32_t lock_bit_mask_ = 1u << lock_bit_offset_;
  static constexpr uint32_t leaf_bit_offset_ = 21;
  static constexpr uint32_t leaf_bit_mask_ = 1u << leaf_bit_offset_;
  static constexpr uint32_t sibling_bit_offset_ = 22;
  static constexpr uint32_t sibling_bit_mask_ = 1u << sibling_bit_offset_;
  static constexpr uint32_t key_meta_bits_offset_ = 0;
  static constexpr uint32_t key_meta_bits_bits_ = 15;
  static constexpr uint32_t key_meta_bits_mask_ = ((1u << key_meta_bits_bits_) - 1) << key_meta_bits_offset_;

  static constexpr size_type max_num_keys_ = node_width - 1;
  static constexpr uint32_t left_half_width_ = (max_num_keys_ + 1) / 2;

  uint16_t num_keys_;
  bool is_locked_;
  bool is_leaf_;
  bool has_sibling_;
  bool key_meta_bit_; // per-lane value. only threads0-14's values are meaningful
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
