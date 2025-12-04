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
                                 const uint32_t metadata,
                                 const uint32_t num_keys,
                                 bool is_locked)
      : node_ptr_(ptr)
      #ifdef NODE_DEBUG
      , node_index_(index)
      #endif
      , lane_elem_(elem)
      , tile_(tile)
      , metadata_(metadata)
      , num_keys_(num_keys) {
    set_metadata_bit<lock_bit_mask_>(metadata_, is_locked);
  }

  DEVICE_QUALIFIER void initialize_root() {
    lane_elem_ = 0;
    // metadata:
    //  num_keys = 0
    //  locked = false
    //  leaf = true
    //  has_sibling = false
    //  key_meta_bits = 0
    metadata_ = 0;
    set_metadata_bit<leaf_bit_mask_, true>(metadata_);
    num_keys_ = 0;
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
    // get per-lane value
    assert((metadata_ & key_meta_reserved_bit_mask_) == 0);
    metadata_ |= (((metadata_ >> (tile_.thread_rank())) & 1u) << key_meta_reserved_bit_offset_);
    // num_keys shortcut
    num_keys_ = (metadata_ & num_keys_mask_) >> num_keys_offset_;
  }
  DEVICE_QUALIFIER void write_metadata_to_registers() {
    metadata_ &= ~(num_keys_mask_ | key_meta_bits_mask_);
    // apply num_keys shortcut
    metadata_ |= (num_keys_ << num_keys_offset_);
    // key_meta_bit_ @ threads 15-31 will be overwritten by other metadata bits.
    uint32_t key_meta_bits = (tile_.ballot(get_this_lane_key_meta_bit()) & key_meta_bits_mask_);
    metadata_ |= key_meta_bits;
    if (tile_.thread_rank() == metadata_lane_) {
      // also unset the key_meta reserved bit to zero
      metadata_ &= ~key_meta_reserved_bit_mask_;
      lane_elem_ = metadata_;
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
  DEVICE_QUALIFIER bool get_key_meta_bit_from_location(const int location) const {
    return tile_.shfl(get_this_lane_key_meta_bit(), get_key_lane_from_location(location));
  }
  DEVICE_QUALIFIER bool is_valid_key_lane() const {
    return tile_.thread_rank() < num_keys_;
  }
  DEVICE_QUALIFIER bool is_valid_value_lane() const {
    // intermediate node has children one more than keys
    return node_width <= tile_.thread_rank() && tile_.thread_rank() < node_width + num_keys_;
  }

  DEVICE_QUALIFIER uint32_t num_keys() const {
    return num_keys_;
  }
  DEVICE_QUALIFIER bool is_leaf() const {
    return (metadata_ & leaf_bit_mask_) != 0;
  }
  DEVICE_QUALIFIER bool is_full() const {
    assert(num_keys_ <= max_num_keys_);
    return (num_keys_ == max_num_keys_);
  }
  DEVICE_QUALIFIER bool has_sibling() const {
    return (metadata_ & sibling_bit_mask_) != 0;
  }
  DEVICE_QUALIFIER bool get_this_lane_key_meta_bit() const {
    return (metadata_ & key_meta_reserved_bit_mask_) != 0;
  }
  template <uint32_t mask, bool bit>
  DEVICE_QUALIFIER void set_metadata_bit(uint32_t& metadata) {
    if constexpr (bit) { metadata |= mask; }
    else { metadata &= ~mask; }
  }
  template <uint32_t mask>
  DEVICE_QUALIFIER void set_metadata_bit(uint32_t& metadata, bool bit) {
    if (bit) { metadata |= mask; }
    else { metadata &= ~mask; }
  }
  DEVICE_QUALIFIER key_type get_high_key() const {
    assert(num_keys_ > 0); // never called
    return tile_.shfl(lane_elem_, num_keys_ - 1);
  }
  DEVICE_QUALIFIER value_type get_sibling_index() const {
    return tile_.shfl(lane_elem_, sibling_ptr_lane_);
  }

  DEVICE_QUALIFIER bool is_locked() const {
    return (metadata_ & lock_bit_mask_) != 0;
  }
  DEVICE_QUALIFIER bool try_lock() {
    elem_type old;
    if (tile_.thread_rank() == metadata_lane_) {
      old = atomicOr(reinterpret_cast<elem_type*>(&node_ptr_[metadata_lane_]),
                     static_cast<elem_type>(lock_bit_mask_));
    }
    old = tile_.shfl(old, metadata_lane_);
    bool is_locked = (old & lock_bit_mask_) == 0; // if previously not locked, now it's locked
    if (is_locked) {
      __threadfence();
    }
    return is_locked;
    // do not need to update registers; if locked, the code will load() again.
    // if lock failed, this node object will be disposed.
  }
  DEVICE_QUALIFIER void lock() {
    while (!try_lock()) {}
    // the code will load() again
  }
  DEVICE_QUALIFIER void unlock() {
    __threadfence();
    elem_type old;
    if (tile_.thread_rank() == metadata_lane_) {
      old = atomicAnd(reinterpret_cast<elem_type*>(&node_ptr_[metadata_lane_]),
                      static_cast<elem_type>(~lock_bit_mask_));
    }
    // the node object can be used after this, so update regsiters
    set_metadata_bit<lock_bit_mask_, false>(metadata_);
    if (tile_.thread_rank() == metadata_lane_) {
      lane_elem_ &= ~lock_bit_mask_;
    }
  }

  DEVICE_QUALIFIER bool traverse_required(const key_type& key) const {
    return has_sibling() && (get_high_key() < key);
  }
  DEVICE_QUALIFIER int find_next_location(const key_type& key) const {
    assert(!is_leaf());
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
    assert(is_leaf());
    auto key_exist = tile_.ballot(is_valid_key_lane() &&
                                  lane_elem_ == key &&
                                  get_this_lane_key_meta_bit() == last_slice);
    return __ffs(key_exist) - 1;
  }
  DEVICE_QUALIFIER bool key_is_in_node(const key_type& key, bool last_slice) const {
    assert(is_leaf());
    auto key_exist = tile_.ballot(is_valid_key_lane() &&
                                  lane_elem_ == key &&
                                  get_this_lane_key_meta_bit() == last_slice);
    return (key_exist != 0);
  }
  DEVICE_QUALIFIER bool ptr_is_in_node(const value_type& ptr) const {
    auto ptr_exist = tile_.ballot(is_valid_value_lane() && lane_elem_ == ptr);
    return (ptr_exist != 0);
  }
  DEVICE_QUALIFIER bool get_key_value_from_node(const key_type& key, value_type& value, bool last_slice) const {
    assert(is_leaf());
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
    if (is_leaf()) {
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
    assert(is_full());
    // prepare the upper half in right sibling
    elem_type right_sibling_elem = tile_.shfl_down(lane_elem_, left_width);
    // metadata shfl required for key_meta_reserved bit
    uint32_t right_sibling_metadata = tile_.shfl_down(metadata_, left_width);

    // reconnect right sibling pointers
    if (tile_.thread_rank() == sibling_ptr_lane_) {
      // right's right sibling = this node's previous right sibling
      right_sibling_elem = lane_elem_;
      // left's right sibling = right
      lane_elem_ = right_sibling_index;
    }
    set_metadata_bit<sibling_bit_mask_>(right_sibling_metadata, has_sibling());
    set_metadata_bit<sibling_bit_mask_, true>(metadata_);

    // create right sibling node
    masstree_node right_sibling_node =
        masstree_node(right_sibling_ptr, right_sibling_index, tile_, right_sibling_elem, 
                      right_sibling_metadata, max_num_keys_ - left_width, make_sibling_locked);
    num_keys_ = left_width;

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
                      metadata_, num_keys_, make_children_locked);
    // if the root was a leaf, now it should be intermediate
    set_metadata_bit<leaf_bit_mask_, false>(metadata_);
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
    assert(!has_sibling()); // root has no sibling
    write_metadata_to_registers();

    // now split the left child
    auto right_child =
        left_child.do_split(right_sibling_index, right_sibling_ptr, left_width, make_children_locked);
    return {left_child, right_child};
  }

  DEVICE_QUALIFIER void do_insert(const key_type& key, const value_type& value, bool last_slice, const size_type& key_location) {
    // shuffle the keys and do the insertion
    num_keys_++;
    const int key_lane = get_key_lane_from_location(key_location);
    const int value_lane = get_value_lane_from_location(key_location + (is_leaf() ? 0 : 1));
    auto up_elem = tile_.shfl_up(lane_elem_, 1);
    bool up_key_meta_bit = tile_.shfl_up(get_this_lane_key_meta_bit(), 1);
    if (is_valid_key_lane()) {
      if (tile_.thread_rank() == key_lane) {
        lane_elem_ = key;
        // embed last_slice into key_meta_bit
        set_metadata_bit<key_meta_reserved_bit_mask_>(metadata_, last_slice);
      }
      else if (tile_.thread_rank() > key_lane) {
        lane_elem_ = up_elem;
        set_metadata_bit<key_meta_reserved_bit_mask_>(metadata_, up_key_meta_bit);
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
    metadata_ = other.metadata_;
    num_keys_ = other.num_keys_;
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
      bool key_meta_bit = tile_.shfl(get_this_lane_key_meta_bit(), get_key_lane_from_location(i));
      if (lead_lane) printf("(%u %u %s) ", key, value, key_meta_bit ? "$" : ":");
    }
    if (lead_lane) printf("%s %s ", is_locked() ? "locked" : "unlocked", is_leaf() ? "leaf" : "nonleaf");
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
  // MSB - [empty:9][has_sibling:1][is_leaf:1][is_locked:1][num_keys:4][reserved_to_zero:1][key_meta_bits:15] - LSB
  //    key_meta_bits: each corresponds to key14-key0 (MSB-LSB).
  //      for intermediate node's child pointers: meaningless
  //      for leaf (border) node's child pointers: bit=0 (root node pointer to next masstree layer), bit=1 (leaf pointer)
  //    in metadata_ variable (in register), bit15 is reserved to store per-lane key_meta_bit (only valid for lane0-14)
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
  static constexpr uint32_t key_meta_reserved_bit_offset_ = 15;
  static constexpr uint32_t key_meta_reserved_bit_mask_ = 1u << key_meta_reserved_bit_offset_;

  static constexpr size_type max_num_keys_ = node_width - 1;
  static constexpr uint32_t left_half_width_ = (max_num_keys_ + 1) / 2;

  uint32_t metadata_;
  uint32_t num_keys_;
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
