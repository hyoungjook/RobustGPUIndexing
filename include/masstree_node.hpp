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
#include <utils.hpp>
#include <suffix.hpp>

template <typename tile_type, typename allocator_type>
struct masstree_node {
  using elem_type = uint32_t;
  using key_type = elem_type;
  using value_type = elem_type;
  using size_type = uint32_t;
  static constexpr int node_width = 16;
  static constexpr key_type max_key = std::numeric_limits<key_type>::max();
  static constexpr uint32_t KEYSTATE_VALUE = 0b00u;
  static constexpr uint32_t KEYSTATE_LINK = 0b01u;
  static constexpr uint32_t KEYSTATE_SUFFIX = 0b11u;
  DEVICE_QUALIFIER masstree_node(const tile_type& tile, allocator_type& allocator)
      : tile_(tile), allocator_(allocator) {}
  DEVICE_QUALIFIER masstree_node(size_type index, const tile_type& tile, allocator_type& allocator)
      : node_index_(index), tile_(tile), allocator_(allocator) {}
  {
    assert(tile_.size() == 2 * node_width);
  }
  DEVICE_QUALIFIER masstree_node(size_type index,
                                 const elem_type elem,
                                 const tile_type& tile,
                                 allocator_type& allocator)
      : node_index_(index)
      , lane_elem_(elem)
      , tile_(tile)
      , allocator_(allocator) {}
  DEVICE_QUALIFIER masstree_node(size_type index,
                                 const elem_type elem,
                                 uint32_t metadata,
                                 uint32_t keystate,
                                 const tile_type& tile,
                                 allocator_type& allocator)
      : node_index_(index)
      , lane_elem_(elem)
      , metadata_(metadata)
      , keystate_(keystate)
      , tile_(tile)
      , allocator_(allocator) {}

  DEVICE_QUALIFIER void initialize_root() {
    lane_elem_ = 0;
    metadata_ = (
      (0u << num_keys_offset_) |  // num_keys = 0;
      (0u & lock_bit_mask_) |     // is_locked = false;
      (border_bit_mask_) |        // is_border = true;
      (0u & sibling_bit_mask_) |  // has_sibling = false;
      (0u & garbage_bit_mask_) |  // is_garbage = false;
      (root_bit_mask_)            // is_root = true;
    );
    keystate_ = 0;
    write_metadata_to_registers();
  }

  template <bool atomic, bool acquire = true>
  DEVICE_QUALIFIER void load() {
    auto node_ptr = reinterpret_cast<elem_type*>(allocator_.address(node_index_));
    if constexpr (atomic) { tile_.sync(); }
    lane_elem_ = utils::memory::load<elem_type, atomic, acquire>(node_ptr + tile_.thread_rank());
    if constexpr (atomic) { tile_.sync(); }
    read_metadata_from_registers();
  }
  template <bool atomic, bool acquire = true>
  DEVICE_QUALIFIER void load_fetchonly() {
    auto node_ptr = reinterpret_cast<elem_type*>(allocator_.address(node_index_));
    if constexpr (atomic) { tile_.sync(); }
    lane_elem_ = utils::memory::load<elem_type, atomic, acquire>(node_ptr + tile_.thread_rank());
    if constexpr (atomic) { tile_.sync(); }
  }
  template <bool atomic, bool release = true>
  DEVICE_QUALIFIER void store() {
    auto node_ptr = reinterpret_cast<elem_type*>(allocator_.address(node_index_));
    if constexpr (atomic) { tile_.sync(); }
    utils::memory::store<elem_type, atomic, release>(node_ptr + tile_.thread_rank(), lane_elem_);
    if constexpr (atomic) { tile_.sync(); }
  }
  template <bool atomic, bool release = true>
  DEVICE_QUALIFIER void store_unlock() {
    assert(is_locked());
    metadata_ &= ~lock_bit_mask_;
    if (tile_.thread_rank() == metadata_lane_) {
      lane_elem_ &= ~lock_bit_mask_;
    }
    auto node_ptr = reinterpret_cast<elem_type*>(allocator_.address(node_index_));
    if constexpr (atomic) { tile_.sync(); }
    utils::memory::store<elem_type, atomic, release>(node_ptr + tile_.thread_rank(), lane_elem_);
    if constexpr (atomic) { tile_.sync(); }
  }

  DEVICE_QUALIFIER void read_metadata_from_registers() {
    metadata_ = tile_.shfl(lane_elem_, metadata_lane_);
    if (is_border()) {
      uint32_t keystate_bits = tile_.shfl(lane_elem_, keystate_bits_lane_);
      keystate_ = ((keystate_bits >> tile_.thread_rank()) & keystate_mask_more_key_) |
                   ((keystate_bits >> (tile_.thread_rank() + (node_width - 1))) & keystate_mask_suffix_);
    }
  }
  DEVICE_QUALIFIER void write_metadata_to_registers() {
    if (tile_.thread_rank() == metadata_lane_) {
      lane_elem_ = metadata_;
    }
    if (is_border()) {
      uint32_t keystate_bits = (tile_.ballot(keystate_ & keystate_mask_more_key_) & 0x0000ffffu) |
                                (tile_.ballot(keystate_ & keystate_mask_suffix_) << node_width);
      if (tile_.thread_rank() == keystate_bits_lane_) {
        lane_elem_ = keystate_bits;
      }
    }
  }

  DEVICE_QUALIFIER elem_type get_lane_elem() const { return lane_elem_; }

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
  DEVICE_QUALIFIER uint32_t get_keystate_from_location(const int location) const {
    return tile_.shfl(keystate_, get_key_lane_from_location(location));
  }
  static DEVICE_QUALIFIER bool keystate_has_more_key(const uint32_t keystate) {
    return (keystate & keystate_mask_more_key_) != 0;
  }
  DEVICE_QUALIFIER bool is_valid_key_lane() const {
    return tile_.thread_rank() < num_keys();
  }
  DEVICE_QUALIFIER bool is_valid_value_lane() const {
    return node_width <= tile_.thread_rank() && tile_.thread_rank() < node_width + num_keys();
  }

  DEVICE_QUALIFIER size_type get_node_index() const {
    return node_index_;
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
  DEVICE_QUALIFIER bool is_root() const { 
    return static_cast<bool>(metadata_ & root_bit_mask_);
  }
  DEVICE_QUALIFIER bool is_full() const {
    assert(is_border() ? (num_keys() <= border_max_num_keys_) : (num_keys() <= interior_max_num_keys_));
    return (is_border() ? (num_keys() == border_max_num_keys_) : (num_keys() == interior_max_num_keys_));
  }
  DEVICE_QUALIFIER bool is_underflow() const {
    return (num_keys() <= underflow_num_keys_) && (!is_root());
  }
  DEVICE_QUALIFIER bool is_mergeable(const masstree_node& sibling_node) const {
    return (num_keys() + sibling_node.num_keys()) < (is_border() ? border_max_num_keys_ : interior_max_num_keys_);
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
  DEVICE_QUALIFIER bool try_lock_load() {
    auto node_ptr = reinterpret_cast<elem_type*>(allocator_.address(node_index_));
    if (tile_.thread_rank() == metadata_lane_) {
      cuda::atomic_ref<elem_type, cuda::thread_scope_device> metadata_ref(node_ptr[metadata_lane_]);
      lane_elem_ = metadata_ref.fetch_or(lock_bit_mask_, cuda::memory_order_relaxed);
    }
    // if previously not locked, now it's locked
    bool is_locked = (tile_.shfl(lane_elem_, metadata_lane_) & lock_bit_mask_) == 0;
    if (is_locked) {
      tile_.sync();
      lane_elem_ = utils::memory::load<elem_type, true, true>(node_ptr + tile_.thread_rank());
      tile_.sync();
      read_metadata_from_registers();
    }
    return is_locked;
  }
  DEVICE_QUALIFIER void lock_load() {
    while (!try_lock_load());
  }
  DEVICE_QUALIFIER void unlock() {
    assert(is_locked());
    auto node_ptr = reinterpret_cast<elem_type*>(allocator_.address(node_index_));
    if (tile_.thread_rank() == metadata_lane_) {
      cuda::atomic_ref<elem_type, cuda::thread_scope_device> metadata_ref(node_ptr[metadata_lane_]);
      metadata_ref.fetch_and(~lock_bit_mask_, cuda::memory_order_release);
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
    return get_value_from_location(next_location);
  }
  DEVICE_QUALIFIER value_type find_next_and_sibling(const key_type& key, value_type& sibling_index, bool& sibling_at_left) const {
    auto next_location = find_next_location(key);
    sibling_at_left = decide_merge_left(next_location);
    sibling_index = get_value_from_location(next_location + (sibling_at_left ? -1 : 1));
    return get_value_from_location(next_location);
  }

  DEVICE_QUALIFIER uint32_t match_key_in_node(const key_type& key, bool more_key) const {
    assert(is_border());
    return tile_.ballot(is_valid_key_lane() &&
                        lane_elem_ == key &&
                        keystate_has_more_key(keystate_) == more_key);
  }
  DEVICE_QUALIFIER uint32_t match_key_in_node(const key_type& key, uint32_t keystate) const {
    assert(is_border());
    return tile_.ballot(is_valid_key_lane() &&
                        lane_elem_ == key &&
                        keystate_ == keystate);
  }
  DEVICE_QUALIFIER bool key_is_in_node(const key_type& key, bool more_key) const {
    assert(is_border());
    auto key_exist = match_key_in_node(key, more_key);
    return (key_exist != 0);
  }
  DEVICE_QUALIFIER bool key_is_in_node(const key_type& key, uint32_t keystate) const {
    assert(is_border());
    auto key_exist = match_key_in_node(key, keystate);
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
  DEVICE_QUALIFIER int get_key_value_from_node(const key_type& key, value_type& value, bool more_key) const {
    // returns keystate if found, else return -1
    assert(is_border());
    auto key_exists = match_key_in_node(key, more_key);
    if (key_exists == 0) return -1;
    int location = __ffs(key_exists) - 1;
    value = get_value_from_location(location);
    return static_cast<int>(get_keystate_from_location(location));
  }
  DEVICE_QUALIFIER bool get_key_value_from_node(const key_type& key, value_type& value, uint32_t keystate) const {
    assert(is_border());
    auto key_exists = match_key_in_node(key, keystate);
    if (key_exists == 0) return false;
    int location = __ffs(key_exists) - 1;
    value = get_value_from_location(location);
    return true;
  }

  // lexicographic comparisons for key entries
  DEVICE_QUALIFIER static bool cmp_key(const key_type& k1, bool morekey1, const key_type& k2, bool morekey2) {
    return (k1 < k2) || ((k1 == k2) && (static_cast<int>(morekey1) <= static_cast<int>(morekey2)));
  }
  template <bool use_upper_key>
  DEVICE_QUALIFIER uint32_t scan(const key_type& lower_key_slice,
                                 const bool lower_key_more,
                                 const key_type* lower_key,         // original argument
                                 const size_type lower_key_length,  // original argument
                                 bool& passed_lower_key,
                                 const key_type& upper_key_slice,
                                 const bool upper_key_more,
                                 const key_type* upper_key,         // original argument
                                 const size_type upper_key_length,  // original argument
                                 const bool ignore_upper_key,
                                 const size_type out_max_count,
                                 int& link_entry_location,
                                 value_type* out_value,
                                 key_type* out_keys,
                                 size_type* out_key_lengths,
                                 const size_type& layer,
                                 const size_type& out_key_max_length) {
    using suffix_type = suffix_node<tile_type, allocator_type>;
    // return values in range, until we meet the link entry
    // the location of first in-range link entry is stored in link_entry_location
    // if there's no link entry in range, link_entry_location = -1
    assert(is_border());
    link_entry_location = -1;
    // compute in_range
    bool in_range = is_valid_key_lane() &&
        cmp_key(lower_key_slice, lower_key_more, lane_elem_, keystate_has_more_key(keystate_)) &&
        (!use_upper_key || (ignore_upper_key || cmp_key(lane_elem_, keystate_has_more_key(keystate_), upper_key_slice, upper_key_more)));
    uint32_t in_range_ballot = tile_.ballot(in_range);
    if (in_range_ballot == 0) {
      return 0;
    }
    // compute first location
    int first_location = __ffs(in_range_ballot) - 1;
    if (!passed_lower_key) {
      const bool same_slice = (get_key_from_location(first_location) == lower_key_slice);
      const auto first_keystate = get_keystate_from_location(first_location);
      passed_lower_key = !(same_slice && first_keystate == KEYSTATE_LINK);
      // adjust required if (suffix < lower_key)
      if (same_slice && first_keystate == KEYSTATE_SUFFIX) {
        auto suffix_index = get_value_from_location(first_location);
        auto suffix = suffix_type(suffix_index, tile_, allocator_);
        suffix.load_head();
        if (suffix.strcmp(lower_key + layer + 1, lower_key_length - layer - 1) > 0) {
          if (tile_.thread_rank() == first_location) { in_range = false; }
          in_range_ballot = tile_.ballot(in_range);
          if (in_range_ballot == 0) {
            return 0;
          }
          first_location = __ffs(in_range_ballot) - 1;
        }
      }
    }
    // compute last location until link
    int last_location = utils::bits::bfind(in_range_ballot) + 1;
    const bool in_range_and_link = in_range && (keystate_ == KEYSTATE_LINK);
    const uint32_t in_range_and_link_ballot = tile_.ballot(in_range_and_link);
    if (in_range_and_link_ballot != 0) {
      link_entry_location = __ffs(in_range_and_link_ballot) - 1;
      last_location = link_entry_location;
    }
    if (use_upper_key && !ignore_upper_key && link_entry_location < 0) {
      // adjust required if (high_key < suffix)
      if (get_key_from_location(last_location - 1) == upper_key_slice &&
          get_keystate_from_location(last_location - 1) == KEYSTATE_SUFFIX) {
        auto suffix_index = get_value_from_location(last_location - 1);
        auto suffix = suffix_type(suffix_index, tile_, allocator_);
        suffix.load_head();
        if (suffix.strcmp(upper_key + layer + 1, upper_key_length - layer - 1) < 0) {
          last_location--;
        }
      }
    }
    // results up to out_max_count
    int count = last_location - first_location;
    if (count > out_max_count) {
      last_location = first_location + out_max_count;
      count = out_max_count;
      link_entry_location = -1;
    }
    // store results
    if (count > 0) {
      in_range = (first_location <= tile_.thread_rank() && tile_.thread_rank() < last_location);
      auto values_to_key_lanes = tile_.shfl_down(lane_elem_, node_width);
      // store values
      if (out_value) {
        if (in_range) {
          out_value[tile_.thread_rank() - first_location] =
            (keystate_ != KEYSTATE_SUFFIX) ? values_to_key_lanes :
              suffix_type::fetch_value_only(values_to_key_lanes, allocator_);
        }
      }
      if (out_key_lengths) {
        // store key lengths
        if (in_range) {
          out_key_lengths[tile_.thread_rank() - first_location] =
            layer + 1 + ((keystate_ != KEYSTATE_SUFFIX) ? 0 :
              suffix_type::fetch_length_only(values_to_key_lanes, allocator_));
        }
      }
      if (out_keys) {
        // store key slice of this layer
        if (in_range) {
          out_keys[(tile_.thread_rank() - first_location) * out_key_max_length + layer] = lane_elem_;
        }
        // for suffix entries, store key slice of later layers too
        bool to_flush = in_range && keystate_ == KEYSTATE_SUFFIX;
        uint32_t flush_queue = tile_.ballot(to_flush);
        while (flush_queue) {
          auto cur_location = __ffs(flush_queue) - 1;
          auto suffix_index = get_value_from_location(cur_location);
          auto suffix = suffix_type(suffix_index, tile_, allocator_);
          suffix.load_head();
          suffix.flush(out_keys + ((cur_location - first_location) * out_key_max_length + layer + 1));
          if (tile_.thread_rank() == cur_location) { to_flush = false; }
          flush_queue = tile_.ballot(to_flush);
        }
      }
    }
    return count;
  }

  DEVICE_QUALIFIER int get_split_left_width() const {
    // normally, location is left_half_width_
    // but a value entry and a link entry with the same key should be in the same node
    // to avoid comparing keys, we just shift if last elem of the left half is value entry
    bool shift_required = is_border() && (get_keystate_from_location(half_node_width_ - 1) == KEYSTATE_VALUE);
    return shift_required ? (half_node_width_ - 1) : half_node_width_;
  }

  DEVICE_QUALIFIER void do_split(masstree_node& right_sibling_node,
                                 const int left_width) {
    // right_sibling_node has valid node_index_. this function fills its lane_elem_.
    // same-key link and value entries (if both exist) must be in the same node
    //assert(!(
    //  is_border() &&
    //  (get_key_from_location(left_width - 1) == get_key_from_location(left_width)) &&
    //  (get_keystate_from_location(left_width - 1) == KEYSTATE_VALUE)
    //));
    assert(is_full());
    // prepare the upper half in right sibling
    right_sibling_node.lane_elem_ = tile_.shfl_down(lane_elem_, left_width);
    right_sibling_node.keystate_ = is_border() ? tile_.shfl_down(keystate_, left_width) : false;
    // reconnect right sibling pointers
    if (tile_.thread_rank() == sibling_ptr_lane_) {
      // right's right sibling = this node's previous right sibling
      right_sibling_node.lane_elem_ = lane_elem_;
      // left's right sibling = right
      lane_elem_ = right_sibling_node.get_node_index();
    }
    // reassign high keys
    if (is_border()) {
      auto pivot_key = get_key_from_location(left_width - 1);
      if (tile_.thread_rank() == border_high_key_lane_) {
        // right's high key = this node's previous high key
        right_sibling_node.lane_elem_ = lane_elem_;
        // left's high key = pivot key
        lane_elem_ = pivot_key;
      }
    }
    // update metadata
    right_sibling_node.metadata_ = (metadata_ & ~(num_keys_mask_ | sibling_bit_mask_));
    right_sibling_node.metadata_ |= ((num_keys() - left_width) << num_keys_offset_); // right.num_keys = num_keys - left_width;
    if (has_sibling()) { right_sibling_node.metadata_ |= sibling_bit_mask_; }  // right.has_sibling = has_sibling;
    set_num_keys(left_width);
    metadata_ |= sibling_bit_mask_; // has_sibling = true;
    // flush metadata
    write_metadata_to_registers();
    right_sibling_node.write_metadata_to_registers();
  }

  // Note parent must be locked before this gets called
  DEVICE_QUALIFIER void split(masstree_node& right_sibling_node,
                              const value_type parent_index,
                              masstree_node& parent_node) {
    // We assume here that the parent is locked
    auto left_width = get_split_left_width();
    do_split(right_sibling_node, left_width);

    // update the parent
    auto pivot_key = get_key_from_location(left_width - 1);
    parent_node.insert(pivot_key, right_sibling_node.get_node_index(), 0);
  }

  DEVICE_QUALIFIER void split_as_root(masstree_node& left_child_node,
                                      masstree_node& right_child_node) {
    // Copy the current node into a child
    assert(is_root());
    left_child_node.lane_elem_ = lane_elem_;
    left_child_node.metadata_ = metadata_;
    left_child_node.keystate_ = keystate_;
    // if the root was a border node, now it should be interior
    metadata_ &= ~border_bit_mask_; // is_border = false;
    // left child is not a root anymore
    left_child_node.metadata_ &= ~root_bit_mask_;  // left_child_node.is_root = false;
    // Make new root
    set_num_keys(2);
    auto left_width = left_child_node.get_split_left_width();
    auto pivot_key = left_child_node.get_key_from_location(left_width - 1);
    if (tile_.thread_rank() == get_key_lane_from_location(0)) {
      lane_elem_ = pivot_key;
    }
    else if (tile_.thread_rank() == get_value_lane_from_location(0)) {
      lane_elem_ = left_child_node.node_index_;
    }
    else if (tile_.thread_rank() == get_key_lane_from_location(1)) {
      lane_elem_ = max_key;
    }
    else if (tile_.thread_rank() == get_value_lane_from_location(1)) {
      lane_elem_ = right_child_node.node_index_;
    }
    assert(!has_sibling()); // root has no sibling
    write_metadata_to_registers();

    // now split the left child
    left_child_node.do_split(right_child_node, left_width);
  }

  DEVICE_QUALIFIER void do_insert(const key_type& key, const value_type& value, uint32_t keystate, const size_type& key_location) {
    // shuffle the keys and do the insertion
    assert(!is_full());
    metadata_++;  // equiv. to num_keys++;
    const int key_lane = get_key_lane_from_location(key_location);
    const int value_lane = get_value_lane_from_location(key_location + (is_border() ? 0 : 1));
    auto up_elem = tile_.shfl_up(lane_elem_, 1);
    uint32_t up_keystate = tile_.shfl_up(keystate_, 1);
    if (is_valid_key_lane()) {
      if (tile_.thread_rank() == key_lane) {
        lane_elem_ = key;
        keystate_ = keystate;
      }
      else if (tile_.thread_rank() > key_lane) {
        lane_elem_ = up_elem;
        keystate_ = up_keystate;
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
                               uint32_t keystate) {
    assert(!is_full());
    const bool key_is_larger = is_valid_key_lane() && (key > lane_elem_);
    uint32_t key_is_larger_bitmap = tile_.ballot(key_is_larger);
    auto key_location = utils::bits::bfind(key_is_larger_bitmap) + 1;
    assert(key_location <= num_keys());
    // if key already exists, this is the location of it.
    if (is_border() && (keystate != KEYSTATE_VALUE) &&
        (key_location < num_keys()) && (get_key_from_location(key_location) == key) &&
        (get_keystate_from_location(key_location) == KEYSTATE_VALUE)) {
      // the (link or suffix) entry should go after the same-key value entry.
      key_location++;
    }
    do_insert(key, value, keystate, key_location);
  }

  DEVICE_QUALIFIER void update(const key_type& key,
                               const value_type& value,
                               uint32_t old_keystate,
                               uint32_t new_keystate) {
    // already checked that (key, old_keystate) exists
    assert(is_border());
    uint32_t key_exists = match_key_in_node(key, old_keystate);
    assert(key_exists != 0);
    uint32_t key_location = __ffs(key_exists) - 1;
    if (tile_.thread_rank() == get_key_lane_from_location(key_location)) {
      keystate_ = new_keystate;
    }
    if (tile_.thread_rank() == get_value_lane_from_location(key_location)) {
      lane_elem_ = value;
    }
    write_metadata_to_registers();
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

  DEVICE_QUALIFIER bool decide_merge_left(int location) const {
    // prioritize merge_left b/c borrow_right incurs additional malloc
    bool merge_with_left_sibling = (location > 0);
    return merge_with_left_sibling;
  }
  DEVICE_QUALIFIER int check_valid_merge_siblings(const size_type& child_index,
                                                  const size_type& sibling_index) {
    // if child and sibling is not adjacent or not exist, return -1
    // if child is at left (child - sibling), return 0
    // if sibling is at left (sibling - child), return 1
    static_assert(underflow_num_keys_ >= 2);
    assert(num_keys() >= 2);
    auto child_exist = match_ptr_in_node(child_index);
    auto sibling_exist = match_ptr_in_node(sibling_index);
    if (child_exist == 0 || sibling_exist == 0) return -1;
    int location_diff = __ffs(child_exist) - __ffs(sibling_exist);
    if (location_diff == 1) return 1;
    else if (location_diff == -1) return 0;
    else return -1;
  }
  DEVICE_QUALIFIER void merge(masstree_node& right_sibling_node,
                              masstree_node& parent_node) {
    // this node is the left sibling
    // copy elements from right sibling node
    elem_type shifted_elem = tile_.shfl_up(right_sibling_node.lane_elem_, num_keys());
    uint32_t shifted_keystate = is_border() ? tile_.shfl_up(right_sibling_node.keystate_, num_keys()) : 0;
    auto new_num_keys = num_keys() + right_sibling_node.num_keys();
    if ((num_keys() <= tile_.thread_rank() && tile_.thread_rank() < new_num_keys) ||
        (node_width + num_keys() <= tile_.thread_rank() && tile_.thread_rank() < node_width + new_num_keys)) {
      lane_elem_ = shifted_elem;
      keystate_ = shifted_keystate;
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
    int right_sibling_lane_in_parent = __ffs(parent_node.match_ptr_in_node(right_sibling_node.node_index_)) - 1;
    parent_node.do_erase(right_sibling_lane_in_parent - (1 + node_width), right_sibling_lane_in_parent);
    // set right sibling as empty node and connect it to this node
    right_sibling_node.make_garbage_node(true, node_index_);
  }

  DEVICE_QUALIFIER void merge_to_root(const value_type& parent_index,
                                      masstree_node& left_child_node,
                                      masstree_node& right_child_node) {
    // this node is parent
    assert(is_root() && num_keys() == 2);
    // copy the children into current node
    lane_elem_ = left_child_node.lane_elem_;
    keystate_ = left_child_node.keystate_;
    set_num_keys(left_child_node.num_keys() + right_child_node.num_keys());
    metadata_ = (metadata_ & ~border_bit_mask_) ^ (left_child_node.metadata_ & border_bit_mask_); // is_border = left.is_border;
    auto right_elem = tile_.shfl_up(right_child_node.lane_elem_, left_child_node.num_keys());
    uint32_t right_keystate = is_border() ? tile_.shfl_up(right_child_node.keystate_, left_child_node.num_keys()) : 0;
    if ((left_child_node.num_keys() <= tile_.thread_rank() && tile_.thread_rank() < num_keys()) ||
        (node_width + left_child_node.num_keys() <= tile_.thread_rank() && tile_.thread_rank() < node_width + num_keys())) {
      lane_elem_ = right_elem;
      keystate_ = right_keystate;
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
                                    masstree_node& parent_node) {
    // compute num shift; adjust similar to get_split_left_width()
    uint32_t num_shift = (sibling_node.num_keys() - num_keys()) / 2;
    if (is_border() && (sibling_node.get_keystate_from_location(sibling_node.num_keys() - num_shift - 1) == KEYSTATE_VALUE)) {
      num_shift++;
    }
    // copy last num_shift entries of the sibling into current
    elem_type shifted_elem = tile_.shfl_up(lane_elem_, num_shift);
    uint32_t shifted_keystate = is_border() ? tile_.shfl_up(keystate_, num_shift) : 0;
    metadata_ += num_shift; // equiv. to num_keys += num_shift;
    if ((tile_.thread_rank() < num_keys()) ||
        (node_width <= tile_.thread_rank() && tile_.thread_rank() < node_width + num_keys())) {
      lane_elem_ = shifted_elem;
      keystate_ = shifted_keystate;
    }
    sibling_node.metadata_ -= num_shift;  // equiv. to num_keys -= num_shift;
    shifted_elem = tile_.shfl_down(sibling_node.lane_elem_, sibling_node.num_keys());
    shifted_keystate = is_border() ? tile_.shfl_down(sibling_node.keystate_, sibling_node.num_keys()) : false;
    if ((tile_.thread_rank() < num_shift) ||
        (node_width <= tile_.thread_rank() && tile_.thread_rank() < node_width + num_shift)) {
      lane_elem_ = shifted_elem;
      keystate_ = shifted_keystate;
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
    int left_sibling_location_in_parent = __ffs(parent_node.match_ptr_in_node(sibling_node.node_index_)) - (1 + node_width);
    if (tile_.thread_rank() == get_key_lane_from_location(left_sibling_location_in_parent)) {
      parent_node.lane_elem_ = pivot_key;
    }
    //assert(!(
    //  is_border() &&
    //  (sibling_node.get_key_from_location(sibling_node.num_keys_ - 1) == get_key_from_location(0)) &&
    //  (sibling_node.get_keystate_from_location(sibling_node.num_keys_ - 1) == KEYSTATE_VALUE)
    //));
  }

  DEVICE_QUALIFIER void borrow_right(masstree_node& sibling_node,
                                     masstree_node& parent_node,
                                     masstree_node& new_sibling_node) {
    // new_sibling_node: only its node_index_ is valid. this fills its lane_elem_.
    // compute num shift; adjust similar to get_split_left_width()
    uint32_t num_shift = (sibling_node.num_keys() - num_keys()) / 2;
    if (is_border() && (sibling_node.get_keystate_from_location(num_shift - 1) == KEYSTATE_VALUE)) {
      num_shift--;
    }
    // copy first num_shift entries of the sibling into current
    elem_type shifted_elem = tile_.shfl_up(sibling_node.lane_elem_, num_keys());
    uint32_t shifted_keystate = is_border() ? tile_.shfl_up(sibling_node.keystate_, num_keys()) : 0;
    if ((num_keys() <= tile_.thread_rank() && tile_.thread_rank() < num_keys() + num_shift) ||
        (node_width + num_keys() <= tile_.thread_rank() && tile_.thread_rank() < node_width + num_keys() + num_shift)) {
      lane_elem_ = shifted_elem;
      keystate_ = shifted_keystate;
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
      lane_elem_ = new_sibling_node.node_index_;
    }
    // copy sibling to new_sibling
    new_sibling_node.lane_elem_ = sibling_node.lane_elem_;
    new_sibling_node.metadata_ = sibling_node.metadata_;
    new_sibling_node.keystate_ = sibling_node.keystate_;
    // remove first num_shift entries from the new_sibling
    shifted_elem = tile_.shfl_down(new_sibling_node.lane_elem_, num_shift);
    shifted_keystate = is_border() ? tile_.shfl_down(new_sibling_node.keystate_, num_shift) : false;
    new_sibling_node.metadata_ -= num_shift;  // equiv. to new_sibling_node.num_keys -= num_shift;
    if ((tile_.thread_rank() < new_sibling_node.num_keys()) ||
        (node_width <= tile_.thread_rank() && tile_.thread_rank() < node_width + new_sibling_node.num_keys())) {
      new_sibling_node.lane_elem_ = shifted_elem;
      new_sibling_node.keystate_ = shifted_keystate;
    }
    new_sibling_node.write_metadata_to_registers();
    // make old sibling empty
    sibling_node.make_garbage_node(true, node_index_);
    // update parent
    int right_sibling_lane_in_parent = __ffs(parent_node.match_ptr_in_node(sibling_node.node_index_)) - 1;
    if (tile_.thread_rank() == right_sibling_lane_in_parent - (1 + node_width)) {
      parent_node.lane_elem_ = pivot_key;
    }
    else if (tile_.thread_rank() == right_sibling_lane_in_parent) {
      parent_node.lane_elem_ = new_sibling_node.node_index_;
    }
    //assert(!(
    //  is_border() &&
    //  (get_key_from_location(num_keys_ - 1) == new_sibling_node.get_key_from_location(0)) &&
    //  (get_keystate_from_location(num_keys_ - 1) == KEYSTATE_VALUE)
    //));
  }

  DEVICE_QUALIFIER void do_erase(int key_lane, int value_lane) {
    assert(num_keys() > 0);
    metadata_--;  // equiv. to num_keys--;
    auto down_elem = tile_.shfl_down(lane_elem_, 1);
    uint32_t down_keystate = tile_.shfl_down(keystate_, 1);
    if (is_valid_key_lane()) {
      if (tile_.thread_rank() >= key_lane) {
        lane_elem_ = down_elem;
        keystate_ = down_keystate;
      }
    }
    else if (is_valid_value_lane()) {
      if (tile_.thread_rank() >= value_lane) {
        lane_elem_ = down_elem;
      }
    }
    write_metadata_to_registers();
  }

  DEVICE_QUALIFIER bool erase(const key_type& key, uint32_t keystate) {
    assert(is_border());
    uint32_t key_exists = match_key_in_node(key, keystate);
    if (key_exists == 0) return false;
    uint32_t key_location = __ffs(key_exists) - 1;
    do_erase(get_key_lane_from_location(key_location),
             get_value_lane_from_location(key_location));
    return true;
  }

  DEVICE_QUALIFIER masstree_node<tile_type, allocator_type>& operator=(
      const masstree_node<tile_type, allocator_type>& other) {
    node_index_ = other.node_index_;
    lane_elem_ = other.lane_elem_;
    metadata_ = other.metadata_;
    keystate_ = other.keystate_;
    return *this;
  }

  DEVICE_QUALIFIER void print() const {
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
    if (is_root()) {
      if (lead_lane) printf("root ");
    }
    if (lead_lane) printf("%u ", num_keys());
    for (size_type i = 0; i < num_keys(); ++i) {
      elem_type key = tile_.shfl(lane_elem_, get_key_lane_from_location(i));
      elem_type value = tile_.shfl(lane_elem_, get_value_lane_from_location(i));
      uint32_t keystate = tile_.shfl(keystate_, get_key_lane_from_location(i));
      if (lead_lane) printf("(%u %u %s) ", key, value, keystate == KEYSTATE_VALUE ? "$" : (keystate == KEYSTATE_LINK ? ":" : "s"));
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
    for (size_type i = 0; i < num_keys(); ++i) {
      uint32_t keystate = tile_.shfl(keystate_, get_key_lane_from_location(i));
      if (keystate == KEYSTATE_SUFFIX) {
        elem_type suffix_index = tile_.shfl(lane_elem_, get_value_lane_from_location(i));
        auto suffix = suffix_node<tile_type, allocator_type>(suffix_index, tile_, allocator_);
        suffix.load_head();
        suffix.print();
      }
    }
  }

 private:
  size_type node_index_;
  elem_type lane_elem_;
  const tile_type& tile_;
  allocator_type& allocator_;
  
  // node consists of 2*node_width elements, each mapped to a lane in the tile.
  //  [key0] [key1] ... [key13] [key14] | [metadata]
  //  [ptr0] [ptr1] ... [ptr13] [ptr14] | [ptr15 = sibling_ptr]

  // for interior nodes,
  //    ptr_i contains subtree with keys: key_(i-1) < key <= key_i.
  //    key[num_keys-1] is high key (upper bound of the subtree), used for B-link traversal.

  // for border nodes,
  //    key14 is high key, used for B-link traversal
  //    ptr14 is keystate_bits; each key 13-0 has 2 bits (total 28 bits used)
  //      (MSB)[empty:2][key_suffix_bits_per_key:14][empty:2][key_more_bits_per_key:14](LSB)
  //      Each (key, value) pair has three options:
  //        (1) key_suffix=0, key_more=0: value is the final value, key ends here
  //        (2) key_suffix=0, key_more=1: value is link to the next layer root, key continues
  //        (3) key_suffix=1, key_more=1: value is link to the suffix info, key continues (but suffix info contains the final value)
  //      We don't allow link after suffix (i.e. prefix compression) following the original Masstree.

  // metadata is 32bits.
  //    (MSB)
  //    [empty:23]
  //    [is_root:1]
  //    [is_garbage:1][has_sibling:1]
  //    [is_border:1][is_locked:1]
  //    [num_keys:4]
  //    (LSB)
  static_assert(sizeof(elem_type) == sizeof(uint32_t));
  static constexpr uint32_t metadata_lane_ = node_width - 1;
  static constexpr uint32_t sibling_ptr_lane_ = node_width * 2 - 1;
  static constexpr uint32_t border_high_key_lane_ = node_width - 2;
  static constexpr uint32_t keystate_bits_lane_ = node_width * 2 - 2;
  
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
  static constexpr uint32_t root_bit_offset_ = 8;
  static constexpr uint32_t root_bit_mask_ = 1u << root_bit_offset_;

  static constexpr uint32_t interior_max_num_keys_ = node_width - 1;
  static constexpr uint32_t border_max_num_keys_ = node_width - 2;
  static constexpr uint32_t underflow_num_keys_ = node_width / 3;
  static constexpr uint32_t half_node_width_ = node_width / 2;

  static_assert(num_keys_offset_ == 0); // this allows (metadata +/- N) equivalent to (num_keys +/- N) within range

  uint32_t metadata_;
  uint32_t keystate_;  // per-lane variable
  static  constexpr uint32_t keystate_mask_more_key_ = 0b01u;
  static  constexpr uint32_t keystate_mask_suffix_ = 0b10u;

};
