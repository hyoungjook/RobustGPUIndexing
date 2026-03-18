/*
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
#include <limits>

#include <macros.hpp>
#include <suffix_subwarp.hpp>
#include <utils.hpp>

template <typename tile_type, typename allocator_type>
struct masstree_node_subwarp {
  using elem_type = uint64_t;
  using key_type = uint32_t;
  using value_type = uint32_t;
  using size_type = uint32_t;

  static constexpr int node_width = 16;
  static constexpr key_type max_key = std::numeric_limits<key_type>::max();
  static constexpr uint32_t KEYSTATE_VALUE = 0b00u;
  static constexpr uint32_t KEYSTATE_LINK = 0b01u;
  static constexpr uint32_t KEYSTATE_SUFFIX = 0b11u;

  DEVICE_QUALIFIER masstree_node_subwarp(const tile_type& tile, allocator_type& allocator)
      : tile_(tile)
      , allocator_(allocator) {
    assert(tile_.size() == node_width);
  }
  DEVICE_QUALIFIER masstree_node_subwarp(size_type index, const tile_type& tile, allocator_type& allocator)
      : node_index_(index)
      , tile_(tile)
      , allocator_(allocator) {
    assert(tile_.size() == node_width);
  }
  DEVICE_QUALIFIER masstree_node_subwarp(size_type index,
                                         const elem_type elem,
                                         const tile_type& tile,
                                         allocator_type& allocator)
      : node_index_(index)
      , lane_elem_(elem)
      , tile_(tile)
      , allocator_(allocator) {
    assert(tile_.size() == node_width);
  }

  DEVICE_QUALIFIER void initialize_root() {
    lane_elem_ = 0;
    metadata_ = (
      (0u << num_keys_offset_) |
      (0u & lock_bit_mask_) |
      border_bit_mask_ |
      (0u & sibling_bit_mask_) |
      (0u & garbage_bit_mask_) |
      root_bit_mask_);
    write_metadata_to_registers();
  }

  template <bool atomic, bool acquire = true>
  DEVICE_QUALIFIER void load() {
    if constexpr (atomic) { tile_.sync(); }
    load_lane_elem<atomic, acquire>();
    if constexpr (atomic) { tile_.sync(); }
    read_metadata_from_registers();
  }
  template <bool atomic, bool acquire = true>
  DEVICE_QUALIFIER void load_fetchonly() {
    if constexpr (atomic) { tile_.sync(); }
    load_lane_elem<atomic, acquire>();
    if constexpr (atomic) { tile_.sync(); }
  }
  template <bool atomic, bool release = true>
  DEVICE_QUALIFIER void store() {
    if constexpr (atomic) { tile_.sync(); }
    store_lane_elem<atomic, release>();
    if constexpr (atomic) { tile_.sync(); }
  }
  template <bool atomic, bool release = true>
  DEVICE_QUALIFIER void store_unlock() {
    assert(is_locked());
    metadata_ &= ~lock_bit_mask_;
    if (tile_.thread_rank() == metadata_lane_) {
      set_lane_key(metadata_);
    }
    if constexpr (atomic) { tile_.sync(); }
    store_lane_elem<atomic, release>();
    if constexpr (atomic) { tile_.sync(); }
  }

  DEVICE_QUALIFIER void read_metadata_from_registers() {
    metadata_ = low_word(shfl_lane_elem(lane_elem_, metadata_lane_));
  }
  DEVICE_QUALIFIER void write_metadata_to_registers() {
    if (tile_.thread_rank() == metadata_lane_) {
      set_lane_key(metadata_);
    }
  }

  DEVICE_QUALIFIER elem_type get_lane_elem() const { return lane_elem_; }

  DEVICE_QUALIFIER int get_key_lane_from_location(const int location) const {
    assert(0 <= location && location < node_width);
    return location;
  }
  DEVICE_QUALIFIER int get_value_lane_from_location(const int location) const {
    assert(0 <= location && location < node_width);
    return location;
  }

  DEVICE_QUALIFIER key_type get_key_from_location(const int location) const {
    return low_word(shfl_lane_elem(lane_elem_, location));
  }
  DEVICE_QUALIFIER value_type get_value_from_location(const int location) const {
    return high_word(shfl_lane_elem(lane_elem_, location));
  }
  DEVICE_QUALIFIER uint32_t get_keystate_from_location(const int location) const {
    auto raw_keystate_bits = get_raw_keystate_bits();
    if (static_cast<uint32_t>(location) >= border_max_num_keys_) { return 0; }
    return (raw_keystate_bits >> (2 * location)) & keystate_mask_;
  }
  static DEVICE_QUALIFIER bool keystate_has_more_key(const uint32_t keystate) {
    return (keystate & keystate_mask_more_key_) != 0;
  }
  DEVICE_QUALIFIER bool is_valid_key_lane() const {
    return tile_.thread_rank() < num_keys();
  }
  DEVICE_QUALIFIER bool is_valid_value_lane() const {
    return tile_.thread_rank() < num_keys();
  }

  DEVICE_QUALIFIER size_type get_node_index() const { return node_index_; }
  DEVICE_QUALIFIER uint32_t num_keys() const { return (metadata_ & num_keys_mask_) >> num_keys_offset_; }
  DEVICE_QUALIFIER void set_num_keys(const uint32_t& value) {
    assert(value <= (num_keys_mask_ >> num_keys_offset_));
    metadata_ &= ~num_keys_mask_;
    metadata_ |= (value << num_keys_offset_);
  }
  DEVICE_QUALIFIER bool is_border() const { return static_cast<bool>(metadata_ & border_bit_mask_); }
  DEVICE_QUALIFIER bool is_root() const { return static_cast<bool>(metadata_ & root_bit_mask_); }
  DEVICE_QUALIFIER bool is_full() const {
    assert(is_border() ? (num_keys() <= border_max_num_keys_) : (num_keys() <= interior_max_num_keys_));
    return is_border() ? (num_keys() == border_max_num_keys_) : (num_keys() == interior_max_num_keys_);
  }
  DEVICE_QUALIFIER bool is_underflow() const {
    return (num_keys() <= underflow_num_keys_) && (!is_root());
  }
  DEVICE_QUALIFIER bool is_mergeable(const masstree_node_subwarp& sibling_node) const {
    return (num_keys() + sibling_node.num_keys()) < (is_border() ? border_max_num_keys_ : interior_max_num_keys_);
  }
  DEVICE_QUALIFIER bool is_garbage() const { return static_cast<bool>(metadata_ & garbage_bit_mask_); }
  DEVICE_QUALIFIER key_type get_high_key() const {
    assert(is_border() || num_keys() > 0);
    return is_border() ? get_key_from_location(border_high_key_lane_) : get_key_from_location(num_keys() - 1);
  }
  DEVICE_QUALIFIER value_type get_sibling_index() const {
    return high_word(shfl_lane_elem(lane_elem_, sibling_ptr_lane_));
  }

  DEVICE_QUALIFIER bool is_locked() const { return static_cast<bool>(metadata_ & lock_bit_mask_); }
  DEVICE_QUALIFIER bool try_lock_load() {
    auto* metadata_ptr = reinterpret_cast<uint32_t*>(allocator_.address(node_index_)) + (2 * metadata_lane_);
    uint32_t old_metadata = 0;
    if (tile_.thread_rank() == metadata_lane_) {
      cuda::atomic_ref<uint32_t, cuda::thread_scope_device> metadata_ref(*metadata_ptr);
      old_metadata = metadata_ref.fetch_or(lock_bit_mask_, cuda::memory_order_relaxed);
    }
    bool acquired = (tile_.shfl(old_metadata, metadata_lane_) & lock_bit_mask_) == 0;
    if (acquired) {
      tile_.sync();
      load_lane_elem<true, true>();
      tile_.sync();
      read_metadata_from_registers();
    }
    return acquired;
  }
  DEVICE_QUALIFIER void lock_load() {
    while (!try_lock_load()) {}
  }
  DEVICE_QUALIFIER void unlock() {
    assert(is_locked());
    auto* metadata_ptr = reinterpret_cast<uint32_t*>(allocator_.address(node_index_)) + (2 * metadata_lane_);
    if (tile_.thread_rank() == metadata_lane_) {
      cuda::atomic_ref<uint32_t, cuda::thread_scope_device> metadata_ref(*metadata_ptr);
      metadata_ref.fetch_and(~lock_bit_mask_, cuda::memory_order_release);
    }
    metadata_ &= ~lock_bit_mask_;
    if (tile_.thread_rank() == metadata_lane_) {
      set_lane_key(metadata_);
    }
  }

  DEVICE_QUALIFIER bool has_sibling() const { return static_cast<bool>(metadata_ & sibling_bit_mask_); }
  DEVICE_QUALIFIER bool traverse_required(const key_type& key) const {
    return has_sibling() && (is_garbage() || (get_high_key() < key));
  }
  DEVICE_QUALIFIER int find_next_location(const key_type& key) const {
    assert(!is_border());
    const bool key_less_equal = is_valid_key_lane() && (key <= lane_key());
    uint32_t key_less_equal_bitmap = ballot(key_less_equal);
    auto next_location = __ffs(key_less_equal_bitmap) - 1;
    assert(0 <= next_location && next_location < static_cast<int>(num_keys()));
    return next_location;
  }
  DEVICE_QUALIFIER value_type find_next(const key_type& key) const {
    return get_value_from_location(find_next_location(key));
  }
  DEVICE_QUALIFIER value_type find_next_and_sibling(const key_type& key, value_type& sibling_index, bool& sibling_at_left) const {
    auto next_location = find_next_location(key);
    sibling_at_left = decide_merge_left(next_location);
    sibling_index = get_value_from_location(next_location + (sibling_at_left ? -1 : 1));
    return get_value_from_location(next_location);
  }

  DEVICE_QUALIFIER uint32_t match_key_in_node(const key_type& key, bool more_key) const {
    assert(is_border());
    if (num_keys() == 0) { return 0; }
    auto lane_keystate = get_keystate_from_location(tile_.thread_rank());
    return ballot(is_valid_key_lane() &&
                  lane_key() == key &&
                  keystate_has_more_key(lane_keystate) == more_key);
  }
  DEVICE_QUALIFIER uint32_t match_key_in_node(const key_type& key, uint32_t keystate) const {
    assert(is_border());
    if (num_keys() == 0) { return 0; }
    auto lane_keystate = get_keystate_from_location(tile_.thread_rank());
    return ballot(is_valid_key_lane() &&
                  lane_key() == key &&
                  lane_keystate == keystate);
  }
  DEVICE_QUALIFIER bool key_is_in_node(const key_type& key, bool more_key) const {
    return match_key_in_node(key, more_key) != 0;
  }
  DEVICE_QUALIFIER bool key_is_in_node(const key_type& key, uint32_t keystate) const {
    return match_key_in_node(key, keystate) != 0;
  }
  DEVICE_QUALIFIER uint32_t match_ptr_in_node(const value_type& ptr) const {
    assert(!is_border());
    if (num_keys() == 0) { return 0; }
    return ballot(is_valid_value_lane() && lane_value() == ptr);
  }
  DEVICE_QUALIFIER bool ptr_is_in_node(const value_type& ptr) const {
    return match_ptr_in_node(ptr) != 0;
  }
  DEVICE_QUALIFIER int get_key_value_from_node(const key_type& key, value_type& value, bool more_key) const {
    assert(is_border());
    auto key_exists = match_key_in_node(key, more_key);
    if (key_exists == 0) { return -1; }
    int location = __ffs(key_exists) - 1;
    value = get_value_from_location(location);
    return static_cast<int>(get_keystate_from_location(location));
  }
  DEVICE_QUALIFIER bool get_key_value_from_node(const key_type& key, value_type& value, uint32_t keystate) const {
    assert(is_border());
    auto key_exists = match_key_in_node(key, keystate);
    if (key_exists == 0) { return false; }
    int location = __ffs(key_exists) - 1;
    value = get_value_from_location(location);
    return true;
  }

  DEVICE_QUALIFIER static bool cmp_key(const key_type& k1, bool morekey1, const key_type& k2, bool morekey2) {
    return (k1 < k2) || ((k1 == k2) && (static_cast<int>(morekey1) <= static_cast<int>(morekey2)));
  }

  template <bool use_upper_key>
  DEVICE_QUALIFIER uint32_t scan(const key_type& lower_key_slice,
                                 const bool lower_key_more,
                                 const key_type* lower_key,
                                 const size_type lower_key_length,
                                 bool& passed_lower_key,
                                 const key_type& upper_key_slice,
                                 const bool upper_key_more,
                                 const key_type* upper_key,
                                 const size_type upper_key_length,
                                 const bool ignore_upper_key,
                                 const size_type out_max_count,
                                 int& link_entry_location,
                                 value_type* out_value,
                                 key_type* out_keys,
                                 size_type* out_key_lengths,
                                 const size_type& layer,
                                 const size_type& out_key_max_length) {
    using suffix_type = suffix_node_subwarp<tile_type, allocator_type>;
    assert(is_border());
    link_entry_location = -1;
    auto lane_keystate = get_keystate_from_location(tile_.thread_rank());
    bool in_range = is_valid_key_lane() &&
        cmp_key(lower_key_slice, lower_key_more, lane_key(), keystate_has_more_key(lane_keystate)) &&
        (!use_upper_key || (ignore_upper_key || cmp_key(lane_key(), keystate_has_more_key(lane_keystate), upper_key_slice, upper_key_more)));
    uint32_t in_range_ballot = ballot(in_range);
    if (in_range_ballot == 0) { return 0; }

    int first_location = __ffs(in_range_ballot) - 1;
    if (!passed_lower_key) {
      const bool same_slice = (get_key_from_location(first_location) == lower_key_slice);
      const auto first_keystate = get_keystate_from_location(first_location);
      passed_lower_key = !(same_slice && first_keystate == KEYSTATE_LINK);
      if (same_slice && first_keystate == KEYSTATE_SUFFIX) {
        auto suffix_index = get_value_from_location(first_location);
        auto suffix = suffix_type(suffix_index, tile_, allocator_);
        suffix.load_head();
        if (suffix.strcmp(lower_key + layer + 1, lower_key_length - layer - 1) > 0) {
          if (tile_.thread_rank() == first_location) { in_range = false; }
          in_range_ballot = ballot(in_range);
          if (in_range_ballot == 0) { return 0; }
          first_location = __ffs(in_range_ballot) - 1;
        }
      }
    }

    int last_location = utils::bits::bfind(in_range_ballot) + 1;
    const bool in_range_and_link = in_range && (lane_keystate == KEYSTATE_LINK);
    const uint32_t in_range_and_link_ballot = ballot(in_range_and_link);
    if (in_range_and_link_ballot != 0) {
      link_entry_location = __ffs(in_range_and_link_ballot) - 1;
      last_location = link_entry_location;
    }
    if (use_upper_key && !ignore_upper_key && link_entry_location < 0) {
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

    int count = last_location - first_location;
    if (count > static_cast<int>(out_max_count)) {
      last_location = first_location + out_max_count;
      count = out_max_count;
      link_entry_location = -1;
    }
    if (count <= 0) { return 0; }

    in_range = (first_location <= static_cast<int>(tile_.thread_rank()) &&
                static_cast<int>(tile_.thread_rank()) < last_location);
    if (out_value && in_range) {
      auto value = lane_value();
      out_value[tile_.thread_rank() - first_location] =
        (lane_keystate != KEYSTATE_SUFFIX) ? value :
          suffix_type::fetch_value_only(value, allocator_);
    }
    if (out_key_lengths && in_range) {
      auto value = lane_value();
      out_key_lengths[tile_.thread_rank() - first_location] =
        layer + 1 + ((lane_keystate != KEYSTATE_SUFFIX) ? 0 :
          suffix_type::fetch_length_only(value, allocator_));
    }
    if (out_keys) {
      if (in_range) {
        out_keys[(tile_.thread_rank() - first_location) * out_key_max_length + layer] = lane_key();
      }
      bool to_flush = in_range && lane_keystate == KEYSTATE_SUFFIX;
      uint32_t flush_queue = ballot(to_flush);
      while (flush_queue) {
        auto cur_location = __ffs(flush_queue) - 1;
        auto suffix_index = get_value_from_location(cur_location);
        auto suffix = suffix_type(suffix_index, tile_, allocator_);
        suffix.load_head();
        suffix.flush(out_keys + ((cur_location - first_location) * out_key_max_length + layer + 1));
        if (tile_.thread_rank() == cur_location) { to_flush = false; }
        flush_queue = ballot(to_flush);
      }
    }
    return count;
  }

  DEVICE_QUALIFIER int get_split_left_width() const {
    bool shift_required = is_border() && (get_keystate_from_location(half_node_width_ - 1) == KEYSTATE_VALUE);
    return shift_required ? (half_node_width_ - 1) : half_node_width_;
  }

  DEVICE_QUALIFIER void do_split(masstree_node_subwarp& right_sibling_node, const int left_width) {
    assert(is_full());
    auto old_num_keys = num_keys();
    right_sibling_node.lane_elem_ = shfl_down_lane_elem(lane_elem_, left_width);

    if (tile_.thread_rank() == metadata_lane_) {
      right_sibling_node.set_lane_value(lane_value());
      set_lane_value(right_sibling_node.get_node_index());
    }
    if (is_border()) {
      auto old_high_key = get_high_key();
      auto old_bits = get_raw_keystate_bits();
      auto left_mask = bit_prefix_mask(2 * left_width);
      if (tile_.thread_rank() == border_high_key_lane_) {
        right_sibling_node.set_lane_key(old_high_key);
        right_sibling_node.set_lane_value(old_bits >> (2 * left_width));
        set_lane_key(get_key_from_location(left_width - 1));
        set_lane_value(old_bits & left_mask);
      }
    }

    right_sibling_node.metadata_ = (metadata_ & ~(num_keys_mask_ | sibling_bit_mask_));
    right_sibling_node.metadata_ |= ((old_num_keys - left_width) << num_keys_offset_);
    if (has_sibling()) { right_sibling_node.metadata_ |= sibling_bit_mask_; }
    set_num_keys(left_width);
    metadata_ |= sibling_bit_mask_;
    write_metadata_to_registers();
    right_sibling_node.write_metadata_to_registers();
  }

  DEVICE_QUALIFIER void split(masstree_node_subwarp& right_sibling_node,
                              const value_type parent_index,
                              masstree_node_subwarp& parent_node) {
    auto left_width = get_split_left_width();
    do_split(right_sibling_node, left_width);
    auto pivot_key = get_key_from_location(left_width - 1);
    parent_node.insert(pivot_key, right_sibling_node.get_node_index(), 0);
  }

  DEVICE_QUALIFIER void split_as_root(masstree_node_subwarp& left_child_node,
                                      masstree_node_subwarp& right_child_node) {
    assert(is_root());
    left_child_node.lane_elem_ = lane_elem_;
    left_child_node.metadata_ = metadata_ & ~root_bit_mask_;

    metadata_ &= ~border_bit_mask_;
    set_num_keys(2);
    lane_elem_ = 0;
    auto left_width = left_child_node.get_split_left_width();
    auto pivot_key = left_child_node.get_key_from_location(left_width - 1);
    if (tile_.thread_rank() == 0) {
      lane_elem_ = pack_pair(pivot_key, left_child_node.node_index_);
    }
    else if (tile_.thread_rank() == 1) {
      lane_elem_ = pack_pair(max_key, right_child_node.node_index_);
    }
    write_metadata_to_registers();
    left_child_node.do_split(right_child_node, left_width);
  }

  DEVICE_QUALIFIER void do_insert(const key_type& key, const value_type& value, uint32_t keystate, const size_type& key_location) {
    assert(!is_full());
    if (is_border()) {
      do_insert_border(key, value, keystate, key_location);
    }
    else {
      do_insert_interior(key, value, key_location);
    }
    write_metadata_to_registers();
  }

  DEVICE_QUALIFIER void insert(const key_type& key,
                               const value_type& value,
                               uint32_t keystate) {
    assert(!is_full());
    const bool key_is_larger = is_valid_key_lane() && (key > lane_key());
    uint32_t key_is_larger_bitmap = ballot(key_is_larger);
    auto key_location = utils::bits::bfind(key_is_larger_bitmap) + 1;
    assert(key_location <= static_cast<int>(num_keys()));
    if (is_border() && (keystate != KEYSTATE_VALUE) &&
        (key_location < static_cast<int>(num_keys())) &&
        (get_key_from_location(key_location) == key) &&
        (get_keystate_from_location(key_location) == KEYSTATE_VALUE)) {
      key_location++;
    }
    do_insert(key, value, keystate, key_location);
  }

  DEVICE_QUALIFIER void update(const key_type& key,
                               const value_type& value,
                               uint32_t old_keystate,
                               uint32_t new_keystate) {
    assert(is_border());
    uint32_t key_exists = match_key_in_node(key, old_keystate);
    assert(key_exists != 0);
    uint32_t key_location = __ffs(key_exists) - 1;
    set_value_at_location(key_location, value);
    auto bits = get_raw_keystate_bits();
    if (tile_.thread_rank() == keystate_bits_lane_) {
      auto key_location_x2 = 2 * key_location;
      bits &= ~(keystate_mask_ << key_location_x2);
      bits |= ((new_keystate & keystate_mask_) << key_location_x2);
      set_lane_value(bits);
    }
    write_metadata_to_registers();
  }

  DEVICE_QUALIFIER void make_garbage_node(bool has_sibling, value_type sibling_index = 0) {
    metadata_ &= ~(num_keys_mask_ | sibling_bit_mask_);
    metadata_ |= garbage_bit_mask_;
    if (has_sibling) { metadata_ |= sibling_bit_mask_; }
    if (has_sibling && tile_.thread_rank() == sibling_ptr_lane_) {
      set_lane_value(sibling_index);
    }
    write_metadata_to_registers();
  }

  DEVICE_QUALIFIER bool decide_merge_left(int location) const {
    return (location > 0);
  }
  DEVICE_QUALIFIER int check_valid_merge_siblings(const size_type& child_index,
                                                  const size_type& sibling_index) {
    static_assert(underflow_num_keys_ >= 2);
    assert(num_keys() >= 2);
    auto child_exist = match_ptr_in_node(child_index);
    auto sibling_exist = match_ptr_in_node(sibling_index);
    if (child_exist == 0 || sibling_exist == 0) { return -1; }
    int location_diff = __ffs(child_exist) - __ffs(sibling_exist);
    if (location_diff == 1) { return 1; }
    if (location_diff == -1) { return 0; }
    return -1;
  }

  DEVICE_QUALIFIER void merge(masstree_node_subwarp& right_sibling_node,
                              masstree_node_subwarp& parent_node) {
    auto old_num_keys = num_keys();
    auto right_num_keys = right_sibling_node.num_keys();
    auto new_num_keys = old_num_keys + right_num_keys;
    auto shifted_elem = shfl_up_lane_elem(right_sibling_node.lane_elem_, old_num_keys);
    if (old_num_keys <= tile_.thread_rank() && tile_.thread_rank() < new_num_keys) {
      lane_elem_ = shifted_elem;
    }
    if (is_border()) {
      auto old_bits = get_raw_keystate_bits();
      auto right_bits = right_sibling_node.get_raw_keystate_bits();
      auto right_high_key = right_sibling_node.get_high_key();
      if (tile_.thread_rank() == border_high_key_lane_) {
        set_lane_key(right_high_key);
        set_lane_value((old_bits & bit_prefix_mask(2 * old_num_keys)) | (right_bits << (2 * old_num_keys)));
      }
    }
    auto right_sibling_index = right_sibling_node.get_sibling_index();
    if (tile_.thread_rank() == metadata_lane_) {
      set_lane_value(right_sibling_index);
    }
    set_num_keys(new_num_keys);
    metadata_ = (metadata_ & ~sibling_bit_mask_) ^ (right_sibling_node.metadata_ & sibling_bit_mask_);
    write_metadata_to_registers();

    int right_location_in_parent = __ffs(parent_node.match_ptr_in_node(right_sibling_node.node_index_)) - 1;
    parent_node.set_key_at_location(right_location_in_parent - 1, get_high_key());
    parent_node.do_erase(right_location_in_parent, right_location_in_parent);
    right_sibling_node.make_garbage_node(true, node_index_);
  }

  DEVICE_QUALIFIER void merge_to_root(const value_type& parent_index,
                                      masstree_node_subwarp& left_child_node,
                                      masstree_node_subwarp& right_child_node) {
    assert(is_root() && num_keys() == 2);
    auto left_num_keys = left_child_node.num_keys();
    auto right_num_keys = right_child_node.num_keys();
    auto new_num_keys = left_num_keys + right_num_keys;

    lane_elem_ = left_child_node.lane_elem_;
    metadata_ = (metadata_ & (lock_bit_mask_ | root_bit_mask_)) |
                (left_child_node.metadata_ & border_bit_mask_);
    set_num_keys(new_num_keys);

    auto right_elem = shfl_up_lane_elem(right_child_node.lane_elem_, left_num_keys);
    if (left_num_keys <= tile_.thread_rank() && tile_.thread_rank() < new_num_keys) {
      lane_elem_ = right_elem;
    }
    if (is_border()) {
      auto left_bits = left_child_node.get_raw_keystate_bits();
      auto right_bits = right_child_node.get_raw_keystate_bits();
      auto right_high_key = right_child_node.get_high_key();
      if (tile_.thread_rank() == border_high_key_lane_) {
        set_lane_key(right_high_key);
        set_lane_value((left_bits & bit_prefix_mask(2 * left_num_keys)) | (right_bits << (2 * left_num_keys)));
      }
    }
    if (right_child_node.has_sibling()) { metadata_ |= sibling_bit_mask_; }
    auto right_sibling_index = right_child_node.get_sibling_index();
    if (tile_.thread_rank() == metadata_lane_) {
      set_lane_value(right_sibling_index);
    }
    write_metadata_to_registers();

    left_child_node.make_garbage_node(true, parent_index);
    right_child_node.make_garbage_node(true, parent_index);
  }

  DEVICE_QUALIFIER void borrow_left(masstree_node_subwarp& sibling_node,
                                    masstree_node_subwarp& parent_node) {
    auto old_num_keys = num_keys();
    auto sibling_old_num_keys = sibling_node.num_keys();
    uint32_t num_shift = (sibling_old_num_keys - old_num_keys) / 2;
    if (is_border() &&
        (sibling_node.get_keystate_from_location(sibling_old_num_keys - num_shift - 1) == KEYSTATE_VALUE)) {
      num_shift++;
    }
    auto new_num_keys = old_num_keys + num_shift;
    auto shifted_elem = shfl_up_lane_elem(lane_elem_, num_shift);
    auto borrowed_elem = shfl_down_lane_elem(sibling_node.lane_elem_, sibling_old_num_keys - num_shift);
    metadata_ += num_shift;
    if (tile_.thread_rank() < num_shift) {
      lane_elem_ = borrowed_elem;
    }
    else if (tile_.thread_rank() < new_num_keys) {
      lane_elem_ = shifted_elem;
    }
    auto old_bits = get_raw_keystate_bits();
    auto sibling_bits = sibling_node.get_raw_keystate_bits();
    if (is_border() && tile_.thread_rank() == border_high_key_lane_) {
      auto tail_bits = (sibling_bits >> (2 * (sibling_old_num_keys - num_shift))) & bit_prefix_mask(2 * num_shift);
      set_lane_value((old_bits << (2 * num_shift)) | tail_bits);
    }
    write_metadata_to_registers();

    sibling_node.metadata_ -= num_shift;
    auto pivot_key = sibling_node.get_key_from_location(sibling_node.num_keys() - 1);
    auto updated_sibling_bits = sibling_node.get_raw_keystate_bits();
    if (sibling_node.is_border() && tile_.thread_rank() == border_high_key_lane_) {
      set_unused();
      sibling_node.set_lane_key(pivot_key);
      sibling_node.set_lane_value(updated_sibling_bits & bit_prefix_mask(2 * sibling_node.num_keys()));
    }
    sibling_node.write_metadata_to_registers();

    int left_sibling_location_in_parent = __ffs(parent_node.match_ptr_in_node(sibling_node.node_index_)) - 1;
    parent_node.set_key_at_location(left_sibling_location_in_parent, pivot_key);
  }

  DEVICE_QUALIFIER void borrow_right(masstree_node_subwarp& sibling_node,
                                     masstree_node_subwarp& parent_node,
                                     masstree_node_subwarp& new_sibling_node) {
    auto old_num_keys = num_keys();
    auto sibling_old_num_keys = sibling_node.num_keys();
    uint32_t num_shift = (sibling_old_num_keys - old_num_keys) / 2;
    if (is_border() && (sibling_node.get_keystate_from_location(num_shift - 1) == KEYSTATE_VALUE)) {
      num_shift--;
    }

    auto shifted_elem = shfl_up_lane_elem(sibling_node.lane_elem_, old_num_keys);
    if (old_num_keys <= tile_.thread_rank() && tile_.thread_rank() < old_num_keys + num_shift) {
      lane_elem_ = shifted_elem;
    }
    auto old_bits = get_raw_keystate_bits();
    auto sibling_bits = sibling_node.get_raw_keystate_bits();
    if (is_border() && tile_.thread_rank() == border_high_key_lane_) {
      set_lane_value((old_bits & bit_prefix_mask(2 * old_num_keys)) | (sibling_bits << (2 * old_num_keys)));
    }
    metadata_ += num_shift;
    auto pivot_key = get_key_from_location(num_keys() - 1);
    if (is_border() && tile_.thread_rank() == border_high_key_lane_) {
      set_lane_key(pivot_key);
    }
    if (tile_.thread_rank() == sibling_ptr_lane_) {
      set_lane_value(new_sibling_node.node_index_);
    }
    write_metadata_to_registers();

    new_sibling_node.lane_elem_ = shfl_down_lane_elem(sibling_node.lane_elem_, num_shift);
    new_sibling_node.metadata_ = sibling_node.metadata_ - num_shift;
    auto sibling_high_key = sibling_node.get_high_key();
    auto sibling_raw_bits = sibling_node.get_raw_keystate_bits();
    if (new_sibling_node.is_border() && tile_.thread_rank() == border_high_key_lane_) {
      new_sibling_node.set_lane_key(sibling_high_key);
      new_sibling_node.set_lane_value(sibling_raw_bits >> (2 * num_shift));
    }
    auto sibling_sibling_index = sibling_node.get_sibling_index();
    if (tile_.thread_rank() == metadata_lane_) {
      new_sibling_node.set_lane_value(sibling_sibling_index);
    }
    new_sibling_node.write_metadata_to_registers();

    sibling_node.make_garbage_node(true, node_index_);
    int right_sibling_location_in_parent = __ffs(parent_node.match_ptr_in_node(sibling_node.node_index_)) - 1;
    parent_node.set_key_at_location(right_sibling_location_in_parent - 1, pivot_key);
    parent_node.set_value_at_location(right_sibling_location_in_parent, new_sibling_node.node_index_);
  }

  DEVICE_QUALIFIER void do_erase(int key_lane, int value_lane) {
    assert(num_keys() > 0);
    auto location = key_lane;
    metadata_--;
    auto down_elem = shfl_down_lane_elem(lane_elem_, 1);
    if (tile_.thread_rank() >= static_cast<uint32_t>(location) &&
        tile_.thread_rank() < num_keys()) {
      lane_elem_ = down_elem;
    }
    if (is_border() && tile_.thread_rank() == keystate_bits_lane_) {
      auto old_bits = get_raw_keystate_bits();
      uint32_t low_mask = bit_prefix_mask(2 * location);
      auto low_bits = old_bits & low_mask;
      auto shifted = (old_bits >> 2) & ~low_mask;
      set_lane_value(low_bits | shifted);
    }
    write_metadata_to_registers();
  }

  DEVICE_QUALIFIER bool erase(const key_type& key, uint32_t keystate) {
    assert(is_border());
    uint32_t key_exists = match_key_in_node(key, keystate);
    if (key_exists == 0) { return false; }
    uint32_t key_location = __ffs(key_exists) - 1;
    do_erase(key_location, key_location);
    return true;
  }

  DEVICE_QUALIFIER masstree_node_subwarp& operator=(const masstree_node_subwarp& other) {
    node_index_ = other.node_index_;
    lane_elem_ = other.lane_elem_;
    metadata_ = other.metadata_;
    return *this;
  }

  DEVICE_QUALIFIER void print() const {
    bool lead_lane = (tile_.thread_rank() == 0);
    if (lead_lane) { printf("node[%u]: {", node_index_); }
    if (num_keys() > interior_max_num_keys_) {
      if (lead_lane) { printf("num_keys too large: skip}\n"); }
      return;
    }
    if (is_garbage()) {
      if (lead_lane) { printf("garbage}\n"); }
      return;
    }
    if (num_keys() == 0) {
      if (lead_lane) { printf("empty}\n"); }
      return;
    }
    if (is_root() && lead_lane) { printf("root "); }
    if (lead_lane) { printf("%u ", num_keys()); }
    for (size_type i = 0; i < num_keys(); ++i) {
      auto key = get_key_from_location(i);
      auto value = get_value_from_location(i);
      auto keystate = get_keystate_from_location(i);
      if (lead_lane) {
        printf("(%u %u %s) ", key, value,
               keystate == KEYSTATE_VALUE ? "$" : (keystate == KEYSTATE_LINK ? ":" : "s"));
      }
    }
    auto high_key = get_high_key();
    auto sibling_index = get_sibling_index();
    if (lead_lane) {
      printf("%s %s ", is_locked() ? "locked" : "unlocked", is_border() ? "border" : "interior");
      if (has_sibling()) { printf("next(%u %u)", high_key, sibling_index); }
      else { printf("nullnext"); }
      printf("}\n");
    }
  }

 private:
  size_type node_index_;
  elem_type lane_elem_ = 0;
  const tile_type& tile_;
  allocator_type& allocator_;

  static_assert(sizeof(elem_type) == sizeof(uint64_t));
  static constexpr uint32_t metadata_lane_ = node_width - 1;
  static constexpr uint32_t sibling_ptr_lane_ = node_width - 1;
  static constexpr uint32_t border_high_key_lane_ = node_width - 2;
  static constexpr uint32_t keystate_bits_lane_ = node_width - 2;

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

  static constexpr uint32_t keystate_mask_more_key_ = 0b01u;
  static constexpr uint32_t keystate_mask_suffix_ = 0b10u;
  static constexpr uint32_t keystate_mask_ = 0b11u;

  uint32_t metadata_ = 0;

  static DEVICE_QUALIFIER elem_type pack_pair(key_type key, value_type value) {
    return static_cast<elem_type>(key) | (static_cast<elem_type>(value) << 32);
  }
  static DEVICE_QUALIFIER key_type low_word(elem_type lane_elem) {
    return static_cast<key_type>(lane_elem);
  }
  static DEVICE_QUALIFIER value_type high_word(elem_type lane_elem) {
    return static_cast<value_type>(lane_elem >> 32);
  }
  DEVICE_QUALIFIER key_type lane_key() const { return low_word(lane_elem_); }
  DEVICE_QUALIFIER value_type lane_value() const { return high_word(lane_elem_); }
  DEVICE_QUALIFIER void set_lane_key(key_type key) {
    lane_elem_ = (lane_elem_ & 0xffffffff00000000ULL) | static_cast<elem_type>(key);
  }
  DEVICE_QUALIFIER void set_lane_value(value_type value) {
    lane_elem_ = (lane_elem_ & 0x00000000ffffffffULL) | (static_cast<elem_type>(value) << 32);
  }
  DEVICE_QUALIFIER void set_lane_pair(key_type key, value_type value) {
    lane_elem_ = pack_pair(key, value);
  }
  DEVICE_QUALIFIER void set_key_at_location(int location, key_type key) {
    if (tile_.thread_rank() == static_cast<uint32_t>(location)) {
      set_lane_key(key);
    }
  }
  DEVICE_QUALIFIER void set_value_at_location(int location, value_type value) {
    if (tile_.thread_rank() == static_cast<uint32_t>(location)) {
      set_lane_value(value);
    }
  }
  DEVICE_QUALIFIER uint32_t get_raw_keystate_bits() const {
    return high_word(shfl_lane_elem(lane_elem_, keystate_bits_lane_));
  }
  static DEVICE_QUALIFIER uint32_t bit_prefix_mask(uint32_t bits) {
    if (bits == 0) { return 0; }
    return (static_cast<uint32_t>(1) << bits) - 1;
  }
  DEVICE_QUALIFIER void set_unused() {}

  DEVICE_QUALIFIER void do_insert_border(const key_type& key,
                                         const value_type& value,
                                         uint32_t keystate,
                                         const size_type key_location) {
    auto old_bits = get_raw_keystate_bits();
    metadata_++;
    auto up_elem = shfl_up_lane_elem(lane_elem_, 1);
    if (tile_.thread_rank() < num_keys()) {
      if (tile_.thread_rank() == key_location) {
        set_lane_pair(key, value);
      }
      else if (tile_.thread_rank() > key_location) {
        lane_elem_ = up_elem;
      }
    }
    if (tile_.thread_rank() == keystate_bits_lane_) {
      auto low_mask = bit_prefix_mask(2 * key_location);
      auto low_bits = old_bits & low_mask;
      auto shifted = (old_bits & ~low_mask) << 2;
      set_lane_value(low_bits | shifted | ((keystate & keystate_mask_) << (2 * key_location)));
    }
  }

  DEVICE_QUALIFIER void do_insert_interior(const key_type& key,
                                           const value_type& value,
                                           const size_type key_location) {
    auto old_num_keys = num_keys();
    assert(key_location < old_num_keys);
    auto old_key = get_key_from_location(key_location);
    metadata_++;
    auto up_elem = shfl_up_lane_elem(lane_elem_, 1);
    if (tile_.thread_rank() < num_keys()) {
      if (tile_.thread_rank() == key_location) {
        set_lane_key(key);
      }
      else if (tile_.thread_rank() == key_location + 1) {
        set_lane_pair(old_key, value);
      }
      else if (tile_.thread_rank() > key_location + 1) {
        lane_elem_ = up_elem;
      }
    }
  }

  template <bool atomic, bool acquire>
  DEVICE_QUALIFIER void load_lane_elem() {
    auto* node_words = reinterpret_cast<uint32_t*>(allocator_.address(node_index_)) + (2 * tile_.thread_rank());
    auto lo = utils::memory::load<uint32_t, atomic, acquire>(node_words);
    auto hi = utils::memory::load<uint32_t, atomic, acquire>(node_words + 1);
    lane_elem_ = pack_pair(lo, hi);
  }

  template <bool atomic, bool release>
  DEVICE_QUALIFIER void store_lane_elem() {
    auto* node_words = reinterpret_cast<uint32_t*>(allocator_.address(node_index_)) + (2 * tile_.thread_rank());
    utils::memory::store<uint32_t, atomic, release>(node_words, low_word(lane_elem_));
    utils::memory::store<uint32_t, atomic, release>(node_words + 1, high_word(lane_elem_));
  }

  DEVICE_QUALIFIER elem_type shfl_lane_elem(elem_type value, uint32_t src_lane) const {
    return pack_pair(tile_.shfl(low_word(value), src_lane), tile_.shfl(high_word(value), src_lane));
  }

  DEVICE_QUALIFIER elem_type shfl_up_lane_elem(elem_type value, uint32_t delta) const {
    return pack_pair(tile_.shfl_up(low_word(value), delta), tile_.shfl_up(high_word(value), delta));
  }

  DEVICE_QUALIFIER elem_type shfl_down_lane_elem(elem_type value, uint32_t delta) const {
    return pack_pair(tile_.shfl_down(low_word(value), delta), tile_.shfl_down(high_word(value), delta));
  }

  DEVICE_QUALIFIER uint32_t ballot(bool predicate) const {
    return tile_.ballot(predicate);
  }
};
