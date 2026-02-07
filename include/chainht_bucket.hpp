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
#include <suffix.hpp>

template <typename tile_type>
struct chainht_bucket {
  using elem_type = uint32_t;
  using key_type = elem_type;
  using value_type = elem_type;
  using size_type = uint32_t;
  static constexpr int node_width = 16;
  DEVICE_QUALIFIER chainht_bucket(const tile_type& tile): tile_(tile) {}
  DEVICE_QUALIFIER chainht_bucket(elem_type* ptr, const tile_type& tile)
      : bucket_ptr_(ptr), tile_(tile)
  {
    assert(tile_.size() == 2 * node_width);
  }
  DEVICE_QUALIFIER void initialize_empty() {
    lane_elem_ = 0;
    metadata_ = (
      (0u << num_keys_offset_) |  // num_keys = 0;
      (0u & lock_bit_mask_) |     // is_locked = false;
      (0u & next_bit_mask_)       // has_next = false;
    );
    write_metadata_to_registers();
  }

  template <cuda_memory_order order>
  DEVICE_QUALIFIER void load() {
    lane_elem_ = cuda_memory<elem_type, order>::load(bucket_ptr_ + tile_.thread_rank());
    read_metadata_from_registers();
  }
  template <cuda_memory_order order>
  DEVICE_QUALIFIER void store() {
    cuda_memory<elem_type, order>::store(bucket_ptr_ + tile_.thread_rank(), lane_elem_);
  }

  DEVICE_QUALIFIER void read_metadata_from_registers() {
    metadata_ = tile_.shfl(lane_elem_, metadata_lane_);
  }
  DEVICE_QUALIFIER void write_metadata_to_registers() {
    if (tile_.thread_rank() == metadata_lane_) {
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
  DEVICE_QUALIFIER bool is_valid_key_lane() const {
    return tile_.thread_rank() < num_keys();
  }
  DEVICE_QUALIFIER bool is_valid_value_lane() const {
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
  DEVICE_QUALIFIER bool is_full() const {
    return (num_keys() == max_num_keys_);
  }
  DEVICE_QUALIFIER bool is_this_lane_suffix() const {
    assert(is_valid_key_lane());
    return (metadata_ >> (suffix_bits_offset_ + tile_.thread_rank())) & 1u;
  }
  DEVICE_QUALIFIER void set_suffix_of_location(int location, bool more_key) {
    auto mask = (1u << (suffix_bits_offset_ + location));
    if (more_key) { metadata_ |= mask; }
    else { metadata_ &= ~mask; }
  }
  DEVICE_QUALIFIER bool has_next() const {
    return static_cast<bool>(metadata_ & next_bit_mask_);
  }
  DEVICE_QUALIFIER void set_has_next() {
    metadata_ |= next_bit_mask_;
    write_metadata_to_registers();
  }
  DEVICE_QUALIFIER value_type get_next_index() const {
    return tile_.shfl(lane_elem_, next_ptr_lane_);
  }
  DEVICE_QUALIFIER void set_next_index(const value_type& index) {
    if (tile_.thread_rank() == next_ptr_lane_) { lane_elem_ = index; }
  }

  DEVICE_QUALIFIER bool is_locked() const {
    return static_cast<bool>(metadata_ & lock_bit_mask_);
  }
  DEVICE_QUALIFIER bool try_lock() {
    elem_type old;
    if (tile_.thread_rank() == metadata_lane_) {
      old = atomicOr(reinterpret_cast<elem_type*>(&bucket_ptr_[metadata_lane_]),
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
      atomicAnd(reinterpret_cast<elem_type*>(&bucket_ptr_[metadata_lane_]),
                static_cast<elem_type>(~lock_bit_mask_));
    }
    // the node object can be used after this, so update regsiters
    metadata_ &= ~lock_bit_mask_;
    if (tile_.thread_rank() == metadata_lane_) {
      lane_elem_ &= ~lock_bit_mask_;
    }
  }
  static DEVICE_QUALIFIER void unlock(elem_type* bucket_ptr, const tile_type& tile) {
    // unlock, only using the pointer, not load the entire register
    __threadfence();
    if (tile.thread_rank() == metadata_lane_) {
      atomicAnd(reinterpret_cast<elem_type*>(&bucket_ptr[metadata_lane_]),
                static_cast<elem_type>(~lock_bit_mask_));
    }
  }

  DEVICE_QUALIFIER uint32_t match_key_in_node(const key_type& key, bool more_key) const {
    return tile_.ballot(is_valid_key_lane() &&
                        lane_elem_ == key &&
                        is_this_lane_suffix() == more_key);
  }

  DEVICE_QUALIFIER void insert(const key_type& key, const value_type& value, bool more_key) {
    assert(!is_full());
    auto location = num_keys();
    if (tile_.thread_rank() == get_key_lane_from_location(location)) {
      lane_elem_ = key;
    }
    if (tile_.thread_rank() == get_value_lane_from_location(location)) {
      lane_elem_ = value;
    }
    set_suffix_of_location(location, more_key);
    metadata_++;    // equiv. to num_keys++
    write_metadata_to_registers();
  }

  DEVICE_QUALIFIER void update(int location, const value_type& value) {
    assert(location < num_keys());
    if (tile_.thread_rank() == get_value_lane_from_location(location)) {
      lane_elem_ = value;
    }
  }

  DEVICE_QUALIFIER void erase(int location) {
    assert(location < num_keys());
    metadata_--;    // equiv. to num_keys--
    bool suffix_bit = static_cast<bool>((metadata_ >> (suffix_bits_offset_ + tile_.thread_rank())) & 1u);
    auto down_elem = tile_.shfl_down(lane_elem_, 1);
    auto down_suffix_bit = tile_.shfl_down(suffix_bit, 1);
    if (is_valid_key_lane()) {
      if (tile_.thread_rank() >= get_key_lane_from_location(location)) {
        lane_elem_ = down_elem;
        suffix_bit = down_suffix_bit;
      }
    }
    else if (is_valid_value_lane()) {
      if (tile_.thread_rank() >= get_value_lane_from_location(location)) {
        lane_elem_ = down_elem;
      }
    }
    auto new_suffix_bits = tile_.ballot(suffix_bit);
    metadata_ &= ~suffix_bits_mask_;
    metadata_ |= ((new_suffix_bits << suffix_bits_offset_) & suffix_bits_mask_);
    write_metadata_to_registers();
  }

  DEVICE_QUALIFIER chainht_bucket<tile_type>& operator=(
      const chainht_bucket<tile_type>& other) {
    bucket_ptr_ = other.bucket_ptr_;
    lane_elem_ = other.lane_elem_;
    metadata_ = other.metadata_;
    return *this;
  }

  template <typename allocator_type>
  DEVICE_QUALIFIER void print(allocator_type& allocator) const {
    bool lead_lane = (tile_.thread_rank() == 0);
    if (lead_lane) printf("bucket[%p]: {", bucket_ptr_);
    if (num_keys() > max_num_keys_) {
      if (lead_lane) printf("num_keys too large: skip}\n");
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
      bool suffix_bit = static_cast<bool>((metadata_ >> (suffix_bits_offset_ + i)) & 1u);
      if (lead_lane) printf("(%u %u %s) ", key, value, suffix_bit ? "s" : "$");
    }
    if (lead_lane) printf("%s ", is_locked() ? "locked" : "unlocked");
    elem_type next_index = get_next_index();
    if (has_next()) {
      if (lead_lane) printf("next(%u)", next_index);
    }
    else {
      if (lead_lane) printf("nullnext");
    }
    if (lead_lane) printf("}\n");
    for (size_type i = 0; i < num_keys(); ++i) {
      bool suffix_bit = static_cast<bool>((metadata_ >> (suffix_bits_offset_ + i)) & 1u);
      if (suffix_bit) {
        elem_type suffix_index = tile_.shfl(lane_elem_, get_value_lane_from_location(i));
        auto suffix = suffix_node<tile_type, allocator_type>(
            reinterpret_cast<elem_type*>(allocator.address(suffix_index)), suffix_index, tile_, allocator);
        suffix.template load_head<cuda_memory_order::weak>();
        suffix.print();
      }
    }
  }

 private:
  elem_type* bucket_ptr_;
  elem_type lane_elem_;
  const tile_type tile_;

  // node consists of 2*node_width elements, each mapped to a lane in the tile.
  //  [key0] [key1] ... [key13] [key14] | [metadata]
  //  [val0] [val1] ... [val13] [val14] | [next]

  // metadata is 32bits.
  //    (MSB)
  //    [empty:11]
  //    [key_suffix_bits_per_key:15]
  //    [has_next:1][is_locked:1]
  //    [num_keys:4]
  //    (LSB)
  static_assert(sizeof(elem_type) == sizeof(uint32_t));
  static constexpr uint32_t metadata_lane_ = node_width - 1;
  static constexpr uint32_t next_ptr_lane_ = node_width * 2 - 1;
  static constexpr uint32_t num_keys_offset_ = 0;
  static constexpr uint32_t num_keys_bits_ = 4;
  static constexpr uint32_t num_keys_mask_ = ((1u << num_keys_bits_) - 1) << num_keys_offset_;
  static constexpr uint32_t lock_bit_offset_ = 4;
  static constexpr uint32_t lock_bit_mask_ = 1u << lock_bit_offset_;
  static constexpr uint32_t next_bit_offset_ = 5;
  static constexpr uint32_t next_bit_mask_ = 1u << next_bit_offset_;
  static constexpr uint32_t suffix_bits_offset_ = 6;
  static constexpr uint32_t suffix_bits_bits_ = 15;
  static constexpr uint32_t suffix_bits_mask_ = ((1u << suffix_bits_bits_) - 1) << suffix_bits_offset_;
  static constexpr uint32_t max_num_keys_ = node_width - 1;
  static_assert(num_keys_offset_ == 0); // this allows (metadata +/- N) equivalent to (num_keys +/- N) within range

  uint32_t metadata_;
};
