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
#include <suffix_node_subwarp.hpp>

template <typename tile_type, typename allocator_type>
struct hashtable_node_subwarp {
  using key_type = uint32_t;
  using value_type = uint32_t;
  using size_type = uint32_t;
  struct __align__(8) elem_type {
    key_type key;
    value_type value;
  };
  using elem_unsigned_type = uint64_t;
  static constexpr int node_width = 16;
  static constexpr int capacity = node_width - 1;
  static_assert(tile_type::size() == node_width);
  DEVICE_QUALIFIER hashtable_node_subwarp(const tile_type& tile, allocator_type& allocator)
      : tile_(tile), allocator_(allocator) {}
  DEVICE_QUALIFIER hashtable_node_subwarp(size_type index, const tile_type& tile, allocator_type& allocator)
      : node_index_(index), tile_(tile), allocator_(allocator) {}
  DEVICE_QUALIFIER void initialize_empty(bool is_head, size_type local_depth = 0, bool is_locked = false) {
    lane_elem_ = {0, 0};
    metadata_ = (
      (0u << num_keys_offset_) |  // num_keys = 0;
      (0u & next_bit_mask_) |     // has_next = false;
      (0u & garbage_bit_mask_)    // is_garbage = false;
    );
    if (is_head) { metadata_ |= head_bit_mask_; }
    if (is_locked) { metadata_ |= lock_bit_mask_; }
    metadata_ |= (local_depth << local_depth_bits_offset_);
    write_metadata_to_registers();
  }

  template <bool atomic, bool acquire = true>
  DEVICE_QUALIFIER void load_from_array(key_type* table_ptr) {
    auto node_ptr = reinterpret_cast<elem_unsigned_type*>(table_ptr + (static_cast<std::size_t>(2 * node_width) * node_index_));
    do_load<atomic, acquire>(node_ptr);
  }
  template <bool atomic, bool acquire = true>
  DEVICE_QUALIFIER void load_from_allocator() {
    auto node_ptr = reinterpret_cast<elem_unsigned_type*>(allocator_.address(node_index_));
    do_load<atomic, acquire>(node_ptr);
  }
  template <bool atomic, bool acquire>
  DEVICE_QUALIFIER void do_load(elem_unsigned_type* node_ptr) {
    if constexpr (atomic) { tile_.sync(); }
    auto elem = utils::memory::load<elem_unsigned_type, atomic, acquire>(node_ptr + tile_.thread_rank());
    lane_elem_ = *reinterpret_cast<elem_type*>(&elem);
    if constexpr (atomic) { tile_.sync(); }
    read_metadata_from_registers();
  }
  template <bool atomic, bool release = true>
  DEVICE_QUALIFIER void store_to_array(key_type* table_ptr) {
    auto node_ptr = table_ptr + (static_cast<std::size_t>(2 * node_width) * node_index_);
    do_store<atomic, release>(reinterpret_cast<elem_unsigned_type*>(node_ptr));
  }
  template <bool atomic, bool release = true>
  DEVICE_QUALIFIER void store_to_allocator() {
    auto node_ptr = reinterpret_cast<elem_unsigned_type*>(allocator_.address(node_index_));
    do_store<atomic, release>(node_ptr);
  }
  template <bool atomic, bool release = true>
  DEVICE_QUALIFIER void store_head_to_array_aux_to_allocator(key_type* table_ptr) {
    auto node_ptr = is_head() ?
        reinterpret_cast<elem_unsigned_type*>(table_ptr + (static_cast<std::size_t>(2 * node_width) * node_index_)) :
        reinterpret_cast<elem_unsigned_type*>(allocator_.address(node_index_));
    do_store<atomic, release>(node_ptr);
  }
  template <bool atomic, bool release>
  DEVICE_QUALIFIER void do_store(elem_unsigned_type* node_ptr) {
    if constexpr (atomic) { tile_.sync(); }
    utils::memory::store<elem_unsigned_type, atomic, release>(node_ptr + tile_.thread_rank(), *reinterpret_cast<elem_unsigned_type*>(&lane_elem_));
    if constexpr (atomic) { tile_.sync(); }
  }

  DEVICE_QUALIFIER void read_metadata_from_registers() {
    metadata_ = tile_.shfl(lane_elem_.key, metadata_lane_);
  }
  DEVICE_QUALIFIER void write_metadata_to_registers() {
    if (tile_.thread_rank() == metadata_lane_) {
      lane_elem_.key = metadata_;
    }
  }

  DEVICE_QUALIFIER key_type get_key_from_location(const int location) const {
    return tile_.shfl(lane_elem_.key, location);
  }
  DEVICE_QUALIFIER value_type get_value_from_location(const int location) const {
    return tile_.shfl(lane_elem_.value, location);
  }
  DEVICE_QUALIFIER bool is_valid_lane() const {
    return tile_.thread_rank() < num_keys();
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
  DEVICE_QUALIFIER bool is_mergeable(const hashtable_node_subwarp& next_node) const {
    return (num_keys() + next_node.num_keys()) <= max_num_keys_;
  }
  DEVICE_QUALIFIER bool get_suffix_of_location(int location) const {
    return (metadata_ >> (suffix_bits_offset_ + location)) & 1u;
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
    return tile_.shfl(lane_elem_.value, next_ptr_lane_);
  }
  DEVICE_QUALIFIER void set_next_index(const value_type& index) {
    if (tile_.thread_rank() == next_ptr_lane_) { lane_elem_.value = index; }
  }
  DEVICE_QUALIFIER bool is_head() const {
    return static_cast<bool>(metadata_ & head_bit_mask_);
  }
  DEVICE_QUALIFIER bool is_garbage() const {
    return static_cast<bool>(metadata_ & garbage_bit_mask_);
  }
  DEVICE_QUALIFIER void make_garbage() {
    metadata_ |= garbage_bit_mask_;
    write_metadata_to_registers();
  }
  DEVICE_QUALIFIER size_type get_local_depth() const {
    return ((metadata_ & local_depth_bits_mask_) >> local_depth_bits_offset_);
  }
  DEVICE_QUALIFIER void set_local_depth(size_type local_depth) {
    metadata_ &= ~local_depth_bits_mask_;
    metadata_ |= (local_depth << local_depth_bits_offset_);
    write_metadata_to_registers();
  }

  DEVICE_QUALIFIER size_type get_node_index() const { return node_index_; }

  // lock/unlock for head nodes embedded in array (chainHT, cuckooHT)
  static DEVICE_QUALIFIER bool try_lock(key_type* table_ptr, size_type bucket_index, const tile_type& tile) {
    key_type old;
    auto bucket_ptr = reinterpret_cast<elem_type*>(table_ptr + (static_cast<std::size_t>(2 * node_width) * bucket_index));
    if (tile.thread_rank() == metadata_lane_) {
      cuda::atomic_ref<key_type, cuda::thread_scope_device> metadata_ref(bucket_ptr[metadata_lane_].key);
      old = metadata_ref.fetch_or(lock_bit_mask_, cuda::memory_order_relaxed);
    }
    // if previously not locked, now it's locked
    bool is_locked = (tile.shfl(old, metadata_lane_) & lock_bit_mask_) == 0;
    if (is_locked) { cuda::atomic_thread_fence(cuda::memory_order_acquire, cuda::thread_scope_device); }
    return is_locked;
  }
  static DEVICE_QUALIFIER void lock(key_type* table_ptr, size_type bucket_index, const tile_type& tile) {
    while (!try_lock(table_ptr, bucket_index, tile));
  }
  template <bool release = true>
  static DEVICE_QUALIFIER void unlock(key_type* table_ptr, size_type bucket_index, const tile_type& tile) {
    // unlock, only using the pointer, not load the entire register
    auto bucket_ptr = reinterpret_cast<elem_type*>(table_ptr + (static_cast<std::size_t>(2 * node_width) * bucket_index));
    if (tile.thread_rank() == metadata_lane_) {
      if constexpr (release) {
        cuda::atomic_ref<key_type, cuda::thread_scope_device> metadata_ref(bucket_ptr[metadata_lane_].key);
        metadata_ref.fetch_and(~lock_bit_mask_, cuda::memory_order_release);
      }
      else {
        bucket_ptr[metadata_lane_].key &= ~lock_bit_mask_;
      }
    }
  }
  // lock/unlock for head nodes in slab allocator (linearHT)
  static DEVICE_QUALIFIER bool try_lock(size_type head_index, const tile_type& tile, allocator_type& allocator) {
    key_type old;
    auto bucket_ptr = reinterpret_cast<elem_type*>(allocator.address(head_index));
    if (tile.thread_rank() == metadata_lane_) {
      cuda::atomic_ref<key_type, cuda::thread_scope_device> metadata_ref(bucket_ptr[metadata_lane_].key);
      old = metadata_ref.fetch_or(lock_bit_mask_, cuda::memory_order_relaxed);
    }
    // if previously not locked, now it's locked
    bool is_locked = (tile.shfl(old, metadata_lane_) & lock_bit_mask_) == 0;
    if (is_locked) { cuda::atomic_thread_fence(cuda::memory_order_acquire, cuda::thread_scope_device); }
    return is_locked;
  }
  static DEVICE_QUALIFIER void lock(size_type head_index, const tile_type& tile, allocator_type& allocator) {
    while (!try_lock(head_index, tile, allocator));
  }
  template <bool release = true>
  static DEVICE_QUALIFIER void unlock(size_type head_index, const tile_type& tile, allocator_type& allocator) {
    // unlock, only using the pointer, not load the entire register
    auto bucket_ptr = reinterpret_cast<elem_type*>(allocator.address(head_index));
    if (tile.thread_rank() == metadata_lane_) {
      if constexpr (release) {
        cuda::atomic_ref<key_type, cuda::thread_scope_device> metadata_ref(bucket_ptr[metadata_lane_].key);
        metadata_ref.fetch_and(~lock_bit_mask_, cuda::memory_order_release);
      }
      else {
        bucket_ptr[metadata_lane_].key &= ~lock_bit_mask_;
      }
    }
  }

  DEVICE_QUALIFIER uint32_t match_key_in_node(const key_type& key, bool more_key) const {
    return tile_.ballot(is_valid_lane() &&
                        lane_elem_.key == key &&
                        get_suffix_of_location(tile_.thread_rank()) == more_key);
  }
  DEVICE_QUALIFIER uint32_t match_key_value_in_node(const key_type& key, const value_type& value, bool more_key) const {
    return tile_.ballot(is_valid_lane() &&
                        lane_elem_.key == key &&
                        lane_elem_.value == value &&
                        get_suffix_of_location(tile_.thread_rank()) == more_key);
  }

  DEVICE_QUALIFIER void insert(const key_type& key, const value_type& value, bool more_key) {
    assert(!is_full());
    auto location = num_keys();
    if (tile_.thread_rank() == location) {
      lane_elem_ = {key, value};
    }
    set_suffix_of_location(location, more_key);
    metadata_++;    // equiv. to num_keys++
    write_metadata_to_registers();
  }

  DEVICE_QUALIFIER void update(int location, const value_type& value) {
    assert(location < num_keys());
    if (tile_.thread_rank() == location) {
      lane_elem_.value = value;
    }
  }

  DEVICE_QUALIFIER void merge(const hashtable_node_subwarp<tile_type, allocator_type>& next_node) {
    assert(is_mergeable(next_node));
    // copy elements from next node
    bool suffix_bit = get_suffix_of_location(tile_.thread_rank());
    elem_type shifted_elem = tile_.shfl_up(next_node.lane_elem_, num_keys());
    bool shifted_suffix_bit = next_node.get_suffix_of_location(tile_.thread_rank() - num_keys());
    auto new_num_keys = num_keys() + next_node.num_keys();
    if (num_keys() <= tile_.thread_rank() && tile_.thread_rank() < new_num_keys) {
      lane_elem_ = shifted_elem;
      suffix_bit = shifted_suffix_bit;
    }
    set_num_keys(new_num_keys);
    auto new_suffix_bits = tile_.ballot(suffix_bit);
    metadata_ &= ~suffix_bits_mask_;
    metadata_ |= ((new_suffix_bits << suffix_bits_offset_) & suffix_bits_mask_);
    // copy next node's next index
    if (tile_.thread_rank() == next_ptr_lane_) {
      lane_elem_.value = next_node.lane_elem_.value;
    }
    // has_next = next.has_next
    metadata_ &= ~next_bit_mask_;
    metadata_ |= (next_node.metadata_ & next_bit_mask_);
    write_metadata_to_registers();
  }

  DEVICE_QUALIFIER void erase(int location) {
    assert(location < num_keys());
    metadata_--;    // equiv. to num_keys--
    bool suffix_bit = get_suffix_of_location(tile_.thread_rank());
    auto down_elem = tile_.shfl_down(lane_elem_, 1);
    auto down_suffix_bit = get_suffix_of_location(tile_.thread_rank() + 1);
    if (is_valid_lane()) {
      if (tile_.thread_rank() >= location) {
        lane_elem_ = down_elem;
        suffix_bit = down_suffix_bit;
      }
    }
    auto new_suffix_bits = tile_.ballot(suffix_bit);
    metadata_ &= ~suffix_bits_mask_;
    metadata_ |= ((new_suffix_bits << suffix_bits_offset_) & suffix_bits_mask_);
    write_metadata_to_registers();
  }

  DEVICE_QUALIFIER hashtable_node_subwarp<tile_type, allocator_type>& operator=(
        const hashtable_node_subwarp<tile_type, allocator_type>& other) {
    node_index_ = other.node_index_;
    lane_elem_ = other.lane_elem_;
    metadata_ = other.metadata_;
    return *this;
  }

  DEVICE_QUALIFIER void print() const {
    bool lead_lane = (tile_.thread_rank() == 0);
    if (lead_lane) printf("node[%u]: {", node_index_);
    if (num_keys() > max_num_keys_) {
      if (lead_lane) printf("num_keys too large: skip}\n");
      return;
    }
    if (num_keys() == 0) {
      if (lead_lane) printf("empty}\n");
      return;
    }
    if (is_head()) {
      if (lead_lane) printf("head ");
    }
    if (is_garbage()) {
      if (lead_lane) printf("garbage ");
    }
    if (lead_lane) printf("ld(%u) ", get_local_depth());
    if (lead_lane) printf("%u ", num_keys());
    for (size_type i = 0; i < num_keys(); ++i) {
      key_type key = get_key_from_location(i);
      value_type value = get_value_from_location(i);
      bool suffix_bit = get_suffix_of_location(i);
      if (lead_lane) printf("(%u %u %s) ", key, value, suffix_bit ? "s" : "$");
    }
    if (lead_lane) printf("%s ", (metadata_ & lock_bit_mask_) ? "locked" : "free");
    size_type next_index = get_next_index();
    if (has_next()) {
      if (lead_lane) printf("next(%u)", next_index);
    }
    else {
      if (lead_lane) printf("nullnext");
    }
    if (lead_lane) printf("}\n");
    for (size_type i = 0; i < num_keys(); ++i) {
      bool suffix_bit = get_suffix_of_location(i);
      if (suffix_bit) {
        size_type suffix_index = get_value_from_location(i);
        auto suffix = suffix_node_subwarp<tile_type, allocator_type>(suffix_index, tile_, allocator_);
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
  //  [key,val0] [key,val1] ... [key,val14] | [metadata,next]

  // metadata is 32bits.
  //    (MSB)
  //    [empty:3]
  //    [local_depth:6]
  //    [key_suffix_bits_per_key:15]
  //    [is_garbage:1][is_head:1]
  //    [has_next:1][is_locked:1]
  //    [num_keys:4]
  //    (LSB)
  //  - is_garbage, is_locked are only valid for head nodes
  //  - local_depth for the node chain is all same

  static_assert(sizeof(elem_type) == sizeof(uint64_t));
  static constexpr uint32_t metadata_lane_ = node_width - 1;  // key
  static constexpr uint32_t next_ptr_lane_ = node_width - 1;  // value
  static constexpr uint32_t num_keys_offset_ = 0;
  static constexpr uint32_t num_keys_bits_ = 4;
  static constexpr uint32_t num_keys_mask_ = ((1u << num_keys_bits_) - 1) << num_keys_offset_;
  static constexpr uint32_t lock_bit_offset_ = 4;
  static constexpr uint32_t lock_bit_mask_ = 1u << lock_bit_offset_;
  static constexpr uint32_t next_bit_offset_ = 5;
  static constexpr uint32_t next_bit_mask_ = 1u << next_bit_offset_;
  static constexpr uint32_t head_bit_offset_ = 6;
  static constexpr uint32_t head_bit_mask_ = 1u << head_bit_offset_;
  static constexpr uint32_t garbage_bit_offset_ = 7;
  static constexpr uint32_t garbage_bit_mask_ = 1u << garbage_bit_offset_;
  static constexpr uint32_t suffix_bits_offset_ = 8;
  static constexpr uint32_t suffix_bits_bits_ = 15;
  static constexpr uint32_t suffix_bits_mask_ = ((1u << suffix_bits_bits_) - 1) << suffix_bits_offset_;
  static constexpr uint32_t local_depth_bits_offset_ = 23;
  static constexpr uint32_t local_depth_bits_bits_ = 6;
  static constexpr uint32_t local_depth_bits_mask_ = ((1u << local_depth_bits_bits_) - 1) << local_depth_bits_offset_;
  static constexpr uint32_t max_num_keys_ = node_width - 1;
  static_assert(num_keys_offset_ == 0); // this allows (metadata +/- N) equivalent to (num_keys +/- N) within range
  static_assert(max_num_keys_ == capacity);

  uint32_t metadata_;
};
