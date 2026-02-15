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
#include <macros.hpp>
#include <memory_utils.hpp>
#include <utils.hpp>

template <typename tile_type, typename allocator_type>
struct suffix_node {
  using elem_type = uint32_t;
  using size_type = uint32_t;
  static constexpr int node_width = 32;
  DEVICE_QUALIFIER suffix_node(const tile_type& tile, allocator_type& allocator): tile_(tile), allocator_(allocator) {}
  DEVICE_QUALIFIER suffix_node(elem_type* ptr, const size_type index, const tile_type& tile, allocator_type& allocator)
      : node_ptr_(ptr)
      , node_index_(index)
      , tile_(tile)
      , allocator_(allocator) {}

  template <cuda_memory_order order>
  DEVICE_QUALIFIER void load_head() {
    lane_elem_ = cuda_memory<elem_type, order>::load(node_ptr_ + tile_.thread_rank());
  }
  template <cuda_memory_order order>
  DEVICE_QUALIFIER void store_head() {
    cuda_memory<elem_type, order>::store(node_ptr_ + tile_.thread_rank(), lane_elem_);
  }

  DEVICE_QUALIFIER size_type get_node_index() const {
    return node_index_;
  }

  DEVICE_QUALIFIER size_type get_next() const {
    return tile_.shfl(lane_elem_, next_lane_);
  }
  DEVICE_QUALIFIER size_type get_key_length() const {
    return tile_.shfl(lane_elem_, head_node_length_lane_);
  }
  DEVICE_QUALIFIER void set_length(const size_type& key_length) {
    if (tile_.thread_rank() == head_node_length_lane_) {
      lane_elem_ = key_length;
    }
  }
  DEVICE_QUALIFIER elem_type get_value() const {
    return tile_.shfl(lane_elem_, head_node_value_lane_);
  }
  DEVICE_QUALIFIER void update_value(const elem_type& value) {
    if (tile_.thread_rank() == head_node_value_lane_) {
      lane_elem_ = value;
    }
  }

  DEVICE_QUALIFIER uint32_t get_num_nodes() const {
    auto length = get_key_length();
    // first two elements are length and value, so (length + 2)
    return ((length + 2) + node_max_len_ - 1) / node_max_len_;
  }

  template <cuda_memory_order order>
  DEVICE_QUALIFIER bool streq(const elem_type* key, uint32_t key_length) const {
    if (get_key_length() != key_length) { return false; }
    // ignore first two elements in head
    key -= 2;
    key_length += 2;
    uint32_t skip_elems = 2;
    // now key_length == this_key_length, compare head node
    auto elem = lane_elem_;
    while (true) {
      bool mismatch = (skip_elems <= tile_.thread_rank()) &&
                      (tile_.thread_rank() < node_max_len_) &&
                      (tile_.thread_rank() < key_length) && 
                      (elem != key[tile_.thread_rank()]);
      uint32_t mismatch_ballot = tile_.ballot(mismatch);
      if (mismatch_ballot != 0) { return false; }
      if (key_length <= node_max_len_) { return true; }
      key_length -= node_max_len_;
      key += node_max_len_;
      auto next_index = tile_.shfl(elem, next_lane_);
      auto* next_ptr = reinterpret_cast<elem_type*>(allocator_.address(next_index));
      elem = cuda_memory<elem_type, order>::load(next_ptr + tile_.thread_rank());
      skip_elems = 0;
    }
    assert(false);
  }

  template <cuda_memory_order order>
  DEVICE_QUALIFIER int strcmp(const elem_type* key, uint32_t key_length, elem_type* mismatch_value = nullptr) const {
    // strcmp(this, key) -> 0 (match), +(this<key), -(this>key)
    // the absolute of return value: (1 + num_matches)
    // NOTE if one is prefix of the other, num_matches is (len(smaller) - 1)
    auto this_length = get_key_length();
    auto cmp_length = min(this_length, key_length);
    auto elem = lane_elem_;
    int total_num_matches = 0;
    // ignore first two elements in head
    key -= 2;
    key_length += 2;
    this_length += 2;
    cmp_length += 2;
    uint32_t skip_elems = 2;
    while (true) {
      // compare elements
      bool this_more = (tile_.thread_rank() < this_length - 1);
      bool key_more = (tile_.thread_rank() < key_length - 1);
      bool valid_cmp = (tile_.thread_rank() < node_max_len_) &&
                       (tile_.thread_rank() < cmp_length);
      elem_type other = valid_cmp ? (key[tile_.thread_rank()]) : 0;
      bool match = valid_cmp &&
                   (elem == other) &&
                   (this_more == key_more);
      match = match || (tile_.thread_rank() < skip_elems);
      uint32_t match_ballot = tile_.ballot(match);
      int num_matches = __ffs(~match_ballot) - 1;
      // if all elements match
      if (num_matches == cmp_length) { return 0; }
      // if found mismatch in this node
      total_num_matches += (num_matches - skip_elems);
      if (num_matches < node_max_len_ && num_matches < cmp_length) {
        // num_matches'th lane has the first mismatch elems
        bool ge = tile_.shfl((elem < other) || (elem == other && this_more < key_more), num_matches);
        if (mismatch_value) { *mismatch_value = tile_.shfl(elem, num_matches); }
        return (ge ? 1 : -1) * static_cast<int>(1 + total_num_matches);
      }
      // proceed to next node
      cmp_length -= node_max_len_;
      this_length -= node_max_len_;
      key_length -= node_max_len_;
      key += node_max_len_;
      auto next_index = tile_.shfl(elem, next_lane_);
      auto* next_ptr = reinterpret_cast<elem_type*>(allocator_.address(next_index));
      elem = cuda_memory<elem_type, order>::load(next_ptr + tile_.thread_rank());
      skip_elems = 0;
    }
    assert(false);
  }

  template <cuda_memory_order order>
  DEVICE_QUALIFIER void compute_polynomial(const uint32_t prime[2],
                                           uint32_t out_hash[2]) const {
    // compute (1 * s[0]) + (p * s[1]) + (p^2 * s[2]) + ... + (p^(l-1) * s[l-1])
    // 1. exponent = [1, p, p^2, ..., p^31]
    uint32_t exponent0 = (tile_.thread_rank() == 0) ? 1 : prime[0];
    uint32_t exponent1 = (tile_.thread_rank() == 0) ? 1 : prime[1];
    for (uint32_t offset = 1; offset < node_width; offset <<= 1) {
      exponent0 *= tile_.shfl_up(exponent0, offset);
      exponent1 *= tile_.shfl_up(exponent1, offset);
    }
    // prime_multiplier = p^31
    const uint32_t prime0_multiplier = tile_.shfl(exponent0, node_max_len_);
    const uint32_t prime1_multiplier = tile_.shfl(exponent1, node_max_len_);
    // 2. compute per-lane value
    auto this_length = get_key_length();
    out_hash[0] = out_hash[1] = 0;
    // ignore first two elements in head;
    //  also make exponent [p^29, p^30, 1, p, p^2, ..., p^28, x]
    {
      auto shifted_exponent = tile_.shfl_down(exponent0, node_max_len_ -2);
      exponent0 = tile_.shfl_up(exponent0, 2);
      if (tile_.thread_rank() < 2) { exponent0 = shifted_exponent; }
      shifted_exponent = tile_.shfl_down(exponent1, node_max_len_ -2);
      exponent1 = tile_.shfl_up(exponent1, 2);
      if (tile_.thread_rank() < 2) { exponent1 = shifted_exponent; }
    }
    this_length += 2;
    uint32_t skip_elems = 2;
    auto elem = lane_elem_;
    while (true) {
      if (skip_elems <= tile_.thread_rank() &&
          tile_.thread_rank() < node_max_len_ &&
          tile_.thread_rank() < this_length) {
        out_hash[0] += exponent0 * elem;
        out_hash[1] += exponent1 * elem;
      }
      if (this_length <= node_max_len_) { break; }
      this_length -= node_max_len_;
      auto next_index = tile_.shfl(elem, next_lane_);
      auto* next_ptr = reinterpret_cast<elem_type*>(allocator_.address(next_index));
      elem = cuda_memory<elem_type, order>::load(next_ptr + tile_.thread_rank());
      if (skip_elems <= tile_.thread_rank()) {
        exponent0 *= prime0_multiplier;
        exponent1 *= prime1_multiplier;
      }
      skip_elems = 0;
    }
    // 3. reduce sum
    for (uint32_t offset = (node_width / 2); offset != 0; offset >>= 1) {
      out_hash[0] += tile_.shfl_down(out_hash[0], offset);
      out_hash[1] += tile_.shfl_down(out_hash[1], offset);
    }
    out_hash[0] = tile_.shfl(out_hash[0], 0);
    out_hash[1] = tile_.shfl(out_hash[1], 0);
  }

  template <cuda_memory_order order>
  DEVICE_QUALIFIER void create_from(const elem_type* key, size_type key_length, elem_type value) {
    // head node metadata
    elem_type elem;
    elem_type* curr_ptr = nullptr;  // NULL if head, else appendix
    if (tile_.thread_rank() == head_node_length_lane_) { elem = key_length; }
    if (tile_.thread_rank() == head_node_value_lane_) { elem = value; }
    // ignore first two elements in head
    key -= 2;
    key_length += 2;
    uint32_t skip_elems = 2;
    while (true) {
      // set elem
      if (tile_.thread_rank() < min(key_length, node_max_len_)) {
        elem = (tile_.thread_rank() >= skip_elems) ? key[tile_.thread_rank()] : elem;
      }
      elem_type* next_ptr;
      if (key_length > node_max_len_) {
        auto next_index = allocator_.allocate(tile_);
        if (tile_.thread_rank() == next_lane_) { elem = next_index; }
        next_ptr = reinterpret_cast<elem_type*>(allocator_.address(next_index));
      }
      // store
      if (curr_ptr) { // !is_head
        cuda_memory<elem_type, order>::store(curr_ptr + tile_.thread_rank(), elem);
      }
      else {  // is_head
        lane_elem_ = elem;
      }
      // proceed
      if (key_length <= node_max_len_) { break; }
      curr_ptr = next_ptr;
      key_length -= node_max_len_;
      key += node_max_len_;
      skip_elems = 0;
    }
  }

  template <cuda_memory_order order>
  DEVICE_QUALIFIER void flush(elem_type* key_buffer) {
    auto this_length = get_key_length();
    auto elem = lane_elem_;
    // ignore first two elements in head
    key_buffer -= 2;
    this_length += 2;
    uint32_t skip_elems = 2;
    while (true) {
      auto count = min(this_length, node_max_len_);
      if (skip_elems <= tile_.thread_rank() && tile_.thread_rank() < count) {
        key_buffer[tile_.thread_rank()] = elem;
      }
      this_length -= count;
      if (this_length == 0) { break; }
      key_buffer += count;
      auto next_index = tile_.shfl(elem, next_lane_);
      auto* next_ptr = reinterpret_cast<elem_type*>(allocator_.address(next_index));
      elem = cuda_memory<elem_type, order>::load(next_ptr + tile_.thread_rank());
      skip_elems = 0;
    }
  }

  template <cuda_memory_order order, typename reclaimer_type>
  DEVICE_QUALIFIER void move_from(suffix_node<tile_type, allocator_type>& src,
                                  uint32_t offset,
                                  reclaimer_type& reclaimer) {
    // move elements from src[offset:] and retire all nodes in src
    auto new_length = src.get_key_length() - offset;
    auto value = src.get_value();
    // skip src nodes until the first element
    reclaimer.retire(src.get_node_index(), tile_);
    while (offset >= node_max_len_) {
      src.node_index_ = src.get_next();
      src.node_ptr_ = reinterpret_cast<elem_type*>(allocator_.address(src.node_index_));
      src.template load_head<order>();
      reclaimer.retire(src.node_index_, tile_);
      offset -= node_max_len_;
    }
    // copy elements into this
    elem_type dst_lane_elem;
    elem_type* dst_ptr = nullptr; // NULL means it's head
    if (tile_.thread_rank() == head_node_length_lane_) { dst_lane_elem = new_length; }
    if (tile_.thread_rank() == head_node_value_lane_) { dst_lane_elem = value; }
    // ignore first two elements in head
    new_length += 2;
    uint32_t skip_elems = 2;
    while (true) {
      // phase 1. copy src[offset:node_max_len) -> dst[0:node_max_len-offset)
      uint32_t copy_count = min(new_length, node_max_len_ - offset);
      auto down_elem = tile_.shfl_down(src.lane_elem_, offset);
      dst_lane_elem = (tile_.thread_rank() >= skip_elems) ? down_elem : dst_lane_elem;
      new_length -= copy_count;
      if (new_length == 0) { break; }
      // phase 2. copy src.next[0:offset) -> dst[node_max_len-offset:node_max_len)
      src.node_index_ = src.get_next();
      src.node_ptr_ = reinterpret_cast<elem_type*>(allocator_.address(src.node_index_));
      src.template load_head<order>();
      reclaimer.retire(src.node_index_, tile_);
      if (offset > 0) {
        copy_count = min(new_length, offset);
        auto up_src_elem = tile_.shfl_up(src.lane_elem_, node_max_len_ - offset);
        if (node_max_len_ - offset <= tile_.thread_rank() &&
            skip_elems <= tile_.thread_rank()) {
          dst_lane_elem = up_src_elem;
        }
        new_length -= copy_count;
        if (new_length == 0) { break; }
      }
      skip_elems = 0;
      // phase 3. store dst & allocate dst.next
      auto dst_index = allocator_.allocate(tile_);
      if (tile_.thread_rank() == next_lane_) {
        dst_lane_elem = dst_index;
      }
      if (dst_ptr) {
        cuda_memory<elem_type, order>::store(dst_ptr + tile_.thread_rank(), dst_lane_elem);
      }
      else {
        lane_elem_ = dst_lane_elem;
      }
      dst_ptr = reinterpret_cast<elem_type*>(allocator_.address(dst_index));
    }
    // flush dst_lane_elem
    if (dst_ptr) {
      cuda_memory<elem_type, order>::store(dst_ptr + tile_.thread_rank(), dst_lane_elem);
    }
    else {
      if (tile_.thread_rank() < node_max_len_ || tile_.thread_rank() == next_lane_) {
        lane_elem_ = dst_lane_elem;
      }
    }
  }

  template <cuda_memory_order order, typename reclaimer_type>
  DEVICE_QUALIFIER void retire(reclaimer_type& reclaimer) {
    reclaimer.retire(node_index_, tile_);
    auto num_nodes = get_num_nodes() - 1;
    auto suffix_index = get_next();
    while (num_nodes > 0) {
      auto* suffix_ptr = reinterpret_cast<elem_type*>(allocator_.address(suffix_index));
      auto next_index = cuda_memory<elem_type, order>::load(suffix_ptr + next_lane_);
      reclaimer.retire(suffix_index, tile_);
      suffix_index = next_index;
      num_nodes--;
    }
  }

  template <cuda_memory_order order>
  static DEVICE_QUALIFIER elem_type fetch_value_only(size_type suffix_index, allocator_type& allocator) {
    auto* ptr = reinterpret_cast<elem_type*>(allocator.address(suffix_index));
    return cuda_memory<elem_type, order>::load(ptr + head_node_value_lane_);
  }

  template <cuda_memory_order order>
  static DEVICE_QUALIFIER elem_type fetch_length_only(size_type suffix_index, allocator_type& allocator) {
    auto* ptr = reinterpret_cast<elem_type*>(allocator.address(suffix_index));
    return cuda_memory<elem_type, order>::load(ptr + head_node_length_lane_);
  }

  DEVICE_QUALIFIER suffix_node<tile_type, allocator_type>& operator=(
      const suffix_node<tile_type, allocator_type>& other) {
    node_ptr_ = other.node_ptr_;
    node_index_ = other.node_index_;
    lane_elem_ = other.lane_elem_;
    return *this;
  }

  DEVICE_QUALIFIER void print() const {
    bool lead_lane = (tile_.thread_rank() == 0);
    auto length = get_key_length();
    auto value = get_value();
    if (lead_lane) printf("node[%u]: s{l=%u v=%u; ", node_index_, length, value);
    length += 2;
    for (uint32_t i = 2; i < min(length, node_max_len_); i++) {
      elem_type key_slice = tile_.shfl(lane_elem_, i);
      if (lead_lane) printf("%u ", key_slice);
    }
    if (length <= node_max_len_) {
      if (lead_lane) printf("}\n");
    }
    else {
      auto next = get_next();
      if (lead_lane) printf("(n=%u)}\n", next);
      length -= node_max_len_;
      while (true) {
        if (lead_lane) printf("node[%u]: s.{", next);
        auto* next_ptr = reinterpret_cast<elem_type*>(allocator_.address(next));
        auto elem = next_ptr[tile_.thread_rank()];
        for (uint32_t i = 0; i < min(length, node_max_len_); i++) {
          elem_type key_slice = tile_.shfl(elem, i);
          if (lead_lane) printf("%u ", key_slice);
        }
        if (length <= node_max_len_) {
          if (lead_lane) printf("}\n");
          break;
        }
        else {
          next = tile_.shfl(elem, next_lane_);
          if (lead_lane) printf("(n=%u)}\n", next);
          length -= node_max_len_;
        }
      }
    }
  }

 private:
  elem_type* node_ptr_;
  size_type node_index_;
  elem_type lane_elem_;
  const tile_type& tile_;
  allocator_type& allocator_;

  // each node consists of 32 elements and stores up to 31 key slices
  // [key0] [key1] ... [key28] [key29] [key30] [next]
  // At the head node, key0 = key_length and key1 = value

  static_assert(sizeof(elem_type) == sizeof(uint32_t));
  static constexpr uint32_t head_node_length_lane_ = 0;
  static constexpr uint32_t head_node_value_lane_ = 1;
  static constexpr uint32_t next_lane_ = node_width - 1;
  static constexpr uint32_t node_max_len_ = node_width - 1;
};
