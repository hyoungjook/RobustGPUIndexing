/*
 *   Copyright 2026 Hyoungjoo Kim, Carnegie Mellon University
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

template <typename tile_type, typename allocator_type>
struct suffix_node_subwarp {
  using size_type = uint32_t;
  using slice_type = uint32_t;
  struct __align__(8) elem_type {
    slice_type first, second;
  };
  using elem_unsigned_type = uint64_t;
  static constexpr int node_width = 16;
  static_assert(tile_type::size() == node_width);
  DEVICE_QUALIFIER suffix_node_subwarp(const tile_type& tile, allocator_type& allocator): tile_(tile), allocator_(allocator) {}
  DEVICE_QUALIFIER suffix_node_subwarp(size_type index, const tile_type& tile, allocator_type& allocator)
      : node_index_(index)
      , tile_(tile)
      , allocator_(allocator) {}
  
  // ALL suffix loads/stores in this file are done as non-atomic, because
  //  - suffix loads are done with the pointer in tree/bucket node, which is loaded with memory_order_acquire
  //  - suffix stores are protected by tree/bucket node's locks, which includes threadfence with memory_order_release
  DEVICE_QUALIFIER void load_head() {
    auto node_ptr = reinterpret_cast<elem_unsigned_type*>(allocator_.address(node_index_));
    auto elem = utils::memory::load<elem_unsigned_type, false>(node_ptr + tile_.thread_rank());
    lane_elem_ = *reinterpret_cast<elem_type*>(&elem);
  }
  DEVICE_QUALIFIER void store_head() {
    auto node_ptr = reinterpret_cast<elem_unsigned_type*>(allocator_.address(node_index_));
    utils::memory::store<elem_unsigned_type, false>(node_ptr + tile_.thread_rank(), *reinterpret_cast<elem_unsigned_type*>(&lane_elem_));
  }

  DEVICE_QUALIFIER size_type get_node_index() const {
    return node_index_;
  }

  DEVICE_QUALIFIER size_type get_next() const {
    return tile_.shfl(lane_elem_.second, next_lane_);
  }
  DEVICE_QUALIFIER size_type get_key_length() const {
    return tile_.shfl(lane_elem_.first, head_node_length_lane_);
  }
  DEVICE_QUALIFIER void set_length(const size_type& key_length) {
    if (tile_.thread_rank() == head_node_length_lane_) {
      lane_elem_.first = key_length;
    }
  }
  DEVICE_QUALIFIER slice_type get_value() const {
    return tile_.shfl(lane_elem_.first, head_node_value_lane_);
  }
  DEVICE_QUALIFIER void update_value(const slice_type& value) {
    if (tile_.thread_rank() == head_node_value_lane_) {
      lane_elem_.first = value;
    }
  }

  DEVICE_QUALIFIER uint32_t get_num_nodes() const {
    auto length = get_key_length();
    // first two elements are length and value, so (length + 2)
    return ((length + 2) + node_max_len_ - 1) / node_max_len_;
  }

  DEVICE_QUALIFIER bool streq(const slice_type* key, uint32_t key_length) const {
    if (get_key_length() != key_length) { return false; }
    // now key_length == this_key_length, compare head node
    // ignore first two slices in head
    key -= 2;
    key_length += 2;
    uint32_t skip_elems = 2;
    auto elem = lane_elem_;
    while (true) {
      // elem.first
      bool mismatch_first = (skip_elems <= tile_.thread_rank()) &&
                            (tile_.thread_rank() < node_max_len_) &&
                            (tile_.thread_rank() < key_length) && 
                            (elem.first != key[tile_.thread_rank()]);
      bool mismatch_second = (node_width + tile_.thread_rank() < node_max_len_) &&
                             (node_width + tile_.thread_rank() < key_length) && 
                             (elem.second != key[node_width + tile_.thread_rank()]);
      uint32_t mismatch_ballot = tile_.ballot(mismatch_first || mismatch_second);
      if (mismatch_ballot != 0) { return false; }
      if (key_length <= node_max_len_) { return true; }
      key_length -= node_max_len_;
      key += node_max_len_;
      auto next_index = tile_.shfl(elem.second, next_lane_);
      auto* next_ptr = reinterpret_cast<elem_unsigned_type*>(allocator_.address(next_index));
      auto tmp_elem = utils::memory::load<elem_unsigned_type, false>(next_ptr + tile_.thread_rank());
      elem = *reinterpret_cast<elem_type*>(&tmp_elem);
      skip_elems = 0;
    }
    assert(false);
  }

  DEVICE_QUALIFIER int strcmp(const slice_type* key, uint32_t key_length, slice_type* mismatch_value = nullptr) const {
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
      // compare elem.first
      bool this_more = (tile_.thread_rank() < this_length - 1);
      bool key_more = (tile_.thread_rank() < key_length - 1);
      bool valid_cmp = (skip_elems <= tile_.thread_rank()) &&
                       (tile_.thread_rank() < cmp_length);
      slice_type other = valid_cmp ? (key[tile_.thread_rank()]) : 0;
      bool match = valid_cmp &&
                   (elem.first == other) &&
                   (this_more == key_more);
      match = match || (tile_.thread_rank() < skip_elems);
      uint32_t match_ballot = tile_.ballot(match);
      int num_matches = __ffs(~match_ballot) - 1;
      // if all elements match
      if (num_matches == cmp_length) { return 0; }
      // if found mismatch in this node
      total_num_matches += (num_matches - skip_elems);
      if (num_matches < node_width && num_matches < cmp_length) {
        // num_matches'th lane has the first mismatch elems
        bool ge = tile_.shfl((elem.first < other) || (elem.first == other && this_more < key_more), num_matches);
        if (mismatch_value) { *mismatch_value = tile_.shfl(elem.first, num_matches); }
        return (ge ? 1 : -1) * static_cast<int>(1 + total_num_matches);
      }
      // compare elem.second
      cmp_length -= node_width;
      this_length -= node_width;
      key_length -= node_width;
      key += node_width;
      this_more = (tile_.thread_rank() < this_length - 1);
      key_more = (tile_.thread_rank() < key_length - 1);
      valid_cmp = (tile_.thread_rank() < (node_max_len_ - node_width)) &&
                  (tile_.thread_rank() < cmp_length);
      other = valid_cmp ? (key[tile_.thread_rank()]) : 0;
      match = valid_cmp &&
              (elem.second == other) &&
              (this_more == key_more);
      match_ballot = tile_.ballot(match);
      num_matches = __ffs(~match_ballot) - 1;
      // if all elements match
      if (num_matches == cmp_length) { return 0; }
      // if found mismatch in this node
      total_num_matches += num_matches;
      if (num_matches < (node_max_len_ - node_width) && num_matches < cmp_length) {
        // num_matches'th lane has the first mismatch elems
        bool ge = tile_.shfl((elem.second < other) || (elem.second == other && this_more < key_more), num_matches);
        if (mismatch_value) { *mismatch_value = tile_.shfl(elem.second, num_matches); }
        return (ge ? 1 : -1) * static_cast<int>(1 + total_num_matches);
      }
      // proceed to next node
      cmp_length -= (node_max_len_ - node_width);
      this_length -= (node_max_len_ - node_width);
      key_length -= (node_max_len_ - node_width);
      key += (node_max_len_ - node_width);
      auto next_index = tile_.shfl(elem.second, next_lane_);
      auto* next_ptr = reinterpret_cast<elem_unsigned_type*>(allocator_.address(next_index));
      auto tmp_elem = utils::memory::load<elem_unsigned_type, false>(next_ptr + tile_.thread_rank());
      elem = *reinterpret_cast<elem_type*>(&tmp_elem);
      skip_elems = 0;
    }
    assert(false);
  }

  template <uint32_t prime0>
  DEVICE_QUALIFIER uint32_t compute_polynomial() const {
    // compute (1 * s[0]) + (p * s[1]) + (p^2 * s[2]) + ... + (p^(l-1) * s[l-1])
    // 1. exponent = [1, p, p^2, ..., p^15]
    uint32_t exponent0 = (tile_.thread_rank() == 0) ? 1 : prime0;
    for (uint32_t offset = 1; offset < node_width; offset <<= 1) {
      auto up_exponent0 = tile_.shfl_up(exponent0, offset);
      if (tile_.thread_rank() >= offset) {
        exponent0 *= up_exponent0;
      }
    }
    // prime_multiplier = p^15, p^16
    static constexpr uint32_t prime0_multiplier15 = utils::constexpr_pow(prime0, node_width - 1);
    static constexpr uint32_t prime0_multiplier16 = utils::constexpr_pow(prime0, node_width);
    // 2. compute per-lane value
    auto this_length = get_key_length();
    uint32_t hash = 0;
    // ignore first two elements in head;
    //  also make exponent [p^14, p^15, 1, p, p^2, ..., p^13]
    {
      auto shifted_exponent = tile_.shfl_down(exponent0, node_width -2);
      exponent0 = tile_.shfl_up(exponent0, 2);
      if (tile_.thread_rank() < 2) { exponent0 = shifted_exponent; }
    }
    this_length += 2;
    uint32_t skip_elems = 2;
    auto elem = lane_elem_;
    while (true) {
      if (skip_elems <= tile_.thread_rank() &&
          tile_.thread_rank() < this_length) {
        hash += exponent0 * elem.first;
      }
      if (this_length <= node_width) { break; }
      this_length -= node_width;
      if (skip_elems <= tile_.thread_rank()) {
        exponent0 *= prime0_multiplier16;
      }
      if (tile_.thread_rank() < (node_max_len_ - node_width) &&
          tile_.thread_rank() < this_length) {
        hash += exponent0 * elem.second;
      }
      if (this_length <= (node_max_len_ - node_width)) { break; }
      this_length -= (node_max_len_ - node_width);
      auto next_index = tile_.shfl(elem.second, next_lane_);
      auto* next_ptr = reinterpret_cast<elem_unsigned_type*>(allocator_.address(next_index));
      auto tmp_elem = utils::memory::load<elem_unsigned_type, false>(next_ptr + tile_.thread_rank());
      elem = *reinterpret_cast<elem_type*>(&tmp_elem);
      exponent0 *= prime0_multiplier15;
      skip_elems = 0;
    }
    // 3. reduce sum
    for (uint32_t offset = (node_width / 2); offset != 0; offset >>= 1) {
      hash += tile_.shfl_down(hash, offset);
    }
    return tile_.shfl(hash, 0);
  }
  template <uint32_t prime0, uint32_t prime1>
  DEVICE_QUALIFIER uint2 compute_polynomialx2() const {
    // compute (1 * s[0]) + (p * s[1]) + (p^2 * s[2]) + ... + (p^(l-1) * s[l-1])
    // 1. exponent = [1, p, p^2, ..., p^15]
    uint32_t exponent0 = (tile_.thread_rank() == 0) ? 1 : prime0;
    uint32_t exponent1 = (tile_.thread_rank() == 0) ? 1 : prime1;
    for (uint32_t offset = 1; offset < node_width; offset <<= 1) {
      auto up_exponent0 = tile_.shfl_up(exponent0, offset);
      auto up_exponent1 = tile_.shfl_up(exponent1, offset);
      if (tile_.thread_rank() >= offset) {
        exponent0 *= up_exponent0;
        exponent1 *= up_exponent1;
      }
    }
    // prime_multiplier = p^15, p^16
    static constexpr uint32_t prime0_multiplier15 = utils::constexpr_pow(prime0, node_width - 1);
    static constexpr uint32_t prime0_multiplier16 = utils::constexpr_pow(prime0, node_width);
    static constexpr uint32_t prime1_multiplier15 = utils::constexpr_pow(prime1, node_width - 1);
    static constexpr uint32_t prime1_multiplier16 = utils::constexpr_pow(prime1, node_width);
    // 2. compute per-lane value
    auto this_length = get_key_length();
    uint32_t hash = 0, hash1 = 0;
    // ignore first two elements in head;
    //  also make exponent [p^14, p^15, 1, p, p^2, ..., p^13]
    {
      auto shifted_exponent = tile_.shfl_down(exponent0, node_width -2);
      exponent0 = tile_.shfl_up(exponent0, 2);
      if (tile_.thread_rank() < 2) { exponent0 = shifted_exponent; }
      shifted_exponent = tile_.shfl_down(exponent1, node_width -2);
      exponent1 = tile_.shfl_up(exponent1, 2);
      if (tile_.thread_rank() < 2) { exponent1 = shifted_exponent; }
    }
    this_length += 2;
    uint32_t skip_elems = 2;
    auto elem = lane_elem_;
    while (true) {
      if (skip_elems <= tile_.thread_rank() &&
          tile_.thread_rank() < this_length) {
        hash += exponent0 * elem.first;
        hash1 += exponent1 * elem.first;
      }
      if (this_length <= node_width) { break; }
      this_length -= node_width;
      if (skip_elems <= tile_.thread_rank()) {
        exponent0 *= prime0_multiplier16;
        exponent1 *= prime1_multiplier16;
      }
      if (tile_.thread_rank() < (node_max_len_ - node_width) &&
          tile_.thread_rank() < this_length) {
        hash += exponent0 * elem.second;
        hash1 += exponent1 * elem.second;
      }
      if (this_length <= (node_max_len_ - node_width)) { break; }
      this_length -= (node_max_len_ - node_width);
      auto next_index = tile_.shfl(elem.second, next_lane_);
      auto* next_ptr = reinterpret_cast<elem_unsigned_type*>(allocator_.address(next_index));
      auto tmp_elem = utils::memory::load<elem_unsigned_type, false>(next_ptr + tile_.thread_rank());
      elem = *reinterpret_cast<elem_type*>(&tmp_elem);
      exponent0 *= prime0_multiplier15;
      exponent1 *= prime1_multiplier15;
      skip_elems = 0;
    }
    // 3. reduce sum
    for (uint32_t offset = (node_width / 2); offset != 0; offset >>= 1) {
      hash += tile_.shfl_down(hash, offset);
      hash1 += tile_.shfl_down(hash1, offset);
    }
    return make_uint2(tile_.shfl(hash, 0), tile_.shfl(hash1, 0));
  }

  DEVICE_QUALIFIER void create_from(const slice_type* key, size_type key_length, slice_type value) {
    // head node metadata
    elem_type elem;
    elem_unsigned_type* curr_ptr = nullptr;  // NULL if head, else appendix
    if (tile_.thread_rank() == head_node_length_lane_) { elem.first = key_length; }
    if (tile_.thread_rank() == head_node_value_lane_) { elem.first = value; }
    // ignore first two elements in head
    key -= 2;
    key_length += 2;
    uint32_t skip_elems = 2;
    while (true) {
      // set elem.first
      if (skip_elems <= tile_.thread_rank() &&
          tile_.thread_rank() < key_length) {
        elem.first = key[tile_.thread_rank()];
      }
      // set elem.second
      if (tile_.thread_rank() < (node_max_len_ - node_width) &&
          (node_width + tile_.thread_rank()) < key_length) {
        elem.second = key[node_width + tile_.thread_rank()];
      }
      elem_unsigned_type* next_ptr;
      if (key_length > node_max_len_) {
        auto next_index = allocator_.allocate(tile_);
        if (tile_.thread_rank() == next_lane_) { elem.second = next_index; }
        next_ptr = reinterpret_cast<elem_unsigned_type*>(allocator_.address(next_index));
      }
      // store
      if (curr_ptr) { // !is_head
        utils::memory::store<elem_unsigned_type, false>(curr_ptr + tile_.thread_rank(), *reinterpret_cast<elem_unsigned_type*>(&elem));
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

  DEVICE_QUALIFIER void flush(slice_type* key_buffer) {
    auto this_length = get_key_length();
    auto elem = lane_elem_;
    // ignore first two elements in head
    key_buffer -= 2;
    this_length += 2;
    uint32_t skip_elems = 2;
    while (true) {
      auto count = min(this_length, node_max_len_);
      if (skip_elems <= tile_.thread_rank() &&
          tile_.thread_rank() < count) {
        key_buffer[tile_.thread_rank()] = elem.first;
      }
      if ((node_width + tile_.thread_rank()) < count) {
        key_buffer[node_width + tile_.thread_rank()] = elem.second;
      }
      this_length -= count;
      if (this_length == 0) { break; }
      key_buffer += count;
      auto next_index = tile_.shfl(elem.second, next_lane_);
      auto* next_ptr = reinterpret_cast<elem_unsigned_type*>(allocator_.address(next_index));
      auto tmp_elem = utils::memory::load<elem_unsigned_type, false>(next_ptr + tile_.thread_rank());
      elem = *reinterpret_cast<elem_type*>(&tmp_elem);
      skip_elems = 0;
    }
  }

  template <typename reclaimer_type>
  DEVICE_QUALIFIER void move_from(suffix_node_subwarp<tile_type, allocator_type>& src,
                                  uint32_t offset,
                                  reclaimer_type& reclaimer) {
    // move elements from src[offset:] and retire all nodes in src
    auto new_length = src.get_key_length() - offset;
    auto value = src.get_value();
    // skip src nodes until the first element
    reclaimer.retire(src.get_node_index(), tile_);
    while (offset >= node_max_len_) {
      src.node_index_ = src.get_next();
      src.load_head();
      reclaimer.retire(src.node_index_, tile_);
      offset -= node_max_len_;
    }
    // copy elements into this
    elem_type dst_lane_elem;
    elem_unsigned_type* dst_ptr = nullptr; // NULL means it's head
    if (tile_.thread_rank() == head_node_length_lane_) { dst_lane_elem.first = new_length; }
    if (tile_.thread_rank() == head_node_value_lane_) { dst_lane_elem.first = value; }
    // ignore first two elements in head
    new_length += 2;
    uint32_t skip_elems = 2;
    while (true) {
      // phase 1. copy src[offset:node_max_len) -> dst[0:node_max_len-offset)
      uint32_t copy_count = min(new_length, node_max_len_ - offset);
      slice_type shift_elem;
      //  simply dst <- shfl_down(src, offset) (for skip_elems <= dst_lane)
      //  but consider transposed half-warp layout
      if (offset < node_width) {
        shift_elem = tile_.shfl_down(src.lane_elem_.first, offset);
        if (skip_elems <= tile_.thread_rank()) {
          dst_lane_elem.first = shift_elem;
        }
        shift_elem = tile_.shfl_down(src.lane_elem_.second, offset);
        dst_lane_elem.second = shift_elem;
        shift_elem = tile_.shfl_up(src.lane_elem_.second, node_width - offset);
        if (skip_elems <= tile_.thread_rank() &&
            node_width - offset <= tile_.thread_rank()) {
          dst_lane_elem.first = shift_elem;
        }
      }
      else {
        shift_elem = tile_.shfl_down(src.lane_elem_.second, offset - node_width);
        if (skip_elems <= tile_.thread_rank()) {
          dst_lane_elem.first = shift_elem;
        }
      }
      new_length -= copy_count;
      if (new_length == 0) { break; }
      // phase 2. copy src.next[0:offset) -> dst[node_max_len-offset:node_max_len)
      src.node_index_ = src.get_next();
      src.load_head();
      reclaimer.retire(src.node_index_, tile_);
      if (offset > 0) {
        copy_count = min(new_length, offset);
        //  simply dst <- shfl_up(src, node_max_len_ - offset)
        //    (for max(skip_elems, node_max_len_ - offset) <= dst_lane)
        //  but consider transposed half-warp layout
        auto shift_offset = node_max_len_ - offset;
        if (shift_offset < node_width) {
          shift_elem = tile_.shfl_up(src.lane_elem_.first, shift_offset);
          if (max(shift_offset, skip_elems) <= tile_.thread_rank()) {
            dst_lane_elem.first = shift_elem;
          }
          shift_elem = tile_.shfl_up(src.lane_elem_.second, shift_offset);
          dst_lane_elem.second = shift_elem;
          shift_elem = tile_.shfl_down(src.lane_elem_.first, node_width - shift_offset);
          if (tile_.thread_rank() < shift_offset) {
            dst_lane_elem.second = shift_elem;
          }
        }
        else {
          shift_elem = tile_.shfl_up(src.lane_elem_.first, shift_offset - node_width);
          if (shift_offset - node_width <= tile_.thread_rank()) {
            dst_lane_elem.second = shift_elem;
          }
        }
        new_length -= copy_count;
        if (new_length == 0) { break; }
      }
      skip_elems = 0;
      // phase 3. store dst & allocate dst.next
      auto dst_index = allocator_.allocate(tile_);
      if (tile_.thread_rank() == next_lane_) {
        dst_lane_elem.second = dst_index;
      }
      if (dst_ptr) {
        utils::memory::store<elem_unsigned_type, false>(dst_ptr + tile_.thread_rank(), *reinterpret_cast<elem_unsigned_type*>(&dst_lane_elem));
      }
      else {
        lane_elem_ = dst_lane_elem;
      }
      dst_ptr = reinterpret_cast<elem_unsigned_type*>(allocator_.address(dst_index));
    }
    // flush dst_lane_elem
    if (dst_ptr) {
      utils::memory::store<elem_unsigned_type, false>(dst_ptr + tile_.thread_rank(), *reinterpret_cast<elem_unsigned_type*>(&dst_lane_elem));
    }
    else {
      lane_elem_ = dst_lane_elem;
    }
  }

  template <typename reclaimer_type>
  DEVICE_QUALIFIER void retire(reclaimer_type& reclaimer) {
    reclaimer.retire(node_index_, tile_);
    auto num_nodes = get_num_nodes() - 1;
    auto suffix_index = get_next();
    while (num_nodes > 0) {
      auto* suffix_ptr = reinterpret_cast<slice_type*>(allocator_.address(suffix_index));
      size_type next_index;
      if (tile_.thread_rank() == 0) {
        next_index = utils::memory::load<size_type, false>(suffix_ptr + (2 * next_lane_ + 1));
      }
      next_index = tile_.shfl(next_index, 0);
      reclaimer.retire(suffix_index, tile_);
      suffix_index = next_index;
      num_nodes--;
    }
  }

  static DEVICE_QUALIFIER slice_type fetch_value_only(size_type suffix_index, allocator_type& allocator) {
    auto* ptr = reinterpret_cast<slice_type*>(allocator.address(suffix_index));
    return utils::memory::load<slice_type, false>(ptr + (2 * head_node_value_lane_));
  }

  static DEVICE_QUALIFIER slice_type fetch_length_only(size_type suffix_index, allocator_type& allocator) {
    auto* ptr = reinterpret_cast<slice_type*>(allocator.address(suffix_index));
    return utils::memory::load<slice_type, false>(ptr + (2 * head_node_length_lane_));
  }

  DEVICE_QUALIFIER suffix_node_subwarp<tile_type, allocator_type>& operator=(
      const suffix_node_subwarp<tile_type, allocator_type>& other) {
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
      slice_type key_slice = tile_.shfl(i < node_width ? lane_elem_.first : lane_elem_.second, i % node_width);
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
        auto* next_ptr = reinterpret_cast<elem_unsigned_type*>(allocator_.address(next));
        auto elem = *reinterpret_cast<elem_type*>(&next_ptr[tile_.thread_rank()]);
        for (uint32_t i = 0; i < min(length, node_max_len_); i++) {
          slice_type key_slice = tile_.shfl(i < node_width ? elem.first : elem.second, i % node_width);
          if (lead_lane) printf("%u ", key_slice);
        }
        if (length <= node_max_len_) {
          if (lead_lane) printf("}\n");
          break;
        }
        else {
          next = tile_.shfl(elem.second, next_lane_);
          if (lead_lane) printf("(n=%u)}\n", next);
          length -= node_max_len_;
        }
      }
    }
  }

 private:
  size_type node_index_;
  elem_type lane_elem_;
  const tile_type& tile_;
  allocator_type& allocator_;

  // each node consists of 16 elements, each element stores 2 key slices
  // in total, one node stores up to 31 key slices
  //  [slice0,slice16] [slice1,slice17] ... [slice14,slice30] [slice15,next]
  // note that the order of slices are TRANSPOSED from the one of suffix_node_warp
  // At the head node, slice0=key_length and slice1=value

  static_assert(sizeof(elem_type) == sizeof(uint64_t));
  static constexpr uint32_t head_node_length_lane_ = 0;       // first
  static constexpr uint32_t head_node_value_lane_ = 1;        // first
  static constexpr uint32_t next_lane_ = node_width - 1;      // second
  static constexpr uint32_t node_max_len_ = 2 * node_width - 1;
};
